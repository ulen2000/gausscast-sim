"""
gc_producer.py
--------------
GaussCast origin producer for the Mini-NDN harness. Serves digest-named segment
Data under the paper's namespace

  /gc/scene/<scid>/ver/<v>/cell/<cid>/layer/<l>/chunk/<k>/seg/<s>/<digest>

The payload of each segment is regenerated deterministically from
``gc_blocks.SceneBlocks`` so the origin is stateless: any segment can be served
on demand and its content matches the digest embedded in the requested name.

The producer counts the segments it actually serves. Because intermediate NFD
forwarders aggregate concurrent Interests in the PIT and satisfy repeats from the
Content Store, this origin count is the *upstream* fetch volume -- the quantity
the paper reports shrinking under shared-base retrieval. On shutdown the count
(and the set of distinct names served) is written to ``--stats`` for the
orchestrator to read.

Pure standard library + python-ndn (matches the node namespace Python).
"""

import argparse
import json
import logging
import os
import signal

from ndn.app import NDNApp
from ndn.encoding import Name
from ndn.security import DigestSha256Signer

from gc_blocks import SceneBlocks

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
)


def parse_seg_name(name_str):
    """Extract (scene, version, cell, layer, seg) from a segment name URI.
    Returns None if the name does not match the GaussCast segment layout."""
    parts = [p for p in name_str.split('/') if p != '']
    # gc scene <scid> ver <v> cell <cid> layer <l> chunk <k> seg <s> <digest>
    try:
        idx = {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1)}
        scene = idx['scene']
        version = idx['ver']
        cell = int(idx['cell'])
        layer = int(idx['layer'])
        seg = int(idx['seg'])
    except (KeyError, ValueError, IndexError):
        return None
    return scene, version, cell, layer, seg


class GaussCastProducer:
    def __init__(self, scene='truck', version='1', n_cells=8,
                 prefix='/gc', stats_path=None):
        os.environ.setdefault('NDN_CLIENT_TRANSPORT', 'tcp://127.0.0.1:6363')
        self.prefix = prefix.rstrip('/')
        self.stats_path = stats_path
        self.scene = scene
        self.version = str(version)
        self.sb = SceneBlocks(scene, version=version, n_cells=n_cells)
        self.app = NDNApp()
        self.served = 0                 # total segments handed out
        self.distinct = set()           # distinct (cell, layer, seg) served

    def on_interest(self, name, _interest_param, _app_param):
        name_str = Name.to_str(name)
        parsed = parse_seg_name(name_str)
        if parsed is None:
            logging.warning(f"unparseable Interest: {name_str}")
            return
        scene, version, cell, layer, seg = parsed
        if scene != self.scene or version != self.version:
            logging.warning(f"scene/ver mismatch: {name_str}")
            return
        payload = self.sb.seg_payload(cell, layer, seg)
        # Serve with the exact requested name (digest component included) so the
        # name->content binding the consumer verifies is preserved end to end.
        self.app.put_data(
            name,
            content=payload,
            freshness_period=600000,
            signer=DigestSha256Signer(),
        )
        self.served += 1
        self.distinct.add((cell, layer, seg))
        # Persist after every serve so the upstream count survives an abrupt
        # shutdown (python-ndn installs its own SIGINT handler).
        self._dump_stats()
        logging.info(f"SERVE {name_str}")

    async def register(self):
        ok = await self.app.register(self.prefix, self.on_interest)
        if not ok:
            raise RuntimeError(f"failed to register {self.prefix}")
        logging.info(f"origin serving {self.scene} under {self.prefix}")

    def _dump_stats(self):
        if not self.stats_path:
            return
        data = {
            'scene': self.scene,
            'version': self.version,
            'served_segments': self.served,
            'distinct_segments': len(self.distinct),
        }
        tmp = self.stats_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(data, fh)
        os.replace(tmp, self.stats_path)        # atomic, never half-written

    def stop(self):
        self._dump_stats()
        self.app.shutdown()

    def run(self):
        def handler(_sig, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        try:
            self.app.run_forever(after_start=self.register())
        except KeyboardInterrupt:
            self.stop()


def main():
    ap = argparse.ArgumentParser(description='GaussCast origin producer')
    ap.add_argument('--scene', default='truck')
    ap.add_argument('--version', default='1')
    ap.add_argument('--n-cells', type=int, default=8)
    ap.add_argument('--prefix', default='/gc')
    ap.add_argument('--stats', default=None, help='write origin counters here')
    args = ap.parse_args()
    GaussCastProducer(
        scene=args.scene, version=args.version, n_cells=args.n_cells,
        prefix=args.prefix, stats_path=args.stats,
    ).run()


if __name__ == '__main__':
    main()
