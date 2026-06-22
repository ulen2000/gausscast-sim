"""
gc_consumer.py
--------------
GaussCast client for the Mini-NDN harness. Reads a per-user plan (an ordered
list of (cell, layer, tag) entries produced by ``gc_planner.plan``), expands it
into segment Interests, and fetches them over NFD.

For every received segment the consumer recomputes SHA-256 over the payload and
checks it against the digest carried in the name's trailing component -- the
digest-linked verification from the paper. Because the name->content binding is
self-certifying, a segment served from any on-path Content Store is accepted
without contacting the origin, yet still verified.

Metrics recorded per run (written to ``--stats`` as JSON):
  * interests           Interests this client expressed
  * received            Data packets received
  * verified            payloads whose digest matched the name
  * ttff_ms             time to first segment (proxy for time-to-first-frame)
  * base_done_ms        time the last base-tag segment arrived (the shared base
                        is what unlocks an initial render in the paper's model)
  * deadline_ms / late  optional soft deadline and how many segments missed it

Pure standard library + python-ndn.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time

from ndn.app import NDNApp
from ndn.encoding import Name
from ndn.types import InterestNack, InterestTimeout

from gc_blocks import SceneBlocks

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
)


def expand_plan(plan_items, sb: SceneBlocks):
    """Expand ordered (cell, layer, tag) blocks into ordered segment requests:
    (name_str, expected_digest, tag). Segment order within a block is ascending,
    preserving the planner's base-first block order."""
    segs = []
    for (cell, layer, tag) in plan_items:
        for s in range(sb.n_segments(cell, layer)):
            segs.append((sb.seg_name(cell, layer, s),
                         sb.seg_digest(cell, layer, s), tag))
    return segs


class GaussCastConsumer:
    def __init__(self, scene='truck', version='1', n_cells=8, plan_path=None,
                 stats_path=None, timeout=4000, deadline_ms=None):
        os.environ.setdefault('NDN_CLIENT_TRANSPORT', 'tcp://127.0.0.1:6363')
        self.sb = SceneBlocks(scene, version=version, n_cells=n_cells)
        self.app = NDNApp()
        self.plan_path = plan_path
        self.stats_path = stats_path
        self.timeout = timeout
        self.deadline_ms = deadline_ms
        self.interests = 0
        self.received = 0
        self.verified = 0
        self.ttff_ms = None
        self.base_done_ms = None
        self.late = 0

    def _load_plan(self):
        with open(self.plan_path) as fh:
            raw = json.load(fh)
        # plan file: list of [cell, layer, tag]
        return [(int(c), int(l), str(t)) for (c, l, t) in raw]

    async def fetch(self):
        plan_items = self._load_plan()
        segs = expand_plan(plan_items, self.sb)
        t0 = time.monotonic()
        try:
            for (name_str, expected, tag) in segs:
                self.interests += 1
                name = Name.from_str(name_str)
                try:
                    _, _, content = await self.app.express_interest(
                        name, must_be_fresh=False, can_be_prefix=False,
                        lifetime=self.timeout)
                except (InterestNack, InterestTimeout) as e:
                    logging.warning(f"miss {name_str}: {e!r}")
                    continue
                payload = bytes(content) if content is not None else b''
                self.received += 1
                now_ms = (time.monotonic() - t0) * 1000.0
                if self.ttff_ms is None:
                    self.ttff_ms = now_ms
                got = hashlib.sha256(payload).hexdigest()
                if got == expected:
                    self.verified += 1
                else:
                    logging.error(f"DIGEST MISMATCH {name_str}")
                if tag == 'base':
                    self.base_done_ms = now_ms
                if self.deadline_ms is not None and now_ms > self.deadline_ms:
                    self.late += 1
        finally:
            self.app.shutdown()
        self._dump_stats()

    def _dump_stats(self):
        data = {
            'interests': self.interests,
            'received': self.received,
            'verified': self.verified,
            'ttff_ms': round(self.ttff_ms, 2) if self.ttff_ms is not None
            else None,
            'base_done_ms': round(self.base_done_ms, 2)
            if self.base_done_ms is not None else None,
            'late': self.late,
        }
        logging.info(f"consumer stats: {data}")
        if self.stats_path:
            with open(self.stats_path, 'w') as fh:
                json.dump(data, fh)

    def run(self):
        try:
            self.app.run_forever(after_start=self.fetch())
        except KeyboardInterrupt:
            logging.info("consumer stopped")


def main():
    ap = argparse.ArgumentParser(description='GaussCast client')
    ap.add_argument('--scene', default='truck')
    ap.add_argument('--version', default='1')
    ap.add_argument('--n-cells', type=int, default=8)
    ap.add_argument('--plan', required=True, help='per-user plan JSON')
    ap.add_argument('--stats', default=None, help='write client metrics here')
    ap.add_argument('--timeout', type=int, default=4000)
    ap.add_argument('--deadline-ms', type=float, default=None)
    args = ap.parse_args()
    GaussCastConsumer(
        scene=args.scene, version=args.version, n_cells=args.n_cells,
        plan_path=args.plan, stats_path=args.stats, timeout=args.timeout,
        deadline_ms=args.deadline_ms,
    ).run()


if __name__ == '__main__':
    main()
