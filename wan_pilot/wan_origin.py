"""
wan_origin.py
-------------
ORIGIN server for the cross-region WAN pilot. Deploy this
on the cloud instance in REGION A (e.g. Singapore). It serves synthetic
equal-content blocks of the exact byte sizes the simulator's origin pulls
require: GET /block/<id> returns <nbytes> bytes (zero-filled). Byte content is
irrelevant -- the pilot measures upstream bytes + TTFF over the real link, not
rendering.

Usage on the origin instance:
    python3 wan_origin.py --manifest wan_manifest.json --port 8080 [--scale 1]

--scale divides every block size by that factor for cheaper/faster pilots on
small instances; use the SAME --scale on wan_edge.py so the ratio is preserved.
Only the Python 3 standard library is used (no pip installs on the cloud box).
Security note: this server is UNAUTHENTICATED and serves arbitrary-size zero
buffers; bind it to the pilot's security-group/VPC only, not the public
internet, and tear it down after the run.
"""
import sys, os, json, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOCK_BYTES = {}
SCALE = 1
ZERO = b"\x00" * (1 << 20)   # 1 MiB reusable zero buffer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path.startswith("/block/"):
            ident = self.path[len("/block/"):]
            nb = BLOCK_BYTES.get(ident)
            if nb is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            nb = max(1, int(nb // SCALE))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(nb))
            self.end_headers()
            sent = 0
            while sent < nb:
                chunk = min(len(ZERO), nb - sent)
                self.wfile.write(ZERO[:chunk])
                sent += chunk
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    global BLOCK_BYTES, SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="wan_manifest.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--scale", type=int, default=1)
    args = ap.parse_args()
    m = json.load(open(args.manifest))
    BLOCK_BYTES = m["block_bytes"]
    SCALE = max(1, args.scale)
    total = sum(BLOCK_BYTES.values()) / SCALE / 1e6
    print(f"origin: {len(BLOCK_BYTES)} blocks, {total:.1f} MB distinct "
          f"(scale={SCALE}); serving on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
