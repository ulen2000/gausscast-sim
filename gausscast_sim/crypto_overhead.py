"""
crypto_overhead.py
------------------
GENUINE measurement of the verification primitives behind GaussCast's
authenticated delivery. Uses the system `cryptography` library to perform real
Ed25519 signatures/verifies and SHA-256 hashing, timed on the local machine.

Two authentication strategies are modeled from the SAME measured primitives:

  * Per-block signature   : one Ed25519 public-key verify per delivered block.
  * Digest-linked manifest: one Ed25519 verify authenticates a manifest of
    `fanout` blocks, and each block is checked with a single SHA-256 digest
    comparison. The amortized public-key verifies/second therefore scale with
    the manifest fanout.

Everything printed is derived directly from the measured per-operation costs and
the manifest fanout: change `--fanout` or `--block-mb` and the numbers move
accordingly.
"""
import os
import time
import json
import argparse

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import hashes

MB = 1024 * 1024


def time_ed25519_verify(n):
    """Mean wall-clock seconds per Ed25519 public-key verify."""
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    msg = os.urandom(64)
    sig = sk.sign(msg)
    t0 = time.perf_counter()
    for _ in range(n):
        pk.verify(sig, msg)
    return (time.perf_counter() - t0) / n


def time_sha256(n, nbytes):
    """Mean wall-clock seconds to SHA-256 a buffer of `nbytes` bytes."""
    buf = os.urandom(int(nbytes))
    t0 = time.perf_counter()
    for _ in range(n):
        h = hashes.Hash(hashes.SHA256())
        h.update(buf)
        h.finalize()
    return (time.perf_counter() - t0) / n


def measure(block_mb=1.0, fanout=32, n_verify=20000, n_hash=2000):
    t_verify = time_ed25519_verify(n_verify)            # s / PK verify
    t_hash_block = time_sha256(n_hash, block_mb * MB)   # s / full-block hash
    t_hash_digest = time_sha256(n_verify, 32)           # s / 32-byte digest cmp

    # Per-block signature: a PK verify + hashing the block content per block.
    cpu_per_block_sig = t_verify + t_hash_block
    pk_per_s_sig = 1.0 / cpu_per_block_sig

    # Digest-linked: PK verify amortized over `fanout` blocks; each block pays a
    # SHA-256 over its content plus a cheap digest comparison.
    cpu_per_block_dl = t_verify / fanout + t_hash_block + t_hash_digest
    # sustained PK verifies/s = verifies issued per second under this workload
    pk_per_s_dl = 1.0 / (cpu_per_block_dl * fanout)

    return {
        "primitives": {
            "ed25519_verify_us": t_verify * 1e6,
            "sha256_block_ms": t_hash_block * 1e3,
            "sha256_digest_us": t_hash_digest * 1e6,
            "block_mb": block_mb,
            "manifest_fanout": fanout,
        },
        "per_block_signature": {
            "pk_verifies_per_s": pk_per_s_sig,
            "cpu_per_block_ms": cpu_per_block_sig * 1e3,
        },
        "digest_linked_manifest": {
            "pk_verifies_per_s": pk_per_s_dl,
            "cpu_per_block_ms": cpu_per_block_dl * 1e3,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-mb", type=float, default=1.0)
    ap.add_argument("--fanout", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "out",
                                                  "crypto_overhead.json"))
    args = ap.parse_args()

    r = measure(block_mb=args.block_mb, fanout=args.fanout)
    p = r["primitives"]
    print(f"ed25519 verify = {p['ed25519_verify_us']:.2f} us | "
          f"sha256({p['block_mb']:.1f}MB) = {p['sha256_block_ms']:.3f} ms | "
          f"sha256(32B) = {p['sha256_digest_us']:.3f} us | "
          f"fanout = {p['manifest_fanout']}")
    print(f"{'strategy':24s}{'PK verifies/s':>16s}{'CPU/block (ms)':>16s}")
    for k in ("per_block_signature", "digest_linked_manifest"):
        print(f"{k:24s}{r[k]['pk_verifies_per_s']:16.1f}"
              f"{r[k]['cpu_per_block_ms']:16.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(r, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
