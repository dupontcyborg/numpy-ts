"""
Microbenchmark: Python dict lookup for operation dispatch.

Run:  python3 benchmarks/scripts/dispatch_lookup_bench.py

Compare output with:  node benchmarks/scripts/dispatch_lookup_bench.mjs
"""

import time

ITERS = 10_000_000
WARMUP = 1_000_000

# Build ~400 keys to mirror the real dispatch tables
keys = [f"op_{i}" for i in range(400)]
fns = [lambda i=i: i for i in range(400)]

# Dict dispatch
dispatch = {k: fn for k, fn in zip(keys, fns)}

# Pick a few keys to cycle through
hot_keys = [keys[0], keys[50], keys[150], keys[300], keys[399]]
hot_len = len(hot_keys)


def bench_dict():
    d = dispatch
    hk = hot_keys
    hl = hot_len
    total = 0
    for i in range(ITERS):
        fn = d[hk[i % hl]]
        total += fn()
    return total


def bench_if_chain():
    ks = keys
    fs = fns
    hk = hot_keys
    hl = hot_len
    total = 0
    for i in range(ITERS):
        key = hk[i % hl]
        for j, k in enumerate(ks):
            if key == k:
                total += fs[j]()
                break
    return total


def run(label, fn):
    # Warmup
    fn_warm = dispatch[hot_keys[0]]
    for _ in range(WARMUP):
        fn_warm()

    t0 = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    ops = int(ITERS / elapsed * 1000)
    print(f"{label:<20} {elapsed:>8.1f} ms   {ops:>15,} ops/sec   (checksum: {result})")


if __name__ == "__main__":
    print(f"\nPython dispatch lookup benchmark — {ITERS:,} iterations, {len(keys)} keys\n")
    run("Dict", bench_dict)
    run("If/elif chain", bench_if_chain)
    print()
