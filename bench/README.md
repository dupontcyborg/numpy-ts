# numpy-ts WASM Performance Benchmark

Compares performance of numerical operations across:
- **Pure TypeScript** (baseline)
- **Rust WASM** (single-threaded)
- **Rust WASM** (multi-threaded with rayon)
- **Zig WASM** (single-threaded)
- **Zig WASM** (multi-threaded)

## Operations Tested

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `add` | Element-wise addition (a + b) | O(n) - memory bound |
| `sin` | Element-wise sine | O(n) - compute bound |
| `sum` | Reduction (sum all elements) | O(n) - memory bound |
| `matmul` | Matrix multiplication | O(n³) - compute bound |

## Array Sizes

| Size | Elements | Matmul Dims | Memory (f64) |
|------|----------|-------------|--------------|
| Small | 1,000 | 64×64 | 8 KB |
| Medium | 100,000 | 512×512 | 800 KB |
| Large | 10,000,000 | 2048×2048 | 80 MB |

## Prerequisites

### Rust (for Rust WASM)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-unknown-unknown

# For multi-threaded builds, need nightly + build-std
rustup install nightly
rustup component add rust-src --toolchain nightly
```

### Zig (for Zig WASM)
```bash
# macOS
brew install zig

# Linux (Ubuntu/Debian)
snap install zig --classic --beta

# Or download from https://ziglang.org/download/
```

### Node.js
```bash
# Node.js 18+ required for WASM support
node --version  # Should be 18.0.0 or higher
```

## Building WASM Modules

### Build Rust WASM (single-threaded)
```bash
cd rust
cargo build --release --target wasm32-unknown-unknown
mkdir -p ../dist
cp target/wasm32-unknown-unknown/release/numpy_bench.wasm ../dist/
```

### Build Rust WASM (multi-threaded)
```bash
cd rust
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
  cargo +nightly build --release --target wasm32-unknown-unknown \
  -Z build-std=std,panic_abort \
  --features threads
cp target/wasm32-unknown-unknown/release/numpy_bench.wasm ../dist/numpy_bench_threads.wasm
```

### Build Zig WASM (single-threaded)
```bash
cd zig
zig build -Doptimize=ReleaseFast
cp zig-out/lib/numpy_bench.wasm ../dist/
```

### Build Zig WASM (multi-threaded)
```bash
cd zig
zig build -Doptimize=ReleaseFast -Dthreads=true
cp zig-out/lib/numpy_bench_threads.wasm ../dist/
```

## Running Benchmarks

### Install dependencies
```bash
npm install
```

### Run full benchmark
```bash
npm run bench
```

### Run quick benchmark (smaller sizes, fewer iterations)
```bash
npm run bench:quick
```

### Run TypeScript-only (no WASM build required)
```bash
npx tsx run.ts
```

## Expected Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     numpy-ts WASM Performance Benchmark                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Comparing: TypeScript vs Rust WASM vs Zig WASM                              ║
║  Operations: add, sin, sum, matmul                                           ║
║  Data types: float32, float64                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

====================================================================================================
Operation: ADD-FLOAT64
====================================================================================================
Implementation        Size          Time (ms)     Std Dev     Throughput    Speedup vs TS
----------------------------------------------------------------------------------------------------
typescript            1.0K          0.012         ±0.002      2.00 GB/s     -
rust-wasm             1.0K          0.003         ±0.001      8.00 GB/s     4.00x
zig-wasm              1.0K          0.003         ±0.001      8.00 GB/s     4.00x
...
```

## Notes on Multi-threading

WASM multi-threading requires:
1. **SharedArrayBuffer** - needs COOP/COEP headers in browsers
2. **Atomics** - for synchronization
3. **Web Workers** - threads are implemented as workers

In Node.js 18+, SharedArrayBuffer is available by default. In browsers, you need:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The Rust multi-threaded build uses `rayon` for parallel iterators. The Zig multi-threaded build currently falls back to single-threaded (true MT would require JS-side Web Worker coordination).

## Interpreting Results

- **Throughput (GB/s)**: Higher is better. Memory bandwidth limit is ~20-50 GB/s on modern systems.
- **Speedup vs TS**: How many times faster than TypeScript baseline.
- **Small arrays**: May show less speedup due to overhead of WASM calls.
- **Large arrays**: Should show maximum speedup, limited by memory bandwidth or compute.

## Architecture

```
bench/
├── src/
│   ├── harness.ts           # Benchmark utilities
│   ├── typescript/ops.ts    # Pure TS implementations
│   └── bridges/
│       └── wasm-loader.ts   # WASM module loader
├── rust/
│   ├── Cargo.toml
│   └── src/lib.rs           # Rust WASM implementations
├── zig/
│   ├── build.zig
│   └── src/main.zig         # Zig WASM implementations
├── dist/                    # Built WASM files go here
├── run.ts                   # Main benchmark runner
└── package.json
```
