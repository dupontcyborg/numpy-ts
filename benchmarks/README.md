# NumPy vs numpy-ts Benchmarks

Performance comparison suite for numpy-ts against Python NumPy.

## <a id="methodology"></a>Methodology (canonical)

This section pins the methodology behind the headline performance numbers
linked from the project README. Anything else (blog posts, screenshots, slack
threads) is either older or summarized — defer to this.

- **Hardware**: latest published numbers are produced on the GitHub Actions
  `ubuntu-latest` runner (x86_64, AVX2 available). Local runs on Apple Silicon
  are kept in `.github/benchmark-history/` for trend continuity.
- **Runtime**: Node.js (LTS — currently Node 24). numpy-ts uses its Zig-WASM
  SIMD kernels via the standard `numpy-ts` import. Pyodide-based browser
  numbers are not part of the headline; they live in their own report.
- **NumPy version**: NumPy 2.x (`numpy>=2`, see `ci.yml`). Different NumPy
  versions are not interchangeable — pinning matters here.
- **Size regime**: "standard" mode (medium arrays — see *Benchmark Modes*
  below). The headline ratio comes from this regime. "quick" and "full" exist
  for CI feedback and stress testing respectively and are not used for the
  published number.
- **Aggregation**: the headline ratio is the **geometric mean** of per-benchmark
  `ratio = numpy-ts mean / NumPy mean`. Geomean is the correct averaging
  method for ratios (arithmetic mean would over-weight outlier slowdowns).
- **Inversion for public display**: `bench:docs` publishes `speedup = 1 / ratio`
  so the landing-page narrative reads "Xx faster" — the underlying ratios are
  identical to what the local HTML report shows.

If you reproduce these numbers in a different environment (Bun, Deno, ARM,
older NumPy), expect different absolute values — the methodology above is what
keeps headline numbers comparable across releases.

## Quick Start

```bash
# Run standard benchmarks (recommended)
pnpm run bench

# Run quick benchmarks (for CI or quick checks)
pnpm run bench:quick

# Run full benchmarks (comprehensive, takes longer)
pnpm run bench:full

# View latest results in browser
pnpm run bench:view
```

## Benchmark Modes

### Quick Mode (`--quick`)
- Small array sizes (50x50, 100 elements)
- Fewer iterations (10 runs per benchmark)
- **Duration**: ~1-2 minutes
- **Use case**: CI, rapid development feedback

### Standard Mode (default)
- Medium array sizes (100x100, 1000 elements)
- Moderate iterations (50 runs per benchmark)
- **Duration**: ~5-10 minutes
- **Use case**: Regular performance testing

### Full Mode (`--full`)
- Large array sizes (1000x1000, 10000 elements)
- Many iterations (100 runs per benchmark)
- **Duration**: ~30-60 minutes
- **Use case**: Comprehensive analysis, pre-release testing

## Category-Specific Benchmarks

Run benchmarks for a specific category:

```bash
# Array creation only
pnpm run bench:category creation

# Linear algebra only
pnpm run bench:category linalg

# Arithmetic operations only
pnpm run bench:category arithmetic

# Reductions only
pnpm run bench:category reductions

# Reshape operations only
pnpm run bench:category reshape
```

## Available Categories

- **creation**: Array creation (zeros, ones, arange, linspace, eye)
- **arithmetic**: Arithmetic operations (add, subtract, multiply, divide)
- **linalg**: Linear algebra (matmul, transpose)
- **reductions**: Reductions (sum, mean, max, min) with and without axis
- **reshape**: Reshape operations (reshape, flatten, ravel)

## Output

### Console Output

Results are printed to console with color-coded time ratios (`numpy-ts / NumPy`).
A ratio < 1 means numpy-ts is faster than NumPy; > 1 means slower:
- 🟢 **Green**: < 2x (at parity or faster)
- 🟡 **Yellow**: 2–5x slower
- 🔴 **Red**: > 5x slower (needs optimization)

Example:
```
[ARITHMETIC]
  add [100x100] + scalar           NumPy:     0.050ms | numpy-ts:     0.120ms |     2.40x
  multiply [100x100] * [100x100]   NumPy:     0.080ms | numpy-ts:     0.350ms |     4.38x

[LINALG]
  matmul [100x100] @ [100x100]     NumPy:     0.450ms | numpy-ts:     2.100ms |     4.67x

SUMMARY
Average slowdown: 3.2x
Median slowdown:  2.8x
Best case:        2.1x
Worst case:       5.3x
```

### JSON Results

Results are saved to:
- `benchmarks/results/latest.json` - Latest benchmark run
- `benchmarks/results/history/benchmark-<timestamp>.json` - Historical results

### HTML Report

Interactive HTML report with charts is generated at:
- `benchmarks/results/plots/latest.html`

View with: `pnpm run bench:view`

Features:
- Summary statistics cards
- Bar charts by category
- Detailed horizontal bar chart for all benchmarks
- Color-coded results tables

### PNG Chart

A static PNG chart is also generated for easy sharing:
- `benchmarks/results/plots/latest.png`

This chart is displayed in the main README and shows the average slowdown by category. It's tracked in git so performance changes are visible in pull requests.

## Requirements

- **Node.js**: >= 20.1.0
- **Python**: >= 3.8 with NumPy installed
- **NumPy**: >= 1.20

Check your setup:
```bash
node --version
python3 --version
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           TypeScript Benchmark Orchestrator             │
│  - Define benchmark specifications                      │
│  - Run TypeScript benchmarks (numpy-ts)                │
│  - Spawn Python script for NumPy benchmarks             │
│  - Collect and compare results                          │
│  - Generate visualizations                              │
└─────────────┬───────────────────────────────────────────┘
              │
              ├─────────────────────────────────────────┐
              │                                         │
     ┌────────▼─────────┐                    ┌─────────▼─────────┐
     │  numpy-ts Timing │                    │  Python/NumPy     │
     │  (TypeScript)    │                    │  Timing Script    │
     │                  │                    │                   │
     │  performance.now │                    │  time.perf_       │
     │  Multiple runs   │                    │  counter()        │
     │  Statistics      │                    │  JSON output      │
     └──────────────────┘                    └───────────────────┘
```

## Benchmark Specifications

Each benchmark includes:
- **Name**: Descriptive name (e.g., "matmul [100x100] @ [100x100]")
- **Category**: Group (creation, arithmetic, linalg, reductions, reshape)
- **Operation**: Function being benchmarked
- **Setup**: Array creation and initialization
- **Iterations**: Number of timing runs
- **Warmup**: Warmup iterations to stabilize JIT

## Adding New Benchmarks

Edit `src/specs.ts`:

```typescript
specs.push({
  name: 'my_operation [size]',
  category: 'mycategory',
  operation: 'my_op',
  setup: {
    a: { shape: [100, 100], fill: 'ones' },
    b: { shape: [100], fill: 'zeros' }
  },
  iterations: 50,
  warmup: 10
});
```

Add operation support in:
- `src/runner.ts` - TypeScript/numpy-ts execution
- `scripts/numpy_benchmark.py` - Python/NumPy execution

## Interpreting Results

### Time Ratio

`ratio = numpy-ts mean time / NumPy mean time`. Lower is better:
- **< 1.0x**: numpy-ts is faster than NumPy
- **1.0x**: parity
- **2.0–5.0x**: meaningfully slower; worth investigating
- **> 5.0x**: priority optimization target

The published landing page (`bench:docs`) inverts this and reports
`speedup = 1 / ratio` so that "higher = faster" — the underlying data is the
same. The console output and HTML report keep the raw ratio form.

### Current Performance (Zig-WASM SIMD backend)

With the SIMD WASM kernels enabled, the headline numbers (see `bench:docs`
output and the landing page) sit around:
- **Overall**: ~1.13x faster than NumPy (geomean across ~7,200 benchmarks)
- **Best case**: order-of-magnitude wins on vectorizable element-wise ops
- **Worst case**: 5–20x slower on operations that haven't been ported off the
  JS-fallback path yet

These numbers move with each release. The benchmark history is checked into
`.github/benchmark-history/` and the canonical "what does it look like today"
view is the [performance page](https://numpyts.dev/performance).

### Optimization Priority

Focus optimization efforts on:
1. **Highest ratio operations**: Biggest performance gap
2. **Most frequently used**: Maximum user impact
3. **Easiest to optimize**: Quick wins

## CI Integration

For continuous integration, use quick mode:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: pnpm run bench:quick

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmarks/results/latest.json
```

## Tracking Performance Over Time

Historical results are saved in `benchmarks/results/history/`.

To compare against a previous run:

```bash
# Compare latest vs a historical result
diff <(jq '.summary' benchmarks/results/latest.json) \
     <(jq '.summary' benchmarks/results/history/benchmark-2025-*.json)
```

## Troubleshooting

### Python not found
```bash
# Ensure Python 3 is installed and accessible
which python3

# Or set custom Python command
PYTHON_CMD=python pnpm run bench
```

### NumPy not found
```bash
# Install NumPy
pip3 install numpy

# Or using conda
conda install numpy
```

### Out of memory
Use quick mode or run category-specific benchmarks:
```bash
pnpm run bench:quick
pnpm run bench:category creation
```

## Future Enhancements

- [ ] Regression detection (fail if slowdown increases)
- [ ] Comparison against multiple NumPy versions
- [ ] Memory usage benchmarks
- [ ] WebAssembly vs Pure JS comparison
- [ ] Browser benchmarks
- [ ] Performance regression CI checks

---

**Generated by numpy-ts Benchmark Suite**
