# Benchmarking Guide

## Overview

NumPy-TS uses an **auto-calibrated benchmarking system** that provides stable, reproducible performance measurements. The system automatically adjusts the number of operations per sample to ensure meaningful measurements.

## Key Features

### 1. **Auto-Calibration**
The benchmark runner automatically determines how many times to run each operation to achieve a minimum measurement time of 100ms per sample. This eliminates sub-millisecond noise and provides stable results.

```
Fast operations (e.g., zeros):   ~3.9M ops/sec  (runs 100K+ times per sample)
Medium operations (e.g., add):   ~25K ops/sec   (runs 2.5K+ times per sample)
Slow operations (e.g., matmul):  ~100 ops/sec   (runs 10+ times per sample)
```

### 2. **Multiple Samples**
Each benchmark collects **20 samples** to calculate robust statistics:
- Mean (primary metric)
- Median (robust to outliers)
- Min/Max (shows variance)
- Standard deviation (measures consistency)

### 3. **Meaningful Metrics**
Results include both time-based and throughput-based metrics:
- **Time per operation** (ms): How long each operation takes
- **Operations per second**: Intuitive throughput measure
- **Total operations**: Confidence in measurement (more = better)

### 4. **Warmup Phase**
Before benchmarking, the runner performs warmup iterations to:
- Let JIT compilation stabilize
- Warm up caches
- Eliminate first-run overhead

## Running Benchmarks

### Quick Benchmarks (~2-3 minutes)
```bash
source ~/.zshrc && conda activate py313
npm run bench:quick
```

**Configuration:**
- Same array sizes as standard mode (1000 elements, 100×100, 500×500)
- **1 sample** per benchmark (50ms minimum per sample)
- Less warmup for faster feedback
- Use for rapid iteration during development

### Standard Benchmarks (~5-10 minutes)
```bash
source ~/.zshrc && conda activate py313
npm run bench
```

**Configuration:**
- Realistic arrays (1000 elements, 100×100, 500×500 matrices)
- **5 samples** per benchmark (100ms minimum per sample)
- More warmup for stable results
- Use for releases and accurate performance tracking

### Category-Specific Benchmarks
```bash
npm run bench:category creation    # Array creation functions
npm run bench:category arithmetic   # Arithmetic operations
npm run bench:category reductions   # Reduction operations
```

## Understanding Results

### Example Output

**numpy-ts (TypeScript):**
```
Running 42 benchmarks with auto-calibration...
Target: 100ms per sample, 5 samples per benchmark

  [1/42] zeros [100]              0.000ms     3,896,376 ops/sec
  [2/42] add [50x50] + [50x50]    0.031ms        32,444 ops/sec
  [3/42] matmul [50x50] @ [50x50] 0.124ms         8,041 ops/sec
```

**NumPy (Python):**
```
Running 42 benchmarks with auto-calibration...
Target: 100ms per sample, 5 samples per benchmark

  [1/42] zeros [100]              0.000ms     9,584,886 ops/sec
  [2/42] add [50x50] + [50x50]    0.009ms       110,573 ops/sec
  [3/42] matmul [50x50] @ [50x50] 0.016ms        61,884 ops/sec
```

**Both outputs now match in format**, making direct comparison easy!

**Interpretation:**
- **zeros [100]**: Extremely fast, runs ~3.9M times per second
- **add [50x50]**: Medium speed, ~32K ops/sec, 0.031ms per operation
- **matmul [100x100]**: Slower operation, ~108 ops/sec, ~9ms per operation

### Regression Detection

The auto-calibration makes regression detection easier:

**Before (flaky):**
```
zeros: 0.000ms ± 0.001ms  (hard to compare)
```

**After (stable):**
```
zeros: 3,904,369 ops/sec  (±2%)
```

A 10% performance regression is now clearly visible:
```
Before: 3,904,369 ops/sec
After:  3,513,932 ops/sec  (-10.0%)
```

## Technical Details

### Algorithm

1. **Setup Phase**: Create test arrays
2. **Warmup Phase**: Run operation 10-20 times to stabilize JIT
3. **Calibration Phase**:
   - Start with 1 operation per sample
   - Measure time taken
   - If < 100ms, increase ops exponentially (×2 or ×10)
   - Repeat until target time reached
4. **Benchmark Phase**:
   - Run calibrated number of ops in tight loop
   - Measure total time
   - Divide by ops count to get per-op time
   - Repeat for 20 samples
5. **Statistics Phase**: Calculate mean, median, std dev, ops/sec

### Configuration

Key constants in `benchmarks/src/runner.ts`:

```typescript
const MIN_SAMPLE_TIME_MS = 100;  // Target time per sample
const TARGET_SAMPLES = 20;        // Number of samples to collect
```

### Preventing Over-Optimization

The runner uses several techniques to prevent compiler optimizations from skewing results:

```typescript
// Keep reference to prevent dead code elimination
const result = executeOperation(operation, arrays);
void result;
```

## Best Practices

### 1. **Always Use Conda Environment**
NumPy validation and comparison require Python:
```bash
source ~/.zshrc && conda activate py313
```

### 2. **Run Full Benchmarks Before Release**
Quick benchmarks are for development; run full benchmarks for releases:
```bash
npm run bench:full
```

### 3. **Compare Against Baseline**
Keep historical benchmark results to track performance over time:
```bash
git checkout main
npm run bench > baseline.txt

git checkout feature-branch
npm run bench > feature.txt

# Compare results
diff baseline.txt feature.txt
```

### 4. **Watch for Variance**
High standard deviation indicates unstable benchmarks:
```
Good:  0.031ms ± 0.002ms  (6% variance)
Bad:   0.031ms ± 0.015ms  (48% variance)
```

High variance suggests:
- Background processes interfering
- Inconsistent operation behavior
- Need for more samples or longer calibration

### 5. **Benchmark in Isolation**
For accurate results:
- Close unnecessary applications
- Disable CPU throttling
- Run on consistent hardware
- Avoid running during system updates

## Troubleshooting

### Benchmark Takes Too Long
- Use `npm run bench:quick` for faster feedback
- Use category-specific benchmarks: `npm run bench:category creation`
- Reduce `TARGET_SAMPLES` in `runner.ts` (at cost of accuracy)

### Inconsistent Results
- Check for background processes consuming CPU
- Ensure system is not thermal throttling
- Increase `TARGET_SAMPLES` for more stable measurements
- Use `median_ms` instead of `mean_ms` (more robust to outliers)

### Conda Environment Issues
```bash
# Ensure conda is activated
echo $CONDA_DEFAULT_ENV  # Should print: py313

# Reactivate if needed
source ~/.zshrc && conda activate py313
```

## Future Improvements

Potential enhancements to the benchmark system:

1. **CI Integration**: Automated regression detection in CI/CD
2. **Historical Tracking**: Database of benchmark results over time
3. **Visualization**: Interactive charts showing performance trends
4. **Comparison Reports**: Automatic comparison against NumPy baselines
5. **Memory Profiling**: Track memory usage alongside performance
6. **Hardware Detection**: Adjust benchmarks based on available resources

## References

- [Benchmark.js Best Practices](https://benchmarkjs.com/)
- [V8 Performance Tips](https://v8.dev/docs/turbofan)
- [NumPy Performance Guide](https://numpy.org/doc/stable/user/performance.html)
