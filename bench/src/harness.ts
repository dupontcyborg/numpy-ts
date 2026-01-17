/**
 * Benchmark Harness
 *
 * Coordinates benchmarks across TypeScript, Rust WASM, and Zig WASM implementations.
 */

export interface BenchmarkResult {
  name: string;
  implementation: 'typescript' | 'rust-wasm' | 'rust-wasm-copy' | 'rust-wasm-mt' | 'zig-wasm' | 'zig-wasm-copy' | 'zig-wasm-mt';
  operation: 'add' | 'sin' | 'sum' | 'matmul';
  dtype: 'float32' | 'float64';
  size: number;
  timeMs: number;
  opsPerSecond: number;
  throughputGBps: number;
}

export interface BenchmarkConfig {
  warmupRuns: number;
  measureRuns: number;
  sizes: {
    small: number;
    medium: number;
    large: number;
  };
  matmulSizes: {
    small: number;
    medium: number;
    large: number;
  };
}

export const DEFAULT_CONFIG: BenchmarkConfig = {
  warmupRuns: 3,
  measureRuns: 10,
  sizes: {
    small: 1_000,
    medium: 100_000,
    large: 10_000_000,
  },
  matmulSizes: {
    small: 64,
    medium: 512,
    large: 2048,
  },
};

export const QUICK_CONFIG: BenchmarkConfig = {
  warmupRuns: 1,
  measureRuns: 3,
  sizes: {
    small: 1_000,
    medium: 100_000,
    large: 1_000_000,
  },
  matmulSizes: {
    small: 64,
    medium: 256,
    large: 512,
  },
};

/**
 * Measure execution time of a function
 */
export async function measure(
  fn: () => void | Promise<void>,
  warmupRuns: number,
  measureRuns: number
): Promise<{ mean: number; std: number; min: number; max: number }> {
  // Warmup
  for (let i = 0; i < warmupRuns; i++) {
    await fn();
  }

  // Force GC if available
  if (global.gc) {
    global.gc();
  }

  // Measure
  const times: number[] = [];
  for (let i = 0; i < measureRuns; i++) {
    const start = performance.now();
    await fn();
    const end = performance.now();
    times.push(end - start);
  }

  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const variance = times.reduce((sum, t) => sum + (t - mean) ** 2, 0) / times.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...times);
  const max = Math.max(...times);

  return { mean, std, min, max };
}

/**
 * Calculate throughput in GB/s
 */
export function calculateThroughput(
  bytesProcessed: number,
  timeMs: number
): number {
  const seconds = timeMs / 1000;
  const gigabytes = bytesProcessed / (1024 * 1024 * 1024);
  return gigabytes / seconds;
}

/**
 * Format benchmark results as a table
 */
export function formatResultsTable(results: BenchmarkResult[]): string {
  const lines: string[] = [];

  // Group by operation
  const byOperation = new Map<string, BenchmarkResult[]>();
  for (const r of results) {
    const key = `${r.operation}-${r.dtype}`;
    if (!byOperation.has(key)) {
      byOperation.set(key, []);
    }
    byOperation.get(key)!.push(r);
  }

  for (const [opKey, opResults] of byOperation) {
    lines.push(`\n${'='.repeat(100)}`);
    lines.push(`Operation: ${opKey.toUpperCase()}`);
    lines.push('='.repeat(100));

    // Header
    lines.push(
      padRight('Implementation', 18) +
      padRight('Size', 14) +
      padRight('Time (ms)', 14) +
      padRight('Std Dev', 12) +
      padRight('Throughput', 14) +
      'Speedup vs TS'
    );
    lines.push('-'.repeat(100));

    // Group by size
    const bySize = new Map<number, BenchmarkResult[]>();
    for (const r of opResults) {
      if (!bySize.has(r.size)) {
        bySize.set(r.size, []);
      }
      bySize.get(r.size)!.push(r);
    }

    for (const [size, sizeResults] of [...bySize.entries()].sort((a, b) => a[0] - b[0])) {
      const tsResult = sizeResults.find(r => r.implementation === 'typescript');
      const tsTime = tsResult?.timeMs ?? 1;

      for (const r of sizeResults.sort((a, b) => a.implementation.localeCompare(b.implementation))) {
        const speedup = tsTime / r.timeMs;
        const speedupStr = r.implementation === 'typescript' ? '-' : `${speedup.toFixed(2)}x`;

        lines.push(
          padRight(r.implementation, 18) +
          padRight(formatSize(r.size), 14) +
          padRight(r.timeMs.toFixed(3), 14) +
          padRight('±' + ((r as any).std?.toFixed(3) ?? '0'), 12) +
          padRight(r.throughputGBps.toFixed(2) + ' GB/s', 14) +
          speedupStr
        );
      }
      lines.push('-'.repeat(100));
    }
  }

  return lines.join('\n');
}

function padRight(str: string, len: number): string {
  return str.padEnd(len);
}

function formatSize(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

/**
 * Export results as JSON
 */
export function exportResultsJSON(results: BenchmarkResult[]): string {
  return JSON.stringify(results, null, 2);
}

/**
 * Create typed arrays for benchmarking
 */
export function createTestArrays(
  size: number,
  dtype: 'float32' | 'float64'
): { a: Float32Array | Float64Array; b: Float32Array | Float64Array; c: Float32Array | Float64Array } {
  const ArrayConstructor = dtype === 'float32' ? Float32Array : Float64Array;

  const a = new ArrayConstructor(size);
  const b = new ArrayConstructor(size);
  const c = new ArrayConstructor(size);

  // Initialize with random values
  for (let i = 0; i < size; i++) {
    a[i] = Math.random() * 2 - 1;
    b[i] = Math.random() * 2 - 1;
  }

  return { a, b, c };
}

/**
 * Create matrices for matmul benchmarking
 */
export function createTestMatrices(
  n: number,
  dtype: 'float32' | 'float64'
): { a: Float32Array | Float64Array; b: Float32Array | Float64Array; c: Float32Array | Float64Array } {
  const ArrayConstructor = dtype === 'float32' ? Float32Array : Float64Array;
  const size = n * n;

  const a = new ArrayConstructor(size);
  const b = new ArrayConstructor(size);
  const c = new ArrayConstructor(size);

  // Initialize with random values
  for (let i = 0; i < size; i++) {
    a[i] = Math.random() * 2 - 1;
    b[i] = Math.random() * 2 - 1;
  }

  return { a, b, c };
}
