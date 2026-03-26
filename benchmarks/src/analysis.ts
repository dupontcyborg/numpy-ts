/**
 * Benchmark results analysis
 */

const DTYPE_RE =
  /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;
const DTYPE_ORDER = [
  'float64',
  'float32',
  'float16',
  'int64',
  'uint64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
  'complex128',
  'complex64',
  'bool',
];

function compareBenchmarkNames(a: string, b: string): number {
  const aMatch = a.match(DTYPE_RE);
  const bMatch = b.match(DTYPE_RE);
  const aBase = aMatch ? a.slice(0, -aMatch[0].length) : a;
  const bBase = bMatch ? b.slice(0, -bMatch[0].length) : b;
  const baseCmp = aBase.localeCompare(bBase);
  if (baseCmp !== 0) return baseCmp;
  const aDtype = aMatch ? aMatch[1]! : 'float64'; // no suffix = implicit float64
  const bDtype = bMatch ? bMatch[1]! : 'float64';
  return DTYPE_ORDER.indexOf(aDtype) - DTYPE_ORDER.indexOf(bDtype);
}

import type {
  BenchmarkTiming,
  BenchmarkComparison,
  BenchmarkSummary,
  BenchmarkCase,
  RuntimeComparison,
} from './types';

export function compareResults(
  specs: BenchmarkCase[],
  numpyResults: BenchmarkTiming[],
  numpyjsResults: BenchmarkTiming[]
): BenchmarkComparison[] {
  const comparisons: BenchmarkComparison[] = [];

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const numpy = numpyResults[i]!;
    const numpyjs = numpyjsResults[i]!;

    comparisons.push({
      name: spec.name,
      category: spec.category,
      numpy,
      numpyjs,
      ratio: numpyjs.mean_ms / numpy.mean_ms,
      wasmUsed: numpyjs.wasmUsed,
    });
  }

  return comparisons;
}

export function calculateSummary(comparisons: BenchmarkComparison[]): BenchmarkSummary {
  const ratios = comparisons.map((c) => c.ratio);

  const sum = ratios.reduce((a, b) => a + b, 0);
  const avg_slowdown = sum / ratios.length;

  const sorted = [...ratios].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median_slowdown =
    sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;

  const best_case = Math.min(...ratios);
  const worst_case = Math.max(...ratios);

  return {
    avg_slowdown,
    median_slowdown,
    best_case,
    worst_case,
    total_benchmarks: comparisons.length,
  };
}

export function groupByCategory(
  comparisons: BenchmarkComparison[]
): Map<string, BenchmarkComparison[]> {
  const groups = new Map<string, BenchmarkComparison[]>();

  for (const comparison of comparisons) {
    const existing = groups.get(comparison.category) || [];
    existing.push(comparison);
    groups.set(comparison.category, existing);
  }

  for (const items of groups.values()) {
    items.sort((a, b) => compareBenchmarkNames(a.name, b.name));
  }

  return groups;
}

export function getCategorySummaries(
  comparisons: BenchmarkComparison[]
): Map<string, { avg_slowdown: number; count: number }> {
  const groups = groupByCategory(comparisons);
  const summaries = new Map<string, { avg_slowdown: number; count: number }>();

  for (const [category, items] of groups) {
    const ratios = items.map((item) => item.ratio);
    const avg_slowdown = ratios.reduce((a, b) => a + b, 0) / ratios.length;

    summaries.set(category, {
      avg_slowdown,
      count: items.length,
    });
  }

  return summaries;
}

export function getDtypeSummaries(
  comparisons: BenchmarkComparison[]
): Map<string, { avg_slowdown: number; median_slowdown: number; count: number }> {
  const dtypeRe =
    /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;
  const groups = new Map<string, number[]>();

  for (const c of comparisons) {
    const m = c.name.match(dtypeRe);
    const dtype = m ? m[1]! : 'float64';
    const existing = groups.get(dtype) || [];
    existing.push(c.ratio);
    groups.set(dtype, existing);
  }

  const summaries = new Map<
    string,
    { avg_slowdown: number; median_slowdown: number; count: number }
  >();
  // Order dtypes consistently
  const dtypeOrder = [
    'float64',
    'float32',
    'float16',
    'complex128',
    'complex64',
    'int64',
    'int32',
    'int16',
    'int8',
    'uint64',
    'uint32',
    'uint16',
    'uint8',
    'bool',
  ];

  for (const dtype of dtypeOrder) {
    const ratios = groups.get(dtype);
    if (!ratios) continue;
    const avg = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    const sorted = [...ratios].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
    summaries.set(dtype, { avg_slowdown: avg, median_slowdown: median, count: ratios.length });
  }

  return summaries;
}

export function getMultiRuntimeDtypeSummaries(
  comparisons: RuntimeComparison[],
  runtimeName: string
): Map<string, { avg_slowdown: number; median_slowdown: number; count: number }> {
  // Convert RuntimeComparison to BenchmarkComparison for the given runtime
  const benchComparisons: BenchmarkComparison[] = [];
  for (const c of comparisons) {
    const entry = c.runtimes[runtimeName];
    if (!entry) continue;
    benchComparisons.push({
      name: c.name,
      category: c.category,
      numpy: c.numpy,
      numpyjs: entry.timing,
      ratio: entry.ratio,
    });
  }
  return getDtypeSummaries(benchComparisons);
}

export function formatDuration(ms: number): string {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(2)}μs`;
  } else if (ms < 1000) {
    return `${ms.toFixed(3)}ms`;
  } else {
    return `${(ms / 1000).toFixed(2)}s`;
  }
}

export function formatRatio(ratio: number): string {
  return `${ratio.toFixed(2)}x`;
}

export function formatOpsPerSec(ops: number): string {
  if (ops >= 1_000_000) {
    return `${(ops / 1_000_000).toFixed(2)}M`;
  } else if (ops >= 1_000) {
    return `${(ops / 1_000).toFixed(1)}K`;
  } else {
    return `${ops.toFixed(0)}`;
  }
}

export function printResults(comparisons: BenchmarkComparison[], summary: BenchmarkSummary): void {
  const groups = groupByCategory(comparisons);

  console.log('\n' + '='.repeat(80));
  console.log('BENCHMARK RESULTS');
  console.log('='.repeat(80));

  // Print by category
  for (const [category, items] of groups) {
    console.log(`\n[${category.toUpperCase()}]`);

    for (const item of items) {
      const { name, numpy, numpyjs, ratio } = item;
      const color = ratio < 2 ? '\x1b[32m' : ratio < 5 ? '\x1b[33m' : '\x1b[31m';
      const reset = '\x1b[0m';

      console.log(
        `  ${name.padEnd(45)} ` +
          `NumPy: ${formatDuration(numpy.mean_ms).padStart(10)} | ` +
          `numpy-ts: ${formatDuration(numpyjs.mean_ms).padStart(10)} | ` +
          `${color}${formatRatio(ratio).padStart(8)}${reset}`
      );
    }
  }

  // Print summary
  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total benchmarks: ${summary.total_benchmarks}`);
  console.log(`Average slowdown: ${formatRatio(summary.avg_slowdown)}`);
  console.log(`Median slowdown:  ${formatRatio(summary.median_slowdown)}`);
  console.log(`Best case:        ${formatRatio(summary.best_case)}`);
  console.log(`Worst case:       ${formatRatio(summary.worst_case)}`);

  // Print category summaries
  const categorySummaries = getCategorySummaries(comparisons);
  console.log('\nBy Category:');
  for (const [category, data] of categorySummaries) {
    console.log(`  ${category.padEnd(15)} ${formatRatio(data.avg_slowdown).padStart(8)}`);
  }

  // Print dtype summaries
  const dtypeSums = getDtypeSummaries(comparisons);
  if (dtypeSums.size > 1) {
    console.log('\nBy DType:');
    for (const [dtype, data] of dtypeSums) {
      console.log(
        `  ${dtype.padEnd(12)} avg ${formatRatio(data.avg_slowdown).padStart(8)}  median ${formatRatio(data.median_slowdown).padStart(8)}  (${data.count} benchmarks)`
      );
    }
  }

  console.log('='.repeat(80) + '\n');
}

// --- Multi-runtime functions ---

export function compareMultiRuntime(
  specs: BenchmarkCase[],
  numpyResults: BenchmarkTiming[],
  runtimeResults: Map<string, BenchmarkTiming[]>
): RuntimeComparison[] {
  const comparisons: RuntimeComparison[] = [];

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const numpy = numpyResults[i]!;
    const runtimes: Record<string, { timing: BenchmarkTiming; ratio: number }> = {};

    for (const [runtimeName, results] of runtimeResults) {
      const timing = results[i]!;
      runtimes[runtimeName] = {
        timing,
        ratio: timing.mean_ms / numpy.mean_ms,
      };
    }

    comparisons.push({
      name: spec.name,
      category: spec.category,
      numpy,
      runtimes,
      wasmUsed: Object.values(runtimes).some((r) => r.timing.wasmUsed),
    });
  }

  return comparisons;
}

export function calculateMultiRuntimeSummaries(
  comparisons: RuntimeComparison[]
): Record<string, BenchmarkSummary> {
  // Collect all runtime names
  const runtimeNames = new Set<string>();
  for (const c of comparisons) {
    for (const name of Object.keys(c.runtimes)) {
      runtimeNames.add(name);
    }
  }

  const summaries: Record<string, BenchmarkSummary> = {};

  for (const runtimeName of runtimeNames) {
    const ratios = comparisons
      .filter((c) => c.runtimes[runtimeName])
      .map((c) => c.runtimes[runtimeName]!.ratio);

    if (ratios.length === 0) continue;

    const sum = ratios.reduce((a, b) => a + b, 0);
    const avg_slowdown = sum / ratios.length;
    const sorted = [...ratios].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median_slowdown =
      sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;

    summaries[runtimeName] = {
      avg_slowdown,
      median_slowdown,
      best_case: Math.min(...ratios),
      worst_case: Math.max(...ratios),
      total_benchmarks: ratios.length,
    };
  }

  return summaries;
}

export function groupMultiRuntimeByCategory(
  comparisons: RuntimeComparison[]
): Map<string, RuntimeComparison[]> {
  const groups = new Map<string, RuntimeComparison[]>();
  for (const comparison of comparisons) {
    const existing = groups.get(comparison.category) || [];
    existing.push(comparison);
    groups.set(comparison.category, existing);
  }
  for (const items of groups.values()) {
    items.sort((a, b) => compareBenchmarkNames(a.name, b.name));
  }
  return groups;
}

export function printMultiRuntimeResults(
  comparisons: RuntimeComparison[],
  summaries: Record<string, BenchmarkSummary>
): void {
  const groups = groupMultiRuntimeByCategory(comparisons);
  const runtimeNames = Object.keys(summaries);

  console.log('\n' + '='.repeat(100));
  console.log('MULTI-RUNTIME BENCHMARK RESULTS');
  console.log('='.repeat(100));

  for (const [category, items] of groups) {
    console.log(`\n[${category.toUpperCase()}]`);

    for (const item of items) {
      const parts = [
        `  ${item.name.padEnd(35)} NumPy: ${formatDuration(item.numpy.mean_ms).padStart(10)}`,
      ];

      for (const rt of runtimeNames) {
        const entry = item.runtimes[rt];
        if (entry) {
          const color = entry.ratio < 2 ? '\x1b[32m' : entry.ratio < 5 ? '\x1b[33m' : '\x1b[31m';
          const reset = '\x1b[0m';
          parts.push(
            `${rt}: ${formatDuration(entry.timing.mean_ms).padStart(10)} (${color}${formatRatio(entry.ratio)}${reset})`
          );
        }
      }

      console.log(parts.join(' | '));
    }
  }

  // Per-runtime summary
  console.log('\n' + '='.repeat(100));
  console.log('SUMMARY BY RUNTIME');
  console.log('='.repeat(100));

  for (const [runtimeName, summary] of Object.entries(summaries)) {
    console.log(`\n  ${runtimeName}:`);
    console.log(`    Average slowdown: ${formatRatio(summary.avg_slowdown)}`);
    console.log(`    Median slowdown:  ${formatRatio(summary.median_slowdown)}`);
    console.log(`    Best case:        ${formatRatio(summary.best_case)}`);
    console.log(`    Worst case:       ${formatRatio(summary.worst_case)}`);
  }

  console.log('='.repeat(100) + '\n');
}
