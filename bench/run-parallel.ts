#!/usr/bin/env npx tsx

/**
 * Parallel WASM Benchmark Runner
 *
 * Compares single-threaded vs parallel (multi-worker) implementations:
 * - TypeScript (single-threaded)
 * - TypeScript (parallel with SharedArrayBuffer)
 * - Rust WASM (single-threaded)
 * - Rust WASM (parallel with workers)
 * - Zig WASM (single-threaded)
 * - Zig WASM (parallel with workers)
 *
 * Usage:
 *   npx tsx run-parallel.ts           # Full benchmark
 *   npx tsx run-parallel.ts --quick   # Quick benchmark
 */

import { cpus } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

import {
  type BenchmarkConfig,
  DEFAULT_CONFIG,
  QUICK_CONFIG,
  measure,
  calculateThroughput,
  createTestArrays,
  createTestMatrices,
} from './src/harness.js';

import * as tsOps from './src/typescript/ops.js';
import { loadWasmModule, WasmOps } from './src/bridges/wasm-loader.js';
import { TSParallelExecutor } from './src/workers/ts-parallel-executor.js';
import { ParallelExecutor } from './src/workers/parallel-executor.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const isQuick = process.argv.includes('--quick');
const config: BenchmarkConfig = isQuick ? QUICK_CONFIG : DEFAULT_CONFIG;
const NUM_WORKERS = Math.min(cpus().length, 4);

console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║             numpy-ts PARALLEL WASM Performance Benchmark                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Comparing: Single-threaded vs Parallel (${NUM_WORKERS} workers)                        ║
║  Operations: add, sin, sum, matmul                                           ║
║  Using SharedArrayBuffer for zero-copy parallel execution                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);

console.log(`Mode: ${isQuick ? 'QUICK' : 'FULL'}`);
console.log(`Workers: ${NUM_WORKERS}`);
console.log(`Array sizes: ${Object.values(config.sizes).join(', ')}`);
console.log(`Matrix sizes: ${Object.values(config.matmulSizes).map(n => `${n}x${n}`).join(', ')}`);
console.log();

interface Result {
  impl: string;
  op: string;
  dtype: string;
  size: number;
  timeMs: number;
  throughput: number;
}

const results: Result[] = [];

type DType = 'float32' | 'float64';

async function runBenchmarks() {
  // Load WASM modules
  console.log('Loading WASM modules...');
  const distDir = join(__dirname, 'dist');

  let rustWasm: WasmOps | null = null;
  let zigWasm: WasmOps | null = null;

  const rustPath = join(distDir, 'numpy_bench.wasm');
  if (existsSync(rustPath)) {
    const mod = await loadWasmModule(rustPath, 'rust-wasm', false);
    if (mod) rustWasm = new WasmOps(mod);
  }

  const zigPaths = [
    join(distDir, 'zig_numpy_bench.wasm'),
    join(__dirname, 'zig', 'zig-out', 'bin', 'numpy_bench.wasm'),
  ];
  for (const p of zigPaths) {
    if (existsSync(p) && !zigWasm) {
      const mod = await loadWasmModule(p, 'zig-wasm', false);
      if (mod) zigWasm = new WasmOps(mod);
    }
  }

  // Initialize parallel executors
  console.log('Initializing parallel executors...');

  const tsParallel = new TSParallelExecutor(NUM_WORKERS);
  await tsParallel.init();

  let rustParallel: ParallelExecutor | null = null;
  let zigParallel: ParallelExecutor | null = null;

  if (existsSync(rustPath)) {
    rustParallel = new ParallelExecutor({
      numWorkers: NUM_WORKERS,
      wasmPath: rustPath,
      name: 'rust-parallel',
    });
    if (!(await rustParallel.init())) rustParallel = null;
  }

  for (const p of zigPaths) {
    if (existsSync(p) && !zigParallel) {
      zigParallel = new ParallelExecutor({
        numWorkers: NUM_WORKERS,
        wasmPath: p,
        name: 'zig-parallel',
      });
      if (!(await zigParallel.init())) zigParallel = null;
      else break;
    }
  }

  const loaded = [
    'ts-single',
    'ts-parallel',
    rustWasm ? 'rust-single' : null,
    rustParallel ? 'rust-parallel' : null,
    zigWasm ? 'zig-single' : null,
    zigParallel ? 'zig-parallel' : null,
  ].filter(Boolean);

  console.log(`Active implementations: ${loaded.join(', ')}\n`);

  const dtypes: DType[] = ['float64', 'float32'];

  for (const dtype of dtypes) {
    console.log(`\n${'═'.repeat(80)}`);
    console.log(`DTYPE: ${dtype}`);
    console.log('═'.repeat(80));

    // Element-wise ADD
    console.log('\n📊 add (a + b)');
    for (const size of Object.values(config.sizes)) {
      const { a, b, c } = createTestArrays(size, dtype);
      const bytes = size * (dtype === 'float64' ? 8 : 4) * 3;

      // TypeScript single
      const tsTime = await measure(() => {
        dtype === 'float64'
          ? tsOps.add_f64(a as Float64Array, b as Float64Array, c as Float64Array)
          : tsOps.add_f32(a as Float32Array, b as Float32Array, c as Float32Array);
      }, config.warmupRuns, config.measureRuns);
      results.push({ impl: 'ts-single', op: 'add', dtype, size, timeMs: tsTime.mean, throughput: calculateThroughput(bytes, tsTime.mean) });

      // TypeScript parallel
      const tsParTime = await measure(
        () => tsParallel.parallelAdd(a, b, c, dtype),
        config.warmupRuns,
        config.measureRuns
      );
      results.push({ impl: 'ts-parallel', op: 'add', dtype, size, timeMs: tsParTime.mean, throughput: calculateThroughput(bytes, tsParTime.mean) });

      // Rust WASM single
      if (rustWasm) {
        const aW = rustWasm.copyToWasm(a);
        const bW = rustWasm.copyToWasm(b);
        const cW = rustWasm.alloc(size, dtype);
        const rustTime = await measure(() => {
          rustWasm!.add(aW.ptr, bW.ptr, cW.ptr, size, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'rust-single', op: 'add', dtype, size, timeMs: rustTime.mean, throughput: calculateThroughput(bytes, rustTime.mean) });
        rustWasm.free(aW.ptr, aW.len, dtype);
        rustWasm.free(bW.ptr, bW.len, dtype);
        rustWasm.free(cW.ptr, cW.len, dtype);
      }

      // Rust parallel
      if (rustParallel) {
        const rustParTime = await measure(
          () => rustParallel!.parallelAdd(a, b, c, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'rust-parallel', op: 'add', dtype, size, timeMs: rustParTime.mean, throughput: calculateThroughput(bytes, rustParTime.mean) });
      }

      // Zig WASM single
      if (zigWasm) {
        const aW = zigWasm.copyToWasm(a);
        const bW = zigWasm.copyToWasm(b);
        const cW = zigWasm.alloc(size, dtype);
        const zigTime = await measure(() => {
          zigWasm!.add(aW.ptr, bW.ptr, cW.ptr, size, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'zig-single', op: 'add', dtype, size, timeMs: zigTime.mean, throughput: calculateThroughput(bytes, zigTime.mean) });
        zigWasm.free(aW.ptr, aW.len, dtype);
        zigWasm.free(bW.ptr, bW.len, dtype);
        zigWasm.free(cW.ptr, cW.len, dtype);
      }

      // Zig parallel
      if (zigParallel) {
        const zigParTime = await measure(
          () => zigParallel!.parallelAdd(a, b, c, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'zig-parallel', op: 'add', dtype, size, timeMs: zigParTime.mean, throughput: calculateThroughput(bytes, zigParTime.mean) });
      }

      printRow('add', dtype, size, results);
    }

    // SUM reduction
    console.log('\n📊 sum');
    for (const size of Object.values(config.sizes)) {
      const { a } = createTestArrays(size, dtype);
      const bytes = size * (dtype === 'float64' ? 8 : 4);

      const tsTime = await measure(() => {
        dtype === 'float64' ? tsOps.sum_f64(a as Float64Array) : tsOps.sum_f32(a as Float32Array);
      }, config.warmupRuns, config.measureRuns);
      results.push({ impl: 'ts-single', op: 'sum', dtype, size, timeMs: tsTime.mean, throughput: calculateThroughput(bytes, tsTime.mean) });

      const tsParTime = await measure(
        () => tsParallel.parallelSum(a, dtype),
        config.warmupRuns,
        config.measureRuns
      );
      results.push({ impl: 'ts-parallel', op: 'sum', dtype, size, timeMs: tsParTime.mean, throughput: calculateThroughput(bytes, tsParTime.mean) });

      if (rustWasm) {
        const aW = rustWasm.copyToWasm(a);
        const rustTime = await measure(() => {
          rustWasm!.sum(aW.ptr, size, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'rust-single', op: 'sum', dtype, size, timeMs: rustTime.mean, throughput: calculateThroughput(bytes, rustTime.mean) });
        rustWasm.free(aW.ptr, aW.len, dtype);
      }

      if (rustParallel) {
        const rustParTime = await measure(
          () => rustParallel!.parallelSum(a, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'rust-parallel', op: 'sum', dtype, size, timeMs: rustParTime.mean, throughput: calculateThroughput(bytes, rustParTime.mean) });
      }

      if (zigWasm) {
        const aW = zigWasm.copyToWasm(a);
        const zigTime = await measure(() => {
          zigWasm!.sum(aW.ptr, size, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'zig-single', op: 'sum', dtype, size, timeMs: zigTime.mean, throughput: calculateThroughput(bytes, zigTime.mean) });
        zigWasm.free(aW.ptr, aW.len, dtype);
      }

      if (zigParallel) {
        const zigParTime = await measure(
          () => zigParallel!.parallelSum(a, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'zig-parallel', op: 'sum', dtype, size, timeMs: zigParTime.mean, throughput: calculateThroughput(bytes, zigParTime.mean) });
      }

      printRow('sum', dtype, size, results);
    }

    // MATMUL
    console.log('\n📊 matmul');
    for (const n of Object.values(config.matmulSizes)) {
      const { a, b, c } = createTestMatrices(n, dtype);
      const bytes = n * n * (dtype === 'float64' ? 8 : 4) * 3;

      const tsTime = await measure(() => {
        dtype === 'float64'
          ? tsOps.matmul_f64(a as Float64Array, b as Float64Array, c as Float64Array, n, n, n)
          : tsOps.matmul_f32(a as Float32Array, b as Float32Array, c as Float32Array, n, n, n);
      }, config.warmupRuns, config.measureRuns);
      results.push({ impl: 'ts-single', op: 'matmul', dtype, size: n * n, timeMs: tsTime.mean, throughput: calculateThroughput(bytes, tsTime.mean) });

      const tsParTime = await measure(
        () => tsParallel.parallelMatmul(a, b, c, n, n, n, dtype),
        config.warmupRuns,
        config.measureRuns
      );
      results.push({ impl: 'ts-parallel', op: 'matmul', dtype, size: n * n, timeMs: tsParTime.mean, throughput: calculateThroughput(bytes, tsParTime.mean) });

      if (rustWasm) {
        const aW = rustWasm.copyToWasm(a);
        const bW = rustWasm.copyToWasm(b);
        const cW = rustWasm.alloc(n * n, dtype);
        const rustTime = await measure(() => {
          rustWasm!.matmul(aW.ptr, bW.ptr, cW.ptr, n, n, n, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'rust-single', op: 'matmul', dtype, size: n * n, timeMs: rustTime.mean, throughput: calculateThroughput(bytes, rustTime.mean) });
        rustWasm.free(aW.ptr, aW.len, dtype);
        rustWasm.free(bW.ptr, bW.len, dtype);
        rustWasm.free(cW.ptr, cW.len, dtype);
      }

      if (rustParallel) {
        const rustParTime = await measure(
          () => rustParallel!.parallelMatmul(a, b, c, n, n, n, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'rust-parallel', op: 'matmul', dtype, size: n * n, timeMs: rustParTime.mean, throughput: calculateThroughput(bytes, rustParTime.mean) });
      }

      if (zigWasm) {
        const aW = zigWasm.copyToWasm(a);
        const bW = zigWasm.copyToWasm(b);
        const cW = zigWasm.alloc(n * n, dtype);
        const zigTime = await measure(() => {
          zigWasm!.matmul(aW.ptr, bW.ptr, cW.ptr, n, n, n, dtype, false);
        }, config.warmupRuns, config.measureRuns);
        results.push({ impl: 'zig-single', op: 'matmul', dtype, size: n * n, timeMs: zigTime.mean, throughput: calculateThroughput(bytes, zigTime.mean) });
        zigWasm.free(aW.ptr, aW.len, dtype);
        zigWasm.free(bW.ptr, bW.len, dtype);
        zigWasm.free(cW.ptr, cW.len, dtype);
      }

      if (zigParallel) {
        const zigParTime = await measure(
          () => zigParallel!.parallelMatmul(a, b, c, n, n, n, dtype),
          config.warmupRuns,
          config.measureRuns
        );
        results.push({ impl: 'zig-parallel', op: 'matmul', dtype, size: n * n, timeMs: zigParTime.mean, throughput: calculateThroughput(bytes, zigParTime.mean) });
      }

      printRow('matmul', dtype, n * n, results);
    }
  }

  // Cleanup
  tsParallel.terminate();
  rustParallel?.terminate();
  zigParallel?.terminate();

  // Print summary
  printSummary();
}

function printRow(op: string, dtype: string, size: number, allResults: Result[]) {
  const relevant = allResults.filter(r => r.op === op && r.dtype === dtype && r.size === size);
  const baseline = relevant.find(r => r.impl === 'ts-single');
  if (!baseline) return;

  const sizeStr = size >= 1_000_000 ? `${(size / 1_000_000).toFixed(1)}M` : size >= 1000 ? `${(size / 1000).toFixed(0)}K` : size.toString();

  console.log(`  ${sizeStr.padEnd(8)} | ${baseline.timeMs.toFixed(3).padStart(10)} ms (baseline)`);
  for (const r of relevant.filter(r => r.impl !== 'ts-single')) {
    const speedup = baseline.timeMs / r.timeMs;
    const speedupStr = speedup >= 1 ? `${speedup.toFixed(2)}x faster` : `${(1 / speedup).toFixed(2)}x slower`;
    console.log(`           | ${r.timeMs.toFixed(3).padStart(10)} ms ${r.impl.padEnd(14)} ${speedupStr}`);
  }
}

function printSummary() {
  console.log(`\n${'═'.repeat(80)}`);
  console.log('SUMMARY: Parallel speedup over single-threaded');
  console.log('═'.repeat(80));

  const impls = ['ts', 'rust', 'zig'];
  const ops = ['add', 'sum', 'matmul'];

  for (const impl of impls) {
    const singleResults = results.filter(r => r.impl === `${impl}-single`);
    const parallelResults = results.filter(r => r.impl === `${impl}-parallel`);

    if (singleResults.length === 0 || parallelResults.length === 0) continue;

    console.log(`\n${impl.toUpperCase()}:`);
    for (const op of ops) {
      const singleOp = singleResults.filter(r => r.op === op);
      const parallelOp = parallelResults.filter(r => r.op === op);

      if (singleOp.length === 0 || parallelOp.length === 0) continue;

      const speedups = singleOp.map((s, i) => s.timeMs / parallelOp[i]!.timeMs);
      const avg = speedups.reduce((a, b) => a + b, 0) / speedups.length;
      const min = Math.min(...speedups);
      const max = Math.max(...speedups);

      console.log(`  ${op.padEnd(8)}: ${avg.toFixed(2)}x avg (${min.toFixed(2)}x - ${max.toFixed(2)}x)`);
    }
  }

  console.log(`\nNote: Parallel speedup is limited by:`);
  console.log(`  - Worker spawning overhead (significant for small arrays)`);
  console.log(`  - Memory copy overhead (SharedArrayBuffer helps but not zero-copy to WASM)`);
  console.log(`  - Memory bandwidth (large arrays become memory-bound)`);
  console.log();
}

runBenchmarks().catch(console.error);
