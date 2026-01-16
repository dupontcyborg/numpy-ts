#!/usr/bin/env npx tsx

/**
 * WASM Benchmark Runner
 *
 * Compares performance of:
 * - Pure TypeScript
 * - Rust WASM (single-threaded)
 * - Rust WASM (multi-threaded)
 * - Zig WASM (single-threaded)
 * - Zig WASM (multi-threaded)
 *
 * Operations tested:
 * - Element-wise: add (a + b), sin(a)
 * - Reduction: sum
 * - Matrix: matmul
 *
 * Usage:
 *   npx tsx run.ts           # Full benchmark
 *   npx tsx run.ts --quick   # Quick benchmark with smaller sizes
 */

import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

import {
  type BenchmarkResult,
  type BenchmarkConfig,
  DEFAULT_CONFIG,
  QUICK_CONFIG,
  measure,
  calculateThroughput,
  formatResultsTable,
  createTestArrays,
  createTestMatrices,
} from './src/harness.js';

import * as tsOps from './src/typescript/ops.js';
import { loadWasmModule, WasmOps } from './src/bridges/wasm-loader.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ============================================================================
// Configuration
// ============================================================================

const isQuick = process.argv.includes('--quick');
const config: BenchmarkConfig = isQuick ? QUICK_CONFIG : DEFAULT_CONFIG;

console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                     numpy-ts WASM Performance Benchmark                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Comparing: TypeScript vs Rust WASM vs Zig WASM                              ║
║  Operations: add, sin, sum, matmul                                           ║
║  Data types: float32, float64                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);

console.log(`Mode: ${isQuick ? 'QUICK' : 'FULL'}`);
console.log(`Warmup runs: ${config.warmupRuns}`);
console.log(`Measure runs: ${config.measureRuns}`);
console.log(`Array sizes: ${Object.values(config.sizes).join(', ')}`);
console.log(`Matrix sizes: ${Object.values(config.matmulSizes).map(n => `${n}x${n}`).join(', ')}`);
console.log();

// ============================================================================
// WASM Module Loading
// ============================================================================

interface WasmModules {
  rust?: WasmOps;
  rustMt?: WasmOps;
  zig?: WasmOps;
  zigMt?: WasmOps;
}

async function loadWasmModules(): Promise<WasmModules> {
  const modules: WasmModules = {};

  const distDir = join(__dirname, 'dist');

  // Try to load Rust WASM
  const rustPath = join(distDir, 'numpy_bench.wasm');
  if (existsSync(rustPath)) {
    const mod = await loadWasmModule(rustPath, 'rust-wasm', false);
    if (mod) modules.rust = new WasmOps(mod);
  }

  // Try to load Rust WASM with threads
  const rustMtPath = join(distDir, 'numpy_bench_threads.wasm');
  if (existsSync(rustMtPath)) {
    const mod = await loadWasmModule(rustMtPath, 'rust-wasm-mt', true);
    if (mod) {
      modules.rustMt = new WasmOps(mod);
      modules.rustMt.initThreadPool(4);
    }
  }

  // Try to load Zig WASM (check multiple locations)
  const zigPaths = [
    join(distDir, 'zig_numpy_bench.wasm'),
    join(__dirname, 'zig', 'zig-out', 'bin', 'numpy_bench.wasm'),
    join(__dirname, 'zig', 'zig-out', 'lib', 'numpy_bench.wasm'),
  ];

  for (const zigPath of zigPaths) {
    if (existsSync(zigPath) && !modules.zig) {
      const mod = await loadWasmModule(zigPath, 'zig-wasm', false);
      if (mod) {
        modules.zig = new WasmOps(mod);
        break;
      }
    }
  }

  // Try to load Zig WASM with threads
  const zigMtPaths = [
    join(distDir, 'zig_numpy_bench_threads.wasm'),
    join(__dirname, 'zig', 'zig-out', 'bin', 'numpy_bench_threads.wasm'),
    join(__dirname, 'zig', 'zig-out', 'lib', 'numpy_bench_threads.wasm'),
  ];

  for (const zigMtPath of zigMtPaths) {
    if (existsSync(zigMtPath) && !modules.zigMt) {
      const mod = await loadWasmModule(zigMtPath, 'zig-wasm-mt', true);
      if (mod) {
        modules.zigMt = new WasmOps(mod);
        break;
      }
    }
  }

  return modules;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

const results: BenchmarkResult[] = [];

type DType = 'float32' | 'float64';
type Implementation = 'typescript' | 'rust-wasm' | 'rust-wasm-mt' | 'zig-wasm' | 'zig-wasm-mt';

async function benchmarkAdd(
  size: number,
  dtype: DType,
  wasm: WasmModules
): Promise<void> {
  const { a, b, c } = createTestArrays(size, dtype);
  const bytesPerElement = dtype === 'float64' ? 8 : 4;
  const bytesProcessed = size * bytesPerElement * 3; // read a, read b, write c

  // TypeScript
  const tsResult = await measure(
    () => {
      if (dtype === 'float64') {
        tsOps.add_f64(a as Float64Array, b as Float64Array, c as Float64Array);
      } else {
        tsOps.add_f32(a as Float32Array, b as Float32Array, c as Float32Array);
      }
    },
    config.warmupRuns,
    config.measureRuns
  );

  results.push({
    name: `add-${dtype}-${size}`,
    implementation: 'typescript',
    operation: 'add',
    dtype,
    size,
    timeMs: tsResult.mean,
    opsPerSecond: size / (tsResult.mean / 1000),
    throughputGBps: calculateThroughput(bytesProcessed, tsResult.mean),
  });

  // WASM implementations
  const wasmImpls: Array<{ wasm: WasmOps | undefined; name: Implementation; mt: boolean }> = [
    { wasm: wasm.rust, name: 'rust-wasm', mt: false },
    { wasm: wasm.rustMt, name: 'rust-wasm-mt', mt: true },
    { wasm: wasm.zig, name: 'zig-wasm', mt: false },
    { wasm: wasm.zigMt, name: 'zig-wasm-mt', mt: true },
  ];

  for (const { wasm: wasmOps, name, mt } of wasmImpls) {
    if (!wasmOps) continue;

    try {
      const aWasm = wasmOps.copyToWasm(a);
      const bWasm = wasmOps.copyToWasm(b);
      const cWasm = wasmOps.alloc(size, dtype);

      const wasmResult = await measure(
        () => {
          wasmOps.add(aWasm.ptr, bWasm.ptr, cWasm.ptr, size, dtype, mt);
        },
        config.warmupRuns,
        config.measureRuns
      );

      results.push({
        name: `add-${dtype}-${size}`,
        implementation: name,
        operation: 'add',
        dtype,
        size,
        timeMs: wasmResult.mean,
        opsPerSecond: size / (wasmResult.mean / 1000),
        throughputGBps: calculateThroughput(bytesProcessed, wasmResult.mean),
      });

      // Cleanup
      wasmOps.free(aWasm.ptr, aWasm.len, dtype);
      wasmOps.free(bWasm.ptr, bWasm.len, dtype);
      wasmOps.free(cWasm.ptr, cWasm.len, dtype);
    } catch (e) {
      console.warn(`  ${name} failed for add-${dtype}-${size}:`, e);
    }
  }
}

async function benchmarkSin(
  size: number,
  dtype: DType,
  wasm: WasmModules
): Promise<void> {
  const { a, c } = createTestArrays(size, dtype);
  const bytesPerElement = dtype === 'float64' ? 8 : 4;
  const bytesProcessed = size * bytesPerElement * 2; // read a, write c

  // TypeScript
  const tsResult = await measure(
    () => {
      if (dtype === 'float64') {
        tsOps.sin_f64(a as Float64Array, c as Float64Array);
      } else {
        tsOps.sin_f32(a as Float32Array, c as Float32Array);
      }
    },
    config.warmupRuns,
    config.measureRuns
  );

  results.push({
    name: `sin-${dtype}-${size}`,
    implementation: 'typescript',
    operation: 'sin',
    dtype,
    size,
    timeMs: tsResult.mean,
    opsPerSecond: size / (tsResult.mean / 1000),
    throughputGBps: calculateThroughput(bytesProcessed, tsResult.mean),
  });

  // WASM implementations
  const wasmImpls: Array<{ wasm: WasmOps | undefined; name: Implementation; mt: boolean }> = [
    { wasm: wasm.rust, name: 'rust-wasm', mt: false },
    { wasm: wasm.rustMt, name: 'rust-wasm-mt', mt: true },
    { wasm: wasm.zig, name: 'zig-wasm', mt: false },
    { wasm: wasm.zigMt, name: 'zig-wasm-mt', mt: true },
  ];

  for (const { wasm: wasmOps, name, mt } of wasmImpls) {
    if (!wasmOps) continue;

    try {
      const aWasm = wasmOps.copyToWasm(a);
      const cWasm = wasmOps.alloc(size, dtype);

      const wasmResult = await measure(
        () => {
          wasmOps.sin(aWasm.ptr, cWasm.ptr, size, dtype, mt);
        },
        config.warmupRuns,
        config.measureRuns
      );

      results.push({
        name: `sin-${dtype}-${size}`,
        implementation: name,
        operation: 'sin',
        dtype,
        size,
        timeMs: wasmResult.mean,
        opsPerSecond: size / (wasmResult.mean / 1000),
        throughputGBps: calculateThroughput(bytesProcessed, wasmResult.mean),
      });

      // Cleanup
      wasmOps.free(aWasm.ptr, aWasm.len, dtype);
      wasmOps.free(cWasm.ptr, cWasm.len, dtype);
    } catch (e) {
      console.warn(`  ${name} failed for sin-${dtype}-${size}:`, e);
    }
  }
}

async function benchmarkSum(
  size: number,
  dtype: DType,
  wasm: WasmModules
): Promise<void> {
  const { a } = createTestArrays(size, dtype);
  const bytesPerElement = dtype === 'float64' ? 8 : 4;
  const bytesProcessed = size * bytesPerElement; // read a

  // TypeScript
  let tsSum = 0;
  const tsResult = await measure(
    () => {
      if (dtype === 'float64') {
        tsSum = tsOps.sum_f64(a as Float64Array);
      } else {
        tsSum = tsOps.sum_f32(a as Float32Array);
      }
    },
    config.warmupRuns,
    config.measureRuns
  );

  results.push({
    name: `sum-${dtype}-${size}`,
    implementation: 'typescript',
    operation: 'sum',
    dtype,
    size,
    timeMs: tsResult.mean,
    opsPerSecond: size / (tsResult.mean / 1000),
    throughputGBps: calculateThroughput(bytesProcessed, tsResult.mean),
  });

  // WASM implementations
  const wasmImpls: Array<{ wasm: WasmOps | undefined; name: Implementation; mt: boolean }> = [
    { wasm: wasm.rust, name: 'rust-wasm', mt: false },
    { wasm: wasm.rustMt, name: 'rust-wasm-mt', mt: true },
    { wasm: wasm.zig, name: 'zig-wasm', mt: false },
    { wasm: wasm.zigMt, name: 'zig-wasm-mt', mt: true },
  ];

  for (const { wasm: wasmOps, name, mt } of wasmImpls) {
    if (!wasmOps) continue;

    try {
      const aWasm = wasmOps.copyToWasm(a);

      let wasmSum = 0;
      const wasmResult = await measure(
        () => {
          wasmSum = wasmOps.sum(aWasm.ptr, size, dtype, mt);
        },
        config.warmupRuns,
        config.measureRuns
      );

      results.push({
        name: `sum-${dtype}-${size}`,
        implementation: name,
        operation: 'sum',
        dtype,
        size,
        timeMs: wasmResult.mean,
        opsPerSecond: size / (wasmResult.mean / 1000),
        throughputGBps: calculateThroughput(bytesProcessed, wasmResult.mean),
      });

      // Cleanup
      wasmOps.free(aWasm.ptr, aWasm.len, dtype);
    } catch (e) {
      console.warn(`  ${name} failed for sum-${dtype}-${size}:`, e);
    }
  }
}

async function benchmarkMatmul(
  n: number,
  dtype: DType,
  wasm: WasmModules
): Promise<void> {
  const { a, b, c } = createTestMatrices(n, dtype);
  const bytesPerElement = dtype === 'float64' ? 8 : 4;
  const bytesProcessed = n * n * bytesPerElement * 3; // read A, read B, write C
  const flops = 2 * n * n * n; // 2*n^3 FLOPs for matmul

  // TypeScript
  const tsResult = await measure(
    () => {
      if (dtype === 'float64') {
        tsOps.matmul_f64(a as Float64Array, b as Float64Array, c as Float64Array, n, n, n);
      } else {
        tsOps.matmul_f32(a as Float32Array, b as Float32Array, c as Float32Array, n, n, n);
      }
    },
    config.warmupRuns,
    config.measureRuns
  );

  results.push({
    name: `matmul-${dtype}-${n}x${n}`,
    implementation: 'typescript',
    operation: 'matmul',
    dtype,
    size: n * n,
    timeMs: tsResult.mean,
    opsPerSecond: flops / (tsResult.mean / 1000),
    throughputGBps: calculateThroughput(bytesProcessed, tsResult.mean),
  });

  // WASM implementations
  const wasmImpls: Array<{ wasm: WasmOps | undefined; name: Implementation; mt: boolean }> = [
    { wasm: wasm.rust, name: 'rust-wasm', mt: false },
    { wasm: wasm.rustMt, name: 'rust-wasm-mt', mt: true },
    { wasm: wasm.zig, name: 'zig-wasm', mt: false },
    { wasm: wasm.zigMt, name: 'zig-wasm-mt', mt: true },
  ];

  for (const { wasm: wasmOps, name, mt } of wasmImpls) {
    if (!wasmOps) continue;

    try {
      const aWasm = wasmOps.copyToWasm(a);
      const bWasm = wasmOps.copyToWasm(b);
      const cWasm = wasmOps.alloc(n * n, dtype);

      const wasmResult = await measure(
        () => {
          wasmOps.matmul(aWasm.ptr, bWasm.ptr, cWasm.ptr, n, n, n, dtype, mt);
        },
        config.warmupRuns,
        config.measureRuns
      );

      results.push({
        name: `matmul-${dtype}-${n}x${n}`,
        implementation: name,
        operation: 'matmul',
        dtype,
        size: n * n,
        timeMs: wasmResult.mean,
        opsPerSecond: flops / (wasmResult.mean / 1000),
        throughputGBps: calculateThroughput(bytesProcessed, wasmResult.mean),
      });

      // Cleanup
      wasmOps.free(aWasm.ptr, aWasm.len, dtype);
      wasmOps.free(bWasm.ptr, bWasm.len, dtype);
      wasmOps.free(cWasm.ptr, cWasm.len, dtype);
    } catch (e) {
      console.warn(`  ${name} failed for matmul-${dtype}-${n}:`, e);
    }
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('Loading WASM modules...');
  const wasm = await loadWasmModules();

  const loadedModules = [
    wasm.rust ? 'rust-wasm' : null,
    wasm.rustMt ? 'rust-wasm-mt' : null,
    wasm.zig ? 'zig-wasm' : null,
    wasm.zigMt ? 'zig-wasm-mt' : null,
  ].filter(Boolean);

  if (loadedModules.length === 0) {
    console.log('\n⚠️  No WASM modules found. Running TypeScript-only benchmarks.');
    console.log('   To build WASM modules:');
    console.log('   - Rust: cd rust && cargo build --release --target wasm32-unknown-unknown');
    console.log('   - Zig:  cd zig && zig build -Doptimize=ReleaseFast');
    console.log();
  } else {
    console.log(`Loaded WASM modules: ${loadedModules.join(', ')}`);
  }

  const dtypes: DType[] = ['float64', 'float32'];
  const sizes = Object.values(config.sizes);
  const matmulSizes = Object.values(config.matmulSizes);

  // Run benchmarks
  for (const dtype of dtypes) {
    console.log(`\n${'─'.repeat(80)}`);
    console.log(`Running ${dtype} benchmarks...`);
    console.log('─'.repeat(80));

    // Element-wise add
    console.log('\n📊 Benchmarking: add (a + b)');
    for (const size of sizes) {
      process.stdout.write(`  Size ${size.toLocaleString().padStart(12)}... `);
      await benchmarkAdd(size, dtype, wasm);
      console.log('done');
    }

    // Element-wise sin
    console.log('\n📊 Benchmarking: sin(a)');
    for (const size of sizes) {
      process.stdout.write(`  Size ${size.toLocaleString().padStart(12)}... `);
      await benchmarkSin(size, dtype, wasm);
      console.log('done');
    }

    // Sum reduction
    console.log('\n📊 Benchmarking: sum');
    for (const size of sizes) {
      process.stdout.write(`  Size ${size.toLocaleString().padStart(12)}... `);
      await benchmarkSum(size, dtype, wasm);
      console.log('done');
    }

    // Matrix multiplication
    console.log('\n📊 Benchmarking: matmul');
    for (const n of matmulSizes) {
      process.stdout.write(`  Size ${n}x${n}`.padEnd(20) + '... ');
      await benchmarkMatmul(n, dtype, wasm);
      console.log('done');
    }
  }

  // Print results
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════════════════════╗');
  console.log('║                              BENCHMARK RESULTS                               ║');
  console.log('╚══════════════════════════════════════════════════════════════════════════════╝');

  console.log(formatResultsTable(results));

  // Summary
  console.log('\n');
  console.log('═'.repeat(100));
  console.log('SUMMARY');
  console.log('═'.repeat(100));

  // Calculate average speedups by implementation
  const implementations = ['rust-wasm', 'rust-wasm-mt', 'zig-wasm', 'zig-wasm-mt'] as const;
  const operations = ['add', 'sin', 'sum', 'matmul'] as const;

  for (const impl of implementations) {
    const implResults = results.filter(r => r.implementation === impl);
    if (implResults.length === 0) continue;

    console.log(`\n${impl}:`);
    for (const op of operations) {
      const opResults = implResults.filter(r => r.operation === op);
      if (opResults.length === 0) continue;

      const speedups = opResults.map(r => {
        const tsResult = results.find(
          t => t.implementation === 'typescript' && t.operation === op && t.size === r.size && t.dtype === r.dtype
        );
        return tsResult ? tsResult.timeMs / r.timeMs : 1;
      });

      const avgSpeedup = speedups.reduce((a, b) => a + b, 0) / speedups.length;
      const minSpeedup = Math.min(...speedups);
      const maxSpeedup = Math.max(...speedups);

      console.log(`  ${op.padEnd(8)}: avg ${avgSpeedup.toFixed(2)}x (${minSpeedup.toFixed(2)}x - ${maxSpeedup.toFixed(2)}x)`);
    }
  }

  console.log('\n');
}

main().catch(console.error);
