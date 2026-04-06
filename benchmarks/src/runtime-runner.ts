/**
 * Standalone benchmark runner for multi-runtime support.
 *
 * This script is spawned as a subprocess under Node, Deno, or Bun.
 * It reads benchmark specs + config from stdin (JSON), executes benchmarks,
 * prints progress to stderr, and outputs BenchmarkTiming[] as JSON to stdout.
 *
 * Uses globalThis.performance.now() so it works across all runtimes without
 * needing to import perf_hooks.
 */

import { wasmConfig, wasmMemoryConfig } from '../../src/common/wasm/config';
import { wasmFreeBytes, configureWasm, resetScratchAllocator } from '../../src/common/wasm/runtime';
import type { BenchmarkCase, BenchmarkTiming } from './types';
import { setupArrays, executeOperation } from './bench-utils';

// Set to true to log WASM heap usage after each benchmark spec
const LOG_HEAP = !!process.env['LOG_HEAP'];

// Benchmark configuration - set from stdin input
let MIN_SAMPLE_TIME_MS = 100;
let TARGET_SAMPLES = 5;

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

function std(arr: number[]): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function releaseResult(result: unknown): void {
  if (result == null) return;
  if (typeof (result as any).dispose === 'function') {
    (result as any).dispose();
  } else if (result instanceof Map) {
    for (const val of result.values()) releaseResult(val);
  } else if (Array.isArray(result)) {
    for (const item of result) (item as any)?.dispose?.();
  } else if (typeof result === 'object' && !(ArrayBuffer.isView(result) || result instanceof ArrayBuffer)) {
    for (const val of Object.values(result as Record<string, unknown>)) {
      releaseResult(val);
    }
  }
}

function calibrateOpsPerSample(
  operation: string,
  arrays: Record<string, any>,
  targetTimeMs: number = MIN_SAMPLE_TIME_MS
): number {
  let opsPerSample = 1;
  let calibrationRuns = 0;
  const maxCalibrationRuns = 10;

  while (calibrationRuns < maxCalibrationRuns) {
    const start = globalThis.performance.now();
    for (let i = 0; i < opsPerSample; i++) {
      releaseResult(executeOperation(operation, arrays));
    }
    const elapsed = globalThis.performance.now() - start;

    if (elapsed >= targetTimeMs) break;

    if (elapsed < targetTimeMs / 10) {
      opsPerSample *= 10;
    } else if (elapsed < targetTimeMs / 2) {
      opsPerSample *= 2;
    } else {
      const targetOps = Math.ceil((opsPerSample * targetTimeMs) / elapsed);
      opsPerSample = Math.max(targetOps, opsPerSample + 1);
      break;
    }
    calibrationRuns++;
  }

  return Math.min(opsPerSample, 100000);
}

function runBenchmark(spec: BenchmarkCase): BenchmarkTiming {
  const { name, operation, setup, warmup } = spec;
  const arrays = setupArrays(setup, operation);

  // Warmup — also used to detect WASM kernel usage
  wasmConfig.wasmCallCount = 0;
  for (let i = 0; i < warmup; i++) {
    releaseResult(executeOperation(operation, arrays));
  }
  const wasmUsed = wasmConfig.wasmCallCount > 0;

  // Calibrate
  const opsPerSample = calibrateOpsPerSample(operation, arrays);

  // Benchmark
  const sampleTimes: number[] = [];
  let totalOps = 0;

  for (let sample = 0; sample < TARGET_SAMPLES; sample++) {
    const start = globalThis.performance.now();
    for (let i = 0; i < opsPerSample; i++) {
      releaseResult(executeOperation(operation, arrays));
    }
    const elapsed = globalThis.performance.now() - start;
    sampleTimes.push(elapsed / opsPerSample);
    totalOps += opsPerSample;
  }

  // Release setup arrays and flush any temp heap allocs from the last kernel
  for (const val of Object.values(arrays)) {
    (val as any)?.dispose?.();
  }
  resetScratchAllocator();

  if (LOG_HEAP) {
    const free = wasmFreeBytes();
    const total = wasmMemoryConfig.maxMemoryBytes;
    const pct = (((total - free) / total) * 100).toFixed(1);
    console.error(`  [heap] ${name}: ${pct}% used, ${(free / (1024 * 1024)).toFixed(1)}MB free`);
  }

  const meanMs = mean(sampleTimes);
  return {
    name,
    mean_ms: meanMs,
    median_ms: median(sampleTimes),
    min_ms: Math.min(...sampleTimes),
    max_ms: Math.max(...sampleTimes),
    std_ms: std(sampleTimes),
    ops_per_sec: 1000 / meanMs,
    total_ops: totalOps,
    total_samples: TARGET_SAMPLES,
    wasmUsed,
  };
}

// --- Main: read stdin, run benchmarks, write stdout ---

async function main() {
  // Read all of stdin
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk));
  }
  const input = JSON.parse(Buffer.concat(chunks).toString('utf-8'));
  const specs: BenchmarkCase[] = input.specs;
  const config = input.config;

  // Bump WASM memory for large array benchmarks (1 GiB, scratch auto-scales to 32 MiB)
  if (config.sizeScale === 'large') {
    configureWasm({ maxMemory: 1024 * 1024 * 1024 });
    console.error('WASM max memory set to 1 GiB for large benchmarks');
  }

  MIN_SAMPLE_TIME_MS = config.minSampleTimeMs;
  TARGET_SAMPLES = config.targetSamples;
  const expectedWasm: Record<string, boolean> = config.expectedWasm ?? {};
  const hasExpectedWasm = Object.keys(expectedWasm).length > 0;
  if (config.noWasm) {
    wasmConfig.thresholdMultiplier = Infinity;
  }

  // Detect runtime name and version
  let runtimeName = 'node';
  let runtimeVersion = typeof process !== 'undefined' ? process.version : 'unknown';

  // Deno sets Deno global, Bun sets Bun global
  if (typeof (globalThis as any).Deno !== 'undefined') {
    runtimeName = 'deno';
    runtimeVersion = (globalThis as any).Deno.version.deno;
  } else if (typeof (globalThis as any).Bun !== 'undefined') {
    runtimeName = 'bun';
    runtimeVersion = (globalThis as any).Bun.version;
  }

  console.error(`${runtimeName} ${runtimeVersion}`);
  console.error(`Running ${specs.length} benchmarks with auto-calibration...`);
  console.error(
    `Target: ${MIN_SAMPLE_TIME_MS}ms per sample, ${TARGET_SAMPLES} samples per benchmark\n`
  );

  const results: BenchmarkTiming[] = [];

  for (let i = 0; i < specs.length; i++) {
    // GC between specs to flush FinalizationRegistry callbacks and free WASM memory
    if (typeof globalThis.gc === 'function') globalThis.gc();
    else if (typeof globalThis.Bun !== 'undefined') (globalThis as any).Bun.gc(true);

    const spec = specs[i]!;
    try {
      const result = runBenchmark(spec);
      results.push(result);

      const wasmKey = spec.name.replace(/\s*\[.*?\]/g, '').replace(/\(.*?\)/g, '()').trim();
      let wasmWarn = '';
      if (hasExpectedWasm) {
        if (!(wasmKey in expectedWasm)) {
          wasmWarn = '  ⚠ no WASM reference';
        } else if (expectedWasm[wasmKey] && !result.wasmUsed) {
          wasmWarn = '  ⚠ WASM regression';
        }
      }
      console.error(
        `  [${i + 1}/${specs.length}] ${spec.name.padEnd(40)} ${result.mean_ms.toFixed(3).padStart(8)}ms  ${Math.round(result.ops_per_sec).toLocaleString().padStart(12)} ops/sec${wasmWarn}`
      );
    } catch (err) {
      // Push a placeholder result so indices stay aligned with Python results
      results.push({
        name: spec.name,
        mean_ms: 0,
        median_ms: 0,
        min_ms: 0,
        max_ms: 0,
        std_ms: 0,
        ops_per_sec: 0,
        total_ops: 0,
        total_samples: 0,
        wasmUsed: false,
      });
      console.error(
        `  [${i + 1}/${specs.length}] ${spec.name.padEnd(40)}   FAILED: ${err instanceof Error ? err.message : err}`
      );
    }
  }

  // Output results as JSON to stdout
  const output = JSON.stringify(results);
  process.stdout.write(output);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
