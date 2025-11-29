/**
 * TypeScript/numpy-ts benchmark runner with auto-calibration
 *
 * Improved benchmarking approach:
 * - Auto-calibrates to run enough ops to hit minimum measurement time (100ms)
 * - Runs operations in batches to reduce measurement overhead
 * - Provides ops/sec for easier interpretation
 * - Uses multiple samples for statistical robustness
 */

import { performance } from 'perf_hooks';
import * as np from '../../src/index';
import { serializeNpy, parseNpy, serializeNpzSync, parseNpzSync } from '../../src/io';
import type { BenchmarkCase, BenchmarkTiming, BenchmarkSetup } from './types';

// Benchmark configuration - can be overridden
let MIN_SAMPLE_TIME_MS = 100; // Minimum time per sample (reduces noise)
let TARGET_SAMPLES = 5; // Number of samples to collect for statistics

export function setBenchmarkConfig(minSampleTimeMs: number, targetSamples: number): void {
  MIN_SAMPLE_TIME_MS = minSampleTimeMs;
  TARGET_SAMPLES = targetSamples;
}

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

function setupArrays(setup: BenchmarkSetup, operation?: string): Record<string, any> {
  const arrays: Record<string, any> = {};

  for (const [key, spec] of Object.entries(setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = spec;

    // Handle scalar values
    if (
      key === 'n' ||
      key === 'axis' ||
      key === 'new_shape' ||
      key === 'shape' ||
      key === 'fill_value'
    ) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape') {
        arrays[key] = shape;
      }
      continue;
    }

    // Create arrays - check 'value' first to avoid default fill creating zeros
    if (value !== undefined) {
      arrays[key] = np.full(shape, value, dtype);
    } else if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'random') {
      // Create random-like data using arange for consistency
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    } else if (fill === 'arange') {
      const size = shape.reduce((a, b) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = flat.reshape(...shape);
    }
  }

  // Pre-serialize data for parsing benchmarks
  if (operation === 'parseNpy' && arrays['a']) {
    arrays['_npyBytes'] = serializeNpy(arrays['a']);
  } else if (operation === 'serializeNpzSync' && arrays['a']) {
    // Create object for NPZ serialization
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzArrays'] = npzArrays;
  } else if (operation === 'parseNpzSync' && arrays['a']) {
    // Create and pre-serialize NPZ
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzBytes'] = serializeNpzSync(npzArrays);
  }

  return arrays;
}

function executeOperation(operation: string, arrays: Record<string, any>): any {
  // Array creation
  if (operation === 'zeros') {
    return np.zeros(arrays['shape']);
  } else if (operation === 'ones') {
    return np.ones(arrays['shape']);
  } else if (operation === 'empty') {
    return np.empty(arrays['shape']);
  } else if (operation === 'full') {
    return np.full(arrays['shape'], arrays['fill_value']);
  } else if (operation === 'arange') {
    return np.arange(0, arrays['n'], 1);
  } else if (operation === 'linspace') {
    return np.linspace(0, 100, arrays['n']);
  } else if (operation === 'logspace') {
    return np.logspace(0, 3, arrays['n']);
  } else if (operation === 'geomspace') {
    return np.geomspace(1, 1000, arrays['n']);
  } else if (operation === 'eye') {
    return np.eye(arrays['n']);
  } else if (operation === 'identity') {
    return np.identity(arrays['n']);
  } else if (operation === 'copy') {
    return np.copy(arrays['a']);
  } else if (operation === 'zeros_like') {
    return np.zeros_like(arrays['a']);
  } else if (operation === 'ones_like') {
    return np.ones_like(arrays['a']);
  } else if (operation === 'empty_like') {
    return np.empty_like(arrays['a']);
  } else if (operation === 'full_like') {
    return np.full_like(arrays['a'], 7);
  }

  // Arithmetic
  else if (operation === 'add') {
    return arrays['a'].add(arrays['b']);
  } else if (operation === 'subtract') {
    return arrays['a'].subtract(arrays['b']);
  } else if (operation === 'multiply') {
    return arrays['a'].multiply(arrays['b']);
  } else if (operation === 'divide') {
    return arrays['a'].divide(arrays['b']);
  } else if (operation === 'mod') {
    return arrays['a'].mod(arrays['b']);
  } else if (operation === 'floor_divide') {
    return arrays['a'].floor_divide(arrays['b']);
  } else if (operation === 'reciprocal') {
    return arrays['a'].reciprocal();
  } else if (operation === 'positive') {
    return arrays['a'].positive();
  }

  // Mathematical operations
  else if (operation === 'sqrt') {
    return arrays['a'].sqrt();
  } else if (operation === 'power') {
    return arrays['a'].power(arrays['b']);
  } else if (operation === 'absolute') {
    return arrays['a'].absolute();
  } else if (operation === 'negative') {
    return arrays['a'].negative();
  } else if (operation === 'sign') {
    return arrays['a'].sign();
  }

  // Linear algebra
  else if (operation === 'dot') {
    return arrays['a'].dot(arrays['b']);
  } else if (operation === 'inner') {
    return arrays['a'].inner(arrays['b']);
  } else if (operation === 'outer') {
    return arrays['a'].outer(arrays['b']);
  } else if (operation === 'tensordot') {
    const axes = arrays['axes'] ?? 2;
    return arrays['a'].tensordot(arrays['b'], axes);
  } else if (operation === 'matmul') {
    return arrays['a'].matmul(arrays['b']);
  } else if (operation === 'trace') {
    return arrays['a'].trace();
  } else if (operation === 'transpose') {
    return arrays['a'].transpose();
  }

  // Reductions
  else if (operation === 'sum') {
    const axis = arrays['axis'];
    return arrays['a'].sum(axis);
  } else if (operation === 'mean') {
    const axis = arrays['axis'];
    return arrays['a'].mean(axis);
  } else if (operation === 'max') {
    const axis = arrays['axis'];
    return arrays['a'].max(axis);
  } else if (operation === 'min') {
    const axis = arrays['axis'];
    return arrays['a'].min(axis);
  } else if (operation === 'prod') {
    const axis = arrays['axis'];
    return arrays['a'].prod(axis);
  } else if (operation === 'argmin') {
    const axis = arrays['axis'];
    return arrays['a'].argmin(axis);
  } else if (operation === 'argmax') {
    const axis = arrays['axis'];
    return arrays['a'].argmax(axis);
  } else if (operation === 'var') {
    const axis = arrays['axis'];
    return arrays['a'].var(axis);
  } else if (operation === 'std') {
    const axis = arrays['axis'];
    return arrays['a'].std(axis);
  } else if (operation === 'all') {
    const axis = arrays['axis'];
    return arrays['a'].all(axis);
  } else if (operation === 'any') {
    const axis = arrays['axis'];
    return arrays['a'].any(axis);
  }

  // Reshape
  else if (operation === 'reshape') {
    return arrays['a'].reshape(...arrays['new_shape']);
  } else if (operation === 'flatten') {
    return arrays['a'].flatten();
  } else if (operation === 'ravel') {
    return arrays['a'].ravel();
  } else if (operation === 'squeeze') {
    return arrays['a'].squeeze();
  }

  // Slicing
  else if (operation === 'slice') {
    return arrays['a'].slice('0:100', '0:100');
  }

  // IO operations
  else if (operation === 'serializeNpy') {
    return serializeNpy(arrays['a']);
  } else if (operation === 'parseNpy') {
    // arrays['_npyBytes'] is pre-serialized in setupArrays
    return parseNpy(arrays['_npyBytes']);
  } else if (operation === 'serializeNpzSync') {
    // arrays['_npzArrays'] contains the object {a, b} for serialization
    return serializeNpzSync(arrays['_npzArrays']);
  } else if (operation === 'parseNpzSync') {
    // arrays['_npzBytes'] is pre-serialized in setupArrays
    return parseNpzSync(arrays['_npzBytes']);
  }

  throw new Error(`Unknown operation: ${operation}`);
}

/**
 * Auto-calibrate: Determine how many operations to run per sample
 * to achieve the target minimum sample time
 */
function calibrateOpsPerSample(
  operation: string,
  arrays: Record<string, any>,
  targetTimeMs: number = MIN_SAMPLE_TIME_MS
): number {
  let opsPerSample = 1;
  let calibrationRuns = 0;
  const maxCalibrationRuns = 10;

  while (calibrationRuns < maxCalibrationRuns) {
    const start = performance.now();

    // Run operations in batch
    for (let i = 0; i < opsPerSample; i++) {
      const result = executeOperation(operation, arrays);
      void result; // Prevent optimization
    }

    const elapsed = performance.now() - start;

    // If we hit the target time, we're done
    if (elapsed >= targetTimeMs) {
      break;
    }

    // If operation is very fast, increase ops exponentially
    if (elapsed < targetTimeMs / 10) {
      opsPerSample *= 10;
    } else if (elapsed < targetTimeMs / 2) {
      opsPerSample *= 2;
    } else {
      // Close enough, calculate exact number needed
      const targetOps = Math.ceil((opsPerSample * targetTimeMs) / elapsed);
      opsPerSample = Math.max(targetOps, opsPerSample + 1);
      break;
    }

    calibrationRuns++;
  }

  // Cap at reasonable maximum to prevent too-long samples
  const maxOpsPerSample = 100000;
  return Math.min(opsPerSample, maxOpsPerSample);
}

export async function runBenchmark(spec: BenchmarkCase): Promise<BenchmarkTiming> {
  const { name, operation, setup, warmup } = spec;

  // Setup arrays (pass operation for IO benchmarks that need pre-serialized data)
  const arrays = setupArrays(setup, operation);

  // Warmup phase - run several times to stabilize JIT
  for (let i = 0; i < warmup; i++) {
    executeOperation(operation, arrays);
  }

  // Calibration phase - determine ops per sample
  const opsPerSample = calibrateOpsPerSample(operation, arrays);

  // Benchmark phase - collect samples
  const sampleTimes: number[] = [];
  let totalOps = 0;

  for (let sample = 0; sample < TARGET_SAMPLES; sample++) {
    const start = performance.now();

    // Run batch of operations
    for (let i = 0; i < opsPerSample; i++) {
      const result = executeOperation(operation, arrays);
      void result; // Prevent optimization
    }

    const elapsed = performance.now() - start;
    const timePerOp = elapsed / opsPerSample;
    sampleTimes.push(timePerOp);
    totalOps += opsPerSample;
  }

  // Calculate statistics
  const meanMs = mean(sampleTimes);
  const medianMs = median(sampleTimes);
  const minMs = Math.min(...sampleTimes);
  const maxMs = Math.max(...sampleTimes);
  const stdMs = std(sampleTimes);
  const opsPerSec = 1000 / meanMs;

  return {
    name,
    mean_ms: meanMs,
    median_ms: medianMs,
    min_ms: minMs,
    max_ms: maxMs,
    std_ms: stdMs,
    ops_per_sec: opsPerSec,
    total_ops: totalOps,
    total_samples: TARGET_SAMPLES,
  };
}

export async function runBenchmarks(specs: BenchmarkCase[]): Promise<BenchmarkTiming[]> {
  const results: BenchmarkTiming[] = [];

  console.error(`Node ${process.version}`);
  console.error(`Running ${specs.length} benchmarks with auto-calibration...`);
  console.error(
    `Target: ${MIN_SAMPLE_TIME_MS}ms per sample, ${TARGET_SAMPLES} samples per benchmark\n`
  );

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const result = await runBenchmark(spec);
    results.push(result);

    console.error(
      `  [${i + 1}/${specs.length}] ${spec.name.padEnd(40)} ${result.mean_ms.toFixed(3).padStart(8)}ms  ${Math.round(result.ops_per_sec).toLocaleString().padStart(12)} ops/sec`
    );
  }

  return results;
}
