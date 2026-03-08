/**
 * Benchmark type definitions
 */

import type { DType } from '../../src/core/dtype';

export interface BenchmarkSetup {
  [key: string]: {
    shape: number[];
    dtype?: DType;
    fill?: 'zeros' | 'ones' | 'random' | 'arange';
    value?: number;
  };
}

export interface BenchmarkCase {
  name: string;
  category: string;
  operation: string;
  setup: BenchmarkSetup;
  iterations: number;
  warmup: number;
  includeInQuick?: boolean;
}

export interface BenchmarkTiming {
  name: string;
  mean_ms: number;
  median_ms: number;
  min_ms: number;
  max_ms: number;
  std_ms: number;
  ops_per_sec: number; // Operations per second
  total_ops: number; // Total operations executed
  total_samples: number; // Number of timing samples taken
}

export interface BenchmarkComparison {
  name: string;
  category: string;
  numpy: BenchmarkTiming;
  numpyjs: BenchmarkTiming;
  ratio: number; // numpyjs / numpy (how many times slower)
}

export interface BenchmarkSummary {
  avg_slowdown: number;
  median_slowdown: number;
  best_case: number;
  worst_case: number;
  total_benchmarks: number;
}

export interface BenchmarkReport {
  timestamp: string;
  environment: {
    node_version: string;
    python_version?: string;
    numpy_version?: string;
    numpyjs_version: string;
    machine?: string;
  };
  results: BenchmarkComparison[];
  summary: BenchmarkSummary;
}

export type BenchmarkMode = 'quick' | 'standard' | 'full';

export type RuntimeName = 'node' | 'deno' | 'bun';

export interface RuntimeInfo {
  name: RuntimeName;
  version: string;
}

export interface RuntimeComparison {
  name: string;
  category: string;
  numpy: BenchmarkTiming;
  runtimes: Record<string, { timing: BenchmarkTiming; ratio: number }>;
}

export interface MultiRuntimeReport {
  timestamp: string;
  environment: {
    python_version?: string;
    numpy_version?: string;
    numpyjs_version: string;
    runtimes: Record<string, string>;
    machine?: string;
  };
  results: RuntimeComparison[];
  summaries: Record<string, BenchmarkSummary>;
}

export interface BenchmarkOptions {
  mode?: BenchmarkMode;
  category?: string;
  output?: string;
  singleThread?: boolean;
  runtimes?: RuntimeName[];
  wasm?: boolean;
}
