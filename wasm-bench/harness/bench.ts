/**
 * WASM Multi-Category Benchmark Harness
 *
 * Compares WASM (Zig + Rust) kernels against numpy-ts pure JS across:
 *   - matmul, reductions, unary elementwise, binary elementwise, sorting
 * Reports timing, throughput metrics, and binary sizes.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { gzipSync } from 'node:zlib';

// Import numpy-ts internals for baseline comparison
import { ArrayStorage } from '../../src/common/storage.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST_DIR = resolve(__dirname, '..', 'dist');

// ─── Runtime Detection ──────────────────────────────────────────────────────

interface RuntimeInfo {
  name: string;    // 'node' | 'deno' | 'bun'
  version: string;
  arch: string;
  platform: string;
}

function detectRuntime(): RuntimeInfo {
  // @ts-ignore - Deno global
  if (typeof Deno !== 'undefined') {
    // @ts-ignore
    return { name: 'deno', version: Deno.version.deno, arch: process.arch, platform: process.platform };
  }
  // @ts-ignore - Bun global
  if (typeof Bun !== 'undefined') {
    // @ts-ignore
    return { name: 'bun', version: Bun.version, arch: process.arch, platform: process.platform };
  }
  return { name: 'node', version: process.version, arch: process.arch, platform: process.platform };
}

// ─── Configuration ───────────────────────────────────────────────────────────

interface SizeConfig {
  size: number;
  iterations: number;
  warmup: number;
}

interface CategoryConfig {
  name: string;
  sizes: SizeConfig[];
  metric: 'GFLOP/s' | 'GB/s' | 'M elem/s';
}

const CATEGORIES: CategoryConfig[] = [
  {
    name: 'matmul',
    sizes: [
      { size: 10, iterations: 20, warmup: 5 },
      { size: 100, iterations: 10, warmup: 5 },
      { size: 1000, iterations: 3, warmup: 2 },
    ],
    metric: 'GFLOP/s',
  },
  {
    name: 'reduction',
    sizes: [
      { size: 1_000, iterations: 100, warmup: 10 },
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 10_000_000, iterations: 5, warmup: 2 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'unary',
    sizes: [
      { size: 1_000, iterations: 100, warmup: 10 },
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 10_000_000, iterations: 5, warmup: 2 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'binary',
    sizes: [
      { size: 1_000, iterations: 100, warmup: 10 },
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 10_000_000, iterations: 5, warmup: 2 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'sort',
    sizes: [
      { size: 1_000, iterations: 50, warmup: 10 },
      { size: 100_000, iterations: 20, warmup: 5 },
      { size: 1_000_000, iterations: 5, warmup: 2 },
    ],
    metric: 'M elem/s',
  },
  {
    name: 'convolve',
    sizes: [
      { size: 100, iterations: 100, warmup: 10 },
      { size: 1_000, iterations: 50, warmup: 5 },
      { size: 10_000, iterations: 10, warmup: 3 },
    ],
    metric: 'M elem/s',
  },
  {
    name: 'linalg',
    sizes: [
      { size: 50, iterations: 50, warmup: 10 },
      { size: 100, iterations: 20, warmup: 5 },
      { size: 500, iterations: 5, warmup: 2 },
    ],
    metric: 'GFLOP/s',
  },
  {
    name: 'arrayops',
    sizes: [
      { size: 1_000, iterations: 100, warmup: 10 },
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 10_000_000, iterations: 5, warmup: 2 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'fft',
    sizes: [
      { size: 32, iterations: 50, warmup: 10 },
      { size: 100, iterations: 20, warmup: 5 },
      { size: 256, iterations: 10, warmup: 3 },
    ],
    metric: 'GFLOP/s',
  },
  // Complex categories
  {
    name: 'binary_complex',
    sizes: [
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 1_000_000, iterations: 10, warmup: 3 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'unary_complex',
    sizes: [
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 1_000_000, iterations: 10, warmup: 3 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'reduction_complex',
    sizes: [
      { size: 100_000, iterations: 30, warmup: 5 },
      { size: 1_000_000, iterations: 10, warmup: 3 },
    ],
    metric: 'GB/s',
  },
  {
    name: 'linalg_complex',
    sizes: [
      { size: 64, iterations: 10, warmup: 3 },
      { size: 128, iterations: 5, warmup: 2 },
      { size: 256, iterations: 3, warmup: 1 },
    ],
    metric: 'GFLOP/s',
  },
  {
    name: 'fft_complex',
    sizes: [
      { size: 256, iterations: 20, warmup: 5 },
      { size: 1024, iterations: 10, warmup: 3 },
      { size: 4096, iterations: 5, warmup: 2 },
    ],
    metric: 'GFLOP/s',
  },
];

// ─── WASM Loading ────────────────────────────────────────────────────────────

interface WasmKernels {
  name: string;
  memory: WebAssembly.Memory;
  baseOffset: number; // first safe byte offset (after stack + data segments)
  // Matmul
  matmul_f64?: (a: number, b: number, c: number, m: number, n: number, k: number) => void;
  matmul_f32?: (a: number, b: number, c: number, m: number, n: number, k: number) => void;
  // Reductions
  sum_f64?: (ptr: number, n: number) => number;
  sum_f32?: (ptr: number, n: number) => number;
  max_f64?: (ptr: number, n: number) => number;
  max_f32?: (ptr: number, n: number) => number;
  min_f64?: (ptr: number, n: number) => number;
  min_f32?: (ptr: number, n: number) => number;
  prod_f64?: (ptr: number, n: number) => number;
  prod_f32?: (ptr: number, n: number) => number;
  mean_f64?: (ptr: number, n: number) => number;
  mean_f32?: (ptr: number, n: number) => number;
  // Unary
  sqrt_f64?: (inp: number, out: number, n: number) => void;
  sqrt_f32?: (inp: number, out: number, n: number) => void;
  exp_f64?: (inp: number, out: number, n: number) => void;
  exp_f32?: (inp: number, out: number, n: number) => void;
  log_f64?: (inp: number, out: number, n: number) => void;
  log_f32?: (inp: number, out: number, n: number) => void;
  sin_f64?: (inp: number, out: number, n: number) => void;
  sin_f32?: (inp: number, out: number, n: number) => void;
  cos_f64?: (inp: number, out: number, n: number) => void;
  cos_f32?: (inp: number, out: number, n: number) => void;
  abs_f64?: (inp: number, out: number, n: number) => void;
  abs_f32?: (inp: number, out: number, n: number) => void;
  neg_f64?: (inp: number, out: number, n: number) => void;
  neg_f32?: (inp: number, out: number, n: number) => void;
  ceil_f64?: (inp: number, out: number, n: number) => void;
  ceil_f32?: (inp: number, out: number, n: number) => void;
  floor_f64?: (inp: number, out: number, n: number) => void;
  floor_f32?: (inp: number, out: number, n: number) => void;
  // Binary
  add_f64?: (a: number, b: number, out: number, n: number) => void;
  add_f32?: (a: number, b: number, out: number, n: number) => void;
  sub_f64?: (a: number, b: number, out: number, n: number) => void;
  sub_f32?: (a: number, b: number, out: number, n: number) => void;
  mul_f64?: (a: number, b: number, out: number, n: number) => void;
  mul_f32?: (a: number, b: number, out: number, n: number) => void;
  div_f64?: (a: number, b: number, out: number, n: number) => void;
  div_f32?: (a: number, b: number, out: number, n: number) => void;
  copysign_f64?: (a: number, b: number, out: number, n: number) => void;
  copysign_f32?: (a: number, b: number, out: number, n: number) => void;
  power_f64?: (a: number, b: number, out: number, n: number) => void;
  power_f32?: (a: number, b: number, out: number, n: number) => void;
  maximum_f64?: (a: number, b: number, out: number, n: number) => void;
  maximum_f32?: (a: number, b: number, out: number, n: number) => void;
  minimum_f64?: (a: number, b: number, out: number, n: number) => void;
  minimum_f32?: (a: number, b: number, out: number, n: number) => void;
  fmax_f64?: (a: number, b: number, out: number, n: number) => void;
  fmax_f32?: (a: number, b: number, out: number, n: number) => void;
  fmin_f64?: (a: number, b: number, out: number, n: number) => void;
  fmin_f32?: (a: number, b: number, out: number, n: number) => void;
  logaddexp_f64?: (a: number, b: number, out: number, n: number) => void;
  logaddexp_f32?: (a: number, b: number, out: number, n: number) => void;
  logical_and_f64?: (a: number, b: number, out: number, n: number) => void;
  logical_and_f32?: (a: number, b: number, out: number, n: number) => void;
  logical_xor_f64?: (a: number, b: number, out: number, n: number) => void;
  logical_xor_f32?: (a: number, b: number, out: number, n: number) => void;
  // Reduction - diff
  diff_f64?: (inp: number, out: number, n: number) => void;
  diff_f32?: (inp: number, out: number, n: number) => void;
  // Sort
  sort_f64?: (ptr: number, n: number) => void;
  sort_f32?: (ptr: number, n: number) => void;
  argsort_f64?: (vals: number, idx: number, n: number) => void;
  argsort_f32?: (vals: number, idx: number, n: number) => void;
  partition_f64?: (ptr: number, n: number, kth: number) => void;
  partition_f32?: (ptr: number, n: number, kth: number) => void;
  argpartition_f64?: (vals: number, idx: number, n: number, kth: number) => void;
  argpartition_f32?: (vals: number, idx: number, n: number, kth: number) => void;
  // Convolve
  correlate_f64?: (a: number, b: number, out: number, na: number, nb: number) => void;
  correlate_f32?: (a: number, b: number, out: number, na: number, nb: number) => void;
  convolve_f64?: (a: number, b: number, out: number, na: number, nb: number) => void;
  convolve_f32?: (a: number, b: number, out: number, na: number, nb: number) => void;
  // Linalg
  matvec_f64?: (a: number, x: number, out: number, m: number, n: number) => void;
  matvec_f32?: (a: number, x: number, out: number, m: number, n: number) => void;
  vecmat_f64?: (x: number, a: number, out: number, m: number, n: number) => void;
  vecmat_f32?: (x: number, a: number, out: number, m: number, n: number) => void;
  vecdot_f64?: (a: number, b: number, out: number, nbatch: number, veclen: number) => void;
  vecdot_f32?: (a: number, b: number, out: number, nbatch: number, veclen: number) => void;
  outer_f64?: (a: number, b: number, out: number, m: number, n: number) => void;
  outer_f32?: (a: number, b: number, out: number, m: number, n: number) => void;
  kron_f64?: (a: number, b: number, out: number, am: number, an: number, bm: number, bn: number) => void;
  kron_f32?: (a: number, b: number, out: number, am: number, an: number, bm: number, bn: number) => void;
  cross_f64?: (a: number, b: number, out: number, n: number) => void;
  cross_f32?: (a: number, b: number, out: number, n: number) => void;
  norm_f64?: (ptr: number, n: number) => number;
  norm_f32?: (ptr: number, n: number) => number;
  // Unary extensions
  tan_f64?: (inp: number, out: number, n: number) => void;
  tan_f32?: (inp: number, out: number, n: number) => void;
  signbit_f64?: (inp: number, out: number, n: number) => void;
  signbit_f32?: (inp: number, out: number, n: number) => void;
  // Reduction extensions
  nanmax_f64?: (ptr: number, n: number) => number;
  nanmax_f32?: (ptr: number, n: number) => number;
  nanmin_f64?: (ptr: number, n: number) => number;
  nanmin_f32?: (ptr: number, n: number) => number;
  // Array ops
  roll_f64?: (inp: number, out: number, n: number, shift: number) => void;
  roll_f32?: (inp: number, out: number, n: number, shift: number) => void;
  flip_f64?: (inp: number, out: number, n: number) => void;
  flip_f32?: (inp: number, out: number, n: number) => void;
  tile_f64?: (inp: number, out: number, n: number, reps: number) => void;
  tile_f32?: (inp: number, out: number, n: number, reps: number) => void;
  pad_f64?: (inp: number, out: number, rows: number, cols: number, pw: number) => void;
  pad_f32?: (inp: number, out: number, rows: number, cols: number, pw: number) => void;
  take_f64?: (data: number, indices: number, out: number, n: number) => void;
  take_f32?: (data: number, indices: number, out: number, n: number) => void;
  gradient_f64?: (inp: number, out: number, n: number) => void;
  gradient_f32?: (inp: number, out: number, n: number) => void;
  // Binary extensions
  mod_f64?: (a: number, b: number, out: number, n: number) => void;
  mod_f32?: (a: number, b: number, out: number, n: number) => void;
  floor_divide_f64?: (a: number, b: number, out: number, n: number) => void;
  floor_divide_f32?: (a: number, b: number, out: number, n: number) => void;
  hypot_f64?: (a: number, b: number, out: number, n: number) => void;
  hypot_f32?: (a: number, b: number, out: number, n: number) => void;
  // Unary extensions (hyperbolic + exp2)
  sinh_f64?: (inp: number, out: number, n: number) => void;
  sinh_f32?: (inp: number, out: number, n: number) => void;
  cosh_f64?: (inp: number, out: number, n: number) => void;
  cosh_f32?: (inp: number, out: number, n: number) => void;
  tanh_f64?: (inp: number, out: number, n: number) => void;
  tanh_f32?: (inp: number, out: number, n: number) => void;
  exp2_f64?: (inp: number, out: number, n: number) => void;
  exp2_f32?: (inp: number, out: number, n: number) => void;
  // Stats (in sort module)
  median_f64?: (ptr: number, n: number) => number;
  median_f32?: (ptr: number, n: number) => number;
  percentile_f64?: (ptr: number, n: number, p: number) => number;
  percentile_f32?: (ptr: number, n: number, p: number) => number;
  quantile_f64?: (ptr: number, n: number, q: number) => number;
  quantile_f32?: (ptr: number, n: number, q: number) => number;
  // Search (in arrayops module)
  nonzero_f64?: (ptr: number, out: number, n: number) => number;
  nonzero_f32?: (ptr: number, out: number, n: number) => number;
  // Linalg extensions
  matrix_power_f64?: (a: number, out: number, scratch: number, n: number, power: number) => void;
  multi_dot3_f64?: (a: number, b: number, c: number, out: number, tmp: number, n: number) => void;
  qr_f64?: (a: number, q: number, r: number, tau: number, scratch: number, m: number, n: number) => void;
  lstsq_f64?: (a: number, b: number, x: number, scratch: number, m: number, n: number) => void;
  // FFT
  rfft2_f64?: (inp: number, out: number, scratch: number, m: number, n: number) => void;
  irfft2_f64?: (inp: number, out: number, scratch: number, m: number, n: number) => void;
  // ─── Integer binary ops ──────
  add_i32?: (a: number, b: number, out: number, n: number) => void;
  sub_i32?: (a: number, b: number, out: number, n: number) => void;
  mul_i32?: (a: number, b: number, out: number, n: number) => void;
  maximum_i32?: (a: number, b: number, out: number, n: number) => void;
  minimum_i32?: (a: number, b: number, out: number, n: number) => void;
  add_i16?: (a: number, b: number, out: number, n: number) => void;
  sub_i16?: (a: number, b: number, out: number, n: number) => void;
  mul_i16?: (a: number, b: number, out: number, n: number) => void;
  maximum_i16?: (a: number, b: number, out: number, n: number) => void;
  minimum_i16?: (a: number, b: number, out: number, n: number) => void;
  add_i8?: (a: number, b: number, out: number, n: number) => void;
  sub_i8?: (a: number, b: number, out: number, n: number) => void;
  mul_i8?: (a: number, b: number, out: number, n: number) => void;
  maximum_i8?: (a: number, b: number, out: number, n: number) => void;
  minimum_i8?: (a: number, b: number, out: number, n: number) => void;
  // ─── Integer reductions ──────
  sum_i32?: (ptr: number, n: number) => number;
  max_i32?: (ptr: number, n: number) => number;
  min_i32?: (ptr: number, n: number) => number;
  sum_i16?: (ptr: number, n: number) => number;
  max_i16?: (ptr: number, n: number) => number;
  min_i16?: (ptr: number, n: number) => number;
  sum_i8?: (ptr: number, n: number) => number;
  max_i8?: (ptr: number, n: number) => number;
  min_i8?: (ptr: number, n: number) => number;
  // ─── Integer sort ────────────
  sort_i32?: (ptr: number, n: number) => void;
  sort_i16?: (ptr: number, n: number) => void;
  sort_i8?: (ptr: number, n: number) => void;
  argsort_i32?: (vals: number, idx: number, n: number) => void;
  argsort_i16?: (vals: number, idx: number, n: number) => void;
  argsort_i8?: (vals: number, idx: number, n: number) => void;
  // ─── Complex binary ops ──────
  add_c128?: (a: number, b: number, out: number, n: number) => void;
  add_c64?: (a: number, b: number, out: number, n: number) => void;
  mul_c128?: (a: number, b: number, out: number, n: number) => void;
  mul_c64?: (a: number, b: number, out: number, n: number) => void;
  // ─── Complex unary ops ───────
  abs_c128?: (inp: number, out: number, n: number) => void;
  abs_c64?: (inp: number, out: number, n: number) => void;
  exp_c128?: (inp: number, out: number, n: number) => void;
  exp_c64?: (inp: number, out: number, n: number) => void;
  // ─── Complex reductions ──────
  sum_c128?: (ptr: number, out: number, n: number) => void;
  sum_c64?: (ptr: number, out: number, n: number) => void;
  // ─── Complex matmul ──────────
  matmul_c128?: (a: number, b: number, c: number, m: number, k: number, n: number) => void;
  matmul_c64?: (a: number, b: number, c: number, m: number, k: number, n: number) => void;
  // ─── Complex FFT ─────────────
  fft_c128?: (inp: number, out: number, scratch: number, n: number) => void;
  ifft_c128?: (inp: number, out: number, scratch: number, n: number) => void;
  fft_c64?: (inp: number, out: number, scratch: number, n: number) => void;
  ifft_c64?: (inp: number, out: number, scratch: number, n: number) => void;
}

async function loadWasm(filename: string, name: string): Promise<WasmKernels | null> {
  const wasmPath = resolve(DIST_DIR, filename);
  let wasmBytes: Buffer;
  try {
    wasmBytes = readFileSync(wasmPath);
  } catch {
    console.warn(`  [skip] ${name}: ${filename} not found`);
    return null;
  }

  const module = await WebAssembly.compile(wasmBytes);
  const imports = WebAssembly.Module.imports(module);
  const moduleExports = WebAssembly.Module.exports(module);

  const memoryImport = imports.find((i) => i.kind === 'memory');
  const memoryExport = moduleExports.find((e) => e.kind === 'memory');

  let importObj: WebAssembly.Imports = {};
  let memory: WebAssembly.Memory;

  if (memoryImport) {
    memory = new WebAssembly.Memory({ initial: 1024, maximum: 65536 });
    importObj = { [memoryImport.module]: { [memoryImport.name]: memory } };
  }

  const instance = await WebAssembly.instantiate(module, importObj);
  const exp = instance.exports as Record<string, unknown>;

  if (memoryExport) {
    memory = exp[memoryExport.name] as WebAssembly.Memory;
  }

  if (!memory!) {
    console.warn(`  [skip] ${name}: no memory found`);
    return null;
  }

  // Record initial memory size as base offset (stack + data segments live there)
  const baseOffset = memory!.buffer.byteLength;

  // Dynamically map all function exports
  const kernels: WasmKernels = { name, memory, baseOffset };
  const fnNames = [
    'matmul_f64', 'matmul_f32',
    'sum_f64', 'sum_f32', 'max_f64', 'max_f32', 'min_f64', 'min_f32',
    'prod_f64', 'prod_f32', 'mean_f64', 'mean_f32',
    'diff_f64', 'diff_f32',
    'sqrt_f64', 'sqrt_f32', 'exp_f64', 'exp_f32', 'log_f64', 'log_f32',
    'sin_f64', 'sin_f32', 'cos_f64', 'cos_f32',
    'abs_f64', 'abs_f32', 'neg_f64', 'neg_f32',
    'ceil_f64', 'ceil_f32', 'floor_f64', 'floor_f32',
    'add_f64', 'add_f32', 'sub_f64', 'sub_f32',
    'mul_f64', 'mul_f32', 'div_f64', 'div_f32',
    'copysign_f64', 'copysign_f32', 'power_f64', 'power_f32',
    'maximum_f64', 'maximum_f32', 'minimum_f64', 'minimum_f32',
    'fmax_f64', 'fmax_f32', 'fmin_f64', 'fmin_f32',
    'logaddexp_f64', 'logaddexp_f32',
    'logical_and_f64', 'logical_and_f32', 'logical_xor_f64', 'logical_xor_f32',
    'sort_f64', 'sort_f32',
    'argsort_f64', 'argsort_f32',
    'partition_f64', 'partition_f32',
    'argpartition_f64', 'argpartition_f32',
    'correlate_f64', 'correlate_f32', 'convolve_f64', 'convolve_f32',
    'matvec_f64', 'matvec_f32', 'vecmat_f64', 'vecmat_f32',
    'vecdot_f64', 'vecdot_f32', 'outer_f64', 'outer_f32',
    'kron_f64', 'kron_f32', 'cross_f64', 'cross_f32',
    'norm_f64', 'norm_f32',
    'tan_f64', 'tan_f32', 'signbit_f64', 'signbit_f32',
    'nanmax_f64', 'nanmax_f32', 'nanmin_f64', 'nanmin_f32',
    'roll_f64', 'roll_f32', 'flip_f64', 'flip_f32',
    'tile_f64', 'tile_f32', 'pad_f64', 'pad_f32',
    'take_f64', 'take_f32', 'gradient_f64', 'gradient_f32',
    'mod_f64', 'mod_f32', 'floor_divide_f64', 'floor_divide_f32',
    'hypot_f64', 'hypot_f32',
    'sinh_f64', 'sinh_f32', 'cosh_f64', 'cosh_f32',
    'tanh_f64', 'tanh_f32', 'exp2_f64', 'exp2_f32',
    'median_f64', 'median_f32', 'percentile_f64', 'percentile_f32',
    'quantile_f64', 'quantile_f32',
    'nonzero_f64', 'nonzero_f32',
    // Linalg extensions
    'matrix_power_f64', 'multi_dot3_f64', 'qr_f64', 'lstsq_f64',
    // FFT
    'rfft2_f64', 'irfft2_f64',
    // Integer binary ops
    'add_i32', 'sub_i32', 'mul_i32', 'maximum_i32', 'minimum_i32',
    'add_i16', 'sub_i16', 'mul_i16', 'maximum_i16', 'minimum_i16',
    'add_i8', 'sub_i8', 'mul_i8', 'maximum_i8', 'minimum_i8',
    // Integer reductions
    'sum_i32', 'max_i32', 'min_i32',
    'sum_i16', 'max_i16', 'min_i16',
    'sum_i8', 'max_i8', 'min_i8',
    // Integer sort
    'sort_i32', 'sort_i16', 'sort_i8',
    'argsort_i32', 'argsort_i16', 'argsort_i8',
    // Complex binary ops
    'add_c128', 'add_c64', 'mul_c128', 'mul_c64',
    // Complex unary ops
    'abs_c128', 'abs_c64', 'exp_c128', 'exp_c64',
    // Complex reductions
    'sum_c128', 'sum_c64',
    // Complex matmul
    'matmul_c128', 'matmul_c64',
    // Complex FFT
    'fft_c128', 'ifft_c128', 'fft_c64', 'ifft_c64',
  ] as const;
  for (const fn of fnNames) {
    if (typeof exp[fn] === 'function') {
      (kernels as Record<string, unknown>)[fn] = exp[fn];
    }
  }
  return kernels;
}

// ─── numpy-ts baseline imports ───────────────────────────────────────────────

async function importBaselines() {
  const [linalgMod, reductionMod, expMod, arithMod, sortMod, trigMod, roundMod, logicMod, gradMod, statsMod] = await Promise.all([
    import('../../src/common/ops/linalg.js'),
    import('../../src/common/ops/reduction.js'),
    import('../../src/common/ops/exponential.js'),
    import('../../src/common/ops/arithmetic.js'),
    import('../../src/common/ops/sorting.js'),
    import('../../src/common/ops/trig.js'),
    import('../../src/common/ops/rounding.js'),
    import('../../src/common/ops/logic.js'),
    import('../../src/common/ops/gradient.js'),
    import('../../src/common/ops/statistics.js'),
  ]);
  return {
    matmul: linalgMod.matmul as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    // Reductions
    sum: reductionMod.sum as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    max: reductionMod.max as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    min: reductionMod.min as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    prod: reductionMod.prod as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    mean: reductionMod.mean as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    diff: gradMod.diff as (a: ArrayStorage) => ArrayStorage,
    // Unary math
    sqrt: expMod.sqrt as (a: ArrayStorage) => ArrayStorage,
    exp: expMod.exp as (a: ArrayStorage) => ArrayStorage,
    log: expMod.log as (a: ArrayStorage) => ArrayStorage,
    sin: trigMod.sin as (a: ArrayStorage) => ArrayStorage,
    cos: trigMod.cos as (a: ArrayStorage) => ArrayStorage,
    absolute: arithMod.absolute as (a: ArrayStorage) => ArrayStorage,
    negative: arithMod.negative as (a: ArrayStorage) => ArrayStorage,
    ceil: roundMod.ceil as (a: ArrayStorage) => ArrayStorage,
    floor: roundMod.floor as (a: ArrayStorage) => ArrayStorage,
    // Binary
    add: arithMod.add as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    subtract: arithMod.subtract as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    multiply: arithMod.multiply as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    divide: arithMod.divide as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    copysign: logicMod.copysign as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    power: expMod.power as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    maximum: arithMod.maximum as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    minimum: arithMod.minimum as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    fmax: arithMod.fmax as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    fmin: arithMod.fmin as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    logaddexp: expMod.logaddexp as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    logical_and: logicMod.logical_and as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    logical_xor: logicMod.logical_xor as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    // Sort
    sort: sortMod.sort as (a: ArrayStorage, axis?: number) => ArrayStorage,
    argsort: sortMod.argsort as (a: ArrayStorage, axis?: number) => ArrayStorage,
    partition: sortMod.partition as (a: ArrayStorage, kth: number, axis?: number) => ArrayStorage,
    argpartition: sortMod.argpartition as (a: ArrayStorage, kth: number, axis?: number) => ArrayStorage,
    // Convolve
    correlate: statsMod.correlate as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    convolve: statsMod.convolve as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
  };
}

type Baselines = Awaited<ReturnType<typeof importBaselines>>;

// ─── Timing Utilities ────────────────────────────────────────────────────────

interface TimingResult {
  impl: string;
  op: string;
  size: number;
  timeMs: number;
  metric: number;
  metricUnit: string;
}

interface DualTimingResult {
  kernel: TimingResult;
  e2e: TimingResult;
}

function dualJsResult(r: TimingResult): DualTimingResult {
  return { kernel: r, e2e: r };
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function ensureMemory(wasm: WasmKernels, bytes: number) {
  const totalNeeded = wasm.baseOffset + bytes;
  const currentBytes = wasm.memory.buffer.byteLength;
  if (totalNeeded > currentBytes) {
    const needed = Math.ceil((totalNeeded - currentBytes) / 65536);
    wasm.memory.grow(needed);
  }
}

function formatSize(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return `${n}`;
}

// ─── Matmul Benchmarks ──────────────────────────────────────────────────────

function computeGflops(n: number, timeMs: number): number {
  return (2 * n * n * n) / (timeMs / 1000) / 1e9;
}

function benchWasmMatmul(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = dtype === 'f64' ? wasm.matmul_f64 : wasm.matmul_f32;
  if (!fn) return null;

  const n = size;
  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const matElems = n * n;
  const matBytes = matElems * bytesPerElem;
  ensureMemory(wasm, 3 * matBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const aData = new ArrayType(matElems);
  const bData = new ArrayType(matElems);
  for (let i = 0; i < matElems; i++) {
    aData[i] = Math.random() * 2 - 1;
    bData[i] = Math.random() * 2 - 1;
  }

  const base = wasm.baseOffset;
  const aOff = base, bOff = base + matBytes, cOff = base + 2 * matBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];

  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, aOff, matElems).set(aData);
    new ArrayType(wasm.memory.buffer, bOff, matElems).set(bData);
    const t0 = performance.now();
    fn(aOff, bOff, cOff, n, n, n);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, cOff, matElems).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'matmul', size, timeMs: kMs, metric: computeGflops(n, kMs), metricUnit: 'GFLOP/s' },
    e2e: { impl, op: 'matmul', size, timeMs: eMs, metric: computeGflops(n, eMs), metricUnit: 'GFLOP/s' },
  };
}

function benchJsMatmul(
  baselines: Baselines, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult {
  const n = size;
  const matElems = n * n;

  // numpy-ts matmul only supports f64, so f32 path includes upcast
  if (dtype === 'f64') {
    const aData = new Float64Array(matElems);
    const bData = new Float64Array(matElems);
    for (let i = 0; i < matElems; i++) {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 - 1;
    }
    const aS = ArrayStorage.fromData(aData, [n, n], 'float64');
    const bS = ArrayStorage.fromData(bData, [n, n], 'float64');

    const times: number[] = [];
    for (let iter = 0; iter < warmup + iterations; iter++) {
      const t0 = performance.now();
      baselines.matmul(aS, bS);
      const t1 = performance.now();
      if (iter >= warmup) times.push(t1 - t0);
    }
    const ms = median(times);
    return dualJsResult({ impl: 'JS (f64)', op: 'matmul', size, timeMs: ms, metric: computeGflops(n, ms), metricUnit: 'GFLOP/s' });
  } else {
    const aData = new Float32Array(matElems);
    const bData = new Float32Array(matElems);
    for (let i = 0; i < matElems; i++) {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 - 1;
    }
    const times: number[] = [];
    for (let iter = 0; iter < warmup + iterations; iter++) {
      const t0 = performance.now();
      const aF64 = Float64Array.from(aData);
      const bF64 = Float64Array.from(bData);
      const aS = ArrayStorage.fromData(aF64, [n, n], 'float64');
      const bS = ArrayStorage.fromData(bF64, [n, n], 'float64');
      baselines.matmul(aS, bS);
      const t1 = performance.now();
      if (iter >= warmup) times.push(t1 - t0);
    }
    const ms = median(times);
    return dualJsResult({ impl: 'JS (f32→f64)', op: 'matmul', size, timeMs: ms, metric: computeGflops(n, ms), metricUnit: 'GFLOP/s' });
  }
}

// ─── Reduction Benchmarks ───────────────────────────────────────────────────

function computeGBs(elems: number, bytesPerElem: number, timeMs: number): number {
  return (elems * bytesPerElem) / (timeMs / 1000) / 1e9;
}

function benchWasmReduction(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((ptr: number, n: number) => number) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  if (opName === 'prod') {
    for (let i = 0; i < size; i++) data[i] = 1 + Math.random() * 0.0001;
  } else {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  }
  const base = wasm.baseOffset;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t1 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, bytesPerElem, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, bytesPerElem, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsReduction(
  baselines: Baselines, opName: 'sum' | 'max' | 'min' | 'prod' | 'mean', size: number, iterations: number, warmup: number
): DualTimingResult {
  // Use small values for prod to avoid overflow
  const data = new Float64Array(size);
  if (opName === 'prod') {
    for (let i = 0; i < size; i++) data[i] = 1 + Math.random() * 0.0001;
  } else {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  }
  const storage = ArrayStorage.fromData(data, [size], 'float64');
  const fn = baselines[opName] as (a: ArrayStorage) => ArrayStorage | number;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(storage);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  });
}

// ─── Unary Benchmarks ───────────────────────────────────────────────────────

function benchWasmUnary(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((inp: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, 2 * bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  if (opName === 'sqrt' || opName === 'log') {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 5 + 0.1;
  } else {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 4 - 2;
  }
  const base = wasm.baseOffset;

  const inOff = base, outOff = base + bufBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(inOff, outOff, size);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, outOff, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, bytesPerElem, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, bytesPerElem, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsUnary(
  baselines: Baselines, opName: string, size: number, iterations: number, warmup: number, displayOp?: string
): DualTimingResult {
  const data = new Float64Array(size);
  // Use positive values for sqrt/log; general range for others
  if (opName === 'sqrt' || opName === 'log') {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 5 + 0.1;
  } else {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 4 - 2;
  }
  const storage = ArrayStorage.fromData(data, [size], 'float64');
  const fn = (baselines as Record<string, (a: ArrayStorage) => ArrayStorage>)[opName];

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(storage);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: displayOp ?? opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  });
}

// ─── Binary Benchmarks ──────────────────────────────────────────────────────

function benchWasmBinary(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((a: number, b: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, 3 * bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const aData = new ArrayType(size);
  const bData = new ArrayType(size);
  for (let i = 0; i < size; i++) {
    if (opName === 'div' || opName === 'floor_divide' || opName === 'mod') {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 + 0.5;
    } else if (opName === 'power') {
      aData[i] = Math.random() * 5 + 0.1;
      bData[i] = Math.random() * 3;
    } else if (opName === 'logaddexp') {
      aData[i] = Math.random() * 10 - 5;
      bData[i] = Math.random() * 10 - 5;
    } else if (opName === 'logical_and' || opName === 'logical_xor') {
      aData[i] = Math.random() > 0.3 ? Math.random() * 5 : 0;
      bData[i] = Math.random() > 0.3 ? Math.random() * 5 : 0;
    } else if (opName === 'hypot') {
      aData[i] = Math.random() * 10;
      bData[i] = Math.random() * 10;
    } else {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 - 1;
    }
  }
  const base = wasm.baseOffset;

  const aOff = base, bOff = base + bufBytes, outOff = base + 2 * bufBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(aData);
    new ArrayType(wasm.memory.buffer, base + bufBytes, size).set(bData);
    const t0 = performance.now();
    fn(aOff, bOff, outOff, size);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, outOff, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, bytesPerElem, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, bytesPerElem, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsBinary(
  baselines: Baselines, opName: string, size: number, iterations: number, warmup: number, displayOp?: string
): DualTimingResult {
  const aData = new Float64Array(size);
  const bData = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    if (opName === 'divide' || opName === 'floor_divide' || opName === 'mod') {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 + 0.5;
    } else if (opName === 'power') {
      aData[i] = Math.random() * 5 + 0.1;
      bData[i] = Math.random() * 3;
    } else if (opName === 'logaddexp') {
      aData[i] = Math.random() * 10 - 5;
      bData[i] = Math.random() * 10 - 5;
    } else if (opName === 'logical_and' || opName === 'logical_xor') {
      aData[i] = Math.random() > 0.3 ? Math.random() * 5 : 0;
      bData[i] = Math.random() > 0.3 ? Math.random() * 5 : 0;
    } else if (opName === 'hypot') {
      aData[i] = Math.random() * 10;
      bData[i] = Math.random() * 10;
    } else {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 - 1;
    }
  }
  const aS = ArrayStorage.fromData(aData, [size], 'float64');
  const bS = ArrayStorage.fromData(bData, [size], 'float64');
  const fn = (baselines as Record<string, (a: ArrayStorage, b: ArrayStorage) => ArrayStorage>)[opName];

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(aS, bS);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: displayOp ?? opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  });
}

// ─── Sort Benchmarks ────────────────────────────────────────────────────────

function computeMElemPerSec(elems: number, timeMs: number): number {
  return elems / (timeMs / 1000) / 1e6;
}

function benchWasmSort(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = dtype === 'f64' ? wasm.sort_f64 : wasm.sort_f32;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const base = wasm.baseOffset;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'sort', size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
    e2e: { impl, op: 'sort', size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
  };
}

function benchJsSort(
  baselines: Baselines, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Create fresh storage each time (sort returns new array, but input matters)
    const storage = ArrayStorage.fromData(new Float64Array(data), [size], 'float64');
    const t0 = performance.now();
    baselines.sort(storage);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: 'sort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  });
}

// ─── Diff Benchmarks ─────────────────────────────────────────────────────────

function benchWasmDiff(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`diff_${dtype}`] as ((inp: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const inBytes = size * bytesPerElem;
  const outBytes = (size - 1) * bytesPerElem;
  ensureMemory(wasm, inBytes + outBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;

  const base = wasm.baseOffset;
  const inOff = base, outOff = base + inBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(inOff, outOff, size);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, outOff, size - 1).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'diff', size, timeMs: kMs, metric: computeGBs(size - 1, bytesPerElem, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: 'diff', size, timeMs: eMs, metric: computeGBs(size - 1, bytesPerElem, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsDiff(
  baselines: Baselines, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  const storage = ArrayStorage.fromData(data, [size], 'float64');

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    baselines.diff(storage);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: 'diff',
    size,
    timeMs: ms,
    metric: computeGBs(size - 1, 8, ms),
    metricUnit: 'GB/s',
  });
}

// ─── Argsort Benchmarks ──────────────────────────────────────────────────────

function benchWasmArgsort(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`argsort_${dtype}`] as ((vals: number, idx: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const dataBytes = size * bytesPerElem;
  const idxBytes = size * 4;
  ensureMemory(wasm, dataBytes + idxBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const base = wasm.baseOffset;
  const dataOff = base;
  const idxOff = base + dataBytes;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, dataOff, size).set(data);
    const t0 = performance.now();
    fn(dataOff, idxOff, size);
    const t1 = performance.now();
    new Int32Array(wasm.memory.buffer, idxOff, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'argsort', size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
    e2e: { impl, op: 'argsort', size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
  };
}

function benchJsArgsort(
  baselines: Baselines, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const storage = ArrayStorage.fromData(new Float64Array(data), [size], 'float64');
    const t0 = performance.now();
    baselines.argsort(storage);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: 'argsort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  });
}

// ─── Partition Benchmarks ────────────────────────────────────────────────────

function benchWasmPartition(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`partition_${dtype}`] as ((ptr: number, n: number, kth: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const kth = Math.floor(size / 2);
  const base = wasm.baseOffset;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size, kth);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, base, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'partition', size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
    e2e: { impl, op: 'partition', size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
  };
}

function benchJsPartition(
  baselines: Baselines, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;
  const kth = Math.floor(size / 2);

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const storage = ArrayStorage.fromData(new Float64Array(data), [size], 'float64');
    const t0 = performance.now();
    baselines.partition(storage, kth);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: 'partition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  });
}

// ─── Argpartition Benchmarks ─────────────────────────────────────────────────

function benchWasmArgpartition(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`argpartition_${dtype}`] as ((vals: number, idx: number, n: number, kth: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const dataBytes = size * bytesPerElem;
  const idxBytes = size * 4;
  ensureMemory(wasm, dataBytes + idxBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const kth = Math.floor(size / 2);
  const base = wasm.baseOffset;
  const dataOff = base;
  const idxOff = base + dataBytes;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, dataOff, size).set(data);
    const t0 = performance.now();
    fn(dataOff, idxOff, size, kth);
    const t1 = performance.now();
    new Int32Array(wasm.memory.buffer, idxOff, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'argpartition', size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
    e2e: { impl, op: 'argpartition', size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
  };
}

function benchJsArgpartition(
  baselines: Baselines, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;
  const kth = Math.floor(size / 2);

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const storage = ArrayStorage.fromData(new Float64Array(data), [size], 'float64');
    const t0 = performance.now();
    baselines.argpartition(storage, kth);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: 'argpartition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  });
}

// ─── Convolve Benchmarks ─────────────────────────────────────────────────────

function benchWasmConvolve(
  wasm: WasmKernels, op: 'correlate' | 'convolve', signalSize: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const kernelSize = 32;
  const outSize = signalSize + kernelSize - 1;
  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const totalBytes = (signalSize + kernelSize + outSize) * bytesPerElem;
  ensureMemory(wasm, totalBytes);

  const B = wasm.baseOffset;
  const aOff = B;
  const bOff = B + signalSize * bytesPerElem;
  const outOff = bOff + kernelSize * bytesPerElem;

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const aData = new ArrayType(signalSize);
  const bData = new ArrayType(kernelSize);
  for (let i = 0; i < signalSize; i++) aData[i] = Math.random() * 2 - 1;
  for (let i = 0; i < kernelSize; i++) bData[i] = Math.random() * 2 - 1;

  const fn_name = `${op}_${dtype}`;
  const fn = (wasm as Record<string, unknown>)[fn_name] as ((a: number, b: number, out: number, na: number, nb: number) => void) | undefined;
  if (!fn) return null;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new ArrayType(wasm.memory.buffer, aOff, signalSize).set(aData);
    new ArrayType(wasm.memory.buffer, bOff, kernelSize).set(bData);
    const t0 = performance.now();
    fn(aOff, bOff, outOff, signalSize, kernelSize);
    const t1 = performance.now();
    new ArrayType(wasm.memory.buffer, outOff, outSize).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  const mkResult = (ms: number): TimingResult => ({
    impl, op, size: signalSize, timeMs: ms,
    metric: outSize / (ms / 1000) / 1e6, metricUnit: 'M elem/s',
  });
  return { kernel: mkResult(kMs), e2e: mkResult(eMs) };
}

function benchJsConvolve(
  baselines: Baselines, op: 'correlate' | 'convolve', signalSize: number, iterations: number, warmup: number
): DualTimingResult {
  const kernelSize = 32;
  const outSize = signalSize + kernelSize - 1;

  const aData = new Float64Array(signalSize);
  const bData = new Float64Array(kernelSize);
  for (let i = 0; i < signalSize; i++) aData[i] = Math.random() * 2 - 1;
  for (let i = 0; i < kernelSize; i++) bData[i] = Math.random() * 2 - 1;

  const aS = ArrayStorage.fromData(aData, [signalSize], 'float64');
  const bS = ArrayStorage.fromData(bData, [kernelSize], 'float64');
  const fn = baselines[op] as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(aS, bS);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op,
    size: signalSize,
    timeMs: ms,
    metric: outSize / (ms / 1000) / 1e6,
    metricUnit: 'M elem/s',
  });
}

// ─── Linalg Benchmarks ──────────────────────────────────────────────────────

function computeGflopsCustom(flops: number, timeMs: number): number {
  return flops / (timeMs / 1000) / 1e9;
}

function benchWasmLinalg(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const N = size;
  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const base = wasm.baseOffset;

  let flops: number;
  let runFn: () => void;
  let copyFn: () => void;
  let readOutFn: () => void = () => {};  // default: no copy-out (scalar)

  switch (opName) {
    case 'matvec': {
      const fn = (wasm as Record<string, unknown>)[`matvec_${dtype}`] as ((a: number, x: number, out: number, m: number, n: number) => void) | undefined;
      if (!fn) return null;
      const matBytes = N * N * bytesPerElem;
      const vecBytes = N * bytesPerElem;
      ensureMemory(wasm, matBytes + vecBytes + vecBytes);
      const aData = new ArrayType(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 2 - 1;
      const xData = new ArrayType(N); for (let i = 0; i < N; i++) xData[i] = Math.random() * 2 - 1;
      const aOff = base, xOff = base + matBytes, outOff = base + matBytes + vecBytes;
      flops = 2 * N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N * N).set(aData); new ArrayType(wasm.memory.buffer, xOff, N).set(xData); };
      runFn = () => fn(aOff, xOff, outOff, N, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N).slice(); };
      break;
    }
    case 'vecmat': {
      const fn = (wasm as Record<string, unknown>)[`vecmat_${dtype}`] as ((x: number, a: number, out: number, m: number, n: number) => void) | undefined;
      if (!fn) return null;
      const matBytes = N * N * bytesPerElem;
      const vecBytes = N * bytesPerElem;
      ensureMemory(wasm, vecBytes + matBytes + vecBytes);
      const xData = new ArrayType(N); for (let i = 0; i < N; i++) xData[i] = Math.random() * 2 - 1;
      const aData = new ArrayType(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 2 - 1;
      const xOff = base, aOff = base + vecBytes, outOff = base + vecBytes + matBytes;
      flops = 2 * N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(xData); new ArrayType(wasm.memory.buffer, aOff, N * N).set(aData); };
      runFn = () => fn(xOff, aOff, outOff, N, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N).slice(); };
      break;
    }
    case 'vecdot': {
      const fn = (wasm as Record<string, unknown>)[`vecdot_${dtype}`] as ((a: number, b: number, out: number, nbatch: number, veclen: number) => void) | undefined;
      if (!fn) return null;
      const totalElems = N * N;
      const dataBytes = totalElems * bytesPerElem;
      const outBytes = N * bytesPerElem;
      ensureMemory(wasm, 2 * dataBytes + outBytes);
      const aData = new ArrayType(totalElems); for (let i = 0; i < totalElems; i++) aData[i] = Math.random() * 2 - 1;
      const bData = new ArrayType(totalElems); for (let i = 0; i < totalElems; i++) bData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + dataBytes, outOff = base + 2 * dataBytes;
      flops = 2 * N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, totalElems).set(aData); new ArrayType(wasm.memory.buffer, bOff, totalElems).set(bData); };
      runFn = () => fn(aOff, bOff, outOff, N, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N).slice(); };
      break;
    }
    case 'outer': {
      const fn = (wasm as Record<string, unknown>)[`outer_${dtype}`] as ((a: number, b: number, out: number, m: number, n: number) => void) | undefined;
      if (!fn) return null;
      const vecBytes = N * bytesPerElem;
      const outBytes = N * N * bytesPerElem;
      ensureMemory(wasm, 2 * vecBytes + outBytes);
      const aData = new ArrayType(N); for (let i = 0; i < N; i++) aData[i] = Math.random() * 2 - 1;
      const bData = new ArrayType(N); for (let i = 0; i < N; i++) bData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + vecBytes, outOff = base + 2 * vecBytes;
      flops = N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(aData); new ArrayType(wasm.memory.buffer, bOff, N).set(bData); };
      runFn = () => fn(aOff, bOff, outOff, N, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N * N).slice(); };
      break;
    }
    case 'kron': {
      if (N > 100) return null;
      const fn = (wasm as Record<string, unknown>)[`kron_${dtype}`] as ((a: number, b: number, out: number, am: number, an: number, bm: number, bn: number) => void) | undefined;
      if (!fn) return null;
      const matBytes = N * N * bytesPerElem;
      const outBytes = N * N * N * N * bytesPerElem;
      ensureMemory(wasm, 2 * matBytes + outBytes);
      const aData = new ArrayType(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 2 - 1;
      const bData = new ArrayType(N * N); for (let i = 0; i < N * N; i++) bData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + matBytes, outOff = base + 2 * matBytes;
      flops = N * N * N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N * N).set(aData); new ArrayType(wasm.memory.buffer, bOff, N * N).set(bData); };
      runFn = () => fn(aOff, bOff, outOff, N, N, N, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N * N * N * N).slice(); };
      break;
    }
    case 'cross': {
      const fn = (wasm as Record<string, unknown>)[`cross_${dtype}`] as ((a: number, b: number, out: number, n: number) => void) | undefined;
      if (!fn) return null;
      const vecBytes = N * 3 * bytesPerElem;
      ensureMemory(wasm, 3 * vecBytes);
      const aData = new ArrayType(N * 3); for (let i = 0; i < N * 3; i++) aData[i] = Math.random() * 2 - 1;
      const bData = new ArrayType(N * 3); for (let i = 0; i < N * 3; i++) bData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + vecBytes, outOff = base + 2 * vecBytes;
      flops = 6 * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N * 3).set(aData); new ArrayType(wasm.memory.buffer, bOff, N * 3).set(bData); };
      runFn = () => fn(aOff, bOff, outOff, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, outOff, N * 3).slice(); };
      break;
    }
    case 'norm': {
      const fn = (wasm as Record<string, unknown>)[`norm_${dtype}`] as ((ptr: number, n: number) => number) | undefined;
      if (!fn) return null;
      const totalElems = N * N;
      const dataBytes = totalElems * bytesPerElem;
      ensureMemory(wasm, dataBytes);
      const data = new ArrayType(totalElems); for (let i = 0; i < totalElems; i++) data[i] = Math.random() * 2 - 1;
      flops = 2 * N * N;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, totalElems).set(data); };
      runFn = () => fn(base, totalElems);
      // scalar output — no copy-out
      break;
    }
    case 'matrix_power': {
      if (dtype !== 'f64') return null;
      const fn = wasm.matrix_power_f64;
      if (!fn) return null;
      const matBytes = N * N * 8;
      const scratchBytes = 2 * N * N * 8;
      ensureMemory(wasm, matBytes + matBytes + scratchBytes);
      const aData = new Float64Array(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 0.1;
      for (let i = 0; i < N; i++) aData[i * N + i] += 1;
      const aOff = base, outOff = base + matBytes, scrOff = base + 2 * matBytes;
      const power = 3;
      flops = 4 * N * N * N;
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, N * N).set(aData); };
      runFn = () => fn(aOff, outOff, scrOff, N, power);
      readOutFn = () => { new Float64Array(wasm.memory.buffer, outOff, N * N).slice(); };
      break;
    }
    case 'multi_dot3': {
      if (dtype !== 'f64') return null;
      const fn = wasm.multi_dot3_f64;
      if (!fn) return null;
      const matBytes = N * N * 8;
      ensureMemory(wasm, 4 * matBytes + matBytes);
      const aData = new Float64Array(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 2 - 1;
      const bDataM = new Float64Array(N * N); for (let i = 0; i < N * N; i++) bDataM[i] = Math.random() * 2 - 1;
      const cData = new Float64Array(N * N); for (let i = 0; i < N * N; i++) cData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + matBytes, cOff = base + 2 * matBytes;
      const outOff = base + 3 * matBytes, tmpOff = base + 4 * matBytes;
      flops = 4 * N * N * N;
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, N * N).set(aData); new Float64Array(wasm.memory.buffer, bOff, N * N).set(bDataM); new Float64Array(wasm.memory.buffer, cOff, N * N).set(cData); };
      runFn = () => fn(aOff, bOff, cOff, outOff, tmpOff, N);
      readOutFn = () => { new Float64Array(wasm.memory.buffer, outOff, N * N).slice(); };
      break;
    }
    case 'qr': {
      if (dtype !== 'f64') return null;
      const fn = wasm.qr_f64;
      if (!fn) return null;
      const matBytes = N * N * 8;
      const qBytes = N * N * 8;
      const rBytes = N * N * 8;
      const tauBytes = N * 8;
      const scrBytes = N * 8;
      ensureMemory(wasm, matBytes + qBytes + rBytes + tauBytes + scrBytes);
      const aData = new Float64Array(N * N); for (let i = 0; i < N * N; i++) aData[i] = Math.random() * 2 - 1;
      const aOff = base, qOff = base + matBytes, rOff = qOff + qBytes;
      const tauOff = rOff + rBytes, scrOff = tauOff + tauBytes;
      flops = (4 * N * N * N) / 3;
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, N * N).set(aData); };
      runFn = () => {
        new Float64Array(wasm.memory.buffer, base, N * N).set(aData);
        fn(aOff, qOff, rOff, tauOff, scrOff, N, N);
      };
      readOutFn = () => { new Float64Array(wasm.memory.buffer, qOff, N * N).slice(); new Float64Array(wasm.memory.buffer, rOff, N * N).slice(); };
      break;
    }
    case 'lstsq': {
      if (dtype !== 'f64') return null;
      const fn = wasm.lstsq_f64;
      if (!fn) return null;
      const M = N, Nc = Math.max(Math.floor(N * 0.8), 1);
      const aBytes = M * Nc * 8;
      const bBytes = M * 8;
      const xBytes = Nc * 8;
      const scrBytes = (M * Nc + M * Nc + Nc * Nc + Nc + Nc + Nc) * 8;
      ensureMemory(wasm, aBytes + bBytes + xBytes + scrBytes);
      const aData = new Float64Array(M * Nc); for (let i = 0; i < M * Nc; i++) aData[i] = Math.random() * 2 - 1;
      const bData = new Float64Array(M); for (let i = 0; i < M; i++) bData[i] = Math.random() * 2 - 1;
      const aOff = base, bOff = base + aBytes, xOff = bOff + bBytes, scrOff = xOff + xBytes;
      flops = 2 * N * N * N;
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, M * Nc).set(aData); new Float64Array(wasm.memory.buffer, bOff, M).set(bData); };
      runFn = () => {
        new Float64Array(wasm.memory.buffer, base, M * Nc).set(aData);
        fn(aOff, bOff, xOff, scrOff, M, Nc);
      };
      readOutFn = () => { new Float64Array(wasm.memory.buffer, xOff, Nc).slice(); };
      break;
    }
    default:
      return null;
  }

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    copyFn();
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    readOutFn();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGflopsCustom(flops, kMs), metricUnit: 'GFLOP/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGflopsCustom(flops, eMs), metricUnit: 'GFLOP/s' },
  };
}

function benchJsLinalg(
  opName: string, size: number, iterations: number, warmup: number
): DualTimingResult {
  const N = size;
  let flops: number;
  let runFn: () => void;

  switch (opName) {
    case 'matvec': {
      const A = new Float64Array(N * N); for (let i = 0; i < N * N; i++) A[i] = Math.random() * 2 - 1;
      const x = new Float64Array(N); for (let i = 0; i < N; i++) x[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N);
      flops = 2 * N * N;
      runFn = () => { for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += A[i * N + j] * x[j]; out[i] = s; } };
      break;
    }
    case 'vecmat': {
      const x = new Float64Array(N); for (let i = 0; i < N; i++) x[i] = Math.random() * 2 - 1;
      const A = new Float64Array(N * N); for (let i = 0; i < N * N; i++) A[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N);
      flops = 2 * N * N;
      runFn = () => { for (let j = 0; j < N; j++) { let s = 0; for (let i = 0; i < N; i++) s += x[i] * A[i * N + j]; out[j] = s; } };
      break;
    }
    case 'vecdot': {
      const a = new Float64Array(N * N); for (let i = 0; i < N * N; i++) a[i] = Math.random() * 2 - 1;
      const b = new Float64Array(N * N); for (let i = 0; i < N * N; i++) b[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N);
      flops = 2 * N * N;
      runFn = () => { for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += a[i * N + j] * b[i * N + j]; out[i] = s; } };
      break;
    }
    case 'outer': {
      const a = new Float64Array(N); for (let i = 0; i < N; i++) a[i] = Math.random() * 2 - 1;
      const b = new Float64Array(N); for (let i = 0; i < N; i++) b[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N * N);
      flops = N * N;
      runFn = () => { for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) out[i * N + j] = a[i] * b[j]; };
      break;
    }
    case 'kron': {
      if (N > 100) return null; // kron output is N²×N², too large for N>100
      const a = new Float64Array(N * N); for (let i = 0; i < N * N; i++) a[i] = Math.random() * 2 - 1;
      const b = new Float64Array(N * N); for (let i = 0; i < N * N; i++) b[i] = Math.random() * 2 - 1;
      const outN = N * N;
      const out = new Float64Array(outN * outN);
      flops = N * N * N * N;
      runFn = () => {
        for (let ai = 0; ai < N; ai++)
          for (let aj = 0; aj < N; aj++)
            for (let bi = 0; bi < N; bi++)
              for (let bj = 0; bj < N; bj++)
                out[(ai * N + bi) * outN + aj * N + bj] = a[ai * N + aj] * b[bi * N + bj];
      };
      break;
    }
    case 'cross': {
      const a = new Float64Array(N * 3); for (let i = 0; i < N * 3; i++) a[i] = Math.random() * 2 - 1;
      const b = new Float64Array(N * 3); for (let i = 0; i < N * 3; i++) b[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N * 3);
      flops = 6 * N;
      runFn = () => {
        for (let i = 0; i < N; i++) {
          const o = i * 3;
          out[o] = a[o + 1] * b[o + 2] - a[o + 2] * b[o + 1];
          out[o + 1] = a[o + 2] * b[o] - a[o] * b[o + 2];
          out[o + 2] = a[o] * b[o + 1] - a[o + 1] * b[o];
        }
      };
      break;
    }
    case 'norm': {
      const totalElems = N * N;
      const data = new Float64Array(totalElems); for (let i = 0; i < totalElems; i++) data[i] = Math.random() * 2 - 1;
      flops = 2 * N * N;
      runFn = () => { let s = 0; for (let i = 0; i < totalElems; i++) s += data[i] * data[i]; Math.sqrt(s); };
      break;
    }
    case 'matrix_power': {
      const A = new Float64Array(N * N); for (let i = 0; i < N * N; i++) A[i] = Math.random() * 0.1;
      for (let i = 0; i < N; i++) A[i * N + i] += 1;
      const out = new Float64Array(N * N);
      const cur = new Float64Array(N * N);
      const tmp = new Float64Array(N * N);
      flops = 4 * N * N * N;
      const naiveMatmul = (a: Float64Array, b: Float64Array, c: Float64Array) => {
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < N; k++) s += a[i * N + k] * b[k * N + j]; c[i * N + j] = s; }
      };
      runFn = () => {
        for (let i = 0; i < N * N; i++) out[i] = 0;
        for (let i = 0; i < N; i++) out[i * N + i] = 1;
        cur.set(A);
        let p = 3;
        while (p > 0) {
          if (p & 1) { naiveMatmul(out, cur, tmp); out.set(tmp); }
          p >>= 1;
          if (p > 0) { naiveMatmul(cur, cur, tmp); cur.set(tmp); }
        }
      };
      break;
    }
    case 'multi_dot3': {
      const a = new Float64Array(N * N); for (let i = 0; i < N * N; i++) a[i] = Math.random() * 2 - 1;
      const b = new Float64Array(N * N); for (let i = 0; i < N * N; i++) b[i] = Math.random() * 2 - 1;
      const c = new Float64Array(N * N); for (let i = 0; i < N * N; i++) c[i] = Math.random() * 2 - 1;
      const tmp2 = new Float64Array(N * N);
      const out2 = new Float64Array(N * N);
      flops = 4 * N * N * N;
      const naiveMatmul2 = (a2: Float64Array, b2: Float64Array, c2: Float64Array) => {
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < N; k++) s += a2[i * N + k] * b2[k * N + j]; c2[i * N + j] = s; }
      };
      runFn = () => { naiveMatmul2(a, b, tmp2); naiveMatmul2(tmp2, c, out2); };
      break;
    }
    case 'qr': {
      const A = new Float64Array(N * N); for (let i = 0; i < N * N; i++) A[i] = Math.random() * 2 - 1;
      const Q = new Float64Array(N * N);
      const R = new Float64Array(N * N);
      flops = (4 * N * N * N) / 3;
      runFn = () => {
        // Simple Gram-Schmidt QR
        const copy = new Float64Array(N * N); copy.set(A);
        for (let j = 0; j < N; j++) {
          for (let i = 0; i < j; i++) {
            let dot = 0; for (let k = 0; k < N; k++) dot += Q[k * N + i] * copy[k * N + j];
            R[i * N + j] = dot;
            for (let k = 0; k < N; k++) copy[k * N + j] -= dot * Q[k * N + i];
          }
          let nrm = 0; for (let k = 0; k < N; k++) nrm += copy[k * N + j] * copy[k * N + j]; nrm = Math.sqrt(nrm);
          R[j * N + j] = nrm;
          if (nrm > 0) for (let k = 0; k < N; k++) Q[k * N + j] = copy[k * N + j] / nrm;
        }
      };
      break;
    }
    case 'lstsq': {
      const M = N, Nc = Math.max(Math.floor(N * 0.8), 1);
      const A = new Float64Array(M * Nc); for (let i = 0; i < M * Nc; i++) A[i] = Math.random() * 2 - 1;
      const bv = new Float64Array(M); for (let i = 0; i < M; i++) bv[i] = Math.random() * 2 - 1;
      const x = new Float64Array(Nc);
      flops = 2 * N * N * N;
      runFn = () => {
        // Simple normal equations: x = (A^T A)^-1 A^T b (for benchmarking only)
        const AtA = new Float64Array(Nc * Nc);
        const Atb = new Float64Array(Nc);
        for (let i = 0; i < Nc; i++) {
          for (let j = 0; j < Nc; j++) { let s = 0; for (let k = 0; k < M; k++) s += A[k * Nc + i] * A[k * Nc + j]; AtA[i * Nc + j] = s; }
          let s = 0; for (let k = 0; k < M; k++) s += A[k * Nc + i] * bv[k]; Atb[i] = s;
        }
        // Solve via back-sub (simplified)
        for (let i = Nc - 1; i >= 0; i--) {
          let s = Atb[i]; for (let j = i + 1; j < Nc; j++) s -= AtA[i * Nc + j] * x[j];
          x[i] = AtA[i * Nc + i] !== 0 ? s / AtA[i * Nc + i] : 0;
        }
      };
      break;
    }
    default:
      flops = 0;
      runFn = () => {};
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: opName,
    size,
    timeMs: ms,
    metric: computeGflopsCustom(flops, ms),
    metricUnit: 'GFLOP/s',
  });
}

// ─── Array Ops Benchmarks ───────────────────────────────────────────────────

function benchWasmArrayOp(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  const N = size;
  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const base = wasm.baseOffset;

  let throughputBytes: number;
  let runFn: () => void;
  let copyFn: () => void;
  let readOutFn: () => void = () => {};

  switch (opName) {
    case 'roll': {
      const fn = (wasm as Record<string, unknown>)[`roll_${dtype}`] as ((inp: number, out: number, n: number, shift: number) => void) | undefined;
      if (!fn) return null;
      const bufBytes = N * bytesPerElem;
      ensureMemory(wasm, 2 * bufBytes);
      const data = new ArrayType(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const shift = Math.floor(N / 3);
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(data); };
      runFn = () => fn(base, base + bufBytes, N, shift);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + bufBytes, N).slice(); };
      break;
    }
    case 'flip': {
      const fn = (wasm as Record<string, unknown>)[`flip_${dtype}`] as ((inp: number, out: number, n: number) => void) | undefined;
      if (!fn) return null;
      const bufBytes = N * bytesPerElem;
      ensureMemory(wasm, 2 * bufBytes);
      const data = new ArrayType(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(data); };
      runFn = () => fn(base, base + bufBytes, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + bufBytes, N).slice(); };
      break;
    }
    case 'tile': {
      const fn = (wasm as Record<string, unknown>)[`tile_${dtype}`] as ((inp: number, out: number, n: number, reps: number) => void) | undefined;
      if (!fn) return null;
      const inBytes = N * bytesPerElem;
      const reps = 4;
      const outBytes = N * reps * bytesPerElem;
      ensureMemory(wasm, inBytes + outBytes);
      const data = new ArrayType(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(data); };
      runFn = () => fn(base, base + inBytes, N, reps);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + inBytes, N * reps).slice(); };
      break;
    }
    case 'pad': {
      const fn = (wasm as Record<string, unknown>)[`pad_${dtype}`] as ((inp: number, out: number, rows: number, cols: number, pw: number) => void) | undefined;
      if (!fn) return null;
      const side = Math.floor(Math.sqrt(N));
      const pw = 2;
      const outSide = side + 2 * pw;
      const inBytes = side * side * bytesPerElem;
      const outBytes = outSide * outSide * bytesPerElem;
      ensureMemory(wasm, inBytes + outBytes);
      const data = new ArrayType(side * side); for (let i = 0; i < side * side; i++) data[i] = Math.random() * 2 - 1;
      throughputBytes = side * side * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, side * side).set(data); };
      runFn = () => fn(base, base + inBytes, side, side, pw);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + inBytes, outSide * outSide).slice(); };
      break;
    }
    case 'take': {
      const fn = (wasm as Record<string, unknown>)[`take_${dtype}`] as ((data: number, indices: number, out: number, n: number) => void) | undefined;
      if (!fn) return null;
      const dataBytes = N * bytesPerElem;
      const idxBytes = N * 4;
      const outBytes = N * bytesPerElem;
      ensureMemory(wasm, dataBytes + idxBytes + outBytes);
      const data = new ArrayType(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const indices = new Int32Array(N); for (let i = 0; i < N; i++) indices[i] = Math.floor(Math.random() * N);
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(data); new Int32Array(wasm.memory.buffer, base + dataBytes, N).set(indices); };
      runFn = () => fn(base, base + dataBytes, base + dataBytes + idxBytes, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + dataBytes + idxBytes, N).slice(); };
      break;
    }
    case 'gradient': {
      const fn = (wasm as Record<string, unknown>)[`gradient_${dtype}`] as ((inp: number, out: number, n: number) => void) | undefined;
      if (!fn) return null;
      const bufBytes = N * bytesPerElem;
      ensureMemory(wasm, 2 * bufBytes);
      const data = new ArrayType(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 10;
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(data); };
      runFn = () => fn(base, base + bufBytes, N);
      readOutFn = () => { new ArrayType(wasm.memory.buffer, base + bufBytes, N).slice(); };
      break;
    }
    case 'nonzero': {
      const fn = (wasm as Record<string, unknown>)[`nonzero_${dtype}`] as ((ptr: number, out: number, n: number) => number) | undefined;
      if (!fn) return null;
      const inputBytes = N * bytesPerElem;
      const outputBytes = N * 4;
      ensureMemory(wasm, inputBytes + outputBytes);
      const srcData = new ArrayType(N);
      for (let i = 0; i < N; i++) srcData[i] = Math.random() > 0.3 ? Math.random() * 10 : 0;
      const outPtr = base + inputBytes;
      throughputBytes = N * bytesPerElem;
      copyFn = () => { new ArrayType(wasm.memory.buffer, base, N).set(srcData); };
      runFn = () => fn(base, outPtr, N);
      readOutFn = () => { new Int32Array(wasm.memory.buffer, outPtr, N).slice(); };
      break;
    }
    default:
      return null;
  }

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    copyFn();
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    readOutFn();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(N, bytesPerElem, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(N, bytesPerElem, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsArrayOp(
  opName: string, size: number, iterations: number, warmup: number
): DualTimingResult {
  const N = size;
  let runFn: () => void;

  switch (opName) {
    case 'roll': {
      const data = new Float64Array(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N);
      const shift = Math.floor(N / 3);
      runFn = () => { for (let i = 0; i < N; i++) out[(i + shift) % N] = data[i]; };
      break;
    }
    case 'flip': {
      const data = new Float64Array(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const out = new Float64Array(N);
      runFn = () => { for (let i = 0; i < N; i++) out[N - 1 - i] = data[i]; };
      break;
    }
    case 'tile': {
      const data = new Float64Array(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const reps = 4;
      const out = new Float64Array(N * reps);
      runFn = () => { for (let r = 0; r < reps; r++) for (let i = 0; i < N; i++) out[r * N + i] = data[i]; };
      break;
    }
    case 'pad': {
      const side = Math.floor(Math.sqrt(N));
      const pw = 2;
      const outSide = side + 2 * pw;
      const data = new Float64Array(side * side); for (let i = 0; i < side * side; i++) data[i] = Math.random() * 2 - 1;
      const out = new Float64Array(outSide * outSide);
      runFn = () => {
        out.fill(0);
        for (let i = 0; i < side; i++)
          for (let j = 0; j < side; j++)
            out[(i + pw) * outSide + (j + pw)] = data[i * side + j];
      };
      break;
    }
    case 'take': {
      const data = new Float64Array(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1;
      const indices = new Int32Array(N); for (let i = 0; i < N; i++) indices[i] = Math.floor(Math.random() * N);
      const out = new Float64Array(N);
      runFn = () => { for (let i = 0; i < N; i++) out[i] = data[indices[i]]; };
      break;
    }
    case 'gradient': {
      const data = new Float64Array(N); for (let i = 0; i < N; i++) data[i] = Math.random() * 10;
      const out = new Float64Array(N);
      runFn = () => {
        out[0] = data[1] - data[0];
        for (let i = 1; i < N - 1; i++) out[i] = (data[i + 1] - data[i - 1]) / 2;
        out[N - 1] = data[N - 1] - data[N - 2];
      };
      break;
    }
    case 'nonzero': {
      const data = new Float64Array(N);
      for (let i = 0; i < N; i++) data[i] = Math.random() > 0.3 ? Math.random() * 10 : 0;
      const indices: number[] = [];
      runFn = () => {
        indices.length = 0;
        for (let i = 0; i < N; i++) { if (data[i] !== 0) indices.push(i); }
      };
      break;
    }
    default:
      runFn = () => {};
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(N, 8, ms),
    metricUnit: 'GB/s',
  });
}

// ─── Stats Benchmarks (sort-category) ────────────────────────────────────────

function benchWasmStats(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32',
  iterations: number, warmup: number
): DualTimingResult | null {
  const fn64 = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((...args: number[]) => number) | undefined;
  if (!fn64) return null;
  const mem = wasm.memory;
  const N = size;
  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const dataBytes = N * bytesPerElem;

  const ptrOffset = wasm.baseOffset;

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const srcData = new ArrayType(N);
  for (let i = 0; i < N; i++) srcData[i] = Math.random() * 200 - 100;

  ensureMemory(wasm, dataBytes);

  // Warmup
  for (let w = 0; w < warmup; w++) {
    new ArrayType(mem.buffer, ptrOffset, N).set(srcData);
    if (opName === 'percentile') fn64(ptrOffset, N, 50.0);
    else if (opName === 'quantile') fn64(ptrOffset, N, 0.5);
    else fn64(ptrOffset, N);
  }

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < iterations; iter++) {
    const te = performance.now();
    new ArrayType(mem.buffer, ptrOffset, N).set(srcData);
    const t0 = performance.now();
    if (opName === 'percentile') fn64(ptrOffset, N, 50.0);
    else if (opName === 'quantile') fn64(ptrOffset, N, 0.5);
    else fn64(ptrOffset, N);
    const t1 = performance.now();
    kernelTimes.push(t1 - t0);
    e2eTimes.push(t1 - te);
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  const mkResult = (ms: number): TimingResult => ({
    impl, op: opName, size: N, timeMs: ms,
    metric: (N / 1e6) / (ms / 1000), metricUnit: 'M elem/s',
  });
  return { kernel: mkResult(kMs), e2e: mkResult(eMs) };
}

function benchJsStats(
  opName: string, size: number, iterations: number, warmup: number
): DualTimingResult {
  const N = size;
  const srcData = new Float64Array(N);
  for (let i = 0; i < N; i++) srcData[i] = Math.random() * 200 - 100;
  const workBuf = new Float64Array(N);

  const runFn = () => {
    workBuf.set(srcData);
    // Sort then pick — JS doesn't have quickselect
    workBuf.sort();
    if (opName === 'median') {
      const mid = Math.floor(N / 2);
      return N % 2 === 1 ? workBuf[mid] : (workBuf[mid - 1] + workBuf[mid]) / 2;
    } else {
      // percentile(50) and quantile(0.5) are same as median
      const idx = 0.5 * (N - 1);
      const lo = Math.floor(idx);
      const t = idx - lo;
      return workBuf[lo] * (1 - t) + workBuf[Math.min(lo + 1, N - 1)] * t;
    }
  };

  for (let w = 0; w < warmup; w++) runFn();
  const times: number[] = [];
  for (let iter = 0; iter < iterations; iter++) {
    const t0 = performance.now();
    runFn();
    times.push(performance.now() - t0);
  }

  const avgMs = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: opName,
    size: N,
    timeMs: avgMs,
    metric: (N / 1e6) / (avgMs / 1000),
    metricUnit: 'M elem/s',
  });
}

// ─── FFT Benchmarks ─────────────────────────────────────────────────────────

function benchWasmFft(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): DualTimingResult | null {
  if (dtype !== 'f64') return null;
  const M = size, N = size;
  const base = wasm.baseOffset;

  let flops: number;
  let runFn: () => void;
  let copyFn: () => void;
  let readOutFn: () => void;

  switch (opName) {
    case 'rfft2': {
      const fn = wasm.rfft2_f64;
      if (!fn) return null;
      const inpBytes = M * N * 8;
      const halfN = Math.floor(N / 2) + 1;
      const outBytes = M * halfN * 2 * 8;
      const scratchBytes = 8 * Math.max(M, N) * Math.max(M, N) * 8;
      ensureMemory(wasm, inpBytes + outBytes + scratchBytes);
      const data = new Float64Array(M * N); for (let i = 0; i < M * N; i++) data[i] = Math.random() * 2 - 1;
      const inpOff = base, outOff = base + inpBytes, scrOff = base + inpBytes + outBytes;
      flops = 5 * M * N * Math.log2(M * N);
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, M * N).set(data); };
      runFn = () => fn(inpOff, outOff, scrOff, M, N);
      readOutFn = () => { new Float64Array(wasm.memory.buffer, outOff, M * halfN * 2).slice(); };
      break;
    }
    case 'irfft2': {
      const fn = wasm.irfft2_f64;
      if (!fn) return null;
      const halfN = Math.floor(N / 2) + 1;
      const inpBytes = M * halfN * 2 * 8;
      const outBytes = M * N * 8;
      const scratchBytes = 8 * Math.max(M, N) * Math.max(M, N) * 8;
      ensureMemory(wasm, inpBytes + outBytes + scratchBytes);
      const data = new Float64Array(M * halfN * 2); for (let i = 0; i < M * halfN * 2; i++) data[i] = Math.random() * 2 - 1;
      const inpOff = base, outOff = base + inpBytes, scrOff = base + inpBytes + outBytes;
      flops = 5 * M * N * Math.log2(M * N);
      copyFn = () => { new Float64Array(wasm.memory.buffer, base, M * halfN * 2).set(data); };
      runFn = () => fn(inpOff, outOff, scrOff, M, N);
      readOutFn = () => { new Float64Array(wasm.memory.buffer, outOff, M * N).slice(); };
      break;
    }
    default:
      return null;
  }

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    copyFn();
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    readOutFn();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGflopsCustom(flops, kMs), metricUnit: 'GFLOP/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGflopsCustom(flops, eMs), metricUnit: 'GFLOP/s' },
  };
}

function benchJsFft(
  opName: string, size: number, iterations: number, warmup: number
): DualTimingResult {
  const M = size, N = size;
  let flops: number;
  let runFn: () => void;

  // Naive JS DFT for baseline
  const naiveDft = (re: Float64Array, im: Float64Array, outRe: Float64Array, outIm: Float64Array, n: number, inv: boolean) => {
    const sign = inv ? 1 : -1;
    for (let k = 0; k < n; k++) {
      let sr = 0, si = 0;
      for (let j = 0; j < n; j++) {
        const angle = sign * 2 * Math.PI * k * j / n;
        const cr = Math.cos(angle), ci = Math.sin(angle);
        sr += re[j] * cr - im[j] * ci;
        si += re[j] * ci + im[j] * cr;
      }
      outRe[k] = inv ? sr / n : sr;
      outIm[k] = inv ? si / n : si;
    }
  };

  switch (opName) {
    case 'rfft2': {
      const data = new Float64Array(M * N); for (let i = 0; i < M * N; i++) data[i] = Math.random() * 2 - 1;
      const halfN = Math.floor(N / 2) + 1;
      const outRe = new Float64Array(M * halfN);
      const outIm = new Float64Array(M * halfN);
      flops = 5 * M * N * Math.log2(M * N);
      runFn = () => {
        // Row FFTs then column FFTs (naive O(N^2) per transform)
        const re = new Float64Array(N), im = new Float64Array(N);
        const ore = new Float64Array(N), oim = new Float64Array(N);
        for (let r = 0; r < M; r++) {
          for (let j = 0; j < N; j++) { re[j] = data[r * N + j]; im[j] = 0; }
          naiveDft(re, im, ore, oim, N, false);
          for (let j = 0; j < halfN; j++) { outRe[r * halfN + j] = ore[j]; outIm[r * halfN + j] = oim[j]; }
        }
      };
      break;
    }
    case 'irfft2': {
      const halfN = Math.floor(N / 2) + 1;
      const inpRe = new Float64Array(M * halfN); for (let i = 0; i < M * halfN; i++) inpRe[i] = Math.random() * 2 - 1;
      const inpIm = new Float64Array(M * halfN); for (let i = 0; i < M * halfN; i++) inpIm[i] = Math.random() * 2 - 1;
      const out = new Float64Array(M * N);
      flops = 5 * M * N * Math.log2(M * N);
      runFn = () => {
        const re = new Float64Array(N), im = new Float64Array(N);
        const ore = new Float64Array(N), oim = new Float64Array(N);
        for (let r = 0; r < M; r++) {
          for (let j = 0; j < halfN; j++) { re[j] = inpRe[r * halfN + j]; im[j] = inpIm[r * halfN + j]; }
          for (let j = halfN; j < N; j++) { re[j] = re[N - j]; im[j] = -im[N - j]; }
          naiveDft(re, im, ore, oim, N, true);
          for (let j = 0; j < N; j++) out[r * N + j] = ore[j];
        }
      };
      break;
    }
    default:
      flops = 0;
      runFn = () => {};
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    runFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: 'JS (f64)',
    op: opName,
    size,
    timeMs: ms,
    metric: computeGflopsCustom(flops, ms),
    metricUnit: 'GFLOP/s',
  });
}

// ─── Integer Benchmark Functions ─────────────────────────────────────────────

type IntDtype = 'i32' | 'i16' | 'i8';
type ComplexDtype = 'c128' | 'c64';

function intArrayType(dtype: IntDtype): typeof Int32Array | typeof Int16Array | typeof Int8Array {
  return dtype === 'i32' ? Int32Array : dtype === 'i16' ? Int16Array : Int8Array;
}

function intBytesPerElem(dtype: IntDtype): number {
  return dtype === 'i32' ? 4 : dtype === 'i16' ? 2 : 1;
}

function intMaxVal(dtype: IntDtype): number {
  return dtype === 'i32' ? 1000 : dtype === 'i16' ? 1000 : 50;
}

function benchWasmBinaryInt(
  wasm: WasmKernels, opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((a: number, b: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bpe = intBytesPerElem(dtype);
  const bufBytes = size * bpe;
  ensureMemory(wasm, 3 * bufBytes);

  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const aData = new AT(size);
  const bData = new AT(size);
  for (let i = 0; i < size; i++) {
    aData[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);
    bData[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);
  }
  const base = wasm.baseOffset;

  const aOff = base, bOff = base + bufBytes, outOff = base + 2 * bufBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new AT(wasm.memory.buffer, base, size).set(aData);
    new AT(wasm.memory.buffer, base + bufBytes, size).set(bData);
    const t0 = performance.now();
    fn(aOff, bOff, outOff, size);
    const t1 = performance.now();
    new AT(wasm.memory.buffer, outOff, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, bpe, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, bpe, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsBinaryInt(
  opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const a = new AT(size), b = new AT(size), out = new AT(size);
  for (let i = 0; i < size; i++) {
    a[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);
    b[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);
  }

  let jsFn: () => void;
  switch (opName) {
    case 'add': jsFn = () => { for (let i = 0; i < size; i++) out[i] = a[i] + b[i]; }; break;
    case 'sub': jsFn = () => { for (let i = 0; i < size; i++) out[i] = a[i] - b[i]; }; break;
    case 'mul': jsFn = () => { for (let i = 0; i < size; i++) out[i] = a[i] * b[i]; }; break;
    case 'maximum': jsFn = () => { for (let i = 0; i < size; i++) out[i] = a[i] > b[i] ? a[i] : b[i]; }; break;
    case 'minimum': jsFn = () => { for (let i = 0; i < size; i++) out[i] = a[i] < b[i] ? a[i] : b[i]; }; break;
    default: jsFn = () => {};
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, intBytesPerElem(dtype), ms),
    metricUnit: 'GB/s',
  });
}

function benchWasmReductionInt(
  wasm: WasmKernels, opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((ptr: number, n: number) => number) | undefined;
  if (!fn) return null;

  const bpe = intBytesPerElem(dtype);
  const bufBytes = size * bpe;
  ensureMemory(wasm, bufBytes);

  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const data = new AT(size);
  for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);
  const base = wasm.baseOffset;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new AT(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t1 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, bpe, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, bpe, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsReductionInt(
  opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const data = new AT(size);
  for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * maxV) - Math.floor(maxV / 2);

  let jsFn: () => number;
  switch (opName) {
    case 'sum': jsFn = () => { let s = 0; for (let i = 0; i < size; i++) s += data[i]; return s; }; break;
    case 'max': jsFn = () => { let m = data[0]; for (let i = 1; i < size; i++) if (data[i] > m) m = data[i]; return m; }; break;
    case 'min': jsFn = () => { let m = data[0]; for (let i = 1; i < size; i++) if (data[i] < m) m = data[i]; return m; }; break;
    default: jsFn = () => 0;
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, intBytesPerElem(dtype), ms),
    metricUnit: 'GB/s',
  });
}

function benchWasmSortInt(
  wasm: WasmKernels, opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((ptr: number, n: number) => void) | undefined;
  if (!fn && opName === 'sort') return null;

  if (opName === 'argsort') {
    const fnA = (wasm as Record<string, unknown>)[`argsort_${dtype}`] as ((vals: number, idx: number, n: number) => void) | undefined;
    if (!fnA) return null;

    const bpe = intBytesPerElem(dtype);
    const valBytes = size * bpe;
    const idxBytes = size * 4;
    ensureMemory(wasm, valBytes + idxBytes);

    const AT = intArrayType(dtype);
    const maxV = intMaxVal(dtype);
    const data = new AT(size);
    for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * maxV);
    const base = wasm.baseOffset;

    const kernelTimes: number[] = [];
    const e2eTimes: number[] = [];
    for (let iter = 0; iter < warmup + iterations; iter++) {
      const te = performance.now();
      new AT(wasm.memory.buffer, base, size).set(data);
      const t0 = performance.now();
      fnA(base, base + valBytes, size);
      const t1 = performance.now();
      new Int32Array(wasm.memory.buffer, base + valBytes, size).slice();
      const t2 = performance.now();
      if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
    }

    const kMs = median(kernelTimes);
    const eMs = median(e2eTimes);
    const impl = `${wasm.name} (${dtype})`;
    return {
      kernel: { impl, op: opName, size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
      e2e: { impl, op: opName, size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
    };
  }

  if (!fn) return null;
  const bpe = intBytesPerElem(dtype);
  const bufBytes = size * bpe;
  ensureMemory(wasm, bufBytes);

  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const data = new AT(size);
  for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * maxV);
  const base = wasm.baseOffset;

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new AT(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    new AT(wasm.memory.buffer, base, size).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeMElemPerSec(size, kMs), metricUnit: 'M elem/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeMElemPerSec(size, eMs), metricUnit: 'M elem/s' },
  };
}

function benchJsSortInt(
  opName: string, dtype: IntDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const AT = intArrayType(dtype);
  const maxV = intMaxVal(dtype);
  const data = new AT(size);
  for (let i = 0; i < size; i++) data[i] = Math.floor(Math.random() * maxV);

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const copy = new AT(data);
    const t0 = performance.now();
    if (opName === 'argsort') {
      // Create index array and sort by values
      const indices = new Uint32Array(size);
      for (let i = 0; i < size; i++) indices[i] = i;
      indices.sort((a, b) => copy[a] - copy[b]);
    } else {
      copy.sort();
    }
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  });
}

// ─── Complex Benchmark Functions ─────────────────────────────────────────────

function complexFloatType(dtype: ComplexDtype): typeof Float64Array | typeof Float32Array {
  return dtype === 'c128' ? Float64Array : Float32Array;
}

function complexBytesPerElem(dtype: ComplexDtype): number {
  return dtype === 'c128' ? 16 : 8; // per complex element
}

function complexScalarBytes(dtype: ComplexDtype): number {
  return dtype === 'c128' ? 8 : 4; // per real/imag component
}

function benchWasmBinaryComplex(
  wasm: WasmKernels, opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((a: number, b: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const scBytes = complexScalarBytes(dtype);
  const bufBytes = size * 2 * scBytes;
  ensureMemory(wasm, 3 * bufBytes);

  const FT = complexFloatType(dtype);
  const a = new FT(size * 2), b = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }
  const base = wasm.baseOffset;

  const aOff = base, bOff = base + bufBytes, outOff = base + 2 * bufBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new FT(wasm.memory.buffer, base, size * 2).set(a);
    new FT(wasm.memory.buffer, base + bufBytes, size * 2).set(b);
    const t0 = performance.now();
    fn(aOff, bOff, outOff, size);
    const t1 = performance.now();
    new FT(wasm.memory.buffer, outOff, size * 2).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  const cbpe = complexBytesPerElem(dtype);
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, cbpe, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, cbpe, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsBinaryComplex(
  opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const FT = complexFloatType(dtype);
  const a = new FT(size * 2), b = new FT(size * 2), out = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }

  let jsFn: () => void;
  if (opName === 'add') {
    jsFn = () => { for (let i = 0; i < size * 2; i++) out[i] = a[i] + b[i]; };
  } else {
    // mul: (ac-bd, ad+bc)
    jsFn = () => {
      for (let i = 0; i < size; i++) {
        const ar = a[2*i], ai = a[2*i+1], br = b[2*i], bi = b[2*i+1];
        out[2*i] = ar*br - ai*bi;
        out[2*i+1] = ar*bi + ai*br;
      }
    };
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, complexBytesPerElem(dtype), ms),
    metricUnit: 'GB/s',
  });
}

function benchWasmUnaryComplex(
  wasm: WasmKernels, opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((inp: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const scBytes = complexScalarBytes(dtype);
  const inpBytes = size * 2 * scBytes;
  const outBytes = opName === 'abs' ? size * scBytes : size * 2 * scBytes;
  ensureMemory(wasm, inpBytes + outBytes);

  const FT = complexFloatType(dtype);
  const inp = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) inp[i] = Math.random() * 2 - 1;
  const base = wasm.baseOffset;

  const inpOff = base, outOff = base + inpBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new FT(wasm.memory.buffer, base, size * 2).set(inp);
    const t0 = performance.now();
    fn(inpOff, outOff, size);
    const t1 = performance.now();
    new FT(wasm.memory.buffer, outOff, opName === 'abs' ? size : size * 2).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  const cbpe = complexBytesPerElem(dtype);
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, cbpe, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, cbpe, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsUnaryComplex(
  opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const FT = complexFloatType(dtype);
  const inp = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) inp[i] = Math.random() * 2 - 1;
  const out = opName === 'abs' ? new FT(size) : new FT(size * 2);

  let jsFn: () => void;
  if (opName === 'abs') {
    jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.sqrt(inp[2*i]*inp[2*i] + inp[2*i+1]*inp[2*i+1]); };
  } else {
    // exp(a+bi) = e^a * (cos(b) + i*sin(b))
    jsFn = () => {
      for (let i = 0; i < size; i++) {
        const ea = Math.exp(inp[2*i]);
        out[2*i] = ea * Math.cos(inp[2*i+1]);
        out[2*i+1] = ea * Math.sin(inp[2*i+1]);
      }
    };
  }

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, complexBytesPerElem(dtype), ms),
    metricUnit: 'GB/s',
  });
}

function benchWasmReductionComplex(
  wasm: WasmKernels, opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((ptr: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const scBytes = complexScalarBytes(dtype);
  const inpBytes = size * 2 * scBytes;
  const outBytes = 2 * scBytes;
  ensureMemory(wasm, inpBytes + outBytes);

  const FT = complexFloatType(dtype);
  const data = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) data[i] = Math.random() * 2 - 1;
  const base = wasm.baseOffset;

  const inpOff = base, outOff = base + inpBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new FT(wasm.memory.buffer, base, size * 2).set(data);
    const t0 = performance.now();
    fn(inpOff, outOff, size);
    const t1 = performance.now();
    new FT(wasm.memory.buffer, outOff, 2).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  const cbpe = complexBytesPerElem(dtype);
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGBs(size, cbpe, kMs), metricUnit: 'GB/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGBs(size, cbpe, eMs), metricUnit: 'GB/s' },
  };
}

function benchJsReductionComplex(
  opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const FT = complexFloatType(dtype);
  const data = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) data[i] = Math.random() * 2 - 1;

  const jsFn = () => {
    let re = 0, im = 0;
    for (let i = 0; i < size; i++) { re += data[2*i]; im += data[2*i+1]; }
    return [re, im];
  };

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, complexBytesPerElem(dtype), ms),
    metricUnit: 'GB/s',
  });
}

function benchWasmLinalgComplex(
  wasm: WasmKernels, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`matmul_${dtype}`] as ((a: number, b: number, c: number, m: number, k: number, n: number) => void) | undefined;
  if (!fn) return null;

  const scBytes = complexScalarBytes(dtype);
  const matElems = size * size;
  const matBytes = matElems * 2 * scBytes;
  ensureMemory(wasm, 3 * matBytes);

  const FT = complexFloatType(dtype);
  const a = new FT(matElems * 2), b = new FT(matElems * 2);
  for (let i = 0; i < matElems * 2; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }
  const base = wasm.baseOffset;

  const aOff = base, bOff = base + matBytes, cOff = base + 2 * matBytes;
  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new FT(wasm.memory.buffer, base, matElems * 2).set(a);
    new FT(wasm.memory.buffer, base + matBytes, matElems * 2).set(b);
    const t0 = performance.now();
    fn(aOff, bOff, cOff, size, size, size);
    const t1 = performance.now();
    new FT(wasm.memory.buffer, cOff, matElems * 2).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const flops = 8 * size * size * size;
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: 'matmul', size, timeMs: kMs, metric: computeGflopsCustom(flops, kMs), metricUnit: 'GFLOP/s' },
    e2e: { impl, op: 'matmul', size, timeMs: eMs, metric: computeGflopsCustom(flops, eMs), metricUnit: 'GFLOP/s' },
  };
}

function benchJsLinalgComplex(
  dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const a = new Float64Array(size * size * 2), b = new Float64Array(size * size * 2);
  const c = new Float64Array(size * size * 2);
  for (let i = 0; i < size * size * 2; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }

  const jsFn = () => {
    c.fill(0);
    for (let i = 0; i < size; i++) {
      for (let k = 0; k < size; k++) {
        const ar = a[(i*size+k)*2], ai = a[(i*size+k)*2+1];
        for (let j = 0; j < size; j++) {
          const br = b[(k*size+j)*2], bi = b[(k*size+j)*2+1];
          c[(i*size+j)*2] += ar*br - ai*bi;
          c[(i*size+j)*2+1] += ar*bi + ai*br;
        }
      }
    }
  };

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  const flops = 8 * size * size * size;
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: 'matmul',
    size,
    timeMs: ms,
    metric: computeGflopsCustom(flops, ms),
    metricUnit: 'GFLOP/s',
  });
}

function benchWasmFftComplex(
  wasm: WasmKernels, opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((inp: number, out: number, scratch: number, n: number) => void) | undefined;
  if (!fn) return null;

  const scBytes = complexScalarBytes(dtype);
  const FT = complexFloatType(dtype);
  const inpBytes = size * 2 * scBytes;
  const outBytes = size * 2 * scBytes;
  const scratchBytes = 8 * size * 8;
  ensureMemory(wasm, inpBytes + outBytes + scratchBytes);

  const data = new FT(size * 2);
  for (let i = 0; i < size * 2; i++) data[i] = Math.random() * 2 - 1;
  const base = wasm.baseOffset;

  const inpOff = base, outOff = base + inpBytes, scrOff = base + inpBytes + outBytes;
  const flops = 5 * size * Math.log2(size);

  const kernelTimes: number[] = [];
  const e2eTimes: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const te = performance.now();
    new FT(wasm.memory.buffer, base, size * 2).set(data);
    const t0 = performance.now();
    fn(inpOff, outOff, scrOff, size);
    const t1 = performance.now();
    new FT(wasm.memory.buffer, outOff, size * 2).slice();
    const t2 = performance.now();
    if (iter >= warmup) { kernelTimes.push(t1 - t0); e2eTimes.push(t2 - te); }
  }

  const kMs = median(kernelTimes);
  const eMs = median(e2eTimes);
  const impl = `${wasm.name} (${dtype})`;
  return {
    kernel: { impl, op: opName, size, timeMs: kMs, metric: computeGflopsCustom(flops, kMs), metricUnit: 'GFLOP/s' },
    e2e: { impl, op: opName, size, timeMs: eMs, metric: computeGflopsCustom(flops, eMs), metricUnit: 'GFLOP/s' },
  };
}

function benchJsFftComplex(
  opName: string, dtype: ComplexDtype, size: number, iterations: number, warmup: number
): DualTimingResult {
  const data = new Float64Array(size * 2);
  for (let i = 0; i < size * 2; i++) data[i] = Math.random() * 2 - 1;
  const outRe = new Float64Array(size), outIm = new Float64Array(size);
  const inRe = new Float64Array(size), inIm = new Float64Array(size);

  const inverse = opName === 'ifft';
  const sign = inverse ? 1 : -1;

  // Naive DFT baseline
  const jsFn = () => {
    for (let i = 0; i < size; i++) { inRe[i] = data[2*i]; inIm[i] = data[2*i+1]; }
    for (let k = 0; k < size; k++) {
      let sr = 0, si = 0;
      for (let j = 0; j < size; j++) {
        const angle = sign * 2 * Math.PI * k * j / size;
        const cr = Math.cos(angle), ci = Math.sin(angle);
        sr += inRe[j] * cr - inIm[j] * ci;
        si += inRe[j] * ci + inIm[j] * cr;
      }
      outRe[k] = inverse ? sr / size : sr;
      outIm[k] = inverse ? si / size : si;
    }
  };

  const flops = 5 * size * Math.log2(size);
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    jsFn();
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return dualJsResult({
    impl: `JS (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGflopsCustom(flops, ms),
    metricUnit: 'GFLOP/s',
  });
}

// ─── Correctness Checks ─────────────────────────────────────────────────────

function verifyAll(wasms: WasmKernels[]): boolean {
  let allOk = true;

  for (const wasm of wasms) {
    const checks: [string, boolean][] = [];
    const B = wasm.baseOffset; // safe data region start

    // Matmul
    if (wasm.matmul_f64) {
      ensureMemory(wasm, 3 * 16 * 8);
      const a = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      const b = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      new Float64Array(wasm.memory.buffer, B, 16).set(a);
      new Float64Array(wasm.memory.buffer, B + 128, 16).set(b);
      wasm.matmul_f64(B, B + 128, B + 256, 4, 4, 4);
      const c = new Float64Array(wasm.memory.buffer, B + 256, 16);
      checks.push(['matmul', b.every((v, i) => Math.abs(c[i] - v) < 1e-10)]);
    }

    // Reductions
    if (wasm.sum_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      checks.push(['sum', Math.abs(wasm.sum_f64(B, 4) - 10) < 1e-10]);
    }
    if (wasm.max_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 6).set([3, 1, 4, 1, 5, 9]);
      checks.push(['max', Math.abs(wasm.max_f64(B, 6) - 9) < 1e-10]);
    }
    if (wasm.min_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 6).set([3, 1, 4, 1, 5, 9]);
      checks.push(['min', Math.abs(wasm.min_f64(B, 6) - 1) < 1e-10]);
    }
    if (wasm.prod_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      checks.push(['prod', Math.abs(wasm.prod_f64(B, 4) - 24) < 1e-10]);
    }
    if (wasm.mean_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      checks.push(['mean', Math.abs(wasm.mean_f64(B, 4) - 2.5) < 1e-10]);
    }

    // Unary
    if (wasm.sqrt_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 1).set([4]);
      wasm.sqrt_f64(B, B + 8, 1);
      checks.push(['sqrt', Math.abs(new Float64Array(wasm.memory.buffer, B + 8, 1)[0] - 2) < 1e-10]);
    }
    if (wasm.exp_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, 1]);
      wasm.exp_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['exp', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - Math.E) < 1e-6]);
    }
    if (wasm.log_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, Math.E]);
      wasm.log_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['log', Math.abs(r[0]) < 1e-10 && Math.abs(r[1] - 1) < 1e-6]);
    }
    if (wasm.sin_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, Math.PI / 2]);
      wasm.sin_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['sin', Math.abs(r[0]) < 1e-10 && Math.abs(r[1] - 1) < 1e-10]);
    }
    if (wasm.cos_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, Math.PI]);
      wasm.cos_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['cos', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] + 1) < 1e-10]);
    }
    if (wasm.abs_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([-3, 5]);
      wasm.abs_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['abs', Math.abs(r[0] - 3) < 1e-10 && Math.abs(r[1] - 5) < 1e-10]);
    }
    if (wasm.neg_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([3, -5]);
      wasm.neg_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['neg', Math.abs(r[0] + 3) < 1e-10 && Math.abs(r[1] - 5) < 1e-10]);
    }
    if (wasm.ceil_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([1.2, -1.8]);
      wasm.ceil_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['ceil', Math.abs(r[0] - 2) < 1e-10 && Math.abs(r[1] + 1) < 1e-10]);
    }
    if (wasm.floor_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([1.8, -1.2]);
      wasm.floor_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['floor', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] + 2) < 1e-10]);
    }

    // Binary
    if (wasm.add_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([3, 4]);
      wasm.add_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['add', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 6) < 1e-10]);
    }
    if (wasm.sub_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([5, 3]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([2, 1]);
      wasm.sub_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['sub', Math.abs(r[0] - 3) < 1e-10 && Math.abs(r[1] - 2) < 1e-10]);
    }
    if (wasm.mul_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([2, 3]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([4, 5]);
      wasm.mul_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['mul', Math.abs(r[0] - 8) < 1e-10 && Math.abs(r[1] - 15) < 1e-10]);
    }
    if (wasm.div_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([10, 9]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([2, 3]);
      wasm.div_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['div', Math.abs(r[0] - 5) < 1e-10 && Math.abs(r[1] - 3) < 1e-10]);
    }

    // Sort
    if (wasm.sort_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 6).set([3, 1, 4, 1, 5, 9]);
      wasm.sort_f64(B, 6);
      const r = new Float64Array(wasm.memory.buffer, B, 6);
      const expected = [1, 1, 3, 4, 5, 9];
      checks.push(['sort', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // New binary ops
    if (wasm.copysign_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([-1, -1, 1]);
      wasm.copysign_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['copysign', Math.abs(r[0] + 1) < 1e-10 && Math.abs(r[1] + 2) < 1e-10 && Math.abs(r[2] - 3) < 1e-10]);
    }
    if (wasm.power_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([4, 9, 16]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([0.5, 0.5, 0.5]);
      wasm.power_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['power', Math.abs(r[0] - 2) < 1e-10 && Math.abs(r[1] - 3) < 1e-10 && Math.abs(r[2] - 4) < 1e-10]);
    }
    if (wasm.maximum_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 5, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([4, 2, 6]);
      wasm.maximum_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['maximum', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 5) < 1e-10 && Math.abs(r[2] - 6) < 1e-10]);
    }
    if (wasm.minimum_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 5, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([4, 2, 6]);
      wasm.minimum_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['minimum', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - 2) < 1e-10 && Math.abs(r[2] - 3) < 1e-10]);
    }
    if (wasm.fmax_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 5, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([4, 2, 6]);
      wasm.fmax_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['fmax', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 5) < 1e-10 && Math.abs(r[2] - 6) < 1e-10]);
    }
    if (wasm.fmin_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 5, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([4, 2, 6]);
      wasm.fmin_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['fmin', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - 2) < 1e-10 && Math.abs(r[2] - 3) < 1e-10]);
    }
    if (wasm.logaddexp_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, 0]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([0, 0]);
      wasm.logaddexp_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      const ln2 = Math.log(2);
      checks.push(['logaddexp', Math.abs(r[0] - ln2) < 1e-6 && Math.abs(r[1] - ln2) < 1e-6]);
    }
    if (wasm.logical_and_f64) {
      ensureMemory(wasm, 72);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 0, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([1, 2, 0]);
      wasm.logical_and_f64(B, B + 24, B + 48, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['logical_and', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1]) < 1e-10 && Math.abs(r[2]) < 1e-10]);
    }
    if (wasm.logical_xor_f64) {
      ensureMemory(wasm, 96);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 0, 3, 0]);
      new Float64Array(wasm.memory.buffer, B + 32, 4).set([1, 2, 0, 0]);
      wasm.logical_xor_f64(B, B + 32, B + 64, 4);
      const r = new Float64Array(wasm.memory.buffer, B + 64, 4);
      checks.push(['logical_xor', Math.abs(r[0]) < 1e-10 && Math.abs(r[1] - 1) < 1e-10 && Math.abs(r[2] - 1) < 1e-10 && Math.abs(r[3]) < 1e-10]);
    }

    // Diff
    if (wasm.diff_f64) {
      ensureMemory(wasm, 56);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 3, 6, 10]);
      wasm.diff_f64(B, B + 32, 4);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 3);
      checks.push(['diff', Math.abs(r[0] - 2) < 1e-10 && Math.abs(r[1] - 3) < 1e-10 && Math.abs(r[2] - 4) < 1e-10]);
    }

    // Argsort
    if (wasm.argsort_f64) {
      ensureMemory(wasm, 60);
      new Float64Array(wasm.memory.buffer, B, 5).set([3, 1, 4, 1, 5]);
      wasm.argsort_f64(B, B + 40, 5);
      const r = new Uint32Array(wasm.memory.buffer, B + 40, 5);
      const expected = [1, 3, 0, 2, 4];
      checks.push(['argsort', expected.every((v, i) => r[i] === v)]);
    }

    // Partition
    if (wasm.partition_f64) {
      ensureMemory(wasm, 56);
      new Float64Array(wasm.memory.buffer, B, 7).set([3, 1, 4, 1, 5, 9, 2]);
      wasm.partition_f64(B, 7, 3);
      const r = new Float64Array(wasm.memory.buffer, B, 7);
      // After partition with kth=3, arr[3] should be the 4th smallest (value 3)
      checks.push(['partition', Math.abs(r[3] - 3) < 1e-10]);
    }

    // Argpartition — skip correctness for now (complex to verify in-place)

    // Correlate
    if (wasm.correlate_f64) {
      ensureMemory(wasm, 80);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 2).set([1, 1]);
      wasm.correlate_f64(B, B + 24, B + 40, 3, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 40, 4);
      const expected = [1, 3, 5, 3];
      checks.push(['correlate', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Convolve
    if (wasm.convolve_f64) {
      ensureMemory(wasm, 80);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 2).set([1, 1]);
      wasm.convolve_f64(B, B + 24, B + 40, 3, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 40, 4);
      const expected = [1, 3, 5, 3];
      checks.push(['convolve', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Linalg — matvec: [[1,2],[3,4]] · [1,1] = [3,7]
    if (wasm.matvec_f64) {
      ensureMemory(wasm, 6 * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      new Float64Array(wasm.memory.buffer, B + 32, 2).set([1, 1]);
      wasm.matvec_f64(B, B + 32, B + 48, 2, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 2);
      checks.push(['matvec', Math.abs(r[0] - 3) < 1e-10 && Math.abs(r[1] - 7) < 1e-10]);
    }

    // Linalg — vecmat: [1,1] · [[1,2],[3,4]] = [4,6]
    if (wasm.vecmat_f64) {
      ensureMemory(wasm, 6 * 8);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, 1]);
      new Float64Array(wasm.memory.buffer, B + 16, 4).set([1, 2, 3, 4]);
      wasm.vecmat_f64(B, B + 16, B + 48, 2, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 2);
      checks.push(['vecmat', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 6) < 1e-10]);
    }

    // Linalg — vecdot: [1,2,3] · [4,5,6] = 32 (1 batch, len 3)
    if (wasm.vecdot_f64) {
      ensureMemory(wasm, 7 * 8);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([4, 5, 6]);
      wasm.vecdot_f64(B, B + 24, B + 48, 1, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 1);
      checks.push(['vecdot', Math.abs(r[0] - 32) < 1e-10]);
    }

    // Linalg — outer: [1,2] ⊗ [3,4] = [3,4,6,8]
    if (wasm.outer_f64) {
      ensureMemory(wasm, 8 * 8);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([3, 4]);
      wasm.outer_f64(B, B + 16, B + 32, 2, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 4);
      checks.push(['outer', Math.abs(r[0] - 3) < 1e-10 && Math.abs(r[1] - 4) < 1e-10 && Math.abs(r[2] - 6) < 1e-10 && Math.abs(r[3] - 8) < 1e-10]);
    }

    // Linalg — kron: I2 ⊗ [[1,2],[3,4]] = [[1,2,0,0],[3,4,0,0],[0,0,1,2],[0,0,3,4]]
    if (wasm.kron_f64) {
      ensureMemory(wasm, (4 + 4 + 16) * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 0, 0, 1]);
      new Float64Array(wasm.memory.buffer, B + 32, 4).set([1, 2, 3, 4]);
      wasm.kron_f64(B, B + 32, B + 64, 2, 2, 2, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 64, 16);
      const expected = [1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4];
      checks.push(['kron', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Linalg — cross: [1,0,0] × [0,1,0] = [0,0,1]
    if (wasm.cross_f64) {
      ensureMemory(wasm, 9 * 8);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 0, 0]);
      new Float64Array(wasm.memory.buffer, B + 24, 3).set([0, 1, 0]);
      wasm.cross_f64(B, B + 24, B + 48, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 48, 3);
      checks.push(['cross', Math.abs(r[0]) < 1e-10 && Math.abs(r[1]) < 1e-10 && Math.abs(r[2] - 1) < 1e-10]);
    }

    // Linalg — norm: norm([3,4]) = 5
    if (wasm.norm_f64) {
      ensureMemory(wasm, 2 * 8);
      new Float64Array(wasm.memory.buffer, B, 2).set([3, 4]);
      const result = wasm.norm_f64(B, 2);
      checks.push(['norm', Math.abs(result - 5) < 1e-10]);
    }

    // Unary extensions — tan(0) = 0, tan(π/4) ≈ 1.0
    if (wasm.tan_f64) {
      ensureMemory(wasm, 4 * 8);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, Math.PI / 4]);
      wasm.tan_f64(B, B + 16, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['tan', Math.abs(r[0]) < 1e-10 && Math.abs(r[1] - 1) < 1e-6]);
    }

    // Unary extensions — signbit(-1) = 1, signbit(1) = 0, signbit(-0) = 1
    if (wasm.signbit_f64) {
      ensureMemory(wasm, 6 * 8);
      new Float64Array(wasm.memory.buffer, B, 3).set([-1, 1, -0]);
      wasm.signbit_f64(B, B + 24, 3);
      const r = new Float64Array(wasm.memory.buffer, B + 24, 3);
      checks.push(['signbit', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1]) < 1e-10 && Math.abs(r[2] - 1) < 1e-10]);
    }

    // Reduction extensions — nanmax([1, NaN, 3, 2]) = 3
    if (wasm.nanmax_f64) {
      ensureMemory(wasm, 4 * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, NaN, 3, 2]);
      const result = wasm.nanmax_f64(B, 4);
      checks.push(['nanmax', Math.abs(result - 3) < 1e-10]);
    }

    // Reduction extensions — nanmin([1, NaN, 3, 2]) = 1
    if (wasm.nanmin_f64) {
      ensureMemory(wasm, 4 * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, NaN, 3, 2]);
      const result = wasm.nanmin_f64(B, 4);
      checks.push(['nanmin', Math.abs(result - 1) < 1e-10]);
    }

    // Array ops — roll([1,2,3,4,5], shift=2) = [4,5,1,2,3]
    if (wasm.roll_f64) {
      ensureMemory(wasm, 10 * 8);
      new Float64Array(wasm.memory.buffer, B, 5).set([1, 2, 3, 4, 5]);
      wasm.roll_f64(B, B + 40, 5, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 40, 5);
      const expected = [4, 5, 1, 2, 3];
      checks.push(['roll', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Array ops — flip([1,2,3,4]) = [4,3,2,1]
    if (wasm.flip_f64) {
      ensureMemory(wasm, 8 * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      wasm.flip_f64(B, B + 32, 4);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 4);
      const expected = [4, 3, 2, 1];
      checks.push(['flip', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Array ops — tile([1,2,3], reps=2) = [1,2,3,1,2,3]
    if (wasm.tile_f64) {
      ensureMemory(wasm, 9 * 8);
      new Float64Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      wasm.tile_f64(B, B + 24, 3, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 24, 6);
      const expected = [1, 2, 3, 1, 2, 3];
      checks.push(['tile', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Array ops — gradient([1,3,6,10]) = [2, 2.5, 3.5, 4]
    if (wasm.gradient_f64) {
      ensureMemory(wasm, 8 * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 3, 6, 10]);
      wasm.gradient_f64(B, B + 32, 4);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 4);
      const expected = [2, 2.5, 3.5, 4];
      checks.push(['gradient', expected.every((v, i) => Math.abs(r[i] - v) < 1e-10)]);
    }

    // Binary extensions — mod: [7,3] mod [2,2] = [1,1]
    if (wasm.mod_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([7, 3]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([2, 2]);
      wasm.mod_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['mod', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - 1) < 1e-10]);
    }

    // Binary extensions — floor_divide: [7,3] floor_divide [2,2] = [3,1]
    if (wasm.floor_divide_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([7, 3]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([2, 2]);
      wasm.floor_divide_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['floor_divide', Math.abs(r[0] - 3) < 1e-10 && Math.abs(r[1] - 1) < 1e-10]);
    }

    // Binary extensions — hypot: [3,0] hypot [4,5] = [5,5]
    if (wasm.hypot_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([3, 0]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([4, 5]);
      wasm.hypot_f64(B, B + 16, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['hypot', Math.abs(r[0] - 5) < 1e-10 && Math.abs(r[1] - 5) < 1e-10]);
    }

    // Unary extensions — sinh(0) = 0
    if (wasm.sinh_f64) {
      ensureMemory(wasm, 16);
      new Float64Array(wasm.memory.buffer, B, 1).set([0]);
      wasm.sinh_f64(B, B + 8, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 8, 1);
      checks.push(['sinh', Math.abs(r[0]) < 1e-10]);
    }

    // Unary extensions — cosh(0) = 1
    if (wasm.cosh_f64) {
      ensureMemory(wasm, 16);
      new Float64Array(wasm.memory.buffer, B, 1).set([0]);
      wasm.cosh_f64(B, B + 8, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 8, 1);
      checks.push(['cosh', Math.abs(r[0] - 1) < 1e-10]);
    }

    // Unary extensions — tanh(0) = 0
    if (wasm.tanh_f64) {
      ensureMemory(wasm, 16);
      new Float64Array(wasm.memory.buffer, B, 1).set([0]);
      wasm.tanh_f64(B, B + 8, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 8, 1);
      checks.push(['tanh', Math.abs(r[0]) < 1e-10]);
    }

    // Unary extensions — exp2(3) = 8
    if (wasm.exp2_f64) {
      ensureMemory(wasm, 16);
      new Float64Array(wasm.memory.buffer, B, 1).set([3]);
      wasm.exp2_f64(B, B + 8, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 8, 1);
      checks.push(['exp2', Math.abs(r[0] - 8) < 1e-10]);
    }

    // Stats — median([1,2,3,4,5]) = 3
    if (wasm.median_f64) {
      ensureMemory(wasm, 5 * 8);
      new Float64Array(wasm.memory.buffer, B, 5).set([1, 2, 3, 4, 5]);
      const result = wasm.median_f64(B, 5);
      checks.push(['median', Math.abs(result - 3) < 1e-10]);
    }

    // Stats — percentile([1,2,3,4,5], 50) = 3
    if (wasm.percentile_f64) {
      ensureMemory(wasm, 5 * 8);
      new Float64Array(wasm.memory.buffer, B, 5).set([1, 2, 3, 4, 5]);
      const result = wasm.percentile_f64(B, 5, 50.0);
      checks.push(['percentile', Math.abs(result - 3) < 1e-10]);
    }

    // Stats — quantile([1,2,3,4,5], 0.25) = 2
    if (wasm.quantile_f64) {
      ensureMemory(wasm, 5 * 8);
      new Float64Array(wasm.memory.buffer, B, 5).set([1, 2, 3, 4, 5]);
      const result = wasm.quantile_f64(B, 5, 0.25);
      checks.push(['quantile', Math.abs(result - 2) < 1e-10]);
    }

    // Search — nonzero([0,1,0,3,0,5]) → count=3, indices=[1,3,5]
    if (wasm.nonzero_f64) {
      const inputBytes = 6 * 8;
      const outputBytes = 6 * 4;
      ensureMemory(wasm, inputBytes + outputBytes);
      new Float64Array(wasm.memory.buffer, B, 6).set([0, 1, 0, 3, 0, 5]);
      const outPtr = B + inputBytes;
      const count = wasm.nonzero_f64(B, outPtr, 6);
      const indices = new Uint32Array(wasm.memory.buffer, outPtr, 3);
      checks.push(['nonzero', count === 3 && indices[0] === 1 && indices[1] === 3 && indices[2] === 5]);
    }

    // Linalg — matrix_power: [[1,1],[0,1]]^3 = [[1,3],[0,1]]
    if (wasm.matrix_power_f64) {
      const sz = 2 * 2;
      ensureMemory(wasm, (sz + sz + 2 * sz) * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 1, 0, 1]);
      const aOff = B, outOff = B + sz * 8, scrOff = B + 2 * sz * 8;
      wasm.matrix_power_f64(aOff, outOff, scrOff, 2, 3);
      const r = new Float64Array(wasm.memory.buffer, outOff, 4);
      checks.push(['matrix_power', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - 3) < 1e-10 && Math.abs(r[2]) < 1e-10 && Math.abs(r[3] - 1) < 1e-10]);
    }

    // Linalg — multi_dot3: I @ A @ I = A for [[1,2],[3,4]]
    if (wasm.multi_dot3_f64) {
      const sz = 4;
      ensureMemory(wasm, 5 * sz * 8);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 0, 0, 1]); // I
      new Float64Array(wasm.memory.buffer, B + 32, 4).set([1, 2, 3, 4]); // A
      new Float64Array(wasm.memory.buffer, B + 64, 4).set([1, 0, 0, 1]); // I
      const outOff = B + 96, tmpOff = B + 128;
      wasm.multi_dot3_f64(B, B + 32, B + 64, outOff, tmpOff, 2);
      const r = new Float64Array(wasm.memory.buffer, outOff, 4);
      checks.push(['multi_dot3', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - 2) < 1e-10 && Math.abs(r[2] - 3) < 1e-10 && Math.abs(r[3] - 4) < 1e-10]);
    }

    // Linalg — qr: 3×3 matrix, verify |Q*R - A| < ε and |Q^T*Q - I| < ε
    if (wasm.qr_f64) {
      const m3 = 3, n3 = 3, k3 = 3;
      const aBytes = m3 * n3 * 8, qBytes = m3 * k3 * 8, rBytes = k3 * n3 * 8;
      const tauBytes = k3 * 8, scrBytes = n3 * 8;
      ensureMemory(wasm, aBytes + qBytes + rBytes + tauBytes + scrBytes);
      const A = [2, -1, 0, -1, 2, -1, 0, -1, 2];
      new Float64Array(wasm.memory.buffer, B, 9).set(A);
      const aOff = B, qOff = B + aBytes, rOff = qOff + qBytes;
      const tauOff = rOff + rBytes, scrOff = tauOff + tauBytes;
      wasm.qr_f64(aOff, qOff, rOff, tauOff, scrOff, m3, n3);
      const Q = new Float64Array(wasm.memory.buffer, qOff, m3 * k3);
      const R = new Float64Array(wasm.memory.buffer, rOff, k3 * n3);
      // Check Q*R ≈ A
      let qrOk = true;
      for (let i = 0; i < m3; i++) for (let j = 0; j < n3; j++) {
        let s = 0; for (let k = 0; k < k3; k++) s += Q[i * k3 + k] * R[k * n3 + j];
        if (Math.abs(s - A[i * n3 + j]) > 1e-8) qrOk = false;
      }
      // Check Q^T*Q ≈ I
      for (let i = 0; i < k3; i++) for (let j = 0; j < k3; j++) {
        let s = 0; for (let k = 0; k < m3; k++) s += Q[k * k3 + i] * Q[k * k3 + j];
        const expected = i === j ? 1 : 0;
        if (Math.abs(s - expected) > 1e-8) qrOk = false;
      }
      checks.push(['qr', qrOk]);
    }

    // Linalg — lstsq: A=[[1,1],[1,2],[1,3]], b=[1,2,2] → x≈[2/3, 1/2]
    if (wasm.lstsq_f64) {
      const mL = 3, nL = 2, kL = 2;
      const aBytes = mL * nL * 8, bBytes = mL * 8, xBytes = nL * 8;
      const scrBytes = (mL * nL + mL * kL + kL * nL + kL + kL + nL + 16) * 8;
      ensureMemory(wasm, aBytes + bBytes + xBytes + scrBytes);
      new Float64Array(wasm.memory.buffer, B, 6).set([1, 1, 1, 2, 1, 3]);
      new Float64Array(wasm.memory.buffer, B + aBytes, 3).set([1, 2, 2]);
      const aOff = B, bOff = B + aBytes, xOff = bOff + bBytes, scrOff = xOff + xBytes;
      wasm.lstsq_f64(aOff, bOff, xOff, scrOff, mL, nL);
      const x = new Float64Array(wasm.memory.buffer, xOff, 2);
      checks.push(['lstsq', Math.abs(x[0] - 2/3) < 1e-6 && Math.abs(x[1] - 0.5) < 1e-6]);
    }

    // FFT — rfft2: 4×4 ones → DC=16, rest≈0
    if (wasm.rfft2_f64) {
      const fM = 4, fN = 4, halfN = fN / 2 + 1;
      const inpBytes = fM * fN * 8;
      const outBytes = fM * halfN * 2 * 8;
      const scrBytes = 8 * fN * fN * 8;
      ensureMemory(wasm, inpBytes + outBytes + scrBytes);
      const ones = new Float64Array(fM * fN); ones.fill(1);
      new Float64Array(wasm.memory.buffer, B, fM * fN).set(ones);
      const inpOff = B, outOff = B + inpBytes, scrOff = B + inpBytes + outBytes;
      wasm.rfft2_f64(inpOff, outOff, scrOff, fM, fN);
      const out = new Float64Array(wasm.memory.buffer, outOff, fM * halfN * 2);
      // DC component (index 0,0) should be 16, rest should be ~0
      let fftOk = Math.abs(out[0] - 16) < 1e-8;
      for (let i = 1; i < fM * halfN; i++) {
        if (Math.abs(out[2 * i]) > 1e-8 || Math.abs(out[2 * i + 1]) > 1e-8) fftOk = false;
      }
      checks.push(['rfft2', fftOk]);
    }

    // FFT — irfft2: roundtrip rfft2→irfft2 recovers original
    if (wasm.rfft2_f64 && wasm.irfft2_f64) {
      const fM = 4, fN = 4, halfN = fN / 2 + 1;
      const inpBytes = fM * fN * 8;
      const outBytes = fM * halfN * 2 * 8;
      const scrBytes = 8 * fN * fN * 8;
      ensureMemory(wasm, inpBytes + outBytes + scrBytes + inpBytes);
      const orig = new Float64Array(fM * fN); for (let i = 0; i < fM * fN; i++) orig[i] = i + 1;
      new Float64Array(wasm.memory.buffer, B, fM * fN).set(orig);
      const inpOff = B, fftOut = B + inpBytes, scrOff = B + inpBytes + outBytes;
      const recoverOff = scrOff + scrBytes;
      wasm.rfft2_f64(inpOff, fftOut, scrOff, fM, fN);
      wasm.irfft2_f64(fftOut, recoverOff, scrOff, fM, fN);
      const recovered = new Float64Array(wasm.memory.buffer, recoverOff, fM * fN);
      let roundtripOk = true;
      for (let i = 0; i < fM * fN; i++) {
        if (Math.abs(recovered[i] - orig[i]) > 1e-6) roundtripOk = false;
      }
      checks.push(['irfft2_roundtrip', roundtripOk]);
    }

    // ─── Integer binary ops ──────
    if (wasm.add_i32) {
      ensureMemory(wasm, 24);
      new Int32Array(wasm.memory.buffer, B, 3).set([1, 2, 3]);
      new Int32Array(wasm.memory.buffer, B + 12, 3).set([4, 5, 6]);
      wasm.add_i32(B, B + 12, B + 24, 3);
      const r = new Int32Array(wasm.memory.buffer, B + 24, 3);
      checks.push(['add_i32', r[0] === 5 && r[1] === 7 && r[2] === 9]);
    }
    if (wasm.mul_i32) {
      ensureMemory(wasm, 24);
      new Int32Array(wasm.memory.buffer, B, 3).set([2, 3, 4]);
      new Int32Array(wasm.memory.buffer, B + 12, 3).set([5, 6, 7]);
      wasm.mul_i32(B, B + 12, B + 24, 3);
      const r = new Int32Array(wasm.memory.buffer, B + 24, 3);
      checks.push(['mul_i32', r[0] === 10 && r[1] === 18 && r[2] === 28]);
    }
    if (wasm.maximum_i32) {
      ensureMemory(wasm, 24);
      new Int32Array(wasm.memory.buffer, B, 3).set([1, 5, 3]);
      new Int32Array(wasm.memory.buffer, B + 12, 3).set([4, 2, 6]);
      wasm.maximum_i32(B, B + 12, B + 24, 3);
      const r = new Int32Array(wasm.memory.buffer, B + 24, 3);
      checks.push(['max_i32', r[0] === 4 && r[1] === 5 && r[2] === 6]);
    }
    if (wasm.add_i16) {
      ensureMemory(wasm, 12);
      new Int16Array(wasm.memory.buffer, B, 3).set([10, 20, 30]);
      new Int16Array(wasm.memory.buffer, B + 6, 3).set([1, 2, 3]);
      wasm.add_i16(B, B + 6, B + 12, 3);
      const r = new Int16Array(wasm.memory.buffer, B + 12, 3);
      checks.push(['add_i16', r[0] === 11 && r[1] === 22 && r[2] === 33]);
    }
    if (wasm.add_i8) {
      ensureMemory(wasm, 12);
      new Int8Array(wasm.memory.buffer, B, 3).set([10, 20, 30]);
      new Int8Array(wasm.memory.buffer, B + 3, 3).set([1, 2, 3]);
      wasm.add_i8(B, B + 3, B + 6, 3);
      const r = new Int8Array(wasm.memory.buffer, B + 6, 3);
      checks.push(['add_i8', r[0] === 11 && r[1] === 22 && r[2] === 33]);
    }

    // ─── Integer reductions ──────
    if (wasm.sum_i32) {
      ensureMemory(wasm, 16);
      new Int32Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      checks.push(['sum_i32', wasm.sum_i32(B, 4) === 10]);
    }
    if (wasm.max_i32) {
      ensureMemory(wasm, 16);
      new Int32Array(wasm.memory.buffer, B, 4).set([3, 1, 4, 2]);
      checks.push(['max_i32r', wasm.max_i32(B, 4) === 4]);
    }
    if (wasm.min_i32) {
      ensureMemory(wasm, 16);
      new Int32Array(wasm.memory.buffer, B, 4).set([3, 1, 4, 2]);
      checks.push(['min_i32r', wasm.min_i32(B, 4) === 1]);
    }
    if (wasm.sum_i16) {
      ensureMemory(wasm, 8);
      new Int16Array(wasm.memory.buffer, B, 4).set([10, 20, 30, 40]);
      checks.push(['sum_i16', wasm.sum_i16(B, 4) === 100]);
    }
    if (wasm.sum_i8) {
      ensureMemory(wasm, 4);
      new Int8Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      checks.push(['sum_i8', wasm.sum_i8(B, 4) === 10]);
    }

    // ─── Integer sort ────────────
    if (wasm.sort_i32) {
      ensureMemory(wasm, 20);
      new Int32Array(wasm.memory.buffer, B, 5).set([3, 1, 4, 1, 5]);
      wasm.sort_i32(B, 5);
      const r = new Int32Array(wasm.memory.buffer, B, 5);
      checks.push(['sort_i32', r[0] === 1 && r[1] === 1 && r[2] === 3 && r[3] === 4 && r[4] === 5]);
    }
    if (wasm.sort_i16) {
      ensureMemory(wasm, 10);
      new Int16Array(wasm.memory.buffer, B, 5).set([3, 1, 4, 1, 5]);
      wasm.sort_i16(B, 5);
      const r = new Int16Array(wasm.memory.buffer, B, 5);
      checks.push(['sort_i16', r[0] === 1 && r[1] === 1 && r[2] === 3 && r[3] === 4 && r[4] === 5]);
    }
    if (wasm.sort_i8) {
      ensureMemory(wasm, 5);
      new Int8Array(wasm.memory.buffer, B, 5).set([3, 1, 4, 1, 5]);
      wasm.sort_i8(B, 5);
      const r = new Int8Array(wasm.memory.buffer, B, 5);
      checks.push(['sort_i8', r[0] === 1 && r[1] === 1 && r[2] === 3 && r[3] === 4 && r[4] === 5]);
    }

    // ─── Complex binary ops ──────
    if (wasm.add_c128) {
      // (1+2i) + (3+4i) = (4+6i)
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([3, 4]);
      wasm.add_c128(B, B + 16, B + 32, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['add_c128', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 6) < 1e-10]);
    }
    if (wasm.mul_c128) {
      // (1+2i) * (3+4i) = (1*3-2*4) + (1*4+2*3)i = (-5+10i)
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float64Array(wasm.memory.buffer, B + 16, 2).set([3, 4]);
      wasm.mul_c128(B, B + 16, B + 32, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['mul_c128', Math.abs(r[0] + 5) < 1e-10 && Math.abs(r[1] - 10) < 1e-10]);
    }
    if (wasm.add_c64) {
      ensureMemory(wasm, 24);
      new Float32Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float32Array(wasm.memory.buffer, B + 8, 2).set([3, 4]);
      wasm.add_c64(B, B + 8, B + 16, 1);
      const r = new Float32Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['add_c64', Math.abs(r[0] - 4) < 1e-5 && Math.abs(r[1] - 6) < 1e-5]);
    }
    if (wasm.mul_c64) {
      ensureMemory(wasm, 24);
      new Float32Array(wasm.memory.buffer, B, 2).set([1, 2]);
      new Float32Array(wasm.memory.buffer, B + 8, 2).set([3, 4]);
      wasm.mul_c64(B, B + 8, B + 16, 1);
      const r = new Float32Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['mul_c64', Math.abs(r[0] + 5) < 1e-5 && Math.abs(r[1] - 10) < 1e-5]);
    }

    // ─── Complex unary ops ───────
    if (wasm.abs_c128) {
      // |3+4i| = 5
      ensureMemory(wasm, 24);
      new Float64Array(wasm.memory.buffer, B, 2).set([3, 4]);
      wasm.abs_c128(B, B + 16, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 1);
      checks.push(['abs_c128', Math.abs(r[0] - 5) < 1e-10]);
    }
    if (wasm.exp_c128) {
      // exp(0+πi) = (-1+0i)  (Euler's formula)
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, B, 2).set([0, Math.PI]);
      wasm.exp_c128(B, B + 16, 1);
      const r = new Float64Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['exp_c128', Math.abs(r[0] + 1) < 1e-10 && Math.abs(r[1]) < 1e-10]);
    }
    if (wasm.abs_c64) {
      ensureMemory(wasm, 12);
      new Float32Array(wasm.memory.buffer, B, 2).set([3, 4]);
      wasm.abs_c64(B, B + 8, 1);
      const r = new Float32Array(wasm.memory.buffer, B + 8, 1);
      checks.push(['abs_c64', Math.abs(r[0] - 5) < 1e-4]);
    }
    if (wasm.exp_c64) {
      ensureMemory(wasm, 16);
      new Float32Array(wasm.memory.buffer, B, 2).set([0, Math.PI]);
      wasm.exp_c64(B, B + 8, 1);
      const r = new Float32Array(wasm.memory.buffer, B + 8, 2);
      checks.push(['exp_c64', Math.abs(r[0] + 1) < 1e-4 && Math.abs(r[1]) < 1e-4]);
    }

    // ─── Complex reductions ──────
    if (wasm.sum_c128) {
      // sum([(1+2i), (3+4i)]) = (4+6i)
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      wasm.sum_c128(B, B + 32, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 32, 2);
      checks.push(['sum_c128', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 6) < 1e-10]);
    }
    if (wasm.sum_c64) {
      ensureMemory(wasm, 24);
      new Float32Array(wasm.memory.buffer, B, 4).set([1, 2, 3, 4]);
      wasm.sum_c64(B, B + 16, 2);
      const r = new Float32Array(wasm.memory.buffer, B + 16, 2);
      checks.push(['sum_c64', Math.abs(r[0] - 4) < 1e-5 && Math.abs(r[1] - 6) < 1e-5]);
    }

    // ─── Complex matmul ──────────
    if (wasm.matmul_c128) {
      // 2×2 identity × [[1+0i,2+0i],[3+0i,4+0i]] = same
      ensureMemory(wasm, 3 * 4 * 16);
      // Identity: (1,0), (0,0), (0,0), (1,0)
      new Float64Array(wasm.memory.buffer, B, 8).set([1, 0, 0, 0, 0, 0, 1, 0]);
      // Matrix: (1,0), (2,0), (3,0), (4,0)
      new Float64Array(wasm.memory.buffer, B + 64, 8).set([1, 0, 2, 0, 3, 0, 4, 0]);
      wasm.matmul_c128(B, B + 64, B + 128, 2, 2, 2);
      const r = new Float64Array(wasm.memory.buffer, B + 128, 8);
      checks.push(['matmul_c128', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1]) < 1e-10
        && Math.abs(r[2] - 2) < 1e-10 && Math.abs(r[4] - 3) < 1e-10 && Math.abs(r[6] - 4) < 1e-10]);
    }
    if (wasm.matmul_c64) {
      ensureMemory(wasm, 3 * 4 * 8);
      new Float32Array(wasm.memory.buffer, B, 8).set([1, 0, 0, 0, 0, 0, 1, 0]);
      new Float32Array(wasm.memory.buffer, B + 32, 8).set([1, 0, 2, 0, 3, 0, 4, 0]);
      wasm.matmul_c64(B, B + 32, B + 64, 2, 2, 2);
      const r = new Float32Array(wasm.memory.buffer, B + 64, 8);
      checks.push(['matmul_c64', Math.abs(r[0] - 1) < 1e-4 && Math.abs(r[2] - 2) < 1e-4
        && Math.abs(r[4] - 3) < 1e-4 && Math.abs(r[6] - 4) < 1e-4]);
    }

    // ─── Complex FFT ─────────────
    if (wasm.fft_c128) {
      // fft([1,0,0,0]) = [1,1,1,1] (all real)
      const n = 4;
      const inpBytes = n * 2 * 8;
      const outBytes = n * 2 * 8;
      const scrBytes = 8 * n * n * 8;
      ensureMemory(wasm, inpBytes + outBytes + scrBytes);
      // Input: (1+0i), (0+0i), (0+0i), (0+0i)
      new Float64Array(wasm.memory.buffer, B, n * 2).set([1, 0, 0, 0, 0, 0, 0, 0]);
      const inpOff = B, outOff = B + inpBytes, scrOff = B + inpBytes + outBytes;
      wasm.fft_c128(inpOff, outOff, scrOff, n);
      const r = new Float64Array(wasm.memory.buffer, outOff, n * 2);
      let fftOk = true;
      for (let i = 0; i < n; i++) {
        if (Math.abs(r[2 * i] - 1) > 1e-8 || Math.abs(r[2 * i + 1]) > 1e-8) fftOk = false;
      }
      checks.push(['fft_c128', fftOk]);
    }
    if (wasm.fft_c128 && wasm.ifft_c128) {
      // Roundtrip: ifft(fft(x)) ≈ x
      const n = 4;
      const inpBytes = n * 2 * 8;
      const outBytes = n * 2 * 8;
      const scrBytes = 8 * n * n * 8;
      ensureMemory(wasm, inpBytes + 2 * outBytes + scrBytes);
      const orig = [1, 2, 3, 4, 5, 6, 7, 8]; // 4 complex numbers
      new Float64Array(wasm.memory.buffer, B, n * 2).set(orig);
      const inpOff = B, fftOut = B + inpBytes, ifftOut = B + inpBytes + outBytes, scrOff = B + inpBytes + 2 * outBytes;
      wasm.fft_c128(inpOff, fftOut, scrOff, n);
      wasm.ifft_c128(fftOut, ifftOut, scrOff, n);
      const r = new Float64Array(wasm.memory.buffer, ifftOut, n * 2);
      let ok = true;
      for (let i = 0; i < n * 2; i++) { if (Math.abs(r[i] - orig[i]) > 1e-6) ok = false; }
      checks.push(['ifft_c128_rt', ok]);
    }
    if (wasm.fft_c64 && wasm.ifft_c64) {
      const n = 4;
      const inpBytes = n * 2 * 4;
      const outBytes = n * 2 * 4;
      const scrBytes = 8 * n * n * 8;
      ensureMemory(wasm, inpBytes + 2 * outBytes + scrBytes);
      const orig = [1, 0, 0, 0, 0, 0, 0, 0];
      new Float32Array(wasm.memory.buffer, B, n * 2).set(orig);
      const inpOff = B, fftOut = B + inpBytes, ifftOut = B + inpBytes + outBytes, scrOff = B + inpBytes + 2 * outBytes;
      wasm.fft_c64(inpOff, fftOut, scrOff, n);
      wasm.ifft_c64(fftOut, ifftOut, scrOff, n);
      const r = new Float32Array(wasm.memory.buffer, ifftOut, n * 2);
      let ok = true;
      for (let i = 0; i < n * 2; i++) { if (Math.abs(r[i] - orig[i]) > 1e-3) ok = false; }
      checks.push(['fft_c64_rt', ok]);
    }

    const allPass = checks.every(([, ok]) => ok);
    const details = checks.map(([name, ok]) => `${name}:${ok ? 'PASS' : 'FAIL'}`).join(' ');
    console.log(`  ${wasm.name.padEnd(12)} ${allPass ? 'PASS' : 'FAIL'}  (${details})`);
    if (!allPass) allOk = false;
  }

  return allOk;
}

// ─── Report Types ────────────────────────────────────────────────────────────

interface BinarySizeEntry {
  kernel: string;
  lang: string;
  raw: number;
  b64gzip: number;
}

interface BenchmarkReport {
  timestamp: string;
  environment: { runtime: string; version: string; arch: string; platform: string };
  categories: Record<string, { metric: string; results: TimingResult[] }>;
  binarySizes: BinarySizeEntry[];
}

// ─── Output Formatting ──────────────────────────────────────────────────────

function printCategoryResults(category: string, metric: string, results: TimingResult[]) {
  const sizes = [...new Set(results.map((r) => r.size))].sort((a, b) => a - b);

  for (const size of sizes) {
    const sizeResults = results.filter((r) => r.size === size);
    const matrixCats = ['matmul', 'linalg', 'fft', 'linalg_complex', 'fft_complex'];
    const sizeLabel = matrixCats.includes(category) ? `${size}x${size}` : formatSize(size);
    console.log(`\n${'═'.repeat(90)}`);
    console.log(`  ${category.toUpperCase()} — Size: ${sizeLabel}`);
    console.log(`${'═'.repeat(90)}`);

    const header = [
      'Implementation'.padEnd(20),
      'Op'.padEnd(8),
      'Time (ms)'.padStart(12),
      metric.padStart(12),
      'vs JS'.padStart(10),
    ].join(' │ ');
    console.log(header);
    console.log('─'.repeat(90));

    const jsBaselines = new Map<string, number>();
    for (const r of sizeResults) {
      if (r.impl.startsWith('JS')) {
        jsBaselines.set(r.op, r.timeMs);
      }
    }

    for (const r of sizeResults) {
      const baseMs = jsBaselines.get(r.op) ?? r.timeMs;
      const speedup = baseMs / r.timeMs;
      const speedupStr = r.impl.startsWith('JS') ? '(base)' : `${speedup.toFixed(2)}x`;

      const row = [
        r.impl.padEnd(20),
        r.op.padEnd(8),
        r.timeMs.toFixed(3).padStart(12),
        r.metric.toFixed(3).padStart(12),
        speedupStr.padStart(10),
      ].join(' │ ');
      console.log(row);
    }
  }
}

function collectBinarySizes(): BinarySizeEntry[] {
  const entries: BinarySizeEntry[] = [];
  for (const kern of KERNEL_NAMES) {
    for (const lang of ['zig', 'rust'] as const) {
      const name = `${kern}_${lang}.wasm`;
      try {
        const raw = readFileSync(resolve(DIST_DIR, name));
        const b64 = Buffer.from(raw.toString('base64'));
        const gzipped = gzipSync(b64, { level: 9 });
        entries.push({ kernel: kern, lang, raw: raw.length, b64gzip: gzipped.length });
      } catch {
        // skip missing files
      }
    }
  }
  return entries;
}

function printBinarySizes(entries: BinarySizeEntry[]) {
  console.log(`\n${'═'.repeat(90)}`);
  console.log('  BINARY SIZES');
  console.log(`${'═'.repeat(90)}`);

  const header = [
    'File'.padEnd(30),
    'Raw (bytes)'.padStart(14),
    'b64+gzip'.padStart(14),
    'Compression'.padStart(14),
  ].join(' │ ');
  console.log(header);
  console.log('─'.repeat(90));

  const totals: Record<string, { raw: number; compressed: number }> = {};

  for (const e of entries) {
    const ratio = ((1 - e.b64gzip / e.raw) * 100).toFixed(1);
    const label = `${e.lang === 'zig' ? 'Zig' : 'Rust'} ${e.kernel}`;
    const filename = `${e.kernel}_${e.lang}.wasm`;
    const row = [
      `${label} (${filename})`.padEnd(30),
      e.raw.toLocaleString().padStart(14),
      e.b64gzip.toLocaleString().padStart(14),
      `${ratio}%`.padStart(14),
    ].join(' │ ');
    console.log(row);

    if (!totals[e.lang]) totals[e.lang] = { raw: 0, compressed: 0 };
    totals[e.lang].raw += e.raw;
    totals[e.lang].compressed += e.b64gzip;
  }

  console.log('─'.repeat(90));
  for (const [lang, t] of Object.entries(totals)) {
    const ratio = ((1 - t.compressed / t.raw) * 100).toFixed(1);
    const row = [
      `${lang === 'zig' ? 'Zig' : 'Rust'} TOTAL`.padEnd(30),
      t.raw.toLocaleString().padStart(14),
      t.compressed.toLocaleString().padStart(14),
      `${ratio}%`.padStart(14),
    ].join(' │ ');
    console.log(row);
  }
}

// ─── JSON + HTML Output ─────────────────────────────────────────────────────

const RESULTS_DIR = resolve(__dirname, '..', 'results');

function saveJSON(report: BenchmarkReport, basename: string) {
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
  const jsonPath = resolve(RESULTS_DIR, `${basename}.json`);
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  console.log(`\nJSON saved to ${jsonPath}`);
}

function generateHTML(report: BenchmarkReport, modeLabel: string, basename: string) {
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
  const htmlPath = resolve(RESULTS_DIR, `${basename}.html`);

  // Build chart data: speedup multipliers grouped by category+op+size
  // Dynamic: discover all unique WASM impl names and assign colors
  const IMPL_COLORS: Record<string, string> = {
    // Zig: shades of blue (darkest → lightest)
    'Zig (f64)': '#1f6feb', 'Zig (f32)': '#58a6ff',
    'Zig (i32)': '#79c0ff', 'Zig (i16)': '#a5d6ff', 'Zig (i8)': '#cae8ff',
    'Zig (c128)': '#388bfd', 'Zig (c64)': '#6cb6ff',
    // Rust: shades of orange (darkest → lightest)
    'Rust (f64)': '#d47616', 'Rust (f32)': '#f0883e',
    'Rust (i32)': '#f4a261', 'Rust (i16)': '#f7ba86', 'Rust (i8)': '#fad2ab',
    'Rust (c128)': '#e8843a', 'Rust (c64)': '#ffb77c',
  };

  interface ChartBar { impl: string; val: number; color: string; }
  interface ChartEntry {
    cat: string;
    entries: { label: string; bars: ChartBar[] }[];
  }
  const chartData: ChartEntry[] = [];

  for (const [catName, catData] of Object.entries(report.categories)) {
    const results = catData.results;
    const ops = [...new Set(results.map((r) => r.op))];
    const sizes = [...new Set(results.map((r) => r.size))].sort((a, b) => a - b);

    const matrixCats = ['matmul', 'linalg', 'fft', 'linalg_complex', 'fft_complex'];

    for (const op of ops) {
      const displayCat = ops.length > 1 ? `${capitalize(catName)} — ${op}` : capitalize(catName);
      const entries: ChartEntry['entries'] = [];

      for (const size of sizes) {
        const sizeLabel = matrixCats.includes(catName) ? `${size}×${size}` : formatSize(size);
        const sizeResults = results.filter((r) => r.size === size && r.op === op);

        // Find JS baseline time (use first JS result for this op — could be different dtypes)
        const jsResult = sizeResults.find((r) => r.impl.startsWith('JS'));
        const baseMs = jsResult?.timeMs ?? 1;

        // Build bars for all non-JS impls
        const bars: ChartBar[] = [];
        for (const r of sizeResults) {
          if (r.impl.startsWith('JS')) continue;
          // Find the JS baseline that matches this impl's dtype for a fair comparison
          const dtypeMatch = r.impl.match(/\(([^)]+)\)/)?.[1] ?? '';
          const matchingJs = sizeResults.find((j) => j.impl === `JS (${dtypeMatch})`);
          const base = matchingJs?.timeMs ?? baseMs;
          bars.push({
            impl: r.impl,
            val: base / r.timeMs,
            color: IMPL_COLORS[r.impl] ?? '#8b949e',
          });
        }

        entries.push({ label: sizeLabel, bars });
      }

      chartData.push({ cat: displayCat, entries });
    }
  }

  // Build binary size data
  const sizesByKernel = new Map<string, { zig: { raw: number; gz: number }; rust: { raw: number; gz: number } }>();
  for (const e of report.binarySizes) {
    if (!sizesByKernel.has(e.kernel)) sizesByKernel.set(e.kernel, { zig: { raw: 0, gz: 0 }, rust: { raw: 0, gz: 0 } });
    const entry = sizesByKernel.get(e.kernel)!;
    entry[e.lang as 'zig' | 'rust'] = { raw: e.raw, gz: e.b64gzip };
  }
  const sizesArray = [...sizesByKernel.entries()].map(([kernel, data]) => ({
    kernel,
    zig: data.zig,
    rust: data.rust,
  }));

  const zigTotal = { raw: 0, gz: 0 };
  const rustTotal = { raw: 0, gz: 0 };
  for (const s of sizesArray) {
    zigTotal.raw += s.zig.raw; zigTotal.gz += s.zig.gz;
    rustTotal.raw += s.rust.raw; rustTotal.gz += s.rust.gz;
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WASM Benchmark Results — ${modeLabel}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 32px; }
  h1 { font-size: 24px; margin-bottom: 4px; color: #f0f6fc; }
  .subtitle { color: #8b949e; margin-bottom: 32px; font-size: 14px; }
  .meta { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; margin-bottom: 24px; font-size: 12px; color: #8b949e; font-family: monospace; }
  .legend { display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .legend-swatch { width: 16px; height: 16px; border-radius: 3px; }
  .category { margin-bottom: 40px; }
  .category h2 { font-size: 18px; color: #f0f6fc; margin-bottom: 16px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  .chart-row { display: flex; align-items: center; margin-bottom: 6px; }
  .chart-label { width: 160px; text-align: right; padding-right: 12px; font-size: 12px; color: #8b949e; flex-shrink: 0; white-space: nowrap; }
  .chart-bars { flex: 1; display: flex; flex-direction: column; gap: 2px; }
  .bar-group { display: flex; gap: 2px; align-items: center; height: 18px; }
  .bar { height: 100%; border-radius: 2px; min-width: 1px; position: relative; display: flex; align-items: center; }
  .bar-val { font-size: 10px; color: #f0f6fc; padding-left: 6px; white-space: nowrap; font-weight: 600; }
  .bar-val.outside { position: absolute; left: calc(100% + 4px); color: #8b949e; }
  .baseline { position: absolute; top: 0; bottom: 0; border-left: 2px dashed #484f58; z-index: 1; }
  .baseline-label { position: absolute; top: -16px; font-size: 10px; color: #484f58; transform: translateX(-50%); }
  .chart-area { position: relative; flex: 1; }
  .group-spacer { height: 10px; }
  .slower-note { font-size: 10px; color: #f85149; font-style: italic; margin-left: 4px; }
  /* Bar colors set dynamically via inline styles */
  .summary { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 32px; }
  .summary h2 { font-size: 18px; color: #f0f6fc; margin-bottom: 16px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  .summary-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .summary-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 16px; }
  .summary-card h3 { font-size: 13px; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .summary-stat { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .summary-stat:last-child { margin-bottom: 0; }
  .summary-label { font-size: 12px; color: #8b949e; }
  .summary-value { font-size: 16px; font-weight: 700; }
  .summary-value.zig { color: #58a6ff; }
  .summary-value.rust { color: #f0883e; }
  .summary-value.neutral { color: #c9d1d9; }
  .summary-detail { font-size: 10px; color: #484f58; margin-top: 2px; }
  .h2h-row { display: flex; align-items: center; margin-bottom: 10px; }
  .h2h-label { font-size: 12px; color: #8b949e; width: 80px; flex-shrink: 0; }
  .h2h-bar-track { flex: 1; height: 24px; display: flex; border-radius: 4px; overflow: hidden; position: relative; }
  .h2h-segment { display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: #f0f6fc; }
  .h2h-center { position: absolute; left: 50%; top: 0; bottom: 0; border-left: 2px dashed #484f58; z-index: 1; }
  .sizes { margin-top: 48px; }
  .sizes h2 { font-size: 18px; color: #f0f6fc; margin-bottom: 16px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  .sizes-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
  .size-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .size-card h3 { font-size: 14px; color: #f0f6fc; margin-bottom: 12px; }
  .size-bar-row { display: flex; align-items: center; margin-bottom: 8px; }
  .size-bar-label { width: 80px; font-size: 11px; color: #8b949e; flex-shrink: 0; }
  .size-bar-track { flex: 1; height: 22px; position: relative; }
  .size-bar { height: 100%; border-radius: 3px; display: flex; align-items: center; position: relative; }
  .size-bar .raw-segment { height: 100%; border-radius: 3px 0 0 3px; opacity: 0.35; }
  .size-bar .gz-segment { height: 100%; border-radius: 0 3px 3px 0; }
  .size-bar-text { font-size: 10px; color: #f0f6fc; font-weight: 600; padding-left: 6px; white-space: nowrap; }
  .size-bar-text.outside { position: absolute; left: calc(100% + 6px); color: #8b949e; }
  .size-total { margin-top: 16px; padding-top: 12px; border-top: 1px solid #30363d; }
  .size-total-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .size-total-val { font-weight: 600; color: #f0f6fc; }
  .size-legend { display: flex; gap: 16px; margin-bottom: 16px; font-size: 12px; color: #8b949e; }
  .size-legend-item { display: flex; align-items: center; gap: 5px; }
  .size-legend-swatch { width: 12px; height: 12px; border-radius: 2px; }
  .size-legend-swatch.raw { opacity: 0.35; }
</style>
</head>
<body>

<h1>WASM vs JS — Speedup Multipliers (${modeLabel})</h1>
<p class="subtitle">Horizontal bars = how many times faster than pure JS (numpy-ts float64). Dashed line = 1× (JS baseline).</p>
<div class="meta">Generated: ${report.timestamp} · ${report.environment.runtime} ${report.environment.version} · ${report.environment.platform}/${report.environment.arch}</div>

<div class="legend" id="legend"></div>

<div id="charts"></div>

<script>
const DATA = ${JSON.stringify(chartData, null, 2)};
const IMPL_COLORS = ${JSON.stringify(IMPL_COLORS)};
const SIZES = ${JSON.stringify(sizesArray, null, 2)};
const zigTotal = ${JSON.stringify(zigTotal)};
const rustTotal = ${JSON.stringify(rustTotal)};

const container = document.getElementById('charts');

// ── Build dynamic legend ──
{
  const legendEl = document.getElementById('legend');
  // Collect all unique impl names that appear in data
  const seen = new Set();
  for (const cat of DATA) for (const e of cat.entries) for (const b of e.bars) seen.add(b.impl);
  for (const impl of seen) {
    const color = IMPL_COLORS[impl] || '#8b949e';
    legendEl.insertAdjacentHTML('beforeend',
      '<div class="legend-item"><div class="legend-swatch" style="background:' + color + '"></div> ' + impl + '</div>');
  }
}

// ── Summary Section ──
{
  // Collect all speedup values (f64 only for apples-to-apples comparison)
  const zigVals = [];
  const rustVals = [];
  const h2hVals = []; // zig/rust ratio (>1 = zig faster)
  const details = []; // {cat, label, zig64, rust64}

  for (const cat of DATA) {
    for (const e of cat.entries) {
      const zigF64 = e.bars.find(b => b.impl === 'Zig (f64)');
      const rustF64 = e.bars.find(b => b.impl === 'Rust (f64)');
      if (zigF64 && rustF64 && zigF64.val > 0 && rustF64.val > 0) {
        zigVals.push({ val: zigF64.val, cat: cat.cat, label: e.label });
        rustVals.push({ val: rustF64.val, cat: cat.cat, label: e.label });
        h2hVals.push({ val: zigF64.val / rustF64.val, cat: cat.cat, label: e.label });
        details.push({ cat: cat.cat, label: e.label, zig64: zigF64.val, rust64: rustF64.val });
      }
    }
  }

  const avg = arr => arr.reduce((s, x) => s + x.val, 0) / arr.length;
  const geoMean = arr => Math.pow(arr.reduce((p, x) => p * x.val, 1), 1 / arr.length);
  const best = arr => arr.reduce((a, b) => a.val > b.val ? a : b);
  const worst = arr => arr.reduce((a, b) => a.val < b.val ? a : b);

  const zigGeo = geoMean(zigVals);
  const rustGeo = geoMean(rustVals);
  const zigBest = best(zigVals);
  const zigWorst = worst(zigVals);
  const rustBest = best(rustVals);
  const rustWorst = worst(rustVals);

  const h2hGeo = geoMean(h2hVals);
  const h2hBest = best(h2hVals); // most zig-favoring
  const h2hWorst = worst(h2hVals); // most rust-favoring

  // Count wins
  let zigWins = 0, rustWins = 0, ties = 0;
  for (const d of details) {
    if (d.zig64 > d.rust64 * 1.01) zigWins++;
    else if (d.rust64 > d.zig64 * 1.01) rustWins++;
    else ties++;
  }
  const total = zigWins + rustWins + ties;

  const fmt = v => v.toFixed(2) + '\\u00d7';
  const fmtDetail = d => d.cat + ' ' + d.label;

  let summaryHtml = '<div class="summary"><h2>Summary (f64 vs JS baseline)</h2>';
  summaryHtml += '<div class="summary-grid">';

  // Zig card
  summaryHtml += '<div class="summary-card"><h3 style="color:#58a6ff">Zig vs JS</h3>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Geo mean speedup</span><span class="summary-value zig">' + fmt(zigGeo) + '</span></div>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Best case</span><span class="summary-value zig">' + fmt(zigBest.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(zigBest) + '</div>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Worst case</span><span class="summary-value zig">' + fmt(zigWorst.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(zigWorst) + '</div>';
  summaryHtml += '</div>';

  // Rust card
  summaryHtml += '<div class="summary-card"><h3 style="color:#f0883e">Rust vs JS</h3>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Geo mean speedup</span><span class="summary-value rust">' + fmt(rustGeo) + '</span></div>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Best case</span><span class="summary-value rust">' + fmt(rustBest.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(rustBest) + '</div>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Worst case</span><span class="summary-value rust">' + fmt(rustWorst.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(rustWorst) + '</div>';
  summaryHtml += '</div>';

  // Head-to-head card
  const h2hWinner = h2hGeo > 1.01 ? 'zig' : h2hGeo < 0.99 ? 'rust' : 'neutral';
  const h2hLabel = h2hGeo >= 1 ? 'Zig ' + fmt(h2hGeo) + ' faster' : 'Rust ' + fmt(1/h2hGeo) + ' faster';
  summaryHtml += '<div class="summary-card"><h3>Zig vs Rust (head-to-head)</h3>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Geo mean</span><span class="summary-value ' + h2hWinner + '">' + h2hLabel + '</span></div>';

  // Win/loss bar
  const zigPct = (zigWins / total * 100).toFixed(0);
  const rustPct = (rustWins / total * 100).toFixed(0);
  const tiePct = (ties / total * 100).toFixed(0);
  summaryHtml += '<div class="h2h-row" style="margin-top:12px"><div class="h2h-bar-track">';
  if (zigWins > 0) summaryHtml += '<div class="h2h-segment" style="width:' + zigPct + '%;background:#58a6ff">Zig ' + zigWins + '</div>';
  if (ties > 0) summaryHtml += '<div class="h2h-segment" style="width:' + tiePct + '%;background:#30363d">Tie ' + ties + '</div>';
  if (rustWins > 0) summaryHtml += '<div class="h2h-segment" style="width:' + rustPct + '%;background:#f0883e">Rust ' + rustWins + '</div>';
  summaryHtml += '</div></div>';

  // Best/worst h2h
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Zig best margin</span><span class="summary-value zig">' + fmt(h2hBest.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(h2hBest) + '</div>';
  summaryHtml += '<div class="summary-stat"><span class="summary-label">Rust best margin</span><span class="summary-value rust">' + fmt(1/h2hWorst.val) + '</span></div>';
  summaryHtml += '<div class="summary-detail">' + fmtDetail(h2hWorst) + '</div>';
  summaryHtml += '</div>';

  summaryHtml += '</div></div>';
  container.insertAdjacentHTML('beforeend', summaryHtml);
}

for (const cat of DATA) {
  let maxVal = 1;
  for (const e of cat.entries) {
    for (const b of e.bars) maxVal = Math.max(maxVal, b.val);
  }
  const scale = maxVal * 1.25;
  const section = document.createElement('div');
  section.className = 'category';
  let html = \`<h2>\${cat.cat}</h2>\`;
  for (const e of cat.entries) {
    html += \`<div class="chart-row"><div class="chart-label">\${e.label}</div><div class="chart-area">\`;
    const baselinePct = (1 / scale) * 100;
    html += \`<div class="baseline" style="left:\${baselinePct}%"><span class="baseline-label">1×</span></div>\`;
    for (const b of e.bars) {
      if (b.val === 0) continue;
      const pct = Math.max((b.val / scale) * 100, 0.5);
      const valStr = b.val.toFixed(2) + '×';
      const isSmall = pct < 8;
      const slowerNote = b.val < 1 ? ' <span class="slower-note">(slower)</span>' : '';
      html += \`<div class="bar-group"><div class="bar" title="\${b.impl}: \${valStr}" style="background:\${b.color};width:\${pct}%"><span class="bar-val\${isSmall ? ' outside' : ''}">\${valStr}\${slowerNote}</span></div></div>\`;
    }
    html += \`</div></div><div class="group-spacer"></div>\`;
  }
  section.innerHTML = html;
  container.appendChild(section);
}

// Binary sizes
const maxRaw = Math.max(...SIZES.flatMap(s => [s.zig.raw, s.rust.raw]));
const barScale = maxRaw * 1.3;
let sizesHtml = \`
<div class="sizes">
  <h2>Binary Sizes (per kernel .wasm)</h2>
  <div class="size-legend">
    <div class="size-legend-item"><div class="size-legend-swatch raw" style="background:#58a6ff"></div> Raw .wasm</div>
    <div class="size-legend-item"><div class="size-legend-swatch" style="background:#58a6ff"></div> b64+gzip (bundle cost)</div>
  </div>
  <div class="sizes-grid">\`;

for (const s of SIZES) {
  const items = [
    { label: "Zig", data: s.zig, color: "#58a6ff" },
    { label: "Rust", data: s.rust, color: "#f0883e" },
  ];
  sizesHtml += \`<div class="size-card"><h3>\${s.kernel}</h3>\`;
  for (const item of items) {
    const rawPct = (item.data.raw / barScale) * 100;
    const gzPct = (item.data.gz / barScale) * 100;
    const isSmall = gzPct < 12;
    const text = \`\${(item.data.gz / 1024).toFixed(1)} KB gz — \${(item.data.raw / 1024).toFixed(1)} KB raw\`;
    sizesHtml += \`
      <div class="size-bar-row">
        <div class="size-bar-label">\${item.label}</div>
        <div class="size-bar-track">
          <div class="size-bar" style="width:\${rawPct}%">
            <div class="raw-segment" style="background:\${item.color};width:100%;position:absolute"></div>
            <div class="gz-segment" style="background:\${item.color};width:\${(item.data.gz/item.data.raw)*100}%;position:relative;z-index:1"></div>
            <span class="size-bar-text\${isSmall ? ' outside' : ''}" style="z-index:2;position:relative">\${text}</span>
          </div>
        </div>
      </div>\`;
  }
  sizesHtml += \`</div>\`;
}

sizesHtml += \`</div>
  <div class="size-total">
    <div class="size-total-row">
      <span>Zig total (all \${SIZES.length} kernels)</span>
      <span class="size-total-val" style="color:#58a6ff">\${(zigTotal.raw/1024).toFixed(1)} KB raw — \${(zigTotal.gz/1024).toFixed(1)} KB b64+gzip</span>
    </div>
    <div class="size-total-row">
      <span>Rust total (all \${SIZES.length} kernels)</span>
      <span class="size-total-val" style="color:#f0883e">\${(rustTotal.raw/1024).toFixed(1)} KB raw — \${(rustTotal.gz/1024).toFixed(1)} KB b64+gzip</span>
    </div>
  </div>
</div>\`;

container.insertAdjacentHTML('beforeend', sizesHtml);
</script>
</body>
</html>`;

  writeFileSync(htmlPath, html);
  console.log(`HTML saved to ${htmlPath}`);
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ─── Per-Category WASM Loading ───────────────────────────────────────────────

const KERNEL_NAMES = ['matmul', 'reduction', 'unary', 'binary', 'sort', 'convolve', 'linalg', 'arrayops', 'fft'] as const;
type KernelName = (typeof KERNEL_NAMES)[number];

// Maps category names to WASM kernel filenames (new int/complex categories reuse existing .wasm files)
const CATEGORY_TO_KERNEL: Record<string, KernelName> = {
  matmul: 'matmul', reduction: 'reduction', unary: 'unary', binary: 'binary',
  sort: 'sort', convolve: 'convolve', linalg: 'linalg', arrayops: 'arrayops', fft: 'fft',
  binary_complex: 'binary', unary_complex: 'unary', reduction_complex: 'reduction',
  linalg_complex: 'linalg', fft_complex: 'fft',
};

async function loadCategoryWasms(): Promise<Record<string, WasmKernels[]>> {
  const result: Record<string, WasmKernels[]> = {};
  // Load each unique kernel file once
  const loaded: Record<string, WasmKernels[]> = {};
  for (const kern of KERNEL_NAMES) {
    const zig = await loadWasm(`${kern}_zig.wasm`, 'Zig');
    const rust = await loadWasm(`${kern}_rust.wasm`, 'Rust');
    loaded[kern] = [zig, rust].filter((w): w is WasmKernels => w !== null);
  }
  // Map all categories (including aliases) to loaded modules
  for (const [cat, kern] of Object.entries(CATEGORY_TO_KERNEL)) {
    result[cat] = loaded[kern];
  }
  return result;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log('\nWASM Multi-Category Benchmark');
  console.log('=============================\n');

  // Load WASM modules (one per kernel per language)
  console.log('Loading WASM modules...');
  const wasmByCategory = await loadCategoryWasms();

  // Load numpy-ts baselines
  console.log('Loading numpy-ts baselines...');
  const baselines = await importBaselines();

  // Correctness checks per kernel (each .wasm file once)
  console.log('\nCorrectness checks...');
  let allOk = true;
  for (const kern of KERNEL_NAMES) {
    const wasms = wasmByCategory[kern];
    if (wasms && wasms.length > 0) {
      const ok = verifyAll(wasms);
      if (!ok) allOk = false;
    }
  }
  if (!allOk) {
    console.error('\nCorrectness check FAILED — aborting benchmarks.');
    process.exit(1);
  }
  console.log();

  // Collect all results for two reports: kernel-only and end-to-end
  const rt = detectRuntime();
  const env = { runtime: rt.name, version: rt.version, arch: rt.arch, platform: rt.platform };
  console.log(`Runtime: ${rt.name} ${rt.version} (${rt.arch}/${rt.platform})`);
  const kernelReport: BenchmarkReport = { timestamp: new Date().toISOString(), environment: env, categories: {}, binarySizes: [] };
  const e2eReport: BenchmarkReport = { timestamp: new Date().toISOString(), environment: env, categories: {}, binarySizes: [] };

  // Run benchmarks per category
  for (const cat of CATEGORIES) {
    const wasmModules = wasmByCategory[cat.name] ?? [];
    console.log(`\nBenchmarking: ${cat.name}...`);
    const kernelResults: TimingResult[] = [];
    const e2eResults: TimingResult[] = [];

    // Helpers to push dual results
    const pushDual = (d: DualTimingResult) => { kernelResults.push(d.kernel); e2eResults.push(d.e2e); };
    const pushWasm = (d: DualTimingResult | null) => { if (d) { kernelResults.push(d.kernel); e2eResults.push(d.e2e); } };
    const pushJsInline = (r: TimingResult) => { kernelResults.push(r); e2eResults.push(r); };

    for (const { size, iterations, warmup } of cat.sizes) {
      const matrixCats = ['matmul', 'linalg', 'fft', 'linalg_complex', 'fft_complex'];
      const sizeLabel = matrixCats.includes(cat.name) ? `${size}x${size}` : formatSize(size);
      console.log(`  ${sizeLabel} (${iterations} iters, ${warmup} warmup)...`);

      try { switch (cat.name) {
        case 'matmul': {
          pushDual(benchJsMatmul(baselines, size, 'f64', iterations, warmup));
          pushDual(benchJsMatmul(baselines, size, 'f32', iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmMatmul(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmMatmul(wasm, size, 'f32', iterations, warmup));
          }
          break;
        }
        case 'reduction': {
          for (const op of ['sum', 'max', 'min', 'prod', 'mean'] as const) {
            pushDual(benchJsReduction(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              pushWasm(benchWasmReduction(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmReduction(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          pushDual(benchJsDiff(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmDiff(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmDiff(wasm, size, 'f32', iterations, warmup));
          }
          for (const op of ['nanmax', 'nanmin'] as const) {
            {
              const data = new Float64Array(size);
              for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
              for (let i = 0; i < size; i += 100) data[i] = NaN;
              const jsFn = op === 'nanmax'
                ? () => { let m = -Infinity; for (let i = 0; i < size; i++) if (!isNaN(data[i]) && data[i] > m) m = data[i]; return m; }
                : () => { let m = Infinity; for (let i = 0; i < size; i++) if (!isNaN(data[i]) && data[i] < m) m = data[i]; return m; };
              const times: number[] = [];
              for (let iter = 0; iter < warmup + iterations; iter++) {
                const t0 = performance.now();
                jsFn();
                const t1 = performance.now();
                if (iter >= warmup) times.push(t1 - t0);
              }
              const ms = median(times);
              pushJsInline({ impl: 'JS (f64)', op, size, timeMs: ms, metric: computeGBs(size, 8, ms), metricUnit: 'GB/s' });
            }
            for (const wasm of wasmModules) {
              pushWasm(benchWasmReduction(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmReduction(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          for (const op of ['sum', 'max', 'min']) {
            for (const dt of ['i32', 'i16', 'i8'] as IntDtype[]) {
              pushDual(benchJsReductionInt(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmReductionInt(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
        case 'unary': {
          const unaryOps: { wasm: string; js: keyof Baselines | null }[] = [
            { wasm: 'sqrt', js: 'sqrt' },
            { wasm: 'exp', js: 'exp' },
            { wasm: 'log', js: 'log' },
            { wasm: 'sin', js: 'sin' },
            { wasm: 'cos', js: 'cos' },
            { wasm: 'abs', js: 'absolute' },
            { wasm: 'neg', js: 'negative' },
            { wasm: 'ceil', js: 'ceil' },
            { wasm: 'floor', js: 'floor' },
            { wasm: 'tan', js: null },
            { wasm: 'signbit', js: null },
            { wasm: 'sinh', js: null },
            { wasm: 'cosh', js: null },
            { wasm: 'tanh', js: null },
            { wasm: 'exp2', js: null },
          ];
          for (const { wasm: wasmOp, js: jsOp } of unaryOps) {
            if (jsOp) {
              pushDual(benchJsUnary(baselines, jsOp as 'sqrt', size, iterations, warmup, wasmOp));
            } else {
              // JS loop baseline for ops without numpy-ts import
              const data = new Float64Array(size);
              for (let i = 0; i < size; i++) data[i] = Math.random() * 4 - 2;
              const out = new Float64Array(size);
              let jsFn: () => void;
              if (wasmOp === 'tan') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.tan(data[i]); };
              } else if (wasmOp === 'sinh') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.sinh(data[i]); };
              } else if (wasmOp === 'cosh') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.cosh(data[i]); };
              } else if (wasmOp === 'tanh') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.tanh(data[i]); };
              } else if (wasmOp === 'exp2') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = 2 ** data[i]; };
              } else {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = (1 / data[i] < 0) ? 1 : 0; }; // signbit
              }
              const times: number[] = [];
              for (let iter = 0; iter < warmup + iterations; iter++) {
                const t0 = performance.now();
                jsFn();
                const t1 = performance.now();
                if (iter >= warmup) times.push(t1 - t0);
              }
              const ms = median(times);
              pushJsInline({ impl: 'JS (f64)', op: wasmOp, size, timeMs: ms, metric: computeGBs(size, 8, ms), metricUnit: 'GB/s' });
            }
            for (const wasm of wasmModules) {
              pushWasm(benchWasmUnary(wasm, wasmOp, size, 'f64', iterations, warmup));
              pushWasm(benchWasmUnary(wasm, wasmOp, size, 'f32', iterations, warmup));
            }
          }
          break;
        }
        case 'binary': {
          const binaryOps: { wasm: string; js: keyof Baselines }[] = [
            { wasm: 'add', js: 'add' },
            { wasm: 'sub', js: 'subtract' },
            { wasm: 'mul', js: 'multiply' },
            { wasm: 'div', js: 'divide' },
            { wasm: 'copysign', js: 'copysign' },
            { wasm: 'power', js: 'power' },
            { wasm: 'maximum', js: 'maximum' },
            { wasm: 'minimum', js: 'minimum' },
            { wasm: 'fmax', js: 'fmax' },
            { wasm: 'fmin', js: 'fmin' },
            { wasm: 'logaddexp', js: 'logaddexp' },
            { wasm: 'logical_and', js: 'logical_and' },
            { wasm: 'logical_xor', js: 'logical_xor' },
            { wasm: 'mod', js: null as unknown as keyof Baselines },
            { wasm: 'floor_divide', js: null as unknown as keyof Baselines },
            { wasm: 'hypot', js: null as unknown as keyof Baselines },
          ];
          for (const { wasm: wasmOp, js: jsOp } of binaryOps) {
            if (!jsOp) {
              // JS loop baseline for binary ops without numpy-ts import
              const aData = new Float64Array(size);
              const bData = new Float64Array(size);
              for (let i = 0; i < size; i++) {
                if (wasmOp === 'mod' || wasmOp === 'floor_divide') {
                  aData[i] = Math.random() * 2 - 1;
                  bData[i] = Math.random() * 2 + 0.5;
                } else {
                  aData[i] = Math.random() * 10;
                  bData[i] = Math.random() * 10;
                }
              }
              const out = new Float64Array(size);
              let jsFn: () => void;
              if (wasmOp === 'mod') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = aData[i] - Math.floor(aData[i] / bData[i]) * bData[i]; };
              } else if (wasmOp === 'floor_divide') {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.floor(aData[i] / bData[i]); };
              } else {
                jsFn = () => { for (let i = 0; i < size; i++) out[i] = Math.hypot(aData[i], bData[i]); };
              }
              const times: number[] = [];
              for (let iter = 0; iter < warmup + iterations; iter++) {
                const t0 = performance.now();
                jsFn();
                const t1 = performance.now();
                if (iter >= warmup) times.push(t1 - t0);
              }
              const ms = median(times);
              pushJsInline({ impl: 'JS (f64)', op: wasmOp, size, timeMs: ms, metric: computeGBs(size, 8, ms), metricUnit: 'GB/s' });
            } else {
              pushDual(benchJsBinary(baselines, jsOp as 'add', size, iterations, warmup, wasmOp));
            }
            for (const wasm of wasmModules) {
              pushWasm(benchWasmBinary(wasm, wasmOp, size, 'f64', iterations, warmup));
              pushWasm(benchWasmBinary(wasm, wasmOp, size, 'f32', iterations, warmup));
            }
          }
          // Integer binary ops (same category)
          for (const op of ['add', 'sub', 'mul', 'maximum', 'minimum']) {
            for (const dt of ['i32', 'i16', 'i8'] as IntDtype[]) {
              pushDual(benchJsBinaryInt(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmBinaryInt(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
        case 'sort': {
          // sort
          pushDual(benchJsSort(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmSort(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmSort(wasm, size, 'f32', iterations, warmup));
          }
          // argsort
          pushDual(benchJsArgsort(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmArgsort(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmArgsort(wasm, size, 'f32', iterations, warmup));
          }
          // partition
          pushDual(benchJsPartition(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmPartition(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmPartition(wasm, size, 'f32', iterations, warmup));
          }
          // argpartition
          pushDual(benchJsArgpartition(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            pushWasm(benchWasmArgpartition(wasm, size, 'f64', iterations, warmup));
            pushWasm(benchWasmArgpartition(wasm, size, 'f32', iterations, warmup));
          }
          // stats: median, percentile, quantile
          const statsOps = ['median', 'percentile', 'quantile'];
          for (const op of statsOps) {
            pushDual(benchJsStats(op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              pushWasm(benchWasmStats(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmStats(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          // Integer sort (same category)
          for (const op of ['sort', 'argsort']) {
            for (const dt of ['i32', 'i16', 'i8'] as IntDtype[]) {
              pushDual(benchJsSortInt(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmSortInt(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
        case 'convolve': {
          for (const op of ['correlate', 'convolve'] as const) {
            pushDual(benchJsConvolve(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              pushWasm(benchWasmConvolve(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmConvolve(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          break;
        }
        case 'linalg': {
          const linalgOps = ['matvec', 'vecmat', 'vecdot', 'outer', 'kron', 'cross', 'norm', 'matrix_power', 'multi_dot3', 'qr', 'lstsq'];
          for (const op of linalgOps) {
            const jsR = benchJsLinalg(op, size, iterations, warmup);
            if (jsR) pushDual(jsR);
            for (const wasm of wasmModules) {
              pushWasm(benchWasmLinalg(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmLinalg(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          break;
        }
        case 'arrayops': {
          const arrayOps = ['roll', 'flip', 'tile', 'pad', 'take', 'gradient', 'nonzero'];
          for (const op of arrayOps) {
            const jsR = benchJsArrayOp(op, size, iterations, warmup);
            if (jsR) pushDual(jsR);
            for (const wasm of wasmModules) {
              pushWasm(benchWasmArrayOp(wasm, op, size, 'f64', iterations, warmup));
              pushWasm(benchWasmArrayOp(wasm, op, size, 'f32', iterations, warmup));
            }
          }
          break;
        }
        case 'fft': {
          const fftOps = ['rfft2', 'irfft2'];
          for (const op of fftOps) {
            const jsR = benchJsFft(op, size, iterations, warmup);
            if (jsR) pushDual(jsR);
            for (const wasm of wasmModules) {
              pushWasm(benchWasmFft(wasm, op, size, 'f64', iterations, warmup));
            }
          }
          break;
        }
        // ─── Complex categories ──────
        case 'binary_complex': {
          const ops = ['add', 'mul'];
          const dtypes: ComplexDtype[] = ['c128', 'c64'];
          for (const op of ops) {
            for (const dt of dtypes) {
              pushDual(benchJsBinaryComplex(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmBinaryComplex(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
        case 'unary_complex': {
          const ops = ['abs', 'exp'];
          const dtypes: ComplexDtype[] = ['c128', 'c64'];
          for (const op of ops) {
            for (const dt of dtypes) {
              pushDual(benchJsUnaryComplex(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmUnaryComplex(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
        case 'reduction_complex': {
          const dtypes: ComplexDtype[] = ['c128', 'c64'];
          for (const dt of dtypes) {
            pushDual(benchJsReductionComplex('sum', dt, size, iterations, warmup));
            for (const wasm of wasmModules) {
              pushWasm(benchWasmReductionComplex(wasm, 'sum', dt, size, iterations, warmup));
            }
          }
          break;
        }
        case 'linalg_complex': {
          const dtypes: ComplexDtype[] = ['c128', 'c64'];
          for (const dt of dtypes) {
            pushDual(benchJsLinalgComplex(dt, size, iterations, warmup));
            for (const wasm of wasmModules) {
              pushWasm(benchWasmLinalgComplex(wasm, dt, size, iterations, warmup));
            }
          }
          break;
        }
        case 'fft_complex': {
          const ops = ['fft', 'ifft'];
          const dtypes: ComplexDtype[] = ['c128', 'c64'];
          for (const op of ops) {
            for (const dt of dtypes) {
              pushDual(benchJsFftComplex(op, dt, size, iterations, warmup));
              for (const wasm of wasmModules) {
                pushWasm(benchWasmFftComplex(wasm, op, dt, size, iterations, warmup));
              }
            }
          }
          break;
        }
      }
      } catch (err) {
        console.error(`  ⚠ Crashed at ${cat.name} size=${size}: ${err instanceof Error ? err.message : err}`);
        console.error(`    Skipping remaining sizes for this category.`);
        break;
      }
    }

    kernelReport.categories[cat.name] = { metric: cat.metric, results: kernelResults };
    e2eReport.categories[cat.name] = { metric: cat.metric, results: e2eResults };
    printCategoryResults(cat.name, cat.metric, kernelResults);
  }

  // Binary sizes
  const binarySizes = collectBinarySizes();
  kernelReport.binarySizes = binarySizes;
  e2eReport.binarySizes = binarySizes;
  printBinarySizes(binarySizes);

  // Save outputs — kernel and e2e, with runtime prefix
  const prefix = rt.name === 'node' ? 'latest' : `latest-${rt.name}`;
  const prefixE2e = rt.name === 'node' ? 'latest-e2e' : `latest-${rt.name}-e2e`;
  saveJSON(kernelReport, prefix);
  generateHTML(kernelReport, `Kernel Only — ${rt.name} ${rt.version}`, prefix);
  saveJSON(e2eReport, prefixE2e);
  generateHTML(e2eReport, `End-to-End (incl. memory copy) — ${rt.name} ${rt.version}`, prefixE2e);

  console.log(`\n${'═'.repeat(90)}`);
  console.log('Notes:');
  console.log('  - Kernel times measure WASM function execution only (post memory copy)');
  console.log('  - E2E times include alloc, copy-in, kernel, and copy-out (JS↔WASM memory)');
  console.log('  - JS baselines use numpy-ts (float64 only); f32 WASM has no direct JS equivalent');
  console.log('  - Sort re-copies random data before each iteration');
  console.log('  - Binary sizes: b64+gzip = base64 encode .wasm, then gzip(level:9) — simulates JS bundle size');
  console.log(`  - Output: results/${prefix}.json + .html (kernel), results/${prefixE2e}.json + .html (e2e)`);
  console.log(`${'═'.repeat(90)}\n`);
}

main().catch(console.error);
