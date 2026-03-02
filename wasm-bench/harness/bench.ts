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
): TimingResult | null {
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
  const times: number[] = [];

  for (let iter = 0; iter < warmup + iterations; iter++) {
    new ArrayType(wasm.memory.buffer, aOff, matElems).set(aData);
    new ArrayType(wasm.memory.buffer, bOff, matElems).set(bData);
    const t0 = performance.now();
    fn(aOff, bOff, cOff, n, n, n);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'matmul',
    size,
    timeMs: ms,
    metric: computeGflops(n, ms),
    metricUnit: 'GFLOP/s',
  };
}

function benchJsMatmul(
  baselines: Baselines, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult {
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
    return { impl: 'JS (f64)', op: 'matmul', size, timeMs: ms, metric: computeGflops(n, ms), metricUnit: 'GFLOP/s' };
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
    return { impl: 'JS (f32→f64)', op: 'matmul', size, timeMs: ms, metric: computeGflops(n, ms), metricUnit: 'GFLOP/s' };
  }
}

// ─── Reduction Benchmarks ───────────────────────────────────────────────────

function computeGBs(elems: number, bytesPerElem: number, timeMs: number): number {
  return (elems * bytesPerElem) / (timeMs / 1000) / 1e9;
}

function benchWasmReduction(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((ptr: number, n: number) => number) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  if (opName === 'prod') {
    // Small values near 1 to avoid overflow
    for (let i = 0; i < size; i++) data[i] = 1 + Math.random() * 0.0001;
  } else {
    for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  }
  const base = wasm.baseOffset;
  new ArrayType(wasm.memory.buffer, base, size).set(data);

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, bytesPerElem, ms),
    metricUnit: 'GB/s',
  };
}

function benchJsReduction(
  baselines: Baselines, opName: 'sum' | 'max' | 'min' | 'prod' | 'mean', size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  };
}

// ─── Unary Benchmarks ───────────────────────────────────────────────────────

function benchWasmUnary(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
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
  new ArrayType(wasm.memory.buffer, base, size).set(data);

  const inOff = base, outOff = base + bufBytes;
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(inOff, outOff, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, bytesPerElem, ms),
    metricUnit: 'GB/s',
  };
}

function benchJsUnary(
  baselines: Baselines, opName: string, size: number, iterations: number, warmup: number, displayOp?: string
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: displayOp ?? opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  };
}

// ─── Binary Benchmarks ──────────────────────────────────────────────────────

function benchWasmBinary(
  wasm: WasmKernels, opName: string, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`${opName}_${dtype}`] as ((a: number, b: number, out: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, 3 * bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const aData = new ArrayType(size);
  const bData = new ArrayType(size);
  for (let i = 0; i < size; i++) {
    if (opName === 'div') {
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
    } else {
      aData[i] = Math.random() * 2 - 1;
      bData[i] = Math.random() * 2 - 1;
    }
  }
  const base = wasm.baseOffset;
  new ArrayType(wasm.memory.buffer, base, size).set(aData);
  new ArrayType(wasm.memory.buffer, base + bufBytes, size).set(bData);

  const aOff = base, bOff = base + bufBytes, outOff = base + 2 * bufBytes;
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(aOff, bOff, outOff, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, bytesPerElem, ms),
    metricUnit: 'GB/s',
  };
}

function benchJsBinary(
  baselines: Baselines, opName: string, size: number, iterations: number, warmup: number, displayOp?: string
): TimingResult {
  const aData = new Float64Array(size);
  const bData = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    if (opName === 'divide') {
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
  return {
    impl: 'JS (f64)',
    op: displayOp ?? opName,
    size,
    timeMs: ms,
    metric: computeGBs(size, 8, ms),
    metricUnit: 'GB/s',
  };
}

// ─── Sort Benchmarks ────────────────────────────────────────────────────────

function computeMElemPerSec(elems: number, timeMs: number): number {
  return elems / (timeMs / 1000) / 1e6;
}

function benchWasmSort(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
  const fn = dtype === 'f64' ? wasm.sort_f64 : wasm.sort_f32;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const bufBytes = size * bytesPerElem;
  ensureMemory(wasm, bufBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const base = wasm.baseOffset;
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Must re-copy random data every iteration (in-place sort!)
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'sort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

function benchJsSort(
  baselines: Baselines, size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: 'sort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

// ─── Diff Benchmarks ─────────────────────────────────────────────────────────

function benchWasmDiff(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
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
  new ArrayType(wasm.memory.buffer, base, size).set(data);

  const inOff = base, outOff = base + inBytes;
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(inOff, outOff, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'diff',
    size,
    timeMs: ms,
    metric: computeGBs(size - 1, bytesPerElem, ms),
    metricUnit: 'GB/s',
  };
}

function benchJsDiff(
  baselines: Baselines, size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: 'diff',
    size,
    timeMs: ms,
    metric: computeGBs(size - 1, 8, ms),
    metricUnit: 'GB/s',
  };
}

// ─── Argsort Benchmarks ──────────────────────────────────────────────────────

function benchWasmArgsort(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`argsort_${dtype}`] as ((vals: number, idx: number, n: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const dataBytes = size * bytesPerElem;
  const idxBytes = size * 4; // u32 indices
  ensureMemory(wasm, dataBytes + idxBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const base = wasm.baseOffset;
  const dataOff = base;
  const idxOff = base + dataBytes;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Re-copy data each iteration
    new ArrayType(wasm.memory.buffer, dataOff, size).set(data);
    const t0 = performance.now();
    fn(dataOff, idxOff, size);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'argsort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

function benchJsArgsort(
  baselines: Baselines, size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: 'argsort',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

// ─── Partition Benchmarks ────────────────────────────────────────────────────

function benchWasmPartition(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
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
  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Re-copy data each iteration (in-place)
    new ArrayType(wasm.memory.buffer, base, size).set(data);
    const t0 = performance.now();
    fn(base, size, kth);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'partition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

function benchJsPartition(
  baselines: Baselines, size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: 'partition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

// ─── Argpartition Benchmarks ─────────────────────────────────────────────────

function benchWasmArgpartition(
  wasm: WasmKernels, size: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
  const fn = (wasm as Record<string, unknown>)[`argpartition_${dtype}`] as ((vals: number, idx: number, n: number, kth: number) => void) | undefined;
  if (!fn) return null;

  const bytesPerElem = dtype === 'f64' ? 8 : 4;
  const dataBytes = size * bytesPerElem;
  const idxBytes = size * 4; // u32 indices
  ensureMemory(wasm, dataBytes + idxBytes);

  const ArrayType = dtype === 'f64' ? Float64Array : Float32Array;
  const data = new ArrayType(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 1000;

  const kth = Math.floor(size / 2);
  const base = wasm.baseOffset;
  const dataOff = base;
  const idxOff = base + dataBytes;

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Re-copy data each iteration
    new ArrayType(wasm.memory.buffer, dataOff, size).set(data);
    const t0 = performance.now();
    fn(dataOff, idxOff, size, kth);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  return {
    impl: `${wasm.name} (${dtype})`,
    op: 'argpartition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

function benchJsArgpartition(
  baselines: Baselines, size: number, iterations: number, warmup: number
): TimingResult {
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
  return {
    impl: 'JS (f64)',
    op: 'argpartition',
    size,
    timeMs: ms,
    metric: computeMElemPerSec(size, ms),
    metricUnit: 'M elem/s',
  };
}

// ─── Convolve Benchmarks ─────────────────────────────────────────────────────

function benchWasmConvolve(
  wasm: WasmKernels, op: 'correlate' | 'convolve', signalSize: number, dtype: 'f64' | 'f32', iterations: number, warmup: number
): TimingResult | null {
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

  new ArrayType(wasm.memory.buffer, aOff, signalSize).set(aData);
  new ArrayType(wasm.memory.buffer, bOff, kernelSize).set(bData);

  const fn_name = `${op}_${dtype}`;
  const fn = (wasm as Record<string, unknown>)[fn_name] as ((a: number, b: number, out: number, na: number, nb: number) => void) | undefined;
  if (!fn) return null;

  // warmup
  for (let i = 0; i < warmup; i++) fn(aOff, bOff, outOff, signalSize, kernelSize);
  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn(aOff, bOff, outOff, signalSize, kernelSize);
  const elapsed = performance.now() - start;
  const timeMs = elapsed / iterations;

  const elemPerSec = outSize / (timeMs / 1000);
  return {
    impl: `${wasm.name} (${dtype})`,
    op,
    size: signalSize,
    timeMs,
    metric: elemPerSec / 1e6,
    metricUnit: 'M elem/s',
  };
}

function benchJsConvolve(
  baselines: Baselines, op: 'correlate' | 'convolve', signalSize: number, iterations: number, warmup: number
): TimingResult {
  const kernelSize = 32;
  const outSize = signalSize + kernelSize - 1;

  const aData = new Float64Array(signalSize);
  const bData = new Float64Array(kernelSize);
  for (let i = 0; i < signalSize; i++) aData[i] = Math.random() * 2 - 1;
  for (let i = 0; i < kernelSize; i++) bData[i] = Math.random() * 2 - 1;

  const aS = ArrayStorage.fromData(aData, [signalSize], 'float64');
  const bS = ArrayStorage.fromData(bData, [kernelSize], 'float64');
  const fn = baselines[op] as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage;

  // warmup
  for (let i = 0; i < warmup; i++) fn(aS, bS);
  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn(aS, bS);
  const elapsed = performance.now() - start;
  const timeMs = elapsed / iterations;

  const elemPerSec = outSize / (timeMs / 1000);
  return {
    impl: 'JS (f64)',
    op,
    size: signalSize,
    timeMs,
    metric: elemPerSec / 1e6,
    metricUnit: 'M elem/s',
  };
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
  environment: { node: string; arch: string; platform: string };
  categories: Record<string, { metric: string; results: TimingResult[] }>;
  binarySizes: BinarySizeEntry[];
}

// ─── Output Formatting ──────────────────────────────────────────────────────

function printCategoryResults(category: string, metric: string, results: TimingResult[]) {
  const sizes = [...new Set(results.map((r) => r.size))].sort((a, b) => a - b);

  for (const size of sizes) {
    const sizeResults = results.filter((r) => r.size === size);
    const sizeLabel = category === 'matmul' ? `${size}x${size}` : formatSize(size);
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

function saveJSON(report: BenchmarkReport) {
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
  const jsonPath = resolve(RESULTS_DIR, 'latest.json');
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  console.log(`\nJSON saved to ${jsonPath}`);
}

function generateHTML(report: BenchmarkReport) {
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });
  const htmlPath = resolve(RESULTS_DIR, 'latest.html');

  // Build chart data: speedup multipliers grouped by category+op+size
  interface ChartEntry {
    cat: string;
    entries: { label: string; zig64: number; zig32: number; rust64: number; rust32: number }[];
  }
  const chartData: ChartEntry[] = [];

  for (const [catName, catData] of Object.entries(report.categories)) {
    const results = catData.results;
    const ops = [...new Set(results.map((r) => r.op))];
    const sizes = [...new Set(results.map((r) => r.size))].sort((a, b) => a - b);

    for (const op of ops) {
      const displayCat = ops.length > 1 ? `${capitalize(catName)} — ${op}` : capitalize(catName);
      const entries: ChartEntry['entries'] = [];

      for (const size of sizes) {
        const sizeLabel = catName === 'matmul' ? `${size}×${size}` : formatSize(size);
        const sizeResults = results.filter((r) => r.size === size && r.op === op);

        // Find JS baseline time
        const jsResult = sizeResults.find((r) => r.impl.startsWith('JS'));
        const baseMs = jsResult?.timeMs ?? 1;

        const get = (lang: string, dtype: string) => {
          const r = sizeResults.find((r) => r.impl === `${lang} (${dtype})`);
          return r ? baseMs / r.timeMs : 0;
        };

        entries.push({
          label: sizeLabel,
          zig64: get('Zig', 'f64'),
          zig32: get('Zig', 'f32'),
          rust64: get('Rust', 'f64'),
          rust32: get('Rust', 'f32'),
        });
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
<title>WASM Benchmark Results — Speedup vs JS</title>
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
  .zig-f64 { background: #58a6ff; }
  .zig-f32 { background: #a5d6ff; }
  .rust-f64 { background: #f0883e; }
  .rust-f32 { background: #ffb77c; }
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

<h1>WASM vs JS — Speedup Multipliers</h1>
<p class="subtitle">Horizontal bars = how many times faster than pure JS (numpy-ts float64). Dashed line = 1× (JS baseline).</p>
<div class="meta">Generated: ${report.timestamp} · Node ${report.environment.node} · ${report.environment.platform}/${report.environment.arch}</div>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch zig-f64"></div> Zig f64</div>
  <div class="legend-item"><div class="legend-swatch zig-f32"></div> Zig f32</div>
  <div class="legend-item"><div class="legend-swatch rust-f64"></div> Rust f64</div>
  <div class="legend-item"><div class="legend-swatch rust-f32"></div> Rust f32</div>
</div>

<div id="charts"></div>

<script>
const DATA = ${JSON.stringify(chartData, null, 2)};
const SIZES = ${JSON.stringify(sizesArray, null, 2)};
const zigTotal = ${JSON.stringify(zigTotal)};
const rustTotal = ${JSON.stringify(rustTotal)};

const container = document.getElementById('charts');

// ── Summary Section ──
{
  // Collect all speedup values (f64 only for apples-to-apples comparison)
  const zigVals = [];
  const rustVals = [];
  const h2hVals = []; // zig/rust ratio (>1 = zig faster)
  const details = []; // {cat, label, zig64, rust64}

  for (const cat of DATA) {
    for (const e of cat.entries) {
      if (e.zig64 > 0 && e.rust64 > 0) {
        zigVals.push({ val: e.zig64, cat: cat.cat, label: e.label });
        rustVals.push({ val: e.rust64, cat: cat.cat, label: e.label });
        h2hVals.push({ val: e.zig64 / e.rust64, cat: cat.cat, label: e.label });
        details.push({ cat: cat.cat, label: e.label, zig64: e.zig64, rust64: e.rust64 });
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
    maxVal = Math.max(maxVal, e.zig64, e.zig32, e.rust64, e.rust32);
  }
  const scale = maxVal * 1.25;
  const section = document.createElement('div');
  section.className = 'category';
  let html = \`<h2>\${cat.cat}</h2>\`;
  for (const e of cat.entries) {
    const bars = [
      { val: e.zig64, cls: 'zig-f64' },
      { val: e.zig32, cls: 'zig-f32' },
      { val: e.rust64, cls: 'rust-f64' },
      { val: e.rust32, cls: 'rust-f32' },
    ];
    html += \`<div class="chart-row"><div class="chart-label">\${e.label}</div><div class="chart-area">\`;
    const baselinePct = (1 / scale) * 100;
    html += \`<div class="baseline" style="left:\${baselinePct}%"><span class="baseline-label">1×</span></div>\`;
    for (const b of bars) {
      if (b.val === 0) continue;
      const pct = Math.max((b.val / scale) * 100, 0.5);
      const valStr = b.val.toFixed(2) + '×';
      const isSmall = pct < 8;
      const slowerNote = b.val < 1 ? ' <span class="slower-note">(slower)</span>' : '';
      html += \`<div class="bar-group"><div class="bar \${b.cls}" style="width:\${pct}%"><span class="bar-val\${isSmall ? ' outside' : ''}">\${valStr}\${slowerNote}</span></div></div>\`;
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

const KERNEL_NAMES = ['matmul', 'reduction', 'unary', 'binary', 'sort', 'convolve'] as const;
type KernelName = (typeof KERNEL_NAMES)[number];

async function loadCategoryWasms(): Promise<Record<KernelName, WasmKernels[]>> {
  const result = {} as Record<KernelName, WasmKernels[]>;
  for (const kern of KERNEL_NAMES) {
    const zig = await loadWasm(`${kern}_zig.wasm`, 'Zig');
    const rust = await loadWasm(`${kern}_rust.wasm`, 'Rust');
    result[kern] = [zig, rust].filter((w): w is WasmKernels => w !== null);
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

  // Correctness checks per category
  console.log('\nCorrectness checks...');
  let allOk = true;
  for (const kern of KERNEL_NAMES) {
    const wasms = wasmByCategory[kern];
    if (wasms.length > 0) {
      const ok = verifyAll(wasms);
      if (!ok) allOk = false;
    }
  }
  if (!allOk) {
    console.error('\nCorrectness check FAILED — aborting benchmarks.');
    process.exit(1);
  }
  console.log();

  // Collect all results for report
  const report: BenchmarkReport = {
    timestamp: new Date().toISOString(),
    environment: {
      node: process.version,
      arch: process.arch,
      platform: process.platform,
    },
    categories: {},
    binarySizes: [],
  };

  // Run benchmarks per category
  for (const cat of CATEGORIES) {
    const catName = cat.name as KernelName;
    const wasmModules = wasmByCategory[catName];
    console.log(`\nBenchmarking: ${cat.name}...`);
    const results: TimingResult[] = [];

    for (const { size, iterations, warmup } of cat.sizes) {
      const sizeLabel = cat.name === 'matmul' ? `${size}x${size}` : formatSize(size);
      console.log(`  ${sizeLabel} (${iterations} iters, ${warmup} warmup)...`);

      switch (cat.name) {
        case 'matmul': {
          results.push(benchJsMatmul(baselines, size, 'f64', iterations, warmup));
          results.push(benchJsMatmul(baselines, size, 'f32', iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmMatmul(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmMatmul(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          break;
        }
        case 'reduction': {
          for (const op of ['sum', 'max', 'min', 'prod', 'mean'] as const) {
            results.push(benchJsReduction(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              const r64 = benchWasmReduction(wasm, op, size, 'f64', iterations, warmup);
              const r32 = benchWasmReduction(wasm, op, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          // diff (outputs n-1 elements, not a scalar)
          results.push(benchJsDiff(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmDiff(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmDiff(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          break;
        }
        case 'unary': {
          const unaryOps: { wasm: string; js: keyof Baselines }[] = [
            { wasm: 'sqrt', js: 'sqrt' },
            { wasm: 'exp', js: 'exp' },
            { wasm: 'log', js: 'log' },
            { wasm: 'sin', js: 'sin' },
            { wasm: 'cos', js: 'cos' },
            { wasm: 'abs', js: 'absolute' },
            { wasm: 'neg', js: 'negative' },
            { wasm: 'ceil', js: 'ceil' },
            { wasm: 'floor', js: 'floor' },
          ];
          for (const { wasm: wasmOp, js: jsOp } of unaryOps) {
            results.push(benchJsUnary(baselines, jsOp as 'sqrt', size, iterations, warmup, wasmOp));
            for (const wasm of wasmModules) {
              const r64 = benchWasmUnary(wasm, wasmOp, size, 'f64', iterations, warmup);
              const r32 = benchWasmUnary(wasm, wasmOp, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
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
          ];
          for (const { wasm: wasmOp, js: jsOp } of binaryOps) {
            results.push(benchJsBinary(baselines, jsOp as 'add', size, iterations, warmup, wasmOp));
            for (const wasm of wasmModules) {
              const r64 = benchWasmBinary(wasm, wasmOp, size, 'f64', iterations, warmup);
              const r32 = benchWasmBinary(wasm, wasmOp, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          break;
        }
        case 'sort': {
          // sort
          results.push(benchJsSort(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmSort(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmSort(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          // argsort
          results.push(benchJsArgsort(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmArgsort(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmArgsort(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          // partition
          results.push(benchJsPartition(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmPartition(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmPartition(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          // argpartition
          results.push(benchJsArgpartition(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmArgpartition(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmArgpartition(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          break;
        }
        case 'convolve': {
          for (const op of ['correlate', 'convolve'] as const) {
            results.push(benchJsConvolve(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              const r64 = benchWasmConvolve(wasm, op, size, 'f64', iterations, warmup);
              const r32 = benchWasmConvolve(wasm, op, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          break;
        }
      }
    }

    report.categories[cat.name] = { metric: cat.metric, results };
    printCategoryResults(cat.name, cat.metric, results);
  }

  // Binary sizes
  report.binarySizes = collectBinarySizes();
  printBinarySizes(report.binarySizes);

  // Save outputs
  saveJSON(report);
  generateHTML(report);

  console.log(`\n${'═'.repeat(90)}`);
  console.log('Notes:');
  console.log('  - WASM times include memory copy overhead for matmul; kernel-only for other categories');
  console.log('  - JS baselines use numpy-ts (float64 only); f32 WASM has no direct JS equivalent');
  console.log('  - Sort re-copies random data before each iteration');
  console.log('  - Binary sizes: b64+gzip = base64 encode .wasm, then gzip(level:9) — simulates JS bundle size');
  console.log(`${'═'.repeat(90)}\n`);
}

main().catch(console.error);
