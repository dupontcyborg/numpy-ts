/**
 * WASM Multi-Category Benchmark Harness
 *
 * Compares WASM (Zig + Rust) kernels against numpy-ts pure JS across:
 *   - matmul, reductions, unary elementwise, binary elementwise, sorting
 * Reports timing, throughput metrics, and binary sizes.
 */

import { readFileSync } from 'node:fs';
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
];

// ─── WASM Loading ────────────────────────────────────────────────────────────

interface WasmKernels {
  name: string;
  memory: WebAssembly.Memory;
  // Matmul
  matmul_f64?: (a: number, b: number, c: number, m: number, n: number, k: number) => void;
  matmul_f32?: (a: number, b: number, c: number, m: number, n: number, k: number) => void;
  // Reductions
  sum_f64?: (ptr: number, n: number) => number;
  sum_f32?: (ptr: number, n: number) => number;
  max_f64?: (ptr: number, n: number) => number;
  max_f32?: (ptr: number, n: number) => number;
  // Unary
  sqrt_f64?: (inp: number, out: number, n: number) => void;
  sqrt_f32?: (inp: number, out: number, n: number) => void;
  exp_f64?: (inp: number, out: number, n: number) => void;
  exp_f32?: (inp: number, out: number, n: number) => void;
  // Binary
  add_f64?: (a: number, b: number, out: number, n: number) => void;
  add_f32?: (a: number, b: number, out: number, n: number) => void;
  mul_f64?: (a: number, b: number, out: number, n: number) => void;
  mul_f32?: (a: number, b: number, out: number, n: number) => void;
  // Sort
  sort_f64?: (ptr: number, n: number) => void;
  sort_f32?: (ptr: number, n: number) => void;
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

  // Dynamically map all function exports
  const kernels: WasmKernels = { name, memory };
  const fnNames = [
    'matmul_f64', 'matmul_f32',
    'sum_f64', 'sum_f32', 'max_f64', 'max_f32',
    'sqrt_f64', 'sqrt_f32', 'exp_f64', 'exp_f32',
    'add_f64', 'add_f32', 'mul_f64', 'mul_f32',
    'sort_f64', 'sort_f32',
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
  const [linalgMod, reductionMod, expMod, arithMod, sortMod] = await Promise.all([
    import('../../src/common/ops/linalg.js'),
    import('../../src/common/ops/reduction.js'),
    import('../../src/common/ops/exponential.js'),
    import('../../src/common/ops/arithmetic.js'),
    import('../../src/common/ops/sorting.js'),
  ]);
  return {
    matmul: linalgMod.matmul as (a: ArrayStorage, b: ArrayStorage) => ArrayStorage,
    sum: reductionMod.sum as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    max: reductionMod.max as (a: ArrayStorage, axis?: number, keepdims?: boolean) => ArrayStorage | number,
    sqrt: expMod.sqrt as (a: ArrayStorage) => ArrayStorage,
    exp: expMod.exp as (a: ArrayStorage) => ArrayStorage,
    add: arithMod.add as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    multiply: arithMod.multiply as (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage,
    sort: sortMod.sort as (a: ArrayStorage, axis?: number) => ArrayStorage,
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
  const currentBytes = wasm.memory.buffer.byteLength;
  if (bytes > currentBytes) {
    const needed = Math.ceil((bytes - currentBytes) / 65536);
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

  const aOff = 0, bOff = matBytes, cOff = 2 * matBytes;
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
  for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  new ArrayType(wasm.memory.buffer, 0, size).set(data);

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(0, size);
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
  baselines: Baselines, opName: 'sum' | 'max', size: number, iterations: number, warmup: number
): TimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
  const storage = ArrayStorage.fromData(data, [size], 'float64');
  const fn = baselines[opName];

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
  for (let i = 0; i < size; i++) data[i] = Math.random() * 5 + 0.1; // positive for sqrt
  new ArrayType(wasm.memory.buffer, 0, size).set(data);

  const inOff = 0, outOff = bufBytes;
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
  baselines: Baselines, opName: 'sqrt' | 'exp', size: number, iterations: number, warmup: number
): TimingResult {
  const data = new Float64Array(size);
  for (let i = 0; i < size; i++) data[i] = Math.random() * 5 + 0.1;
  const storage = ArrayStorage.fromData(data, [size], 'float64');
  const fn = baselines[opName];

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
    aData[i] = Math.random() * 2 - 1;
    bData[i] = Math.random() * 2 - 1;
  }
  new ArrayType(wasm.memory.buffer, 0, size).set(aData);
  new ArrayType(wasm.memory.buffer, bufBytes, size).set(bData);

  const aOff = 0, bOff = bufBytes, outOff = 2 * bufBytes;
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
  baselines: Baselines, opName: 'add' | 'multiply', size: number, iterations: number, warmup: number
): TimingResult {
  const aData = new Float64Array(size);
  const bData = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    aData[i] = Math.random() * 2 - 1;
    bData[i] = Math.random() * 2 - 1;
  }
  const aS = ArrayStorage.fromData(aData, [size], 'float64');
  const bS = ArrayStorage.fromData(bData, [size], 'float64');
  const fn = baselines[opName];

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    const t0 = performance.now();
    fn(aS, bS);
    const t1 = performance.now();
    if (iter >= warmup) times.push(t1 - t0);
  }

  const ms = median(times);
  // Map WASM op names to display: add/mul
  const displayOp = opName === 'multiply' ? 'mul' : opName;
  return {
    impl: 'JS (f64)',
    op: displayOp,
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

  const times: number[] = [];
  for (let iter = 0; iter < warmup + iterations; iter++) {
    // Must re-copy random data every iteration (in-place sort!)
    new ArrayType(wasm.memory.buffer, 0, size).set(data);
    const t0 = performance.now();
    fn(0, size);
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

// ─── Correctness Checks ─────────────────────────────────────────────────────

function verifyAll(wasms: WasmKernels[]): boolean {
  let allOk = true;

  for (const wasm of wasms) {
    const checks: [string, boolean][] = [];

    // Matmul: skip detailed check here (kept from original, just basic)
    if (wasm.matmul_f64) {
      ensureMemory(wasm, 3 * 16 * 8);
      const a = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); // I4
      const b = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      new Float64Array(wasm.memory.buffer, 0, 16).set(a);
      new Float64Array(wasm.memory.buffer, 128, 16).set(b);
      wasm.matmul_f64(0, 128, 256, 4, 4, 4);
      const c = new Float64Array(wasm.memory.buffer, 256, 16);
      // I * B = B
      const ok = b.every((v, i) => Math.abs(c[i] - v) < 1e-10);
      checks.push(['matmul', ok]);
    }

    // Reductions
    if (wasm.sum_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, 0, 4).set([1, 2, 3, 4]);
      const result = wasm.sum_f64(0, 4);
      checks.push(['sum', Math.abs(result - 10) < 1e-10]);
    }
    if (wasm.max_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, 0, 6).set([3, 1, 4, 1, 5, 9]);
      const result = wasm.max_f64(0, 6);
      checks.push(['max', Math.abs(result - 9) < 1e-10]);
    }

    // Unary
    if (wasm.sqrt_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, 0, 1).set([4]);
      wasm.sqrt_f64(0, 8, 1);
      const result = new Float64Array(wasm.memory.buffer, 8, 1)[0];
      checks.push(['sqrt', Math.abs(result - 2) < 1e-10]);
    }
    if (wasm.exp_f64) {
      ensureMemory(wasm, 32);
      new Float64Array(wasm.memory.buffer, 0, 2).set([0, 1]);
      wasm.exp_f64(0, 16, 2);
      const r = new Float64Array(wasm.memory.buffer, 16, 2);
      checks.push(['exp', Math.abs(r[0] - 1) < 1e-10 && Math.abs(r[1] - Math.E) < 1e-6]);
    }

    // Binary
    if (wasm.add_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, 0, 2).set([1, 2]);
      new Float64Array(wasm.memory.buffer, 16, 2).set([3, 4]);
      wasm.add_f64(0, 16, 32, 2);
      const r = new Float64Array(wasm.memory.buffer, 32, 2);
      checks.push(['add', Math.abs(r[0] - 4) < 1e-10 && Math.abs(r[1] - 6) < 1e-10]);
    }
    if (wasm.mul_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, 0, 2).set([2, 3]);
      new Float64Array(wasm.memory.buffer, 16, 2).set([4, 5]);
      wasm.mul_f64(0, 16, 32, 2);
      const r = new Float64Array(wasm.memory.buffer, 32, 2);
      checks.push(['mul', Math.abs(r[0] - 8) < 1e-10 && Math.abs(r[1] - 15) < 1e-10]);
    }

    // Sort
    if (wasm.sort_f64) {
      ensureMemory(wasm, 48);
      new Float64Array(wasm.memory.buffer, 0, 6).set([3, 1, 4, 1, 5, 9]);
      wasm.sort_f64(0, 6);
      const r = new Float64Array(wasm.memory.buffer, 0, 6);
      const expected = [1, 1, 3, 4, 5, 9];
      const ok = expected.every((v, i) => Math.abs(r[i] - v) < 1e-10);
      checks.push(['sort', ok]);
    }

    const allPass = checks.every(([, ok]) => ok);
    const details = checks.map(([name, ok]) => `${name}:${ok ? 'PASS' : 'FAIL'}`).join(' ');
    console.log(`  ${wasm.name.padEnd(12)} ${allPass ? 'PASS' : 'FAIL'}  (${details})`);
    if (!allPass) allOk = false;
  }

  return allOk;
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

    // Find JS baseline for each op
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

function printBinarySizes() {
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

  // All per-category wasm files
  const files: { name: string; label: string }[] = [];
  for (const kern of KERNEL_NAMES) {
    files.push({ name: `${kern}_zig.wasm`, label: `Zig ${kern}` });
    files.push({ name: `${kern}_rust.wasm`, label: `Rust ${kern}` });
  }

  const zigTotal = { raw: 0, compressed: 0 };
  const rustTotal = { raw: 0, compressed: 0 };

  for (const { name, label } of files) {
    try {
      const raw = readFileSync(resolve(DIST_DIR, name));
      // b64 first (how it's embedded in JS), then gzip (transport compression)
      const b64 = Buffer.from(raw.toString('base64'));
      const gzipped = gzipSync(b64, { level: 9 });
      const ratio = ((1 - gzipped.length / raw.length) * 100).toFixed(1);

      const row = [
        `${label} (${name})`.padEnd(30),
        raw.length.toLocaleString().padStart(14),
        gzipped.length.toLocaleString().padStart(14),
        `${ratio}%`.padStart(14),
      ].join(' │ ');
      console.log(row);

      if (name.includes('_zig')) {
        zigTotal.raw += raw.length;
        zigTotal.compressed += gzipped.length;
      } else {
        rustTotal.raw += raw.length;
        rustTotal.compressed += gzipped.length;
      }
    } catch {
      console.log(`  ${label}: not found`);
    }
  }

  // Totals
  console.log('─'.repeat(90));
  for (const [lang, totals] of [['Zig', zigTotal], ['Rust', rustTotal]] as const) {
    if (totals.raw > 0) {
      const ratio = ((1 - totals.compressed / totals.raw) * 100).toFixed(1);
      const row = [
        `${lang} TOTAL`.padEnd(30),
        totals.raw.toLocaleString().padStart(14),
        totals.compressed.toLocaleString().padStart(14),
        `${ratio}%`.padStart(14),
      ].join(' │ ');
      console.log(row);
    }
  }
}

// ─── Per-Category WASM Loading ───────────────────────────────────────────────

const KERNEL_NAMES = ['matmul', 'reduction', 'unary', 'binary', 'sort'] as const;
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
          for (const op of ['sum', 'max'] as const) {
            results.push(benchJsReduction(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              const r64 = benchWasmReduction(wasm, op, size, 'f64', iterations, warmup);
              const r32 = benchWasmReduction(wasm, op, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          break;
        }
        case 'unary': {
          for (const op of ['sqrt', 'exp'] as const) {
            results.push(benchJsUnary(baselines, op, size, iterations, warmup));
            for (const wasm of wasmModules) {
              const r64 = benchWasmUnary(wasm, op, size, 'f64', iterations, warmup);
              const r32 = benchWasmUnary(wasm, op, size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          break;
        }
        case 'binary': {
          const wasmOps = ['add', 'mul'] as const;
          const jsOps = ['add', 'multiply'] as const;
          for (let oi = 0; oi < wasmOps.length; oi++) {
            results.push(benchJsBinary(baselines, jsOps[oi], size, iterations, warmup));
            for (const wasm of wasmModules) {
              const r64 = benchWasmBinary(wasm, wasmOps[oi], size, 'f64', iterations, warmup);
              const r32 = benchWasmBinary(wasm, wasmOps[oi], size, 'f32', iterations, warmup);
              if (r64) results.push(r64);
              if (r32) results.push(r32);
            }
          }
          break;
        }
        case 'sort': {
          results.push(benchJsSort(baselines, size, iterations, warmup));
          for (const wasm of wasmModules) {
            const r64 = benchWasmSort(wasm, size, 'f64', iterations, warmup);
            const r32 = benchWasmSort(wasm, size, 'f32', iterations, warmup);
            if (r64) results.push(r64);
            if (r32) results.push(r32);
          }
          break;
        }
      }
    }

    printCategoryResults(cat.name, cat.metric, results);
  }

  // Binary sizes
  printBinarySizes();

  console.log(`\n${'═'.repeat(90)}`);
  console.log('Notes:');
  console.log('  - WASM times include memory copy overhead for matmul; kernel-only for other categories');
  console.log('  - JS baselines use numpy-ts (float64 only); f32 WASM has no direct JS equivalent');
  console.log('  - Sort re-copies random data before each iteration');
  console.log('  - Binary sizes: b64+gzip = base64 encode .wasm, then gzip(level:9) — simulates JS bundle size');
  console.log(`${'═'.repeat(90)}\n`);
}

main().catch(console.error);
