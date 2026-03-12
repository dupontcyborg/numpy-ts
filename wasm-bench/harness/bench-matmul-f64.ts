/**
 * Focused f64 matmul kernel benchmark
 * Compares: Zig-ikj (original) vs Zig-ijkacc (register-acc) vs Zig-micro (4×4 register-blocked) vs Rust-ikj
 *
 * Tests symmetric and asymmetric shapes. Reports GFLOPs and time.
 */

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST_DIR = resolve(__dirname, '..', 'dist');

// ─── Load WASM modules ──────────────────────────────────────────────────────

const PAGE_SIZE = 65536;

interface WasmModule {
  exports: Record<string, any>;
  memory: WebAssembly.Memory;
}

async function loadWasm(file: string): Promise<WasmModule> {
  const buf = readFileSync(resolve(DIST_DIR, file));
  const mod = await WebAssembly.compile(buf);
  const inst = await WebAssembly.instantiate(mod, {});
  const memory = inst.exports.memory as WebAssembly.Memory;
  return { exports: inst.exports as Record<string, any>, memory };
}

function ensureMemory(memory: WebAssembly.Memory, bytesNeeded: number) {
  const current = memory.buffer.byteLength;
  if (bytesNeeded > current) {
    const pagesNeeded = Math.ceil((bytesNeeded - current) / PAGE_SIZE);
    memory.grow(pagesNeeded);
  }
}

// ─── Benchmark config ────────────────────────────────────────────────────────

interface Shape {
  label: string;
  M: number;
  N: number;
  K: number;
}

// Symmetric sizes
const SYMMETRIC: Shape[] = [
  { label: '32×32×32',     M: 32,   N: 32,   K: 32 },
  { label: '64×64×64',     M: 64,   N: 64,   K: 64 },
  { label: '100×100×100',  M: 100,  N: 100,  K: 100 },
  { label: '128×128×128',  M: 128,  N: 128,  K: 128 },
  { label: '256×256×256',  M: 256,  N: 256,  K: 256 },
  { label: '384×384×384',  M: 384,  N: 384,  K: 384 },
  { label: '500×500×500',  M: 500,  N: 500,  K: 500 },
  { label: '512×512×512',  M: 512,  N: 512,  K: 512 },
  { label: '640×640×640',  M: 640,  N: 640,  K: 640 },
  { label: '768×768×768',  M: 768,  N: 768,  K: 768 },
  { label: '1000×1000×1000', M: 1000, N: 1000, K: 1000 },
  { label: '1024×1024×1024', M: 1024, N: 1024, K: 1024 },
];

// Asymmetric: tall, wide, deep-K, rectangular
const ASYMMETRIC: Shape[] = [
  // Tall-skinny (big M, small N/K)
  { label: '1000×32×32',   M: 1000, N: 32,   K: 32 },
  { label: '1000×100×100', M: 1000, N: 100,  K: 100 },
  { label: '4096×16×16',   M: 4096, N: 16,   K: 16 },
  // Wide (small M, big N)
  { label: '32×1000×32',   M: 32,   N: 1000, K: 32 },
  { label: '100×1000×100', M: 100,  N: 1000, K: 100 },
  { label: '16×4096×16',   M: 16,   N: 4096, K: 16 },
  // Deep K (small M/N, big K)
  { label: '32×32×1000',   M: 32,   N: 32,   K: 1000 },
  { label: '100×100×1000', M: 100,  N: 100,  K: 1000 },
  { label: '16×16×4096',   M: 16,   N: 16,   K: 4096 },
  // Rectangular mixes
  { label: '500×64×500',   M: 500,  N: 64,   K: 500 },
  { label: '64×500×500',   M: 64,   N: 500,  K: 500 },
  { label: '500×500×64',   M: 500,  N: 500,  K: 64 },
  { label: '256×1000×128', M: 256,  N: 1000, K: 128 },
  { label: '1000×128×256', M: 1000, N: 128,  K: 256 },
];

const WARMUP = 3;
const MIN_TIME_MS = 500;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function matmulFlops(M: number, N: number, K: number): number {
  return 2 * M * N * K;
}

function fillRandom(arr: Float64Array) {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = Math.random() * 2 - 1;
  }
}

function verify(c1: Float64Array, c2: Float64Array, label: string, scale: number): boolean {
  let maxErr = 0;
  for (let i = 0; i < c1.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(c1[i] - c2[i]));
  }
  const tol = scale * 1e-12;
  if (maxErr > tol) {
    console.error(`  MISMATCH ${label}: max error = ${maxErr.toExponential(2)} (tol=${tol.toExponential(2)})`);
    return false;
  }
  return true;
}

function benchKernel(
  fn: Function,
  aPtr: number, bPtr: number, cPtr: number,
  M: number, N: number, K: number,
): { ms: number; iters: number } {
  for (let w = 0; w < WARMUP; w++) {
    fn(aPtr, bPtr, cPtr, M, N, K);
  }
  let totalMs = 0;
  let iters = 0;
  while (totalMs < MIN_TIME_MS) {
    const t0 = performance.now();
    fn(aPtr, bPtr, cPtr, M, N, K);
    totalMs += performance.now() - t0;
    iters++;
  }
  return { ms: totalMs / iters, iters };
}

function printHeader() {
  const hdr = [
    'Shape'.padEnd(20),
    'ikj ms'.padStart(10),
    'acc ms'.padStart(10),
    'micro ms'.padStart(10),
    'rust ms'.padStart(10),
    'ikj GF/s'.padStart(10),
    'acc GF/s'.padStart(10),
    'micro GF/s'.padStart(10),
    'rust GF/s'.padStart(10),
    'micro/acc'.padStart(10),
    'micro/rust'.padStart(11),
  ];
  console.log(hdr.join(' │ '));
  console.log('─'.repeat(hdr.join(' │ ').length));
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const zigMod = await loadWasm('matmul_zig.wasm');
  const rustMod = await loadWasm('matmul_rust.wasm');

  console.log('f64 Matmul Kernel Benchmark');
  console.log('═'.repeat(140));
  console.log('Kernels: ikj (streaming), ijkacc (register-acc), micro (4×4 register-blocked), rust (ikj)');
  console.log(`Tile: 64, Min time per measurement: ${MIN_TIME_MS}ms, warmup: ${WARMUP} calls`);
  console.log();

  function runShape(shape: Shape) {
    const { label, M, N, K } = shape;
    const aElems = M * K;
    const bElems = K * N;
    const cElems = M * N;
    const bytesNeeded = (aElems + bElems + cElems) * 8;

    ensureMemory(zigMod.memory, bytesNeeded);
    ensureMemory(rustMod.memory, bytesNeeded);

    const aPtr = 0;
    const bPtr = aElems * 8;
    const cPtr = (aElems + bElems) * 8;

    // Fill A and B
    const jsA = new Float64Array(aElems);
    const jsB = new Float64Array(bElems);
    fillRandom(jsA);
    fillRandom(jsB);

    new Float64Array(zigMod.memory.buffer, aPtr, aElems).set(jsA);
    new Float64Array(zigMod.memory.buffer, bPtr, bElems).set(jsB);
    new Float64Array(rustMod.memory.buffer, aPtr, aElems).set(jsA);
    new Float64Array(rustMod.memory.buffer, bPtr, bElems).set(jsB);

    // Benchmark all 4 kernels
    const zigIkj = benchKernel(zigMod.exports.matmul_f64, aPtr, bPtr, cPtr, M, N, K);
    const zigAcc = benchKernel(zigMod.exports.matmul_f64_ijkacc, aPtr, bPtr, cPtr, M, N, K);
    const zigMicro = benchKernel(zigMod.exports.matmul_f64_micro, aPtr, bPtr, cPtr, M, N, K);
    const rustIkj = benchKernel(rustMod.exports.matmul_f64, aPtr, bPtr, cPtr, M, N, K);

    // Verify correctness against ikj reference
    zigMod.exports.matmul_f64(aPtr, bPtr, cPtr, M, N, K);
    const refC = Float64Array.from(new Float64Array(zigMod.memory.buffer, cPtr, cElems));

    zigMod.exports.matmul_f64_ijkacc(aPtr, bPtr, cPtr, M, N, K);
    verify(refC, new Float64Array(zigMod.memory.buffer, cPtr, cElems), 'zig-acc', K);

    zigMod.exports.matmul_f64_micro(aPtr, bPtr, cPtr, M, N, K);
    verify(refC, new Float64Array(zigMod.memory.buffer, cPtr, cElems), 'zig-micro', K);

    rustMod.exports.matmul_f64(aPtr, bPtr, cPtr, M, N, K);
    verify(refC, new Float64Array(rustMod.memory.buffer, cPtr, cElems), 'rust', K);

    // Report
    const gf = matmulFlops(M, N, K) / 1e9;
    const zigIkjGF = gf / (zigIkj.ms / 1000);
    const zigAccGF = gf / (zigAcc.ms / 1000);
    const zigMicroGF = gf / (zigMicro.ms / 1000);
    const rustGF = gf / (rustIkj.ms / 1000);
    const microVsAcc = zigAcc.ms / zigMicro.ms;
    const microVsRust = rustIkj.ms / zigMicro.ms;

    const row = [
      label.padEnd(20),
      zigIkj.ms.toFixed(2).padStart(10),
      zigAcc.ms.toFixed(2).padStart(10),
      zigMicro.ms.toFixed(2).padStart(10),
      rustIkj.ms.toFixed(2).padStart(10),
      zigIkjGF.toFixed(2).padStart(10),
      zigAccGF.toFixed(2).padStart(10),
      zigMicroGF.toFixed(2).padStart(10),
      rustGF.toFixed(2).padStart(10),
      `${microVsAcc.toFixed(2)}x`.padStart(10),
      `${microVsRust.toFixed(2)}x`.padStart(11),
    ];
    console.log(row.join(' │ '));
  }

  // ── Symmetric ──
  console.log('▸ Symmetric (M=N=K)');
  printHeader();
  for (const shape of SYMMETRIC) {
    runShape(shape);
  }

  console.log();

  // ── Asymmetric ──
  console.log('▸ Asymmetric');
  printHeader();
  for (const shape of ASYMMETRIC) {
    runShape(shape);
  }

  console.log();
  console.log('micro/acc  = ijkacc time / micro time (>1 means micro is faster)');
  console.log('micro/rust = rust time / micro time   (>1 means micro is faster)');
}

main().catch(console.error);
