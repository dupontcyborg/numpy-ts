/**
 * WASM Single-thread vs Multi-thread matmul benchmark (f64 and f32)
 *
 * Each worker pre-loads its A-row-slice + full B into its own WASM memory.
 * Timed iterations measure compute-only (no data copy per iteration).
 * Workers are coordinated via SharedArrayBuffer + Atomics.
 *
 * Run:
 *   tsx harness/bench-matmul-threaded.ts
 *   tsx harness/bench-matmul-threaded.ts --workers=4
 */

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import os from 'node:os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const DIST_DIR = resolve(__dirname, '..', 'dist');

// ─── Worker thread ────────────────────────────────────────────────────────────

if (!isMainThread) {
  const {
    wasmBuf,        // ArrayBuffer of .wasm file
    sharedA,        // SharedArrayBuffer with full A
    sharedB,        // SharedArrayBuffer with full B
    sharedC,        // SharedArrayBuffer for output C
    M, N, K,
    startRow, endRow,
    dtype,          // 'f32' | 'f64'
    syncBuf,        // SharedArrayBuffer: Int32Array[2] — [cmd, doneCount]
    workerId,
    numWorkers,
  } = workerData as {
    wasmBuf: ArrayBuffer;
    sharedA: SharedArrayBuffer;
    sharedB: SharedArrayBuffer;
    sharedC: SharedArrayBuffer;
    M: number; N: number; K: number;
    startRow: number; endRow: number;
    dtype: 'f32' | 'f64';
    syncBuf: SharedArrayBuffer;
    workerId: number;
    numWorkers: number;
  };

  const PAGE_SIZE = 65536;
  const mod = new WebAssembly.Module(wasmBuf);
  const memory = new WebAssembly.Memory({ initial: 1 });
  const inst = new WebAssembly.Instance(mod, {});
  const exports = inst.exports as Record<string, any>;
  const wasmMemory = exports.memory as WebAssembly.Memory;

  const rowCount = endRow - startRow;
  const bpe = dtype === 'f64' ? 8 : 4;
  const bytesNeeded = (rowCount * K + K * N + rowCount * N) * bpe;
  const currentBytes = wasmMemory.buffer.byteLength;
  if (bytesNeeded > currentBytes) {
    wasmMemory.grow(Math.ceil((bytesNeeded - currentBytes) / PAGE_SIZE));
  }

  const aPtr = 0;
  const bPtr = rowCount * K * bpe;
  const cPtr = (rowCount * K + K * N) * bpe;

  // Pre-load A-slice and B into WASM memory (not timed)
  if (dtype === 'f64') {
    new Float64Array(wasmMemory.buffer, aPtr, rowCount * K)
      .set(new Float64Array(sharedA, startRow * K * 8, rowCount * K));
    new Float64Array(wasmMemory.buffer, bPtr, K * N)
      .set(new Float64Array(sharedB));
  } else {
    new Float32Array(wasmMemory.buffer, aPtr, rowCount * K)
      .set(new Float32Array(sharedA, startRow * K * 4, rowCount * K));
    new Float32Array(wasmMemory.buffer, bPtr, K * N)
      .set(new Float32Array(sharedB));
  }

  const kernel = dtype === 'f64'
    ? exports.matmul_f64_micro as (a: number, b: number, c: number, m: number, n: number, k: number) => void
    : exports.matmul_f32 as (a: number, b: number, c: number, m: number, n: number, k: number) => void;

  const sync = new Int32Array(syncBuf);
  // sync[0] = command: 0=wait, 1=compute, 2=stop
  // sync[1] = done counter (Atomics.add)

  parentPort!.postMessage('ready');

  // Event loop
  while (true) {
    // Wait for command
    Atomics.wait(sync, 0, 0);
    const cmd = Atomics.load(sync, 0);
    if (cmd === 2) break;

    // cmd === 1: compute
    kernel(aPtr, bPtr, cPtr, rowCount, N, K);

    // Write back C slice
    if (dtype === 'f64') {
      new Float64Array(sharedC, startRow * N * 8, rowCount * N)
        .set(new Float64Array(wasmMemory.buffer, cPtr, rowCount * N));
    } else {
      new Float32Array(sharedC, startRow * N * 4, rowCount * N)
        .set(new Float32Array(wasmMemory.buffer, cPtr, rowCount * N));
    }

    // Signal done
    const prev = Atomics.add(sync, 1, 1);
    if (prev + 1 === numWorkers) {
      Atomics.notify(sync, 1, 1); // wake main thread
    }
  }

  process.exit(0);
}

// ─── Main thread ──────────────────────────────────────────────────────────────

interface Shape { label: string; M: number; N: number; K: number }

const SHAPES: Shape[] = [
  { label: '64×64×64',       M: 64,   N: 64,   K: 64 },
  { label: '128×128×128',    M: 128,  N: 128,  K: 128 },
  { label: '256×256×256',    M: 256,  N: 256,  K: 256 },
  { label: '512×512×512',    M: 512,  N: 512,  K: 512 },
  { label: '1024×1024×1024', M: 1024, N: 1024, K: 1024 },
];

const WARMUP = 3;
const MIN_TIME_MS = 500;
const PAGE_SIZE = 65536;

function flops(M: number, N: number, K: number) { return 2 * M * N * K; }

const cpuCount = os.cpus().length;
const argWorkers = process.argv.find(a => a.startsWith('--workers='));
const MAX_WORKERS = argWorkers ? parseInt(argWorkers.split('=')[1]) : cpuCount;

// ─── Single-thread baseline ───────────────────────────────────────────────────

function loadWasmSync(file: string) {
  const buf = readFileSync(resolve(DIST_DIR, file));
  const mod = new WebAssembly.Module(buf);
  const inst = new WebAssembly.Instance(mod, {});
  const exports = inst.exports as Record<string, any>;
  const memory = exports.memory as WebAssembly.Memory;
  return { exports, memory };
}

function ensureMemory(memory: WebAssembly.Memory, bytesNeeded: number) {
  const cur = memory.buffer.byteLength;
  if (bytesNeeded > cur) memory.grow(Math.ceil((bytesNeeded - cur) / PAGE_SIZE));
}

function benchSingle(
  fn: Function, memory: WebAssembly.Memory,
  data: Float64Array | Float32Array,
  M: number, N: number, K: number, bpe: number,
): { ms: number } {
  const aElems = M * K, bElems = K * N, cElems = M * N;
  ensureMemory(memory, (aElems + bElems + cElems) * bpe);
  const aPtr = 0, bPtr = aElems * bpe, cPtr = (aElems + bElems) * bpe;

  const Arr = bpe === 8 ? Float64Array : Float32Array;
  const half = aElems + bElems;
  (new Arr(memory.buffer, aPtr, aElems) as any).set(data.subarray(0, aElems));
  (new Arr(memory.buffer, bPtr, bElems) as any).set(data.subarray(aElems, half));

  for (let w = 0; w < WARMUP; w++) fn(aPtr, bPtr, cPtr, M, N, K);
  let total = 0, iters = 0;
  while (total < MIN_TIME_MS) {
    const t0 = performance.now();
    fn(aPtr, bPtr, cPtr, M, N, K);
    total += performance.now() - t0;
    iters++;
  }
  return { ms: total / iters };
}

// ─── Multi-thread bench ───────────────────────────────────────────────────────

interface WorkerCtx {
  worker: Worker;
  sync: Int32Array;
}

async function spawnWorkers(
  wasmBuf: ArrayBuffer,
  sharedA: SharedArrayBuffer, sharedB: SharedArrayBuffer, sharedC: SharedArrayBuffer,
  M: number, N: number, K: number, dtype: 'f32' | 'f64', nWorkers: number,
): Promise<{ workers: Worker[]; sync: Int32Array; syncBuf: SharedArrayBuffer }> {
  const syncBuf = new SharedArrayBuffer(8); // [cmd, doneCount]
  const sync = new Int32Array(syncBuf);

  const readyPromises: Promise<void>[] = [];
  const workers: Worker[] = [];

  for (let i = 0; i < nWorkers; i++) {
    const startRow = Math.floor(i * M / nWorkers);
    const endRow = Math.floor((i + 1) * M / nWorkers);
    const w = new Worker(__filename, {
      workerData: {
        wasmBuf, sharedA, sharedB, sharedC,
        M, N, K, startRow, endRow, dtype,
        syncBuf, workerId: i, numWorkers: nWorkers,
      },
      transferList: [wasmBuf],
    });
    workers.push(w);
    readyPromises.push(new Promise(resolve => w.once('message', resolve)));
    // wasmBuf is transferred to first worker only; subsequent workers need their own copy
    // → actually we need to pass copies. Fix below.
  }

  await Promise.all(readyPromises);
  return { workers, sync, syncBuf };
}

// Because ArrayBuffer is transferred (neutered) to the first worker,
// we need separate copies per worker.
async function spawnWorkersSafe(
  wasmPath: string,
  sharedA: SharedArrayBuffer, sharedB: SharedArrayBuffer, sharedC: SharedArrayBuffer,
  M: number, N: number, K: number, dtype: 'f32' | 'f64', nWorkers: number,
): Promise<{ workers: Worker[]; sync: Int32Array; syncBuf: SharedArrayBuffer }> {
  const syncBuf = new SharedArrayBuffer(8);
  const sync = new Int32Array(syncBuf);
  const workers: Worker[] = [];
  const readyPromises: Promise<void>[] = [];

  const wasmBytes = readFileSync(wasmPath);

  for (let i = 0; i < nWorkers; i++) {
    const startRow = Math.floor(i * M / nWorkers);
    const endRow = Math.floor((i + 1) * M / nWorkers);
    // Each worker gets its own copy of the WASM bytes
    const wasmBuf = wasmBytes.buffer.slice(
      wasmBytes.byteOffset,
      wasmBytes.byteOffset + wasmBytes.byteLength,
    );
    const w = new Worker(__filename, {
      workerData: {
        wasmBuf, sharedA, sharedB, sharedC,
        M, N, K, startRow, endRow, dtype,
        syncBuf, workerId: i, numWorkers: nWorkers,
      },
    });
    workers.push(w);
    readyPromises.push(new Promise(resolve => w.once('message', resolve)));
  }

  await Promise.all(readyPromises);
  return { workers, sync, syncBuf };
}

async function dispatchAndWait(sync: Int32Array, nWorkers: number): Promise<void> {
  Atomics.store(sync, 1, 0);          // reset done counter
  Atomics.store(sync, 0, 1);          // set cmd = compute
  Atomics.notify(sync, 0, nWorkers);  // wake all workers

  // Atomics.waitAsync is the non-blocking variant allowed on the main thread
  while (Atomics.load(sync, 1) < nWorkers) {
    const result = (Atomics as any).waitAsync(sync, 1, Atomics.load(sync, 1));
    if (result.async) await result.value;
  }
  Atomics.store(sync, 0, 0); // reset cmd
}

async function benchMulti(
  sync: Int32Array, nWorkers: number,
): Promise<{ ms: number }> {
  // Warmup
  for (let w = 0; w < WARMUP; w++) {
    await dispatchAndWait(sync, nWorkers);
  }

  let total = 0, iters = 0;
  while (total < MIN_TIME_MS) {
    const t0 = performance.now();
    await dispatchAndWait(sync, nWorkers);
    total += performance.now() - t0;
    iters++;
  }
  return { ms: total / iters };
}

function stopWorkers(workers: Worker[], sync: Int32Array) {
  Atomics.store(sync, 0, 2);
  Atomics.notify(sync, 0, workers.length);
}

// ─── Run ──────────────────────────────────────────────────────────────────────

async function runBench(dtype: 'f32' | 'f64') {
  const bpe = dtype === 'f64' ? 8 : 4;
  const Arr = dtype === 'f64' ? Float64Array : Float32Array;
  const wasmFile = dtype === 'f64' ? 'matmul_zig.wasm' : 'matmul_rust.wasm';
  const wasmPath = resolve(DIST_DIR, wasmFile);
  const singleMod = loadWasmSync(wasmFile);
  const singleFn = dtype === 'f64'
    ? singleMod.exports.matmul_f64_micro
    : singleMod.exports.matmul_f32;
  const kernel = dtype === 'f64' ? 'Zig micro' : 'Rust ikj';

  const workerCounts = [1, 2, 4, 8, 16].filter(n => n <= MAX_WORKERS * 2).slice(0, 5);

  const hdr = [
    'Shape'.padEnd(20),
    '1-thread ms'.padStart(13),
    '1-thread GF/s'.padStart(14),
    ...workerCounts.map(n => `${n}w ms`.padStart(10)),
    ...workerCounts.map(n => `${n}w GF/s`.padStart(11)),
    ...workerCounts.map(n => `${n}w speedup`.padStart(12)),
  ];

  const label = dtype === 'f64'
    ? `▸ f64  (${kernel} WASM — single thread vs multi-thread)`
    : `▸ f32  (${kernel} WASM — single thread vs multi-thread)`;

  console.log(label);
  console.log(hdr.join(' │ '));
  console.log('─'.repeat(hdr.join(' │ ').length));

  for (const { label: shapeLabel, M, N, K } of SHAPES) {
    // Fill shared data
    const sharedA = new SharedArrayBuffer(M * K * bpe);
    const sharedB = new SharedArrayBuffer(K * N * bpe);
    const sharedC = new SharedArrayBuffer(M * N * bpe);
    const arrA = new Arr(sharedA);
    const arrB = new Arr(sharedB);
    for (let i = 0; i < arrA.length; i++) arrA[i] = Math.random() * 2 - 1;
    for (let i = 0; i < arrB.length; i++) arrB[i] = Math.random() * 2 - 1;

    // Single-thread
    const combined = new Arr(M * K + K * N);
    combined.set(arrA); combined.set(arrB, M * K);
    const single = benchSingle(singleFn, singleMod.memory, combined, M, N, K, bpe);
    const gf = flops(M, N, K) / 1e9;
    const singleGF = gf / (single.ms / 1000);

    // Multi-thread
    const multiResults: Array<{ ms: number }> = [];
    for (const nw of workerCounts) {
      const { workers, sync } = await spawnWorkersSafe(
        wasmPath, sharedA, sharedB, sharedC, M, N, K, dtype, nw,
      );
      const res = await benchMulti(sync, nw);
      multiResults.push(res);
      stopWorkers(workers, sync);
      await Promise.all(workers.map(w => new Promise(r => w.once('exit', r))));
    }

    const row = [
      shapeLabel.padEnd(20),
      single.ms.toFixed(2).padStart(13),
      singleGF.toFixed(2).padStart(14),
      ...multiResults.map(r => r.ms.toFixed(2).padStart(10)),
      ...multiResults.map(r => (gf / (r.ms / 1000)).toFixed(2).padStart(11)),
      ...multiResults.map(r => `${(single.ms / r.ms).toFixed(2)}x`.padStart(12)),
    ];
    console.log(row.join(' │ '));
  }

  console.log();
}

async function main() {
  console.log('WASM Single-thread vs Multi-thread Matmul Benchmark');
  console.log('═'.repeat(80));
  console.log(`CPUs: ${cpuCount}, max workers: ${MAX_WORKERS}`);
  console.log(`Warmup: ${WARMUP} iters, min measurement time: ${MIN_TIME_MS}ms`);
  console.log('Timed phase: compute-only (data pre-loaded into each worker\'s WASM memory)');
  console.log();

  await runBench('f64');
  await runBench('f32');
}

main().catch(console.error);
