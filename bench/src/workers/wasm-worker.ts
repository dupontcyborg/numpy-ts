/**
 * WASM Worker for parallel execution
 *
 * This worker loads a WASM module and performs operations on data chunks.
 * Uses SharedArrayBuffer for zero-copy data sharing with main thread.
 */

import { parentPort, workerData } from 'worker_threads';
import { readFile } from 'fs/promises';

interface WorkerConfig {
  wasmPath: string;
  workerId: number;
  numWorkers: number;
}

interface TaskMessage {
  type: 'add' | 'sin' | 'sum' | 'matmul';
  dtype: 'float32' | 'float64';
  // For SharedArrayBuffer operations
  sharedBuffer?: SharedArrayBuffer;
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  length?: number;
  // For matmul
  m?: number;
  n?: number;
  k?: number;
  // Chunk info for parallel reduction
  startIdx?: number;
  endIdx?: number;
}

interface WasmExports {
  memory: WebAssembly.Memory;
  alloc_f64(len: number): number;
  alloc_f32(len: number): number;
  free_f64(ptr: number, len: number): void;
  free_f32(ptr: number, len: number): void;
  add_f64(a: number, b: number, c: number, len: number): void;
  add_f32(a: number, b: number, c: number, len: number): void;
  sin_f64(a: number, c: number, len: number): void;
  sin_f32(a: number, c: number, len: number): void;
  sum_f64(a: number, len: number): number;
  sum_f32(a: number, len: number): number;
  matmul_f64(a: number, b: number, c: number, m: number, n: number, k: number): void;
  matmul_f32(a: number, b: number, c: number, m: number, n: number, k: number): void;
}

let wasmExports: WasmExports | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

async function initWasm(wasmPath: string): Promise<void> {
  const wasmBuffer = await readFile(wasmPath);

  wasmMemory = new WebAssembly.Memory({
    initial: 256,
    maximum: 4096,
  });

  const importObject = {
    env: { memory: wasmMemory },
    wasi_snapshot_preview1: {
      fd_write: () => 0,
      fd_close: () => 0,
      fd_seek: () => 0,
      proc_exit: () => {},
    },
  };

  const wasmModule = await WebAssembly.compile(wasmBuffer);
  const instance = await WebAssembly.instantiate(wasmModule, importObject);
  wasmExports = instance.exports as unknown as WasmExports;

  if (wasmExports.memory) {
    wasmMemory = wasmExports.memory;
  }
}

function copyToWasm(
  data: Float64Array | Float32Array,
  startIdx: number,
  endIdx: number
): { ptr: number; len: number } {
  if (!wasmExports || !wasmMemory) throw new Error('WASM not initialized');

  const len = endIdx - startIdx;
  const isF64 = data instanceof Float64Array;
  const ptr = isF64 ? wasmExports.alloc_f64(len) : wasmExports.alloc_f32(len);

  const wasmArray = isF64
    ? new Float64Array(wasmMemory.buffer, ptr, len)
    : new Float32Array(wasmMemory.buffer, ptr, len);

  // Copy chunk from shared buffer
  for (let i = 0; i < len; i++) {
    wasmArray[i] = data[startIdx + i]!;
  }

  return { ptr, len };
}

function copyFromWasm(
  ptr: number,
  len: number,
  dest: Float64Array | Float32Array,
  destOffset: number
): void {
  if (!wasmMemory) throw new Error('WASM not initialized');

  const isF64 = dest instanceof Float64Array;
  const wasmArray = isF64
    ? new Float64Array(wasmMemory.buffer, ptr, len)
    : new Float32Array(wasmMemory.buffer, ptr, len);

  for (let i = 0; i < len; i++) {
    dest[destOffset + i] = wasmArray[i]!;
  }
}

function handleTask(task: TaskMessage): number | void {
  if (!wasmExports || !wasmMemory) throw new Error('WASM not initialized');

  const startIdx = task.startIdx ?? 0;
  const endIdx = task.endIdx ?? task.length ?? 0;
  const chunkLen = endIdx - startIdx;

  if (task.type === 'add') {
    const sharedA = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);
    const sharedB = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.bOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.bOffset, task.length);
    const sharedC = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.cOffset, task.length);

    // Copy chunk to WASM
    const aWasm = copyToWasm(sharedA, startIdx, endIdx);
    const bWasm = copyToWasm(sharedB, startIdx, endIdx);
    const cPtr = task.dtype === 'float64'
      ? wasmExports.alloc_f64(chunkLen)
      : wasmExports.alloc_f32(chunkLen);

    // Perform operation
    if (task.dtype === 'float64') {
      wasmExports.add_f64(aWasm.ptr, bWasm.ptr, cPtr, chunkLen);
    } else {
      wasmExports.add_f32(aWasm.ptr, bWasm.ptr, cPtr, chunkLen);
    }

    // Copy result back to shared buffer
    copyFromWasm(cPtr, chunkLen, sharedC, startIdx);

    // Free WASM memory
    if (task.dtype === 'float64') {
      wasmExports.free_f64(aWasm.ptr, aWasm.len);
      wasmExports.free_f64(bWasm.ptr, bWasm.len);
      wasmExports.free_f64(cPtr, chunkLen);
    } else {
      wasmExports.free_f32(aWasm.ptr, aWasm.len);
      wasmExports.free_f32(bWasm.ptr, bWasm.len);
      wasmExports.free_f32(cPtr, chunkLen);
    }
  } else if (task.type === 'sin') {
    const sharedA = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);
    const sharedC = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.cOffset, task.length);

    const aWasm = copyToWasm(sharedA, startIdx, endIdx);
    const cPtr = task.dtype === 'float64'
      ? wasmExports.alloc_f64(chunkLen)
      : wasmExports.alloc_f32(chunkLen);

    if (task.dtype === 'float64') {
      wasmExports.sin_f64(aWasm.ptr, cPtr, chunkLen);
    } else {
      wasmExports.sin_f32(aWasm.ptr, cPtr, chunkLen);
    }

    copyFromWasm(cPtr, chunkLen, sharedC, startIdx);

    if (task.dtype === 'float64') {
      wasmExports.free_f64(aWasm.ptr, aWasm.len);
      wasmExports.free_f64(cPtr, chunkLen);
    } else {
      wasmExports.free_f32(aWasm.ptr, aWasm.len);
      wasmExports.free_f32(cPtr, chunkLen);
    }
  } else if (task.type === 'sum') {
    const sharedA = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);

    const aWasm = copyToWasm(sharedA, startIdx, endIdx);

    let partialSum: number;
    if (task.dtype === 'float64') {
      partialSum = wasmExports.sum_f64(aWasm.ptr, chunkLen);
      wasmExports.free_f64(aWasm.ptr, aWasm.len);
    } else {
      partialSum = wasmExports.sum_f32(aWasm.ptr, chunkLen);
      wasmExports.free_f32(aWasm.ptr, aWasm.len);
    }

    return partialSum;
  } else if (task.type === 'matmul') {
    // For matmul, each worker computes a subset of rows
    const m = task.m!;
    const n = task.n!;
    const k = task.k!;

    const sharedA = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, m * k)
      : new Float32Array(task.sharedBuffer!, task.aOffset, m * k);
    const sharedB = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.bOffset, k * n)
      : new Float32Array(task.sharedBuffer!, task.bOffset, k * n);
    const sharedC = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, m * n)
      : new Float32Array(task.sharedBuffer!, task.cOffset, m * n);

    // Worker computes rows [startIdx, endIdx) of C
    const rowStart = startIdx;
    const rowEnd = endIdx;
    const numRows = rowEnd - rowStart;

    // Copy full B matrix and relevant rows of A to WASM
    const bPtr = task.dtype === 'float64'
      ? wasmExports.alloc_f64(k * n)
      : wasmExports.alloc_f32(k * n);
    const bWasm = task.dtype === 'float64'
      ? new Float64Array(wasmMemory!.buffer, bPtr, k * n)
      : new Float32Array(wasmMemory!.buffer, bPtr, k * n);
    bWasm.set(sharedB);

    const aPtr = task.dtype === 'float64'
      ? wasmExports.alloc_f64(numRows * k)
      : wasmExports.alloc_f32(numRows * k);
    const aWasm = task.dtype === 'float64'
      ? new Float64Array(wasmMemory!.buffer, aPtr, numRows * k)
      : new Float32Array(wasmMemory!.buffer, aPtr, numRows * k);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < k; j++) {
        aWasm[i * k + j] = sharedA[(rowStart + i) * k + j]!;
      }
    }

    const cPtr = task.dtype === 'float64'
      ? wasmExports.alloc_f64(numRows * n)
      : wasmExports.alloc_f32(numRows * n);

    // Compute partial matmul
    if (task.dtype === 'float64') {
      wasmExports.matmul_f64(aPtr, bPtr, cPtr, numRows, n, k);
    } else {
      wasmExports.matmul_f32(aPtr, bPtr, cPtr, numRows, n, k);
    }

    // Copy result back
    const cWasm = task.dtype === 'float64'
      ? new Float64Array(wasmMemory!.buffer, cPtr, numRows * n)
      : new Float32Array(wasmMemory!.buffer, cPtr, numRows * n);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < n; j++) {
        sharedC[(rowStart + i) * n + j] = cWasm[i * n + j]!;
      }
    }

    // Free WASM memory
    if (task.dtype === 'float64') {
      wasmExports.free_f64(aPtr, numRows * k);
      wasmExports.free_f64(bPtr, k * n);
      wasmExports.free_f64(cPtr, numRows * n);
    } else {
      wasmExports.free_f32(aPtr, numRows * k);
      wasmExports.free_f32(bPtr, k * n);
      wasmExports.free_f32(cPtr, numRows * n);
    }
  }
}

// Main worker loop
async function main() {
  const config = workerData as WorkerConfig;

  try {
    await initWasm(config.wasmPath);
    parentPort?.postMessage({ type: 'ready' });

    parentPort?.on('message', (task: TaskMessage | { type: 'exit' }) => {
      if (task.type === 'exit') {
        process.exit(0);
      }

      try {
        const result = handleTask(task as TaskMessage);
        parentPort?.postMessage({ type: 'done', result });
      } catch (error) {
        parentPort?.postMessage({ type: 'error', error: String(error) });
      }
    });
  } catch (error) {
    parentPort?.postMessage({ type: 'error', error: String(error) });
    process.exit(1);
  }
}

main();
