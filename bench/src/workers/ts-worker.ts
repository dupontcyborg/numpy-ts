/**
 * TypeScript Worker for parallel execution
 *
 * Pure TypeScript implementations running in worker threads.
 */

import { parentPort, workerData } from 'worker_threads';

interface TaskMessage {
  type: 'add' | 'sin' | 'sum' | 'matmul';
  dtype: 'float32' | 'float64';
  sharedBuffer?: SharedArrayBuffer;
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  length?: number;
  m?: number;
  n?: number;
  k?: number;
  startIdx?: number;
  endIdx?: number;
}

function handleTask(task: TaskMessage): number | void {
  const startIdx = task.startIdx ?? 0;
  const endIdx = task.endIdx ?? task.length ?? 0;

  if (task.type === 'add') {
    const a = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);
    const b = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.bOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.bOffset, task.length);
    const c = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.cOffset, task.length);

    for (let i = startIdx; i < endIdx; i++) {
      c[i] = a[i]! + b[i]!;
    }
  } else if (task.type === 'sin') {
    const a = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);
    const c = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.cOffset, task.length);

    if (task.dtype === 'float64') {
      for (let i = startIdx; i < endIdx; i++) {
        c[i] = Math.sin(a[i]!);
      }
    } else {
      for (let i = startIdx; i < endIdx; i++) {
        c[i] = Math.fround(Math.sin(a[i]!));
      }
    }
  } else if (task.type === 'sum') {
    const a = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, task.length)
      : new Float32Array(task.sharedBuffer!, task.aOffset, task.length);

    let sum = 0;
    for (let i = startIdx; i < endIdx; i++) {
      sum += a[i]!;
    }
    return sum;
  } else if (task.type === 'matmul') {
    const m = task.m!;
    const n = task.n!;
    const k = task.k!;

    const a = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.aOffset, m * k)
      : new Float32Array(task.sharedBuffer!, task.aOffset, m * k);
    const b = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.bOffset, k * n)
      : new Float32Array(task.sharedBuffer!, task.bOffset, k * n);
    const c = task.dtype === 'float64'
      ? new Float64Array(task.sharedBuffer!, task.cOffset, m * n)
      : new Float32Array(task.sharedBuffer!, task.cOffset, m * n);

    // Worker computes rows [startIdx, endIdx) of C
    for (let i = startIdx; i < endIdx; i++) {
      for (let kk = 0; kk < k; kk++) {
        const aik = a[i * k + kk]!;
        for (let j = 0; j < n; j++) {
          c[i * n + j] += aik * b[kk * n + j]!;
        }
      }
    }
  }
}

// Signal ready immediately
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
