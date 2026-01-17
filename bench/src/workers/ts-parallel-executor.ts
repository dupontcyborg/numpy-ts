/**
 * TypeScript Parallel Executor
 *
 * Manages a pool of workers for parallel TypeScript execution.
 */

import { Worker } from 'worker_threads';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface WorkerWrapper {
  worker: Worker;
  ready: boolean;
}

export class TSParallelExecutor {
  private workers: WorkerWrapper[] = [];
  private numWorkers: number;
  private initialized = false;
  public name = 'typescript-parallel';

  constructor(numWorkers: number = 4) {
    this.numWorkers = numWorkers;
  }

  async init(): Promise<boolean> {
    if (this.initialized) return true;

    const workerPath = join(__dirname, 'ts-worker.ts');
    const initPromises: Promise<void>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(workerPath, {
        execArgv: ['--import', 'tsx'],
      });

      const wrapper: WorkerWrapper = { worker, ready: false };
      this.workers.push(wrapper);

      initPromises.push(
        new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error(`Worker ${i} init timeout`)), 5000);

          worker.once('message', (msg) => {
            if (msg.type === 'ready') {
              clearTimeout(timeout);
              wrapper.ready = true;
              resolve();
            }
          });

          worker.once('error', (err) => {
            clearTimeout(timeout);
            reject(err);
          });
        })
      );
    }

    try {
      await Promise.all(initPromises);
      this.initialized = true;
      return true;
    } catch (error) {
      console.warn(`Failed to initialize TS parallel executor: ${error}`);
      this.terminate();
      return false;
    }
  }

  terminate(): void {
    for (const wrapper of this.workers) {
      wrapper.worker.postMessage({ type: 'exit' });
      wrapper.worker.terminate();
    }
    this.workers = [];
    this.initialized = false;
  }

  async parallelAdd(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement * 3);
    const aOffset = 0;
    const bOffset = len * bytesPerElement;
    const cOffset = len * bytesPerElement * 2;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);
    const sharedB = dtype === 'float64'
      ? new Float64Array(sharedBuffer, bOffset, len)
      : new Float32Array(sharedBuffer, bOffset, len);

    sharedA.set(a);
    sharedB.set(b);

    const chunkSize = Math.ceil(len / this.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);
      if (startIdx >= len) break;

      promises.push(this.executeOnWorker(i, {
        type: 'add', dtype, sharedBuffer, aOffset, bOffset, cOffset, length: len, startIdx, endIdx,
      }));
    }

    await Promise.all(promises);

    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, len)
      : new Float32Array(sharedBuffer, cOffset, len);
    c.set(sharedC);
  }

  async parallelSin(
    a: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement * 2);
    const aOffset = 0;
    const cOffset = len * bytesPerElement;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);
    sharedA.set(a);

    const chunkSize = Math.ceil(len / this.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);
      if (startIdx >= len) break;

      promises.push(this.executeOnWorker(i, {
        type: 'sin', dtype, sharedBuffer, aOffset, cOffset, length: len, startIdx, endIdx,
      }));
    }

    await Promise.all(promises);

    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, len)
      : new Float32Array(sharedBuffer, cOffset, len);
    c.set(sharedC);
  }

  async parallelSum(
    a: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<number> {
    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement);
    const aOffset = 0;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);
    sharedA.set(a);

    const chunkSize = Math.ceil(len / this.numWorkers);
    const promises: Promise<number>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);
      if (startIdx >= len) break;

      promises.push(
        this.executeOnWorker(i, {
          type: 'sum', dtype, sharedBuffer, aOffset, length: len, startIdx, endIdx,
        }).then((r) => r as number)
      );
    }

    const partialSums = await Promise.all(promises);
    return partialSums.reduce((a, b) => a + b, 0);
  }

  async parallelMatmul(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    m: number,
    n: number,
    k: number,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    const bytesPerElement = dtype === 'float64' ? 8 : 4;
    const totalSize = m * k + k * n + m * n;
    const sharedBuffer = new SharedArrayBuffer(totalSize * bytesPerElement);

    const aOffset = 0;
    const bOffset = m * k * bytesPerElement;
    const cOffset = (m * k + k * n) * bytesPerElement;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, m * k)
      : new Float32Array(sharedBuffer, aOffset, m * k);
    const sharedB = dtype === 'float64'
      ? new Float64Array(sharedBuffer, bOffset, k * n)
      : new Float32Array(sharedBuffer, bOffset, k * n);
    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, m * n)
      : new Float32Array(sharedBuffer, cOffset, m * n);

    sharedA.set(a);
    sharedB.set(b);
    sharedC.fill(0); // Zero output

    const rowsPerWorker = Math.ceil(m / this.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const startRow = i * rowsPerWorker;
      const endRow = Math.min(startRow + rowsPerWorker, m);
      if (startRow >= m) break;

      promises.push(this.executeOnWorker(i, {
        type: 'matmul', dtype, sharedBuffer, aOffset, bOffset, cOffset, m, n, k,
        startIdx: startRow, endIdx: endRow,
      }));
    }

    await Promise.all(promises);
    c.set(sharedC);
  }

  private executeOnWorker(workerId: number, task: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const wrapper = this.workers[workerId];
      if (!wrapper?.ready) {
        reject(new Error(`Worker ${workerId} not ready`));
        return;
      }

      const handler = (msg: any) => {
        if (msg.type === 'done') {
          wrapper.worker.removeListener('message', handler);
          resolve(msg.result);
        } else if (msg.type === 'error') {
          wrapper.worker.removeListener('message', handler);
          reject(new Error(msg.error));
        }
      };

      wrapper.worker.on('message', handler);
      wrapper.worker.postMessage(task);
    });
  }
}
