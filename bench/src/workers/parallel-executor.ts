/**
 * Parallel WASM Executor
 *
 * Manages a pool of workers for parallel WASM execution.
 * Uses SharedArrayBuffer for zero-copy data sharing.
 */

import { Worker } from 'worker_threads';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface ParallelExecutorConfig {
  numWorkers: number;
  wasmPath: string;
  name: string;
}

interface WorkerWrapper {
  worker: Worker;
  ready: boolean;
  busy: boolean;
}

export class ParallelExecutor {
  private workers: WorkerWrapper[] = [];
  private config: ParallelExecutorConfig;
  private initialized = false;
  public name: string;

  constructor(config: ParallelExecutorConfig) {
    this.config = config;
    this.name = config.name;
  }

  async init(): Promise<boolean> {
    if (this.initialized) return true;

    if (!existsSync(this.config.wasmPath)) {
      console.warn(`WASM file not found: ${this.config.wasmPath}`);
      return false;
    }

    const workerPath = join(__dirname, 'wasm-worker.ts');

    const initPromises: Promise<void>[] = [];

    for (let i = 0; i < this.config.numWorkers; i++) {
      const worker = new Worker(workerPath, {
        workerData: {
          wasmPath: this.config.wasmPath,
          workerId: i,
          numWorkers: this.config.numWorkers,
        },
        execArgv: ['--import', 'tsx'],
      });

      const wrapper: WorkerWrapper = { worker, ready: false, busy: false };
      this.workers.push(wrapper);

      initPromises.push(
        new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error(`Worker ${i} init timeout`));
          }, 10000);

          worker.once('message', (msg) => {
            if (msg.type === 'ready') {
              clearTimeout(timeout);
              wrapper.ready = true;
              resolve();
            } else if (msg.type === 'error') {
              clearTimeout(timeout);
              reject(new Error(msg.error));
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
      console.warn(`Failed to initialize parallel executor: ${error}`);
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

  get numWorkers(): number {
    return this.config.numWorkers;
  }

  /**
   * Execute parallel element-wise add: c = a + b
   */
  async parallelAdd(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    if (!this.initialized) throw new Error('Executor not initialized');

    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    // Create shared buffer for all three arrays
    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement * 3);
    const aOffset = 0;
    const bOffset = len * bytesPerElement;
    const cOffset = len * bytesPerElement * 2;

    // Copy input data to shared buffer
    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);
    const sharedB = dtype === 'float64'
      ? new Float64Array(sharedBuffer, bOffset, len)
      : new Float32Array(sharedBuffer, bOffset, len);

    sharedA.set(a);
    sharedB.set(b);

    // Distribute work across workers
    const chunkSize = Math.ceil(len / this.config.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.config.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);

      if (startIdx >= len) break;

      promises.push(
        this.executeOnWorker(i, {
          type: 'add',
          dtype,
          sharedBuffer,
          aOffset,
          bOffset,
          cOffset,
          length: len,
          startIdx,
          endIdx,
        })
      );
    }

    await Promise.all(promises);

    // Copy result from shared buffer
    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, len)
      : new Float32Array(sharedBuffer, cOffset, len);
    c.set(sharedC);
  }

  /**
   * Execute parallel element-wise sin: c = sin(a)
   */
  async parallelSin(
    a: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    if (!this.initialized) throw new Error('Executor not initialized');

    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement * 2);
    const aOffset = 0;
    const cOffset = len * bytesPerElement;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);

    sharedA.set(a);

    const chunkSize = Math.ceil(len / this.config.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.config.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);

      if (startIdx >= len) break;

      promises.push(
        this.executeOnWorker(i, {
          type: 'sin',
          dtype,
          sharedBuffer,
          aOffset,
          cOffset,
          length: len,
          startIdx,
          endIdx,
        })
      );
    }

    await Promise.all(promises);

    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, len)
      : new Float32Array(sharedBuffer, cOffset, len);
    c.set(sharedC);
  }

  /**
   * Execute parallel sum reduction
   */
  async parallelSum(
    a: Float64Array | Float32Array,
    dtype: 'float32' | 'float64'
  ): Promise<number> {
    if (!this.initialized) throw new Error('Executor not initialized');

    const len = a.length;
    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    const sharedBuffer = new SharedArrayBuffer(len * bytesPerElement);
    const aOffset = 0;

    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, len)
      : new Float32Array(sharedBuffer, aOffset, len);

    sharedA.set(a);

    const chunkSize = Math.ceil(len / this.config.numWorkers);
    const promises: Promise<number>[] = [];

    for (let i = 0; i < this.config.numWorkers; i++) {
      const startIdx = i * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, len);

      if (startIdx >= len) break;

      promises.push(
        this.executeOnWorker(i, {
          type: 'sum',
          dtype,
          sharedBuffer,
          aOffset,
          length: len,
          startIdx,
          endIdx,
        }).then((result) => result as number)
      );
    }

    const partialSums = await Promise.all(promises);
    return partialSums.reduce((a, b) => a + b, 0);
  }

  /**
   * Execute parallel matrix multiplication: C = A * B
   * Parallelizes by distributing rows of C across workers
   */
  async parallelMatmul(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    c: Float64Array | Float32Array,
    m: number,
    n: number,
    k: number,
    dtype: 'float32' | 'float64'
  ): Promise<void> {
    if (!this.initialized) throw new Error('Executor not initialized');

    const bytesPerElement = dtype === 'float64' ? 8 : 4;

    // Create shared buffer for A, B, and C
    const totalSize = m * k + k * n + m * n;
    const sharedBuffer = new SharedArrayBuffer(totalSize * bytesPerElement);

    const aOffset = 0;
    const bOffset = m * k * bytesPerElement;
    const cOffset = (m * k + k * n) * bytesPerElement;

    // Copy input matrices
    const sharedA = dtype === 'float64'
      ? new Float64Array(sharedBuffer, aOffset, m * k)
      : new Float32Array(sharedBuffer, aOffset, m * k);
    const sharedB = dtype === 'float64'
      ? new Float64Array(sharedBuffer, bOffset, k * n)
      : new Float32Array(sharedBuffer, bOffset, k * n);

    sharedA.set(a);
    sharedB.set(b);

    // Distribute rows across workers
    const rowsPerWorker = Math.ceil(m / this.config.numWorkers);
    const promises: Promise<void>[] = [];

    for (let i = 0; i < this.config.numWorkers; i++) {
      const startRow = i * rowsPerWorker;
      const endRow = Math.min(startRow + rowsPerWorker, m);

      if (startRow >= m) break;

      promises.push(
        this.executeOnWorker(i, {
          type: 'matmul',
          dtype,
          sharedBuffer,
          aOffset,
          bOffset,
          cOffset,
          m,
          n,
          k,
          startIdx: startRow,
          endIdx: endRow,
        })
      );
    }

    await Promise.all(promises);

    // Copy result
    const sharedC = dtype === 'float64'
      ? new Float64Array(sharedBuffer, cOffset, m * n)
      : new Float32Array(sharedBuffer, cOffset, m * n);
    c.set(sharedC);
  }

  private executeOnWorker(workerId: number, task: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const wrapper = this.workers[workerId];
      if (!wrapper || !wrapper.ready) {
        reject(new Error(`Worker ${workerId} not ready`));
        return;
      }

      const handler = (msg: any) => {
        if (msg.type === 'done') {
          wrapper.worker.removeListener('message', handler);
          wrapper.busy = false;
          resolve(msg.result);
        } else if (msg.type === 'error') {
          wrapper.worker.removeListener('message', handler);
          wrapper.busy = false;
          reject(new Error(msg.error));
        }
      };

      wrapper.worker.on('message', handler);
      wrapper.busy = true;
      wrapper.worker.postMessage(task);
    });
  }
}

/**
 * Create parallel executors for available WASM modules
 */
export async function createParallelExecutors(
  numWorkers: number = 4
): Promise<Map<string, ParallelExecutor>> {
  const executors = new Map<string, ParallelExecutor>();

  const benchDir = join(__dirname, '..', '..');
  const distDir = join(benchDir, 'dist');

  // Rust WASM
  const rustPath = join(distDir, 'numpy_bench.wasm');
  if (existsSync(rustPath)) {
    const executor = new ParallelExecutor({
      numWorkers,
      wasmPath: rustPath,
      name: 'rust-wasm-parallel',
    });
    if (await executor.init()) {
      executors.set('rust-wasm-parallel', executor);
    }
  }

  // Zig WASM
  const zigPaths = [
    join(distDir, 'zig_numpy_bench.wasm'),
    join(benchDir, 'zig', 'zig-out', 'bin', 'numpy_bench.wasm'),
  ];

  for (const zigPath of zigPaths) {
    if (existsSync(zigPath)) {
      const executor = new ParallelExecutor({
        numWorkers,
        wasmPath: zigPath,
        name: 'zig-wasm-parallel',
      });
      if (await executor.init()) {
        executors.set('zig-wasm-parallel', executor);
        break;
      }
    }
  }

  return executors;
}
