/**
 * WASM Module Loader
 *
 * Loads and instantiates WASM modules with proper memory management.
 */

import { readFile } from 'fs/promises';
import { join } from 'path';

export interface WasmExports {
  memory: WebAssembly.Memory;

  // Memory management
  alloc_f64(len: number): number;
  alloc_f32(len: number): number;
  free_f64(ptr: number, len: number): void;
  free_f32(ptr: number, len: number): void;

  // Single-threaded operations
  add_f64(a: number, b: number, c: number, len: number): void;
  add_f32(a: number, b: number, c: number, len: number): void;
  sin_f64(a: number, c: number, len: number): void;
  sin_f32(a: number, c: number, len: number): void;
  sum_f64(a: number, len: number): number;
  sum_f32(a: number, len: number): number;
  matmul_f64(a: number, b: number, c: number, m: number, n: number, k: number): void;
  matmul_f32(a: number, b: number, c: number, m: number, n: number, k: number): void;

  // Multi-threaded operations (may be same as single-threaded for some builds)
  add_f64_mt?(a: number, b: number, c: number, len: number): void;
  add_f32_mt?(a: number, b: number, c: number, len: number): void;
  sin_f64_mt?(a: number, c: number, len: number): void;
  sin_f32_mt?(a: number, c: number, len: number): void;
  sum_f64_mt?(a: number, len: number): number;
  sum_f32_mt?(a: number, len: number): number;
  matmul_f64_mt?(a: number, b: number, c: number, m: number, n: number, k: number): void;
  matmul_f32_mt?(a: number, b: number, c: number, m: number, n: number, k: number): void;
  init_thread_pool?(num_threads: number): void;

  // Blocked matmul (optional, used by Zig)
  matmul_blocked_f64?(a: number, b: number, c: number, m: number, n: number, k: number): void;
  matmul_blocked_f32?(a: number, b: number, c: number, m: number, n: number, k: number): void;
}

export interface WasmModule {
  exports: WasmExports;
  name: string;
  hasThreads: boolean;
}

/**
 * Load a WASM module from file
 */
export async function loadWasmModule(
  wasmPath: string,
  name: string,
  hasThreads: boolean = false
): Promise<WasmModule | null> {
  try {
    const wasmBuffer = await readFile(wasmPath);

    // Create memory (shared if threads enabled)
    const memory = new WebAssembly.Memory({
      initial: 256, // 16MB
      maximum: 4096, // 256MB
      shared: hasThreads,
    });

    const importObject = {
      env: {
        memory,
      },
      wasi_snapshot_preview1: {
        // Minimal WASI stubs for Zig
        fd_write: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0,
        proc_exit: () => {},
      },
    };

    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const instance = await WebAssembly.instantiate(wasmModule, importObject);

    const exports = instance.exports as unknown as WasmExports;

    // Use module's memory if it exports one
    if (exports.memory) {
      // Module uses its own memory
    }

    return {
      exports,
      name,
      hasThreads,
    };
  } catch (error) {
    console.warn(`Failed to load WASM module ${name} from ${wasmPath}:`, error);
    return null;
  }
}

/**
 * High-level WASM operations wrapper
 */
export class WasmOps {
  private exports: WasmExports;
  private memory: WebAssembly.Memory;
  public name: string;
  public hasThreads: boolean;

  constructor(module: WasmModule) {
    this.exports = module.exports;
    this.memory = module.exports.memory;
    this.name = module.name;
    this.hasThreads = module.hasThreads;
  }

  /**
   * Copy TypedArray to WASM memory and return pointer
   */
  copyToWasm(data: Float64Array | Float32Array): { ptr: number; len: number } {
    const len = data.length;
    const isF64 = data instanceof Float64Array;

    const ptr = isF64 ? this.exports.alloc_f64(len) : this.exports.alloc_f32(len);

    if (ptr === 0) {
      throw new Error('WASM allocation failed');
    }

    // Copy data to WASM memory
    const wasmArray = isF64
      ? new Float64Array(this.memory.buffer, ptr, len)
      : new Float32Array(this.memory.buffer, ptr, len);

    wasmArray.set(data);

    return { ptr, len };
  }

  /**
   * Allocate array in WASM memory
   */
  alloc(len: number, dtype: 'float32' | 'float64'): { ptr: number; len: number } {
    const ptr =
      dtype === 'float64' ? this.exports.alloc_f64(len) : this.exports.alloc_f32(len);

    if (ptr === 0) {
      throw new Error('WASM allocation failed');
    }

    return { ptr, len };
  }

  /**
   * Get a direct view into WASM memory (no copy)
   * WARNING: This view becomes invalid if WASM memory grows
   */
  getView(
    ptr: number,
    len: number,
    dtype: 'float32' | 'float64'
  ): Float64Array | Float32Array {
    return dtype === 'float64'
      ? new Float64Array(this.memory.buffer, ptr, len)
      : new Float32Array(this.memory.buffer, ptr, len);
  }

  /**
   * Copy data from WASM memory to TypedArray
   */
  copyFromWasm(
    ptr: number,
    len: number,
    dtype: 'float32' | 'float64'
  ): Float64Array | Float32Array {
    const result =
      dtype === 'float64'
        ? new Float64Array(this.memory.buffer, ptr, len)
        : new Float32Array(this.memory.buffer, ptr, len);

    // Return a copy (not a view)
    return dtype === 'float64'
      ? new Float64Array(result)
      : new Float32Array(result);
  }

  /**
   * Free WASM memory
   */
  free(ptr: number, len: number, dtype: 'float32' | 'float64'): void {
    if (dtype === 'float64') {
      this.exports.free_f64(ptr, len);
    } else {
      this.exports.free_f32(ptr, len);
    }
  }

  /**
   * Element-wise add
   */
  add(
    aPtr: number,
    bPtr: number,
    cPtr: number,
    len: number,
    dtype: 'float32' | 'float64',
    multithreaded: boolean = false
  ): void {
    if (dtype === 'float64') {
      if (multithreaded && this.exports.add_f64_mt) {
        this.exports.add_f64_mt(aPtr, bPtr, cPtr, len);
      } else {
        this.exports.add_f64(aPtr, bPtr, cPtr, len);
      }
    } else {
      if (multithreaded && this.exports.add_f32_mt) {
        this.exports.add_f32_mt(aPtr, bPtr, cPtr, len);
      } else {
        this.exports.add_f32(aPtr, bPtr, cPtr, len);
      }
    }
  }

  /**
   * Element-wise sin
   */
  sin(
    aPtr: number,
    cPtr: number,
    len: number,
    dtype: 'float32' | 'float64',
    multithreaded: boolean = false
  ): void {
    if (dtype === 'float64') {
      if (multithreaded && this.exports.sin_f64_mt) {
        this.exports.sin_f64_mt(aPtr, cPtr, len);
      } else {
        this.exports.sin_f64(aPtr, cPtr, len);
      }
    } else {
      if (multithreaded && this.exports.sin_f32_mt) {
        this.exports.sin_f32_mt(aPtr, cPtr, len);
      } else {
        this.exports.sin_f32(aPtr, cPtr, len);
      }
    }
  }

  /**
   * Sum reduction
   */
  sum(
    aPtr: number,
    len: number,
    dtype: 'float32' | 'float64',
    multithreaded: boolean = false
  ): number {
    if (dtype === 'float64') {
      if (multithreaded && this.exports.sum_f64_mt) {
        return this.exports.sum_f64_mt(aPtr, len);
      } else {
        return this.exports.sum_f64(aPtr, len);
      }
    } else {
      if (multithreaded && this.exports.sum_f32_mt) {
        return this.exports.sum_f32_mt(aPtr, len);
      } else {
        return this.exports.sum_f32(aPtr, len);
      }
    }
  }

  /**
   * Matrix multiplication
   */
  matmul(
    aPtr: number,
    bPtr: number,
    cPtr: number,
    m: number,
    n: number,
    k: number,
    dtype: 'float32' | 'float64',
    multithreaded: boolean = false
  ): void {
    if (dtype === 'float64') {
      if (multithreaded && this.exports.matmul_f64_mt) {
        this.exports.matmul_f64_mt(aPtr, bPtr, cPtr, m, n, k);
      } else {
        this.exports.matmul_f64(aPtr, bPtr, cPtr, m, n, k);
      }
    } else {
      if (multithreaded && this.exports.matmul_f32_mt) {
        this.exports.matmul_f32_mt(aPtr, bPtr, cPtr, m, n, k);
      } else {
        this.exports.matmul_f32(aPtr, bPtr, cPtr, m, n, k);
      }
    }
  }

  /**
   * Initialize thread pool
   */
  initThreadPool(numThreads: number): void {
    if (this.exports.init_thread_pool) {
      this.exports.init_thread_pool(numThreads);
    }
  }
}
