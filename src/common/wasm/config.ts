/**
 * Global WASM configuration.
 *
 * Controls whether WASM kernels are used and their size thresholds.
 * Useful for testing (force WASM path with multiplier=0, force JS with Infinity).
 */

/**
 * Internal WASM memory configuration. Use configureWasm() to change these
 * before any array operations.
 */
export const wasmMemoryConfig = {
  /** Total WASM linear memory size in bytes. Default 256 MiB. */
  maxMemoryBytes: 256 * 1024 * 1024,
  /** Scratch region for bump-allocating JS-fallback copy-ins. Default 8 MiB. */
  scratchBytes: 8 * 1024 * 1024,
  /** When true, fall back to JS-backed TypedArrays when WASM memory is full. */
  fallbackToJS: true,
  /**
   * Warn once, on stderr, the first time the WASM heap cannot satisfy an
   * allocation.
   *
   * That moment silently changes the library's performance characteristics —
   * every later result is JS-backed, measurably slower, and the heap does not
   * recover within the process. Set false to silence it (the counter below still
   * increments).
   */
  warnOnHeapExhaustion: true,
  /**
   * Number of allocations that fell back to JS because the WASM heap was full.
   * Non-zero means the process crossed the cliff above; a test or benchmark can
   * assert on it. Never reset automatically.
   */
  heapExhaustedCount: 0,
};

export interface ConfigureWasmOptions {
  /** Total WASM linear memory in bytes. Default 256 MiB. */
  maxMemory?: number;
  /** Scratch region size in bytes. Default: 1/16 of maxMemory (~6% for temp kernel buffers). */
  scratchSize?: number;
}

export const wasmConfig = {
  /**
   * Multiplier applied to all WASM size thresholds.
   * - 1 (default): normal behavior, WASM used above optimal thresholds
   * - 0: always use WASM when structurally possible (for testing)
   * - Infinity: disable WASM entirely, always fall back to JS
   */
  thresholdMultiplier: 1,

  /**
   * Incremented each time a WASM kernel successfully executes.
   * Reset to 0 by callers (e.g. benchmark runner) to detect per-operation WASM usage.
   */
  wasmCallCount: 0,

  /**
   * Controls relaxed SIMD kernel usage (FMA: f32x4.relaxed_madd / f64x2.relaxed_madd).
   * - 'auto' (default): detect via WebAssembly.validate probe, use if supported
   * - true: force relaxed kernels (will fail on unsupported runtimes)
   * - false: force baseline kernels even if relaxed is available
   */
  useRelaxedSimd: 'auto' as 'auto' | boolean,
};
