/**
 * Global WASM configuration.
 *
 * Controls whether WASM kernels are used and their size thresholds.
 * Useful for testing (force WASM path with multiplier=0, force JS with Infinity).
 */

export const wasmConfig = {
  /**
   * Multiplier applied to all WASM size thresholds.
   * - 1 (default): normal behavior, WASM used above optimal thresholds
   * - 0: always use WASM when structurally possible (for testing)
   * - Infinity: disable WASM entirely, always fall back to JS
   */
  thresholdMultiplier: 1,
};
