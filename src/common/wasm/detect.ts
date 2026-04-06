/**
 * Relaxed SIMD feature detection.
 *
 * Probes for relaxed SIMD support using WebAssembly.validate() with a
 * minimal module containing f32x4.relaxed_madd. Sync, cached, zero cost
 * after first call.
 */

import { wasmConfig } from './config';

// Minimal WASM module containing f32x4.relaxed_madd (37 bytes).
// Generated from: (module (func (param v128 v128 v128) (result v128)
//   local.get 0 local.get 1 local.get 2 f32x4.relaxed_madd))
// prettier-ignore
const RELAXED_SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x01, 0x60,
  0x03, 0x7b, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0d,
  0x01, 0x0b, 0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0x85, 0x02,
  0x0b,
]);

let _supported: boolean | null = null;

/** Returns true if the runtime supports WASM relaxed SIMD. Cached after first call. */
export function supportsRelaxedSimd(): boolean {
  if (_supported === null) {
    try {
      _supported = typeof WebAssembly !== 'undefined' && WebAssembly.validate(RELAXED_SIMD_PROBE);
    } catch {
      _supported = false;
    }
  }
  return _supported;
}

/**
 * Returns true if relaxed SIMD kernels should be used, respecting config override.
 * - 'auto' (default): detect via probe
 * - true: force relaxed (will fail instantiation on unsupported runtimes)
 * - false: force baseline
 */
export function useRelaxedKernels(): boolean {
  const cfg = wasmConfig.useRelaxedSimd;
  if (cfg === true) return true;
  if (cfg === false) return false;
  return supportsRelaxedSimd();
}
