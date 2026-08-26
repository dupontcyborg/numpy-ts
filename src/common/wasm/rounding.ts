/**
 * WASM-accelerated element-wise rounding: floor, ceil, trunc, rint, around.
 *
 * Unary: out[i] = round(a[i]) for the relevant rounding mode. Integer dtypes
 * never reach here — `ops/rounding.ts` returns a copy for those, since an
 * integer is already rounded.
 *
 * Native float16 is deliberately left to JS: rounding needs f32 arithmetic, and
 * the convert-up/convert-back round-trip measured slower than the JS loop it
 * would replace (14.4us vs 8.5us for ceil at [100x100]), on a dtype that was
 * already several times faster than NumPy.
 *
 * Returns null if WASM can't handle the case (complex, non-contiguous, too small).
 */

import { type DType, effectiveDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  around_f32,
  around_f64,
  ceil_f32,
  ceil_f64,
  floor_f32,
  floor_f64,
  rint_f32,
  rint_f64,
  trunc_f32,
  trunc_f64,
} from './bins/rounding.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 32;

/** Rounding modes this module can accelerate. */
export type RoundingKind = 'floor' | 'ceil' | 'trunc' | 'rint';

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;
type ScaledFn = (aPtr: number, outPtr: number, N: number, m: number) => void;

const f64Kernels: Record<RoundingKind, UnaryFn> = {
  floor: floor_f64,
  ceil: ceil_f64,
  trunc: trunc_f64,
  rint: rint_f64,
};

const f32Kernels: Record<RoundingKind, UnaryFn> = {
  floor: floor_f32,
  ceil: ceil_f32,
  trunc: trunc_f32,
  rint: rint_f32,
};

/**
 * Run a rounding kernel over `a`.
 *
 * `multiplier`, when given, selects the scaled `around` kernels instead: the
 * value is rounded as `rint(x * m) / m`, matching what NumPy's `around` does for
 * a non-zero `decimals`. A multiplier of 1 is the same as plain `rint`, so
 * callers pass undefined for that and keep the cheaper kernel.
 */
function runRounding(
  a: ArrayStorage,
  kind: RoundingKind,
  multiplier?: number,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // effectiveDType maps float16 to float32 on runtimes without Float16Array, in
  // which case the f32 kernels below handle it natively.
  const dtype = effectiveDType(a.dtype);
  if (dtype !== 'float64' && dtype !== 'float32') return null;

  // Native float16 stays in JS. Rounding needs f32 arithmetic, so the kernel
  // would have to convert up and back, and measured at [100x100] that round-trip
  // costs more than the JS loop it replaces: 14.4us against 8.5us for ceil. The
  // JS path was already ~6x faster than NumPy here, so there is nothing to win.

  const scaled = multiplier !== undefined;
  if (scaled && kind !== 'rint') return null; // only `around` scales
  // Scaled rounding is float64-only on purpose. NumPy performs the multiply and
  // divide at the array's own precision: for float32, 2.675f * 100 is exactly
  // 267.5f and the tie rule yields 2.68, while widening to f64 gives
  // 267.4999952 and 2.67. float16 narrows further still (1.35 * 10 rounds to
  // 13.5 in f16, giving 1.4 rather than 1.3). This kernel computes f16 inputs in
  // f32, so it cannot reproduce either narrowing, and the JS fallback widens to
  // f64 — they would disagree either side of the 32-element threshold. f64 has
  // no narrowing to reproduce, so the two agree exactly there.
  //
  // `decimals === 0` does no scaling at all and is unaffected: callers route it
  // to the plain rint kernels, which are exact for every float dtype.
  if (scaled && dtype !== 'float64') return null;

  const isF64 = dtype === 'float64';
  const bpe = isF64 ? 8 : 4;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  if (scaled) {
    const k: ScaledFn = isF64 ? around_f64 : around_f32;
    k(aPtr, outRegion.ptr, size, multiplier);
  } else {
    (isF64 ? f64Kernels : f32Kernels)[kind](aPtr, outRegion.ptr, size);
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype as DType,
    outRegion,
    size,
    (isF64 ? Float64Array : Float32Array) as unknown as new (
      buf: ArrayBuffer,
      off: number,
      len: number,
    ) => TypedArray,
  );
}

/** out[i] = floor(a[i]) */
export function wasmFloor(a: ArrayStorage): ArrayStorage | null {
  return runRounding(a, 'floor');
}

/** out[i] = ceil(a[i]) */
export function wasmCeil(a: ArrayStorage): ArrayStorage | null {
  return runRounding(a, 'ceil');
}

/** out[i] = trunc(a[i]) — also backs `fix`, which is the same operation. */
export function wasmTrunc(a: ArrayStorage): ArrayStorage | null {
  return runRounding(a, 'trunc');
}

/** out[i] = rint(a[i]), round-half-to-even. */
export function wasmRint(a: ArrayStorage): ArrayStorage | null {
  return runRounding(a, 'rint');
}

/**
 * out[i] = rint(a[i] * m) / m, the `around`/`round` family.
 *
 * Pass the multiplier the caller would have used in JS (10 ** decimals). For
 * decimals === 0 prefer {@link wasmRint}: it is the same result without the
 * redundant multiply and divide.
 */
export function wasmAround(a: ArrayStorage, multiplier: number): ArrayStorage | null {
  return runRounding(a, 'rint', multiplier);
}
