/**
 * WASM-accelerated modulo family with a scalar divisor: mod (floor remainder),
 * floor_divide, and fmod (truncated remainder). Same dtype in/out.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */

import { type DType, isComplexDType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  floordiv_f32,
  floordiv_f64,
  floordiv_i8,
  floordiv_i16,
  floordiv_i32,
  floordiv_i64,
  floordiv_scalar_f32,
  floordiv_scalar_f64,
  floordiv_scalar_i8,
  floordiv_scalar_i16,
  floordiv_scalar_i32,
  floordiv_scalar_i64,
  floordiv_scalar_u8,
  floordiv_scalar_u16,
  floordiv_scalar_u32,
  floordiv_scalar_u64,
  floordiv_u8,
  floordiv_u16,
  floordiv_u32,
  floordiv_u64,
  fmod_f32,
  fmod_f64,
  fmod_i8,
  fmod_i16,
  fmod_i32,
  fmod_i64,
  fmod_scalar_f32,
  fmod_scalar_f64,
  fmod_scalar_i8,
  fmod_scalar_i16,
  fmod_scalar_i32,
  fmod_scalar_i64,
  fmod_scalar_u8,
  fmod_scalar_u16,
  fmod_scalar_u32,
  fmod_scalar_u64,
  fmod_u8,
  fmod_u16,
  fmod_u32,
  fmod_u64,
  mod_f32,
  mod_f64,
  mod_i8,
  mod_i16,
  mod_i32,
  mod_i64,
  mod_scalar_f32,
  mod_scalar_f64,
  mod_scalar_i8,
  mod_scalar_i16,
  mod_scalar_i32,
  mod_scalar_i64,
  mod_scalar_u8,
  mod_scalar_u16,
  mod_scalar_u32,
  mod_scalar_u64,
  mod_u8,
  mod_u16,
  mod_u32,
  mod_u64,
} from './bins/modulo.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 32;

type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

const ctorMap: Partial<Record<DType, new (length: number) => TypedArray>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const modKernels: Partial<Record<DType, ScalarFn>> = {
  float64: mod_scalar_f64,
  float32: mod_scalar_f32,
  int64: mod_scalar_i64,
  uint64: mod_scalar_u64,
  int32: mod_scalar_i32,
  uint32: mod_scalar_u32,
  int16: mod_scalar_i16,
  uint16: mod_scalar_u16,
  int8: mod_scalar_i8,
  uint8: mod_scalar_u8,
};

const floorDivKernels: Partial<Record<DType, ScalarFn>> = {
  float64: floordiv_scalar_f64,
  float32: floordiv_scalar_f32,
  int64: floordiv_scalar_i64,
  uint64: floordiv_scalar_u64,
  int32: floordiv_scalar_i32,
  uint32: floordiv_scalar_u32,
  int16: floordiv_scalar_i16,
  uint16: floordiv_scalar_u16,
  int8: floordiv_scalar_i8,
  uint8: floordiv_scalar_u8,
};

const fmodKernels: Partial<Record<DType, ScalarFn>> = {
  float64: fmod_scalar_f64,
  float32: fmod_scalar_f32,
  int64: fmod_scalar_i64,
  uint64: fmod_scalar_u64,
  int32: fmod_scalar_i32,
  uint32: fmod_scalar_u32,
  int16: fmod_scalar_i16,
  uint16: fmod_scalar_u16,
  int8: fmod_scalar_i8,
  uint8: fmod_scalar_u8,
};

function runScalar(
  a: ArrayStorage,
  scalar: number,
  kernels: Partial<Record<DType, ScalarFn>>,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  const kernel = kernels[dtype];
  const bpe = bpeMap[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !bpe || !Ctor) return null;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  kernel(aPtr, outRegion.ptr, size, scalar);

  const CtorT = Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray;
  return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, outRegion, size, CtorT);
}

/** mod / remainder with a scalar divisor (floor modulo, sign of divisor). */
export function wasmModScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  return runScalar(a, scalar, modKernels);
}

/** floor_divide with a scalar divisor. */
export function wasmFloorDivScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  return runScalar(a, scalar, floorDivKernels);
}

/** fmod with a scalar divisor (truncated, sign of dividend). */
export function wasmFmodScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  return runScalar(a, scalar, fmodKernels);
}

type ArrayFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;

const modArrKernels: Partial<Record<DType, ArrayFn>> = {
  float64: mod_f64,
  float32: mod_f32,
  int64: mod_i64,
  uint64: mod_u64,
  int32: mod_i32,
  uint32: mod_u32,
  int16: mod_i16,
  uint16: mod_u16,
  int8: mod_i8,
  uint8: mod_u8,
};

const floorDivArrKernels: Partial<Record<DType, ArrayFn>> = {
  float64: floordiv_f64,
  float32: floordiv_f32,
  int64: floordiv_i64,
  uint64: floordiv_u64,
  int32: floordiv_i32,
  uint32: floordiv_u32,
  int16: floordiv_i16,
  uint16: floordiv_u16,
  int8: floordiv_i8,
  uint8: floordiv_u8,
};

const fmodArrKernels: Partial<Record<DType, ArrayFn>> = {
  float64: fmod_f64,
  float32: fmod_f32,
  int64: fmod_i64,
  uint64: fmod_u64,
  int32: fmod_i32,
  uint32: fmod_u32,
  int16: fmod_i16,
  uint16: fmod_u16,
  int8: fmod_i8,
  uint8: fmod_u8,
};

/**
 * Array / array form. Requires matching dtype and shape; anything else falls
 * back to JS. Same-dtype-in/out, so no promotion happens here.
 */
function runArray(
  a: ArrayStorage,
  b: ArrayStorage,
  kernels: Partial<Record<DType, ArrayFn>>,
): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  if (a.dtype !== b.dtype) return null;
  if (a.size !== b.size) return null;
  if (a.shape.length !== b.shape.length) return null;
  for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  const kernel = kernels[dtype];
  const bpe = bpeMap[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !bpe || !Ctor) return null;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);

  kernel(aPtr, bPtr, outRegion.ptr, size);

  const CtorT = Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray;
  return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, outRegion, size, CtorT);
}

/** mod / remainder, array divisor (floor modulo, sign of divisor). */
export function wasmMod(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  return runArray(a, b, modArrKernels);
}

/** floor_divide, array divisor. */
export function wasmFloorDiv(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  return runArray(a, b, floorDivArrKernels);
}

/** fmod, array divisor (truncated, sign of dividend). */
export function wasmFmod(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  return runArray(a, b, fmodArrKernels);
}
