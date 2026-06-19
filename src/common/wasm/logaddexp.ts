/**
 * WASM-accelerated element-wise logaddexp.
 *
 * Binary: out[i] = log(exp(a[i]) + exp(b[i]))  (same-shape contiguous arrays)
 * Scalar: out[i] = log(exp(a[i]) + exp(scalar))
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import { type DType, effectiveDType, hasFloat16, promoteDTypes, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import * as laeBase from './bins/logaddexp.wasm';
import * as laeRelaxed from './bins/logaddexp-relaxed.wasm';
import { wasmConfig } from './config';
import { useRelaxedKernels } from './detect';
import {
  f16InputToScratchF32,
  f32OutputToF16Region,
  resetScratchAllocator,
  resolveInputPtr,
  wasmMalloc,
} from './runtime';

const BASE_THRESHOLD = 32;

// Relaxed-SIMD selection: the float cores (expv/log1pv) lower their Horner
// chains to relaxed_madd in the -relaxed build. Selected once, lazily.
let _bins: typeof laeBase | null = null;
function bins(): typeof laeBase {
  _bins ??= useRelaxedKernels() ? laeRelaxed : laeBase;
  return _bins;
}

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: (...a) => bins().logaddexp_f64(...a),
  float32: (...a) => bins().logaddexp_f32(...a),
  float16: (...a) => bins().logaddexp_f32(...a),
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: (...a) => bins().logaddexp_scalar_f64(...a),
  float32: (...a) => bins().logaddexp_scalar_f32(...a),
  float16: (...a) => bins().logaddexp_scalar_f32(...a),
};

// Large int → f64 output (i32/u32/i64/u64 need f64 precision)
const largeIntBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: (...a) => bins().logaddexp_i64(...a),
  uint64: (...a) => bins().logaddexp_u64(...a),
  int32: (...a) => bins().logaddexp_i32(...a),
  uint32: (...a) => bins().logaddexp_u32(...a),
};

// Small int → f32 output (i8/u8/i16/u16 → f32, then optionally downcast to f16)
const smallIntBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int16: (...a) => bins().logaddexp_i16(...a),
  uint16: (...a) => bins().logaddexp_u16(...a),
  int8: (...a) => bins().logaddexp_i8(...a),
  uint8: (...a) => bins().logaddexp_u8(...a),
};

const largeIntScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: (...a) => bins().logaddexp_scalar_i64(...a),
  uint64: (...a) => bins().logaddexp_scalar_u64(...a),
  int32: (...a) => bins().logaddexp_scalar_i32(...a),
  uint32: (...a) => bins().logaddexp_scalar_u32(...a),
};

const smallIntScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int16: (...a) => bins().logaddexp_scalar_i16(...a),
  uint16: (...a) => bins().logaddexp_scalar_u16(...a),
  int8: (...a) => bins().logaddexp_scalar_i8(...a),
  uint8: (...a) => bins().logaddexp_scalar_u8(...a),
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * WASM-accelerated element-wise logaddexp of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmLogaddexp(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));

  // Float path
  const floatKernel = binaryKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();

    let aPtr: number;
    let bPtr: number;
    if (dtype === 'float16') {
      aPtr = f16InputToScratchF32(a, size);
      bPtr = f16InputToScratchF32(b, size);
    } else {
      aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
      bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
    }

    floatKernel(aPtr, bPtr, outRegion.ptr, size);

    if (dtype === 'float16') {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        dtype,
        f16Region,
        size,
        Float16Array as unknown as new (
          buf: ArrayBuffer,
          off: number,
          len: number,
        ) => TypedArray,
      );
    }

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntBinaryKernels[dtype];
  if (smallKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inputBpe);

    smallKernel(aPtr, bPtr, outRegion.ptr, size);

    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (
          buf: ArrayBuffer,
          off: number,
          len: number,
        ) => TypedArray,
      );
    }

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      outRegion,
      size,
      Float32Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  // Large int path: i32/u32/i64/u64 → f64 output
  const largeKernel = largeIntBinaryKernels[dtype];
  if (largeKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inputBpe);

    largeKernel(aPtr, bPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float64',
      outRegion,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  return null;
}

/**
 * WASM-accelerated element-wise logaddexp with scalar.
 * Returns null if WASM can't handle.
 */
export function wasmLogaddexpScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);

  // Float path
  const floatKernel = scalarKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();

    let aPtr: number;
    if (dtype === 'float16') {
      aPtr = f16InputToScratchF32(a, size);
    } else {
      aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    }

    floatKernel(aPtr, outRegion.ptr, size, scalar);

    if (dtype === 'float16') {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        dtype,
        f16Region,
        size,
        Float16Array as unknown as new (
          buf: ArrayBuffer,
          off: number,
          len: number,
        ) => TypedArray,
      );
    }

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntScalarKernels[dtype];
  if (smallKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    smallKernel(aPtr, outRegion.ptr, size, scalar);

    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (
          buf: ArrayBuffer,
          off: number,
          len: number,
        ) => TypedArray,
      );
    }

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      outRegion,
      size,
      Float32Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  // Large int path: i32/u32/i64/u64 → f64 output
  const largeKernel = largeIntScalarKernels[dtype];
  if (largeKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    largeKernel(aPtr, outRegion.ptr, size, scalar);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float64',
      outRegion,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number,
      ) => TypedArray,
    );
  }

  return null;
}
