/**
 * WASM-accelerated fused divmod (scalar divisor).
 *
 * Computes floor quotient and floor remainder in a single pass.
 * Returns [quotient, remainder] or null if WASM can't handle.
 */

import {
  divmod_scalar_f64,
  divmod_scalar_f32,
  divmod_scalar_i64,
  divmod_scalar_u64,
  divmod_scalar_i32,
  divmod_scalar_u32,
  divmod_scalar_i16,
  divmod_scalar_u16,
  divmod_scalar_i8,
  divmod_scalar_u8,
} from './bins/divmod.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type DivmodScalarFn = (aPtr: number, outQ: number, outR: number, N: number, scalar: number) => void;

// Float scalar kernels (same-dtype in, same-dtype out)
const floatKernels: Partial<Record<DType, DivmodScalarFn>> = {
  float64: divmod_scalar_f64,
  float32: divmod_scalar_f32,
};

// Integer scalar kernels (same-dtype in/out)
const intKernels: Partial<Record<DType, DivmodScalarFn>> = {
  int64: divmod_scalar_i64,
  uint64: divmod_scalar_u64,
  int32: divmod_scalar_i32,
  uint32: divmod_scalar_u32,
  int16: divmod_scalar_i16,
  uint16: divmod_scalar_u16,
  int8: divmod_scalar_i8,
  uint8: divmod_scalar_u8,
};

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

/**
 * Fused divmod with scalar divisor.
 * Returns [quotient, remainder] or null if WASM can't handle.
 */
export function wasmDivmodScalar(
  a: ArrayStorage,
  scalar: number
): [ArrayStorage, ArrayStorage] | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // Float path — same dtype in/out
  const floatKernel = floatKernels[dtype];
  if (floatKernel) {
    const bpe = bpeMap[dtype]!;
    const outBytes = size * bpe;
    const Ctor = bpe === 4 ? Float32Array : Float64Array;

    const qRegion = wasmMalloc(outBytes);
    if (!qRegion) return null;
    const rRegion = wasmMalloc(outBytes);
    if (!rRegion) {
      qRegion.release();
      return null;
    }

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

    floatKernel(aPtr, qRegion.ptr, rRegion.ptr, size, scalar);

    const shape = Array.from(a.shape);
    const CtorT = Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray;
    return [
      ArrayStorage.fromWasmRegion(shape, dtype, qRegion, size, CtorT),
      ArrayStorage.fromWasmRegion(shape, dtype, rRegion, size, CtorT),
    ];
  }

  // Integer path — same-dtype in/out
  const intKernel = intKernels[dtype];
  const inBpe = bpeMap[dtype];
  if (!intKernel || !inBpe) return null;

  const intCtorMap: Partial<Record<DType, new (length: number) => TypedArray>> = {
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
  };
  const IntCtor = intCtorMap[dtype];
  if (!IntCtor) return null;

  const outBytes = size * inBpe;
  const qRegion = wasmMalloc(outBytes);
  if (!qRegion) return null;
  const rRegion = wasmMalloc(outBytes);
  if (!rRegion) {
    qRegion.release();
    return null;
  }

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

  intKernel(aPtr, qRegion.ptr, rRegion.ptr, size, scalar);

  const shape = Array.from(a.shape);
  const CtorT = IntCtor as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray;
  return [
    ArrayStorage.fromWasmRegion(shape, dtype, qRegion, size, CtorT),
    ArrayStorage.fromWasmRegion(shape, dtype, rRegion, size, CtorT),
  ];
}
