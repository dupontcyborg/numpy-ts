/**
 * WASM-accelerated element-wise sinc: sin(πx)/(πx), sinc(0)=1.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 * Result dtype: float16/float32 preserved; all other types → float64
 * (NumPy promotes every integer dtype to float64 for sinc).
 */

import { type DType, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  sinc_f32,
  sinc_f64,
  sinc_i8,
  sinc_i16,
  sinc_i32,
  sinc_i64,
  sinc_u8,
  sinc_u16,
  sinc_u32,
  sinc_u64,
} from './bins/sinc.wasm';
import { wasmConfig } from './config';
import {
  f16InputToScratchF32,
  f32OutputToF16Region,
  resetScratchAllocator,
  resolveInputPtr,
  wasmMalloc,
} from './runtime';

const BASE_THRESHOLD = 32;

type UnaryFn = (xPtr: number, outPtr: number, N: number) => void;

// Same-dtype-ish float kernels.
const floatKernels: Partial<Record<DType, UnaryFn>> = {
  float64: sinc_f64,
  float32: sinc_f32,
  float16: sinc_f32,
};

// Integer kernels → float64 output.
const intKernels: Partial<Record<DType, UnaryFn>> = {
  int64: sinc_i64,
  uint64: sinc_u64,
  int32: sinc_i32,
  uint32: sinc_u32,
  int16: sinc_i16,
  uint16: sinc_u16,
  int8: sinc_i8,
  uint8: sinc_u8,
};

const inBpe: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  float16: 2,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

export function wasmSinc(x: ArrayStorage): ArrayStorage | null {
  if (!x.isCContiguous) return null;
  const size = x.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = x.dtype;
  const bpe = inBpe[dtype];
  if (!bpe) return null;

  // Float path (float64/float32 same dtype; float16 computes in f32 then downcasts).
  const floatKernel = floatKernels[dtype];
  if (floatKernel) {
    const outElemBytes = dtype === 'float64' ? 8 : 4;
    const outRegion = wasmMalloc(size * outElemBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const xPtr =
      dtype === 'float16'
        ? f16InputToScratchF32(x, size)
        : resolveInputPtr(x.data, x.isWasmBacked, x.wasmPtr, x.offset, size, bpe);

    floatKernel(xPtr, outRegion.ptr, size);

    if (dtype === 'float16') {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(x.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray,
      );
    }
    const Ctor = (dtype === 'float64' ? Float64Array : Float32Array) as unknown as new (
      b: ArrayBuffer,
      o: number,
      l: number,
    ) => TypedArray;
    return ArrayStorage.fromWasmRegion(Array.from(x.shape), dtype, outRegion, size, Ctor);
  }

  // Integer path → float64 output.
  const intKernel = intKernels[dtype];
  if (!intKernel) return null;

  const outRegion = wasmMalloc(size * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const xPtr = resolveInputPtr(x.data, x.isWasmBacked, x.wasmPtr, x.offset, size, bpe);

  intKernel(xPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(x.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray,
  );
}
