/**
 * WASM-accelerated flattened cumulative scan: cumsum / cumprod.
 * Returns null if WASM can't handle it (complex/float16/bool, non-contiguous,
 * too small). Only the `axis === undefined` (flattened) case is supported.
 *
 * Output dtype follows reductionAccumDtype: small ints widen to int64/uint64,
 * floats are preserved. float16 and bool/complex fall back to JS.
 */

import { type DType, reductionAccumDtype, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  cumprod_f32,
  cumprod_f64,
  cumprod_i8,
  cumprod_i16,
  cumprod_i32,
  cumprod_i64,
  cumprod_u8,
  cumprod_u16,
  cumprod_u32,
  cumprod_u64,
  cumsum_f32,
  cumsum_f64,
  cumsum_i8,
  cumsum_i16,
  cumsum_i32,
  cumsum_i64,
  cumsum_u8,
  cumsum_u16,
  cumsum_u32,
  cumsum_u64,
} from './bins/cumulative.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 32;

type ScanFn = (aPtr: number, outPtr: number, N: number) => void;

// Input bytes-per-element for each supported dtype.
const inBpe: Partial<Record<DType, number>> = {
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

function run(
  a: ArrayStorage,
  cumsumKernels: Partial<Record<DType, ScanFn>>,
  cumprodKernels: Partial<Record<DType, ScanFn>>,
  cumprod: boolean,
): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const bpe = inBpe[dtype];
  if (!bpe) return null;

  const kernel = (cumprod ? cumprodKernels : cumsumKernels)[dtype];
  if (!kernel) return null;

  const accumDtype = reductionAccumDtype(dtype);
  // Only the flattened (axis === undefined) case is handled here, so the
  // result is 1-D [size] — NumPy flattens before scanning when axis is omitted.
  // int64/uint64/float64 → 8 bytes, float32 → 4 bytes.
  const outElemBytes = accumDtype === 'float32' ? 4 : 8;
  const outRegion = wasmMalloc(size * outElemBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

  kernel(aPtr, outRegion.ptr, size);

  let Ctor: new (b: ArrayBuffer, o: number, l: number) => TypedArray;
  switch (accumDtype) {
    case 'int64':
      Ctor = BigInt64Array as unknown as typeof Ctor;
      break;
    case 'uint64':
      Ctor = BigUint64Array as unknown as typeof Ctor;
      break;
    case 'float32':
      Ctor = Float32Array as unknown as typeof Ctor;
      break;
    default:
      Ctor = Float64Array as unknown as typeof Ctor;
  }

  return ArrayStorage.fromWasmRegion([size], accumDtype, outRegion, size, Ctor);
}

const cumsumKernels: Partial<Record<DType, ScanFn>> = {
  float64: cumsum_f64,
  float32: cumsum_f32,
  int8: cumsum_i8,
  int16: cumsum_i16,
  int32: cumsum_i32,
  int64: cumsum_i64,
  uint8: cumsum_u8,
  uint16: cumsum_u16,
  uint32: cumsum_u32,
  uint64: cumsum_u64,
};

const cumprodKernels: Partial<Record<DType, ScanFn>> = {
  float64: cumprod_f64,
  float32: cumprod_f32,
  int8: cumprod_i8,
  int16: cumprod_i16,
  int32: cumprod_i32,
  int64: cumprod_i64,
  uint8: cumprod_u8,
  uint16: cumprod_u16,
  uint32: cumprod_u32,
  uint64: cumprod_u64,
};

export function wasmCumsum(a: ArrayStorage): ArrayStorage | null {
  return run(a, cumsumKernels, cumprodKernels, false);
}

export function wasmCumprod(a: ArrayStorage): ArrayStorage | null {
  return run(a, cumsumKernels, cumprodKernels, true);
}
