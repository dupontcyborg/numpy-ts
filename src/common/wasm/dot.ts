/**
 * WASM-accelerated 1D dot product.
 *
 * Computes sum_k a[k] * b[k] for contiguous 1D arrays.
 * Returns null if WASM can't handle this case.
 */

import * as floatBase from './bins/dot_float.wasm';
import * as floatRelaxed from './bins/dot_float-relaxed.wasm';
import { dot_i64, dot_i32, dot_i16, dot_i8 } from './bins/dot_int.wasm';
import { useRelaxedKernels } from './detect';
import { resetScratchAllocator, resolveInputPtr, scratchAlloc, getSharedMemory } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { Complex } from '../complex';

import { wasmConfig } from './config';

let _float: typeof floatBase | null = null;
function float(): typeof floatBase {
  return (_float ??= useRelaxedKernels() ? floatRelaxed : floatBase);
}

const BASE_THRESHOLD = 32; // Minimum K for WASM to be worth it

type WasmDotFn = (aPtr: number, bPtr: number, outPtr: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmDotFn>> = {
  float64: (...a) => float().dot_f64(...a),
  float32: (...a) => float().dot_f32(...a),
  // float16 excluded: NumPy accumulates in f16 precision (overflows to inf), f32 WASM gives different results
  complex128: (...a) => float().dot_c128(...a),
  complex64: (...a) => float().dot_c64(...a),
  int64: dot_i64,
  uint64: dot_i64,
  int32: dot_i32,
  uint32: dot_i32,
  int16: dot_i16,
  uint16: dot_i16,
  int8: dot_i8,
  uint8: dot_i8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  // float16: excluded (see above)
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated 1D dot product. Returns null if WASM can't handle.
 * Both a and b must be 1D, contiguous, same-length.
 */
export function wasmDot1D(a: ArrayStorage, b: ArrayStorage): number | Complex | null {
  if (a.ndim !== 1 || b.ndim !== 1) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const K = a.shape[0]!;
  if (K !== b.shape[0]!) return null;
  if (K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;

  const bytesPerElement = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = 1 * factor * bytesPerElement;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    K * factor,
    bytesPerElement
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    K * factor,
    bytesPerElement
  );
  const outPtr = scratchAlloc(outBytes);

  kernel(aPtr, bPtr, outPtr, K);

  // Read scalar result directly from WASM memory
  const mem = getSharedMemory();
  const outView = new (Ctor as unknown as new (
    buffer: ArrayBuffer,
    byteOffset: number,
    length: number
  ) => TypedArray)(mem.buffer, outPtr, 1 * factor);

  // Complex scalar: read re + im from the 2-element output buffer
  if (factor === 2) {
    return new Complex(
      Number((outView as Float64Array | Float32Array)[0]!),
      Number((outView as Float64Array | Float32Array)[1]!)
    );
  }

  return Number(outView[0]!);
}
