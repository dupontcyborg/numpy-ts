/**
 * WASM-accelerated 2D tile.
 *
 * Tiles a [rows x cols] matrix by [rep_rows x rep_cols].
 * Returns null if WASM can't handle this case.
 */

import {
  tile_2d_f64,
  tile_2d_f32,
  tile_2d_i64,
  tile_2d_i32,
  tile_2d_i16,
  tile_2d_i8,
} from './bins/tile.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type Tile2DFn = (
  aPtr: number,
  outPtr: number,
  rows: number,
  cols: number,
  repRows: number,
  repCols: number
) => void;

const kernels: Partial<Record<DType, Tile2DFn>> = {
  float64: tile_2d_f64,
  float32: tile_2d_f32,
  int64: tile_2d_i64,
  uint64: tile_2d_i64,
  int32: tile_2d_i32,
  uint32: tile_2d_i32,
  int16: tile_2d_i16,
  uint16: tile_2d_i16,
  int8: tile_2d_i8,
  uint8: tile_2d_i8,
  float16: tile_2d_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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
  float16: Float32Array,
};

/**
 * WASM-accelerated 2D tile.
 * Only handles 2D C-contiguous arrays with 2-element reps.
 * Returns null if WASM can't handle.
 */
export function wasmTile2D(a: ArrayStorage, repRows: number, repCols: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.shape.length !== 2) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const rows = a.shape[0]!;
  const cols = a.shape[1]!;
  const outSize = rows * repRows * cols * repCols;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = outSize * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Float16: tile as raw i16 bytes (same 2-byte layout, no conversion needed)
  if (isF16) {
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 2);
    tile_2d_i16(aPtr, outRegion.ptr, rows, cols, repRows, repCols);
    return ArrayStorage.fromWasmRegion(
      [rows * repRows, cols * repCols],
      dtype,
      outRegion,
      outSize,
      Float16Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, rows, cols, repRows, repCols);

  return ArrayStorage.fromWasmRegion(
    [rows * repRows, cols * repCols],
    dtype,
    outRegion,
    outSize,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
