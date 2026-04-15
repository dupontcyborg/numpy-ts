/**
 * WASM-accelerated packbits / unpackbits (1D, big-endian bit order, uint8 only).
 *
 * Returns null if WASM can't handle the case (non-contiguous, too small, wrong dtype).
 */

import { packbits_u8, unpackbits_u8 } from './bins/packbits.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

/**
 * WASM-accelerated packbits for 1D contiguous uint8 arrays, big-endian bit order.
 * Returns null if WASM can't handle.
 */
export function wasmPackbits(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;
  if (a.dtype !== 'uint8' && a.dtype !== 'bool') return null;

  const packedSize = Math.ceil(size / 8);
  const outRegion = wasmMalloc(packedSize);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 1);

  packbits_u8(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    [packedSize],
    'uint8',
    outRegion,
    packedSize,
    Uint8Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}

/**
 * WASM-accelerated unpackbits for 1D contiguous uint8 arrays, big-endian bit order.
 * count is the desired output length.
 * Returns null if WASM can't handle.
 */
export function wasmUnpackbits(a: ArrayStorage, count: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.dtype !== 'uint8') return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const outRegion = wasmMalloc(count);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, 1);

  unpackbits_u8(aPtr, outRegion.ptr, size, count);

  return ArrayStorage.fromWasmRegion(
    [count],
    'uint8',
    outRegion,
    count,
    Uint8Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
