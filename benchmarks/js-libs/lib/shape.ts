/** Reshape a row-major flat array into nested 1-D / 2-D JS arrays. */

import type { ArrayData } from './types';

export function toNested(d: ArrayData): number[] | number[][] {
  if (d.shape.length <= 1) return d.data;
  if (d.shape.length === 2) {
    const [rows, cols] = d.shape as [number, number];
    const out: number[][] = new Array(rows);
    for (let r = 0; r < rows; r++) out[r] = d.data.slice(r * cols, (r + 1) * cols);
    return out;
  }
  throw new Error(`unsupported rank ${d.shape.length}`);
}

export function is2D(d: ArrayData): boolean {
  return d.shape.length === 2;
}

export function is1D(d: ArrayData): boolean {
  return d.shape.length <= 1;
}

const TYPED: Record<string, { from(a: number[]): ArrayLike<number> }> = {
  float64: Float64Array, float32: Float32Array,
  int32: Int32Array, int16: Int16Array, int8: Int8Array,
  uint32: Uint32Array, uint16: Uint16Array, uint8: Uint8Array,
  bool: Uint8Array,
};

/** Build a typed array matching the dtype (Float64Array fallback). */
export function toTyped(d: ArrayData): ArrayLike<number> {
  return (TYPED[d.dtype] ?? Float64Array).from(d.data);
}

/** tfjs dtype name for a numpy dtype (tfjs storage is float32/int32/bool/complex64). */
export function tfjsDtype(dtype: string): 'float32' | 'int32' | 'bool' {
  if (dtype === 'bool') return 'bool';
  if (dtype.startsWith('int') || dtype.startsWith('uint')) return 'int32';
  return 'float32';
}

/** @d4c/numjs dtype name (no 64-bit ints, no bool). */
export function numjsDtype(dtype: string): string {
  return dtype === 'float32' ? 'float32'
    : dtype.startsWith('int') || dtype.startsWith('uint') ? dtype
    : 'float64';
}
