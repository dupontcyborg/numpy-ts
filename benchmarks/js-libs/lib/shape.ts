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

/** Build a typed array matching a regime float dtype (else Float64Array). */
export function toTyped(d: ArrayData): Float64Array | Float32Array {
  return d.dtype === 'float32' ? Float32Array.from(d.data) : Float64Array.from(d.data);
}
