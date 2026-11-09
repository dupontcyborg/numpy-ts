/**
 * Shared helpers for oracle tests that sweep every dtype: seeded random samples that are built
 * identically in numpy-ts and in Python, and a comparison that checks dtype, shape and values.
 */

import { expect } from 'vitest';
import { array, type NDArray } from '../../src';
import { Complex } from '../../src/common/complex';
import {
  DTYPES as ALL_DTYPES,
  type DType,
  isComplexDType,
  isFloatDType,
} from '../../src/common/dtype';
import type { NumPyResult } from './numpy-oracle';

export const DTYPES: readonly DType[] = ALL_DTYPES;

export function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

/** True for a mixed-dtype pair that a sweep should skip, to bound the oracle calls; same-dtype pairs are always kept. */
export function skipMixed(rng: () => number, a: DType, b: DType, keep: number): boolean {
  return a !== b && rng() > keep;
}

export interface Sample {
  re: number[];
  im?: number[];
  shape: number[];
}

const FLOAT_POOL = [-2, -1, 0, 0, 1, 1, 2, 3, 0.5, Number.NaN, Infinity, -Infinity, -0];
const IM_POOL = [0, 0, 1, -1, 2];
const BYTE_POOL = [-3, -2, -1, 0, 0, 1, 1, 2, 3, 4];
const UBYTE_POOL = [0, 0, 1, 1, 2, 3, 4, 5, 6];
// Wide pools put integer spans past NumPy's table method limit, so the sort method is reached.
const WIDE_SIGNED = [-30000, -7, -1, 0, 1, 5, 30000, 12345];
const WIDE_UNSIGNED = [0, 1, 5, 7, 12345, 30000, 60000];

function pool(dtype: DType, wide: boolean): number[] {
  if (dtype === 'bool') return [0, 1];
  if (isFloatDType(dtype) || isComplexDType(dtype)) return FLOAT_POOL;
  const unsigned = dtype.startsWith('uint');
  if (dtype === 'int8' || dtype === 'uint8' || !wide) return unsigned ? UBYTE_POOL : BYTE_POOL;
  return unsigned ? WIDE_UNSIGNED : WIDE_SIGNED;
}

function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Random values from a small pool so that two samples overlap and repeat. */
export function sample(
  rng: () => number,
  dtype: DType,
  shape: number[],
  opts: { wide?: boolean; nan?: boolean; finite?: boolean } = {},
): Sample {
  const base = pool(dtype, opts.wide ?? false);
  const choices = opts.finite
    ? base.filter((v) => Number.isFinite(v))
    : opts.nan === false
      ? base.filter((v) => !Number.isNaN(v))
      : base;
  const n = product(shape);
  const re: number[] = [];
  const im: number[] | undefined = isComplexDType(dtype) ? [] : undefined;
  for (let i = 0; i < n; i++) {
    const r = choices[Math.floor(rng() * choices.length)]!;
    re.push(r);
    // A NaN real part always pairs with a zero imaginary part: NumPy orders NaNs of complex
    // values by their imaginary part, which no caller here needs to depend on.
    if (im) im.push(Number.isNaN(r) ? 0 : IM_POOL[Math.floor(rng() * IM_POOL.length)]!);
  }
  return { re, im, shape };
}

/** Distinct values (a random subset of the pool, each at most once), for assume_unique inputs. */
export function uniqueSample(rng: () => number, dtype: DType, n: number, wide = false): Sample {
  const seen = new Set<string>();
  const re: number[] = [];
  const im: number[] | undefined = isComplexDType(dtype) ? [] : undefined;
  const base = pool(dtype, wide);
  for (let tries = 0; tries < 200 && re.length < n; tries++) {
    const r = base[Math.floor(rng() * base.length)]!;
    const i = im ? (Number.isNaN(r) ? 0 : IM_POOL[Math.floor(rng() * IM_POOL.length)]!) : 0;
    const key = `${r === 0 ? 0 : r},${i}`;
    if (seen.has(key)) continue;
    seen.add(key);
    re.push(r);
    im?.push(i);
  }
  return { re, im, shape: [re.length] };
}

export function toJs(s: Sample, dtype: DType): NDArray {
  if (s.im) {
    const values = s.re.map((r, i) => new Complex(r, s.im![i]!));
    const flat = array(values, dtype);
    return s.shape.length === 1 ? flat : flat.reshape(s.shape);
  }
  const flat = array(s.re, 'float64');
  const shaped = s.shape.length === 1 ? flat : flat.reshape(s.shape);
  return dtype === 'float64' ? shaped : shaped.astype(dtype);
}

function pyNumber(v: number): string {
  if (Number.isNaN(v)) return "float('nan')";
  if (v === Infinity) return "float('inf')";
  if (v === -Infinity) return "float('-inf')";
  return Object.is(v, -0) ? '-0.0' : String(v);
}

export function toPy(s: Sample, dtype: DType): string {
  const shape = `(${s.shape.join(', ')}${s.shape.length === 1 ? ',' : ''})`;
  if (s.im) {
    const items = s.re.map((r, i) => `complex(${pyNumber(r)}, ${pyNumber(s.im![i]!)})`);
    return `np.array([${items.join(', ')}], dtype='${dtype}').reshape(${shape})`;
  }
  return `np.array([${s.re.map(pyNumber).join(', ')}], dtype='float64').reshape(${shape}).astype('${dtype}')`;
}

function flatten(x: unknown): unknown[] {
  return Array.isArray(x) ? x.flatMap(flatten) : [x];
}

function numberEqual(g: number, w: number): boolean {
  return g === w || (Number.isNaN(g) && Number.isNaN(w));
}

function scalarEqual(got: unknown, want: unknown): boolean {
  if (got instanceof Complex || want instanceof Complex) {
    const g = got instanceof Complex ? got : new Complex(Number(got), 0);
    const w = want instanceof Complex ? want : new Complex(Number(want), 0);
    return numberEqual(g.re, w.re) && numberEqual(g.im, w.im);
  }
  return numberEqual(Number(got), Number(want));
}

/** Compare a numpy-ts array to the NumPy result: dtype, shape, and every element. */
export function expectSame(got: NDArray, want: NumPyResult, label: string): void {
  expect(got.dtype, `${label}: dtype`).toBe(want.dtype);
  const gotValues = got.toArray();
  expect(Array.from(got.shape), `${label}: shape`).toEqual(want.shape);
  const g = flatten(gotValues);
  const w = flatten(want.value);
  expect(g.length, `${label}: length`).toBe(w.length);
  for (let i = 0; i < g.length; i++) {
    if (!scalarEqual(g[i], w[i])) {
      throw new Error(`${label}: element ${i} is ${String(g[i])} but NumPy gives ${String(w[i])}`);
    }
  }
}
