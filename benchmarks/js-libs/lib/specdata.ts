/**
 * Build library-neutral SpecData from a benchmark spec, reusing numpy-ts's own
 * setupArrays() so every library benchmarks on byte-identical input values.
 */

import * as np from '../../../src/core';
import { setupArrays } from '../../src/bench-utils';
import type { BenchmarkCase } from '../../src/types';
import type { ArrayData, Regime, SpecData } from './types';

function isNdArray(v: unknown): boolean {
  return (
    v != null &&
    typeof v === 'object' &&
    typeof (v as any).shape !== 'undefined' &&
    typeof (v as any).dtype === 'string' &&
    typeof (v as any).size === 'number'
  );
}

/** Flatten a numpy-ts array to a row-major number[] (bigint coerced to Number). */
function flatten(arr: any): number[] {
  const flat = np.tolist(np.ravel(arr)) as unknown[];
  const out = new Array<number>(flat.length);
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    out[i] = typeof v === 'bigint' ? Number(v) : (v as number);
  }
  return out;
}

/**
 * Extract SpecData for a spec. Returns null if the spec uses dtypes/operands we
 * don't bridge cross-library (complex, npy/npz byte blobs, etc.).
 */
export function extractSpecData(spec: BenchmarkCase, regime: Regime): SpecData | null {
  let arraysRaw: Record<string, any>;
  try {
    arraysRaw = setupArrays(spec.setup, spec.operation);
  } catch {
    return null;
  }

  const arrays: Record<string, ArrayData> = {};
  const params: Record<string, unknown> = {};

  for (const [key, val] of Object.entries(arraysRaw)) {
    if (key.startsWith('_')) {
      // Internal byte-blob / posdef helpers — not cross-library bridgeable.
      params[key] = val;
      continue;
    }
    if (isNdArray(val)) {
      const dtype = (val as any).dtype as string;
      if (dtype.startsWith('complex')) return null; // out of cross-lib scope
      let data: number[];
      try {
        data = flatten(val);
      } catch {
        return null;
      }
      arrays[key] = {
        shape: [...((val as any).shape as number[])],
        data,
        dtype: regime === 'float32' && dtype.startsWith('float') ? 'float32' : dtype,
      };
    } else {
      params[key] = val;
    }
  }

  // Dispose the numpy-ts setup arrays we only used for extraction; the numpy-ts
  // adapter rebuilds its own from the same SpecData for a fair comparison.
  for (const val of Object.values(arraysRaw)) (val as any)?.dispose?.();

  return { arrays, params };
}
