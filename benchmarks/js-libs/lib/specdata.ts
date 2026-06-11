/**
 * Build library-neutral SpecData from a benchmark spec, reusing numpy-ts's own
 * setupArrays() so every library benchmarks on byte-identical input values.
 */

import * as np from '../../../src/core';
import { setupArrays } from '../../src/bench-utils';
import type { BenchmarkCase } from '../../src/types';
import type { ArrayData, Dtype, SpecData } from './types';

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
 * Extract SpecData for a spec, with every array forced to `target`. Overriding
 * the dtype before setupArrays() makes numpy-ts produce dtype-correct values, so
 * the comparison is genuinely "every participant at `target`" — no coercion.
 * Returns null for dtypes we don't bridge cross-library yet (complex), or if
 * setup fails for this op at this dtype.
 */
export function extractSpecData(spec: BenchmarkCase, target: Dtype): SpecData | null {
  const isComplex = target.startsWith('complex');

  // Clone setup and force each array operand to the target dtype.
  const setup = structuredClone(spec.setup) as Record<string, any>;
  for (const k of Object.keys(setup)) {
    if (setup[k] && typeof setup[k] === 'object' && 'shape' in setup[k]) setup[k].dtype = target;
  }

  let arraysRaw: Record<string, any>;
  try {
    arraysRaw = setupArrays(setup as any, spec.operation);
  } catch {
    return null;
  }

  const arrays: Record<string, ArrayData> = {};
  const params: Record<string, unknown> = {};

  for (const [key, val] of Object.entries(arraysRaw)) {
    if (key.startsWith('_')) {
      params[key] = val; // internal byte-blob / posdef helpers — not bridgeable
      continue;
    }
    if (isNdArray(val)) {
      // For complex targets we bridge the real component only — the complex
      // kernel cost is identical regardless of the imaginary values, so this is
      // timing-faithful. Participants build a complex array from these reals.
      let data: number[];
      try {
        if (isComplex) {
          const re = np.real(val);
          data = flatten(re);
          (re as any)?.dispose?.();
        } else {
          data = flatten(val);
        }
      } catch {
        return null;
      }
      arrays[key] = { shape: [...((val as any).shape as number[])], data, dtype: target };
    } else {
      params[key] = val;
    }
  }

  // Dispose the numpy-ts setup arrays we only used for extraction; the numpy-ts
  // adapter rebuilds its own from the same SpecData for a fair comparison.
  for (const val of Object.values(arraysRaw)) (val as any)?.dispose?.();

  return { arrays, params };
}
