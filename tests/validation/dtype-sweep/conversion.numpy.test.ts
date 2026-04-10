/**
 * DType Sweep: Array conversion (.astype).
 * Validates that dtype-to-dtype casting matches NumPy for values, dtype, and shape.
 * Tests multiple "starting states" across all 13×13 dtype pairs.
 *
 * Uses a single batched NumPy call to avoid per-test subprocess overhead.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPy,
  checkNumPyAvailable,
  npDtype,
  isInt,
  isComplex,
  isBool,
} from './_helpers';
import { hasFloat16 } from '../../../src';

const { array } = np;

/** Range limits per dtype so source values don't overflow during array creation. */
const DTYPE_RANGE: Record<string, { min: number; max: number }> = {
  bool: { min: 0, max: 1 },
  int8: { min: -128, max: 127 },
  uint8: { min: 0, max: 255 },
  int16: { min: -32768, max: 32767 },
  uint16: { min: 0, max: 65535 },
  int32: { min: -2147483648, max: 2147483647 },
  uint32: { min: 0, max: 4294967295 },
  int64: { min: -2147483648, max: 2147483647 },
  uint64: { min: 0, max: 4294967295 },
  float16: { min: -65504, max: 65504 },
  float32: { min: -1e38, max: 1e38 },
  float64: { min: -1e100, max: 1e100 },
  complex64: { min: -1e38, max: 1e38 },
  complex128: { min: -1e100, max: 1e100 },
};

const RAW_STATES: { label: string; values: number[] }[] = [
  { label: 'small-positive', values: [0, 1, 2, 3] },
  { label: 'mixed-sign', values: [-2, -1, 0, 1] },
  { label: 'fractional', values: [0.25, 1.5, 2.7, 3.9] },
  { label: 'large', values: [100, 200, 1000, 10000] },
  { label: 'boundary', values: [0, 1, 126, 127] },
];

function clampForDtype(values: number[], dtype: string): number[] {
  const range = DTYPE_RANGE[dtype];
  if (isBool(dtype)) return values.map((v) => (v ? 1 : 0));
  let out = values;
  if (isInt(dtype) || dtype.startsWith('uint')) {
    out = out.map((v) => Math.trunc(v));
  }
  return out.map((v) => Math.max(range.min, Math.min(range.max, v)));
}

/** Key for the oracle lookup map. */
function oracleKey(src: string, dst: string, label: string): string {
  return `${src}|${dst}|${label}`;
}

/**
 * Pre-compute all conversions in a single NumPy subprocess.
 * Returns a map from oracleKey → { value, dtype, shape } | { error: string }.
 */
function buildOracleMap(): Record<string, any> {
  // Build a Python script that computes every conversion and emits JSON
  const lines: string[] = [`results = {}`];

  for (const src of ALL_DTYPES) {
    for (const { label, values } of RAW_STATES) {
      const srcValues = clampForDtype(values, src);
      const pyValues = `[${srcValues.join(', ')}]`;
      const pySrc = npDtype(src);

      for (const dst of ALL_DTYPES) {
        const pyDst = npDtype(dst);
        const key = oracleKey(src, dst, label);
        lines.push(`try:`);
        lines.push(`    _a = np.array(${pyValues}, dtype=${pySrc})`);
        lines.push(`    _orig = _a.astype(${pyDst})`);
        lines.push(
          `    _cmp = _orig.astype(np.complex128) if np.issubdtype(_orig.dtype, np.complexfloating) else _orig.astype(np.float64)`
        );
        lines.push(
          `    results["${key}"] = {"value": serialize_value(_cmp), "dtype": str(_orig.dtype), "shape": list(_orig.shape)}`
        );
        lines.push(`except Exception as e:`);
        lines.push(`    results["${key}"] = {"error": str(e)[:200]}`);
      }
    }
  }

  lines.push(`result = results`);

  // Use the oracle's serialize_value, then output via result
  // runNumPy wraps in try/except and serializes `result`
  const pyCode = lines.join('\n');
  const raw = runNumPy(pyCode);
  return raw.value;
}

let oracle: Record<string, any>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  oracle = buildOracleMap();
});

describe('DType Sweep: Conversion (astype)', () => {
  for (const src of ALL_DTYPES) {
    describe(`from ${src}`, () => {
      for (const dst of ALL_DTYPES) {
        for (const { label, values } of RAW_STATES) {
          it(`${src} → ${dst} (${label})`, () => {
            const srcValues = clampForDtype(values, src);
            const key = oracleKey(src, dst, label);
            const pyEntry = oracle[key];

            // Check if NumPy errored on this conversion
            const npError = pyEntry?.error;

            let jsError: string | null = null;
            let jsResult: any;
            try {
              jsResult = array(srcValues, src).astype(dst);
            } catch (e: any) {
              jsError = e.message;
            }

            // Both error → pass
            if (npError && jsError) return;

            // Divergent error behavior
            if (jsError && !npError) {
              expect.unreachable(
                `JS throws but NumPy succeeds.\nJS error: ${jsError.slice(0, 150)}`
              );
            }
            if (npError && !jsError) {
              expect.unreachable(
                `NumPy rejects but JS succeeds.\nNumPy error: ${npError.slice(0, 150)}`
              );
            }

            // Both succeed — compare
            expect(jsResult.dtype, `dtype: JS '${jsResult.dtype}' vs expected '${dst}'`).toBe(dst);
            expect(Array.from(jsResult.shape)).toEqual(pyEntry.shape);

            // Value comparison: cast JS result to same comparison type
            // No native Float16Array: float16 arrays use float32 precision, skip value check
            if (!hasFloat16 && (src === 'float16' || dst === 'float16')) return;

            const jsCompare = isComplex(dst) ? jsResult : jsResult.astype('float64');
            const jsArr = jsCompare.toArray();
            const npArr = pyEntry.value;

            // Use arraysClose for float comparison
            const close = arraysCloseLocal(jsArr, npArr, 1e-5, 1e-8);
            expect(
              close,
              `value mismatch for ${src} → ${dst} (${label}): ` +
                `JS=${JSON.stringify(jsArr.slice(0, 4))} vs ` +
                `NumPy=${JSON.stringify(npArr?.slice?.(0, 4) ?? npArr)}`
            ).toBe(true);
          });
        }
      }
    });
  }
});

/** Inline arraysClose to avoid import issues with Complex values from oracle. */
function arraysCloseLocal(a: any[], b: any[], rtol: number, atol: number): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const ai = typeof a[i] === 'object' && a[i] !== null ? Number(a[i].re ?? a[i]) : Number(a[i]);
    const bi = typeof b[i] === 'object' && b[i] !== null ? Number(b[i].re ?? b[i]) : Number(b[i]);
    if (Number.isNaN(ai) && Number.isNaN(bi)) continue;
    if (!Number.isFinite(ai) && ai === bi) continue;
    if (Math.abs(ai - bi) > atol + rtol * Math.abs(bi)) return false;
    // For complex: also check imaginary
    if (typeof a[i] === 'object' || typeof b[i] === 'object') {
      const aiIm = typeof a[i] === 'object' && a[i] !== null ? Number(a[i].im ?? 0) : 0;
      const biIm = typeof b[i] === 'object' && b[i] !== null ? Number(b[i].im ?? 0) : 0;
      if (Math.abs(aiIm - biIm) > atol + rtol * Math.abs(biIm)) return false;
    }
  }
  return true;
}
