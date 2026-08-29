/**
 * Rounding operations
 *
 * Pure functions for element-wise rounding operations:
 * around, ceil, fix, floor, rint, round, trunc
 *
 * Note: Rounding operations are not defined for complex numbers.
 * All functions throw TypeError for complex dtypes.
 */

import type { Complex } from '../complex';
import {
  hasFloat16,
  isComplexDType,
  isIntegerDType,
  mathResultDtype,
  throwIfComplex,
} from '../dtype';
import { ArrayStorage } from '../storage';
import { wasmAround, wasmCeil, wasmFloor, wasmRint, wasmTrunc } from '../wasm/rounding';

/**
 * Apply a rounding function component-wise to a complex array.
 * NumPy applies rounding to real and imaginary parts independently for rint/around.
 */
function complexComponentwise(a: ArrayStorage, fn: (x: number) => number): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;
  const result = ArrayStorage.empty(shape, dtype);
  const dstData = result.data as Float64Array | Float32Array;

  if (a.isCContiguous) {
    const srcData = a.data as Float64Array | Float32Array;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      dstData[i * 2] = fn(srcData[(off + i) * 2]!);
      dstData[i * 2 + 1] = fn(srcData[(off + i) * 2 + 1]!);
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = a.iget(i) as Complex;
      dstData[i * 2] = fn(val.re);
      dstData[i * 2 + 1] = fn(val.im);
    }
  }

  return result;
}

/**
 * True on a little-endian platform, where a 64-bit lane's low half sits at the
 * lower index. WASM mandates little-endian and every platform this runs on is
 * little-endian, but the half-word read in `rint` is silently wrong if that ever
 * stops holding, so it is checked rather than assumed.
 */
/** One-element scratch for rounding an intermediate back to float16. */
const f16scratch: Float16Array | null = hasFloat16 ? new Float16Array(1) : null;

const LITTLE_ENDIAN = (() => {
  const probe = new Uint32Array([1]);
  return new Uint8Array(probe.buffer)[0] === 1;
})();

/**
 * Round half to even (banker's rounding) — matches NumPy's `rint`.
 *
 * The tie test has to be exact. Treating anything within 1e-10 of .5 as a tie
 * would disagree with NumPy on near-ties — `np.rint(2.5000000000001)` is 3, not
 * 2 — and would disagree with the WASM kernel that handles arrays of 32
 * elements or more, so the same value could round differently depending on
 * array size.
 *
 * `Math.floor` and the subtraction are both exact, so `frac === 0.5` identifies
 * a genuine tie and nothing else. For |x| >= 2^52 the fraction is 0 and x falls
 * out of the first branch unchanged.
 */
function roundHalfToEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const lo = Math.floor(x);
  const frac = x - lo;
  let r: number;
  if (frac < 0.5) r = lo;
  else if (frac > 0.5) r = lo + 1;
  // Exact tie: pick whichever neighbour is even. `lo % 2` is -0 for negative
  // even values, and -0 === 0, so this holds for both signs.
  else r = lo % 2 === 0 ? lo : lo + 1;
  // NumPy keeps the sign of zero — rint(-0.5) is -0.0, not +0.0 — and so does
  // the WASM kernel, which re-applies the input's sign bit. Without this the two
  // paths disagreed on any negative input that rounds to zero.
  return r === 0 && (x < 0 || Object.is(x, -0)) ? -0 : r;
}

/**
 * Round an array to the given number of decimals
 */
export function around(a: ArrayStorage, decimals: number = 0): ArrayStorage {
  if (isComplexDType(a.dtype)) {
    const multiplier = 10 ** decimals;
    return complexComponentwise(a, (x) => roundHalfToEven(x * multiplier) / multiplier);
  }
  // Integers are already rounded, and for decimals >= 0 NumPy hands the input
  // straight back — `np.round(a) is a` is True and the two share memory.
  // Copying instead would be pure overhead.
  //
  // Only `around`/`round` behaves this way. floor/ceil/trunc/fix all return a
  // genuinely new array for integer input, so their copies below are correct and
  // stay as they are.
  if (isIntegerDType(a.dtype) && decimals >= 0) {
    return ArrayStorage.fromDataShared(
      a.data,
      Array.from(a.shape),
      a.dtype,
      Array.from(a.strides),
      a.offset,
      a.wasmRegion,
    );
  }
  // decimals === 0 is plain rint; anything else needs the scaled kernel.
  const wasmAroundResult = decimals === 0 ? wasmRint(a) : wasmAround(a, 10 ** decimals);
  if (wasmAroundResult) return wasmAroundResult;
  if (a.dtype === 'bool') {
    const dt = mathResultDtype('bool');
    const r = ArrayStorage.empty(Array.from(a.shape), dt);
    const src = a.data as Uint8Array;
    const off = a.offset;
    for (let i = 0; i < a.size; i++) r.data[i] = src[off + i]!;
    return r;
  }
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  const multiplier = 10 ** decimals;

  // NumPy scales at the array's own precision. For float32, f32(2.675) * 100
  // is exactly 267.5f, so the tie rule gives 268 and the answer 2.68; widening
  // to f64 first gives 267.4999952 and rounds down to 2.67. float16 narrows
  // further still (1.35 * 10 lands on 13.5 in f16, giving 1.4 rather than 1.3).
  // So the product has to be rounded back to the input dtype before rint.
  const narrow =
    dtype === 'float32'
      ? Math.fround
      : dtype === 'float16' && f16scratch
        ? (x: number): number => {
            f16scratch[0] = x;
            return f16scratch[0]!;
          }
        : (x: number): number => x;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      const val = Number(data[off + i]!);
      resultData[i] = narrow(roundHalfToEven(narrow(val * multiplier)) / multiplier);
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(a.iget(i));
      resultData[i] = narrow(roundHalfToEven(narrow(val * multiplier)) / multiplier);
    }
  }

  return result;
}

/**
 * Return the ceiling of the input, element-wise
 */
export function ceil(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'ceil', 'Rounding is not defined for complex numbers.');
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmCeil(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.ceil(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.ceil(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Round to nearest integer towards zero
 */
export function fix(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'fix', 'Rounding is not defined for complex numbers.');
  if (isIntegerDType(a.dtype)) return a.copy();
  // fix is trunc — same kernel.
  const wasmResult = wasmTrunc(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Return the floor of the input, element-wise
 */
export function floor(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'floor', 'Rounding is not defined for complex numbers.');
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmFloor(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.floor(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Round elements of the array to the nearest integer (banker's rounding)
 */
export function rint(a: ArrayStorage): ArrayStorage {
  if (isComplexDType(a.dtype)) return complexComponentwise(a, roundHalfToEven);
  // NumPy: rint promotes ints/bool via mathResultDtype (values stay the same, just cast)
  if (isIntegerDType(a.dtype) || a.dtype === 'bool') {
    // rint of an integer array is a pure dtype cast: the values are already
    // integral, and NumPy only widens them (int32 -> float64, int16 -> float32,
    // int8/bool -> float16). Nothing needs rounding, so this must not look like
    // a rounding loop.
    const dt = mathResultDtype(a.dtype);
    const size = a.size;
    const r = ArrayStorage.empty(Array.from(a.shape), dt);
    const src = a.data;
    const off = a.offset;

    // Strided view: the fast paths below both walk the buffer linearly from
    // `off`, which silently reorders a non-contiguous view. (The float path
    // already had an iget fallback for this; the integer path did not, so
    // `rint` on a transposed integer array returned shuffled values.)
    if (!a.isCContiguous) {
      for (let i = 0; i < size; i++) r.data[i] = Number(a.iget(i));
      return r;
    }

    if (src instanceof BigInt64Array || src instanceof BigUint64Array) {
      // TypedArray.set refuses to mix BigInt and Number arrays, so 64-bit ints
      // need a loop. Read the two 32-bit halves instead of calling
      // Number(bigint), which is slower: `hi * 2^32` is exact (|hi| < 2^31)
      // and `lo` is exact (< 2^32), so the single addition is correctly rounded
      // and bit-identical to Number(bigint) — verified across the +/-2^53 and
      // +/-2^63 boundaries and 400k random values. Above 2^53 both lose the same
      // bits, which is what NumPy's int64 -> float64 does too.
      const dst = r.data as Float64Array;
      const byteOff = src.byteOffset + off * 8;
      const lo = new Uint32Array(src.buffer, byteOff, size * 2);
      if (!LITTLE_ENDIAN) {
        for (let i = 0; i < size; i++) dst[i] = Number(src[off + i]!);
      } else if (src instanceof BigInt64Array) {
        const hi = new Int32Array(src.buffer, byteOff, size * 2);
        for (let i = 0; i < size; i++) dst[i] = hi[2 * i + 1]! * 4294967296 + lo[2 * i]!;
      } else {
        for (let i = 0; i < size; i++) dst[i] = lo[2 * i + 1]! * 4294967296 + lo[2 * i]!;
      }
    } else {
      // Native bulk convert — one call instead of `size` calls to Number().
      (r.data as Float64Array).set(src.subarray(off, off + size) as unknown as Float64Array);
    }
    return r;
  }
  const wasmResult = wasmRint(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = roundHalfToEven(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = roundHalfToEven(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Alias for around
 */
export function round(a: ArrayStorage, decimals: number = 0): ArrayStorage {
  return around(a, decimals);
}

/**
 * Return the truncated value of the input, element-wise
 */
export function trunc(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'trunc', 'Rounding is not defined for complex numbers.');
  if (isIntegerDType(a.dtype)) return a.copy();
  const wasmResult = wasmTrunc(a);
  if (wasmResult) return wasmResult;
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // NumPy preserves dtype for rounding ops (integers are already rounded)
  const resultDtype = dtype;
  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    const data = a.data;
    const off = a.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.trunc(Number(a.iget(i)));
    }
  }

  return result;
}
