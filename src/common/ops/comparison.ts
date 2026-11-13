/**
 * Comparison operations
 *
 * Element-wise comparison operations that return boolean arrays:
 * greater, greater_equal, less, less_equal, equal, not_equal,
 * isclose, allclose
 */

import { broadcastTo, computeBroadcastShape } from '../broadcasting';
import { Complex } from '../complex';
import { type DType, isBigIntDType, isComplexDType, isFloatDType, promoteDTypes } from '../dtype';
import { float16BitsToNumber, numberToFloat16Bits } from '../float16-conv';
import { allElementsEqual, elementwiseComparisonOp } from '../internal/compute';
import { ArrayStorage } from '../storage';

// Helper to get complex value at index
function getComplexAt(data: Float64Array | Float32Array, i: number): [number, number] {
  return [data[i * 2]!, data[i * 2 + 1]!];
}

// Helper to get value as [re, im] from any storage
function getAsComplex(storage: ArrayStorage, i: number): [number, number] {
  if (isComplexDType(storage.dtype)) {
    return getComplexAt(storage.data as Float64Array | Float32Array, i);
  }
  const val = storage.iget(i);
  if (val instanceof Complex) {
    return [val.re, val.im];
  }
  return [Number(val), 0];
}

/**
 * Complex comparison helper for arrays
 * Uses lexicographic ordering (real first, then imaginary) matching NumPy 2.x
 */
function complexComparisonOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (aRe: number, aIm: number, bRe: number, bIm: number) => boolean,
): ArrayStorage {
  const outputShape = computeBroadcastShape([Array.from(a.shape), Array.from(b.shape)]);
  if (!outputShape) {
    throw new Error('Cannot broadcast arrays together');
  }

  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);
  const size = outputShape.reduce((x, y) => x * y, 1);
  const result = ArrayStorage.empty(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  for (let i = 0; i < size; i++) {
    const [aRe, aIm] = getAsComplex(aBroadcast, i);
    const [bRe, bIm] = getAsComplex(bBroadcast, i);
    resultData[i] = op(aRe, aIm, bRe, bIm) ? 1 : 0;
  }

  return result;
}

/**
 * Element-wise greater than comparison (a > b)
 * For complex: uses lexicographic ordering (real first, then imaginary)
 */
export function greater(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return greaterScalar(a, b);
  }
  // Use complex comparison if either operand is complex
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      // Lexicographic: compare real first, then imaginary
      if (aRe !== bRe) return aRe > bRe;
      return aIm > bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x > y, 'gt');
}

/**
 * Element-wise greater than or equal comparison (a >= b)
 * For complex: uses lexicographic ordering (real first, then imaginary)
 */
export function greaterEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return greaterEqualScalar(a, b);
  }
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      if (aRe !== bRe) return aRe >= bRe;
      return aIm >= bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x >= y, 'ge');
}

/**
 * Element-wise less than comparison (a < b)
 * For complex: uses lexicographic ordering (real first, then imaginary)
 */
export function less(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return lessScalar(a, b);
  }
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      if (aRe !== bRe) return aRe < bRe;
      return aIm < bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x < y, 'lt');
}

/**
 * Element-wise less than or equal comparison (a <= b)
 * For complex: uses lexicographic ordering (real first, then imaginary)
 */
export function lessEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return lessEqualScalar(a, b);
  }
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      if (aRe !== bRe) return aRe <= bRe;
      return aIm <= bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x <= y, 'le');
}

/**
 * Element-wise equality comparison (a == b)
 * For complex: both real and imaginary parts must be equal
 */
export function equal(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return equalScalar(a, b);
  }
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      return aRe === bRe && aIm === bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x === y, 'eq');
}

/**
 * Element-wise inequality comparison (a != b)
 * For complex: either real or imaginary parts must differ
 */
export function notEqual(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return notEqualScalar(a, b);
  }
  if (isComplexDType(a.dtype) || isComplexDType(b.dtype)) {
    return complexComparisonOp(a, b, (aRe, aIm, bRe, bIm) => {
      return aRe !== bRe || aIm !== bIm;
    });
  }
  return elementwiseComparisonOp(a, b, (x, y) => x !== y, 'ne');
}

type Operand = { re: ArrayLike<number>; im?: ArrayLike<number> };

/** An operand broadcast to the output shape, read as float64 (real and imaginary parts apart). */
function toOperand(storage: ArrayStorage, outShape: number[]): Operand {
  const n = outShape.reduce((x, y) => x * y, 1);
  const complex = isComplexDType(storage.dtype);
  const sameShape =
    storage.shape.length === outShape.length && storage.shape.every((d, i) => d === outShape[i]);
  if (sameShape && storage.isCContiguous && !complex && !isBigIntDType(storage.dtype)) {
    const data = storage.data as unknown as { subarray(b: number, e: number): ArrayLike<number> };
    return { re: data.subarray(storage.offset, storage.offset + n) };
  }
  const view = broadcastTo(storage, outShape);
  const data = view.data as unknown as { [i: number]: number | bigint };
  const re = new Float64Array(n);
  const im = complex ? new Float64Array(n) : undefined;
  const nd = outShape.length;
  const coord = new Array<number>(nd).fill(0);
  for (let i = 0; i < n; i++) {
    let pos = view.offset;
    for (let d = 0; d < nd; d++) pos += coord[d]! * view.strides[d]!;
    if (im) {
      re[i] = data[pos * 2] as number;
      im[i] = data[pos * 2 + 1] as number;
    } else {
      re[i] = Number(data[pos]!);
    }
    for (let d = nd - 1; d >= 0; d--) {
      if (++coord[d]! < outShape[d]!) break;
      coord[d] = 0;
    }
  }
  return im ? { re, im } : { re };
}

/** Rounding to the precision of a float dtype, for arithmetic NumPy does in that dtype. */
function rounderFor(dtype: DType): (v: number) => number {
  if (dtype === 'float32' || dtype === 'complex64') return Math.fround;
  if (dtype === 'float16') return (v) => float16BitsToNumber(numberToFloat16Bits(v));
  return (v) => v;
}

function inexactDType(dtype: DType): DType {
  return isFloatDType(dtype) || isComplexDType(dtype) ? dtype : 'float64';
}

/**
 * Element-wise "close" comparison with tolerance, NumPy's formula: close where
 * |a - b| <= atol + rtol * |b| and b is finite, or where a == b (so equal infinities are
 * close). With equalNan, a NaN in both counts as close. Tolerances may be arrays and broadcast
 * with the inputs. The arithmetic runs in the promoted dtype, so float32 inputs are compared
 * at float32 precision as in NumPy; integers are compared as float64.
 */
export function isclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number | ArrayStorage = 1e-5,
  atol: number | ArrayStorage = 1e-8,
  equalNan: boolean = false,
): ArrayStorage {
  const arrays = [a];
  for (const v of [b, rtol, atol]) if (typeof v !== 'number') arrays.push(v);
  const outShape = computeBroadcastShape(arrays.map((s) => Array.from(s.shape)));
  if (!outShape) {
    throw new Error('Cannot broadcast arrays together');
  }

  // A scalar operand is weak: it takes the dtype of the array it meets instead of widening it.
  const yDtype = typeof b === 'number' ? undefined : inexactDType(b.dtype);
  const dtype = yDtype ? promoteDTypes(a.dtype, yDtype) : inexactDType(a.dtype);
  const rndDiff = rounderFor(dtype);
  let thrDtype: DType = yDtype ?? inexactDType(a.dtype);
  if (isComplexDType(thrDtype)) thrDtype = thrDtype === 'complex64' ? 'float32' : 'float64';
  for (const t of [rtol, atol]) {
    if (typeof t !== 'number') thrDtype = promoteDTypes(thrDtype, inexactDType(t.dtype));
  }
  const rndThr = rounderFor(thrDtype);

  const x = toOperand(a, outShape);
  const y = typeof b === 'number' ? undefined : toOperand(b, outShape);
  const rt = typeof rtol === 'number' ? undefined : toOperand(rtol, outShape);
  const at = typeof atol === 'number' ? undefined : toOperand(atol, outShape);
  const scalarB = typeof b === 'number' ? rndDiff(b) : 0;
  const rtolScalar = typeof rtol === 'number' ? rndThr(rtol) : 0;
  const atolScalar = typeof atol === 'number' ? rndThr(atol) : 0;

  const n = outShape.reduce((p, q) => p * q, 1);
  const result = ArrayStorage.empty(outShape, 'bool');
  const out = result.data as Uint8Array;
  for (let i = 0; i < n; i++) {
    const xr = x.re[i]!;
    const xi = x.im ? x.im[i]! : 0;
    const yr = y ? y.re[i]! : scalarB;
    const yi = y?.im ? y.im[i]! : 0;
    const complex = x.im !== undefined || y?.im !== undefined;

    const absDiff = complex
      ? rndDiff(Math.hypot(rndDiff(xr - yr), rndDiff(xi - yi)))
      : rndDiff(Math.abs(rndDiff(xr - yr)));
    const absY = complex ? rndThr(Math.hypot(yr, yi)) : Math.abs(yr);
    const rtolV = rt ? rt.re[i]! : rtolScalar;
    const atolV = at ? at.re[i]! : atolScalar;
    const threshold = rndThr(atolV + rndThr(rtolV * absY));

    const finite = Number.isFinite(yr) && Number.isFinite(yi);
    let close = (absDiff <= threshold && finite) || (xr === yr && xi === yi);
    if (!close && equalNan) {
      close = (xr !== xr || xi !== xi) && (yr !== yr || yi !== yi);
    }
    out[i] = close ? 1 : 0;
  }
  return result;
}

/**
 * Check if all elements are close (scalar result)
 * Returns true if all elements satisfy the isclose condition
 */
export function allclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number | ArrayStorage = 1e-5,
  atol: number | ArrayStorage = 1e-8,
  equalNan: boolean = false,
): boolean {
  const closeResult = isclose(a, b, rtol, atol, equalNan);
  try {
    const data = closeResult.data as Uint8Array;

    // Check if all values are 1 (true)
    for (let i = 0; i < closeResult.size; i++) {
      if (data[i] === 0) {
        return false;
      }
    }
    return true;
  } finally {
    // Free the intermediate mask on both exits, including the early return.
    closeResult.dispose();
  }
}

/**
 * Returns True if two arrays are element-wise equal within a tolerance.
 * Unlike array_equal, this function broadcasts the arrays before comparison.
 *
 * NumPy behavior: Broadcasts arrays before comparing, returns True if shapes
 * are broadcast-compatible and all elements are equal.
 *
 * @param a1 - First input array
 * @param a2 - Second input array
 * @returns True if arrays are equivalent (after broadcasting), False otherwise
 */
export function arrayEquiv(a1: ArrayStorage, a2: ArrayStorage): boolean {
  // Check if arrays can be broadcast together
  const shapes = [Array.from(a1.shape), Array.from(a2.shape)];
  const broadcastShape = computeBroadcastShape(shapes);

  if (broadcastShape === null) {
    // If shapes are incompatible for broadcasting, arrays are not equivalent
    return false;
  }

  // Identical shapes, same dtype, contiguous: a dtype-specialised loop, instead
  // of decomposing the flat index into per-axis indices for every element below.
  // The shape test gates the loop rather than filtering its result — [1,6] vs
  // [6,1] have equal sizes but must broadcast to [6,6], not compare pairwise.
  // `array_equiv` has no equal_nan option, so NaN never compares equal here.
  if (a1.ndim === a2.ndim && a1.shape.every((d, i) => d === a2.shape[i])) {
    const fast = allElementsEqual(a1, a2, false);
    if (fast !== null) return fast;
  }

  // Broadcast both arrays to the common shape
  const b1 = broadcastTo(a1, broadcastShape);
  const b2 = broadcastTo(a2, broadcastShape);

  // Compare element by element using proper multi-dimensional indexing
  const ndim = broadcastShape.length;
  const size = broadcastShape.reduce((acc, d) => acc * d, 1);

  // Handle different dtypes
  const isBigInt1 = isBigIntDType(b1.dtype);
  const isBigInt2 = isBigIntDType(b2.dtype);

  // Iterate over all elements using multi-dimensional indices
  for (let flatIdx = 0; flatIdx < size; flatIdx++) {
    // Convert flat index to multi-dimensional indices
    let temp = flatIdx;
    const indices: number[] = new Array(ndim);
    for (let i = ndim - 1; i >= 0; i--) {
      indices[i] = temp % broadcastShape[i]!;
      temp = Math.floor(temp / broadcastShape[i]!);
    }

    // Get values using proper indexing
    const val1 = b1.get(...indices);
    const val2 = b2.get(...indices);

    // Compare values
    if (isBigInt1 || isBigInt2) {
      const v1 = typeof val1 === 'bigint' ? val1 : BigInt(Number(val1));
      const v2 = typeof val2 === 'bigint' ? val2 : BigInt(Number(val2));
      if (v1 !== v2) {
        return false;
      }
    } else if (val1 instanceof Complex || val2 instanceof Complex) {
      // `val1 !== val2` on two Complex instances compares object identity, so
      // this returned false for *any* complex input, however equal.
      const re1 = val1 instanceof Complex ? val1.re : Number(val1);
      const im1 = val1 instanceof Complex ? val1.im : 0;
      const re2 = val2 instanceof Complex ? val2.re : Number(val2);
      const im2 = val2 instanceof Complex ? val2.im : 0;
      if (re1 !== re2 || im1 !== im2) {
        return false;
      }
    } else {
      if (val1 !== val2) {
        return false;
      }
    }
  }

  return true;
}

// Scalar comparison optimized paths
// For complex arrays: scalar is treated as (scalar + 0i)
// Lexicographic ordering: compare real first, then imaginary

function greaterScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(storage.dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = (re !== scalar ? re > scalar : im > 0) ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = (val.re !== scalar ? val.re > scalar : val.im > 0) ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = (srcData[off + i] as number) > scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) > scalar ? 1 : 0;
    }
  }

  return result;
}

function greaterEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(storage.dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = (re !== scalar ? re >= scalar : im >= 0) ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = (val.re !== scalar ? val.re >= scalar : val.im >= 0) ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = (srcData[off + i] as number) >= scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) >= scalar ? 1 : 0;
    }
  }

  return result;
}

function lessScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(storage.dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = (re !== scalar ? re < scalar : im < 0) ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = (val.re !== scalar ? val.re < scalar : val.im < 0) ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = (srcData[off + i] as number) < scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) < scalar ? 1 : 0;
    }
  }

  return result;
}

function lessEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(storage.dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = (re !== scalar ? re <= scalar : im <= 0) ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = (val.re !== scalar ? val.re <= scalar : val.im <= 0) ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = (srcData[off + i] as number) <= scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) <= scalar ? 1 : 0;
    }
  }

  return result;
}

function equalScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const dtype = storage.dtype;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = re === scalar && im === 0 ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = val.re === scalar && val.im === 0 ? 1 : 0;
      }
    }
  } else if (isBigIntDType(dtype)) {
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const srcData = storage.data as BigInt64Array | BigUint64Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        resultData[i] = srcData[off + i]! === scalarBig ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = (storage.iget(i) as bigint) === scalarBig ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = srcData[off + i]! === scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) === scalar ? 1 : 0;
    }
  }

  return result;
}

function notEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), 'bool');
  const resultData = result.data as Uint8Array;
  const size = storage.size;
  const dtype = storage.dtype;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(dtype)) {
    if (contiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        resultData[i] = re !== scalar || im !== 0 ? 1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        resultData[i] = val.re !== scalar || val.im !== 0 ? 1 : 0;
      }
    }
  } else if (contiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      resultData[i] = srcData[off + i]! !== scalar ? 1 : 0;
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Number(storage.iget(i)) !== scalar ? 1 : 0;
    }
  }

  return result;
}
