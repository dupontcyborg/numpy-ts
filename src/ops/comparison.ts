/**
 * Comparison operations
 *
 * Element-wise comparison operations that return boolean arrays:
 * greater, greater_equal, less, less_equal, equal, not_equal,
 * isclose, allclose
 */

import { ArrayStorage } from '../core/storage';
import { isBigIntDType, isComplexDType } from '../core/dtype';
import { elementwiseComparisonOp } from '../internal/compute';
import { computeBroadcastShape, broadcastTo } from '../core/broadcasting';
import { Complex } from '../core/complex';

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
  op: (aRe: number, aIm: number, bRe: number, bIm: number) => boolean
): ArrayStorage {
  const outputShape = computeBroadcastShape([Array.from(a.shape), Array.from(b.shape)]);
  if (!outputShape) {
    throw new Error('Cannot broadcast arrays together');
  }

  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);
  const size = outputShape.reduce((x, y) => x * y, 1);
  const resultData = new Uint8Array(size);

  for (let i = 0; i < size; i++) {
    const [aRe, aIm] = getAsComplex(aBroadcast, i);
    const [bRe, bIm] = getAsComplex(bBroadcast, i);
    resultData[i] = op(aRe, aIm, bRe, bIm) ? 1 : 0;
  }

  return ArrayStorage.fromData(resultData, outputShape, 'bool');
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
  return elementwiseComparisonOp(a, b, (x, y) => x > y);
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
  return elementwiseComparisonOp(a, b, (x, y) => x >= y);
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
  return elementwiseComparisonOp(a, b, (x, y) => x < y);
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
  return elementwiseComparisonOp(a, b, (x, y) => x <= y);
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
  return elementwiseComparisonOp(a, b, (x, y) => x === y);
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
  return elementwiseComparisonOp(a, b, (x, y) => x !== y);
}

/**
 * Element-wise "close" comparison with tolerance
 * Returns true where |a - b| <= atol + rtol * |b|
 */
export function isclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): ArrayStorage {
  if (typeof b === 'number') {
    return iscloseScalar(a, b, rtol, atol);
  }
  return elementwiseComparisonOp(a, b, (x, y) => {
    const diff = Math.abs(x - y);
    const threshold = atol + rtol * Math.abs(y);
    return diff <= threshold;
  });
}

/**
 * Check if all elements are close (scalar result)
 * Returns true if all elements satisfy isclose condition
 */
export function allclose(
  a: ArrayStorage,
  b: ArrayStorage | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  const closeResult = isclose(a, b, rtol, atol);
  const data = closeResult.data as Uint8Array;

  // Check if all values are 1 (true)
  for (let i = 0; i < closeResult.size; i++) {
    if (data[i] === 0) {
      return false;
    }
  }
  return true;
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
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  if (isComplexDType(storage.dtype)) {
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      // Lexicographic: compare real first, if equal compare imaginary (vs 0)
      data[i] = (re !== scalar ? re > scalar : im > 0) ? 1 : 0;
    }
  } else {
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! > scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function greaterEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  if (isComplexDType(storage.dtype)) {
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      data[i] = (re !== scalar ? re >= scalar : im >= 0) ? 1 : 0;
    }
  } else {
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! >= scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function lessScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  if (isComplexDType(storage.dtype)) {
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      data[i] = (re !== scalar ? re < scalar : im < 0) ? 1 : 0;
    }
  } else {
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! < scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function lessEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;

  if (isComplexDType(storage.dtype)) {
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      data[i] = (re !== scalar ? re <= scalar : im <= 0) ? 1 : 0;
    }
  } else {
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! <= scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function equalScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const dtype = storage.dtype;

  if (isComplexDType(dtype)) {
    // Complex equals scalar only if real==scalar and imag==0
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      data[i] = re === scalar && im === 0 ? 1 : 0;
    }
  } else if (isBigIntDType(dtype)) {
    // BigInt comparison: convert scalar to BigInt
    const scalarBig = BigInt(Math.round(scalar));
    const typedData = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < storage.size; i++) {
      data[i] = typedData[i]! === scalarBig ? 1 : 0;
    }
  } else {
    // Regular comparison
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! === scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function notEqualScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const dtype = storage.dtype;

  if (isComplexDType(dtype)) {
    // Complex not-equals scalar if real!=scalar or imag!=0
    const complexData = thisData as Float64Array | Float32Array;
    for (let i = 0; i < storage.size; i++) {
      const [re, im] = getComplexAt(complexData, i);
      data[i] = re !== scalar || im !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < storage.size; i++) {
      data[i] = thisData[i]! !== scalar ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}

function iscloseScalar(
  storage: ArrayStorage,
  scalar: number,
  rtol: number,
  atol: number
): ArrayStorage {
  const data = new Uint8Array(storage.size);
  const thisData = storage.data;
  const dtype = storage.dtype;

  if (isBigIntDType(dtype)) {
    // For BigInt, convert to Number for comparison
    const thisTyped = thisData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < storage.size; i++) {
      const a = Number(thisTyped[i]!);
      const diff = Math.abs(a - scalar);
      const threshold = atol + rtol * Math.abs(scalar);
      data[i] = diff <= threshold ? 1 : 0;
    }
  } else {
    // Regular numeric types
    for (let i = 0; i < storage.size; i++) {
      const a = Number(thisData[i]!);
      const diff = Math.abs(a - scalar);
      const threshold = atol + rtol * Math.abs(scalar);
      data[i] = diff <= threshold ? 1 : 0;
    }
  }

  return ArrayStorage.fromData(data, Array.from(storage.shape), 'bool');
}
