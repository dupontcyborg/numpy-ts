/**
 * Arithmetic operations
 *
 * Pure functions for element-wise arithmetic operations:
 * add, subtract, multiply, divide
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../storage';
import { Complex } from '../complex';
import {
  isBigIntDType,
  isComplexDType,
  isIntegerDType,
  getComplexComponentDType,
  promoteDTypes,
  throwIfComplex,
  throwIfBool,
  isFloatDType,
  mathResultDtype,
  boolArithmeticDtype,
} from '../dtype';
import { elementwiseBinaryOp } from '../internal/compute';
import { wasmAdd, wasmAddScalar } from '../wasm/add';
import { wasmSub, wasmSubScalar } from '../wasm/sub';
import { wasmMul, wasmMulScalar } from '../wasm/mul';
import { wasmDiv, wasmDivScalar } from '../wasm/divide';
import { wasmNeg } from '../wasm/neg';
import { wasmAbs } from '../wasm/abs';
import { wasmSign } from '../wasm/sign';
import { wasmMin, wasmMinScalar } from '../wasm/min';
import { wasmMax, wasmMaxScalar } from '../wasm/max';
import { wasmClip } from '../wasm/clip';
import { wasmSquare } from '../wasm/square';
import { wasmReciprocal } from '../wasm/reciprocal';
import { wasmHeavisideScalar, wasmHeaviside } from '../wasm/heaviside';
import { wasmLdexpScalar } from '../wasm/ldexp';
import { wasmFrexp } from '../wasm/frexp';
import { wasmGcdScalar, wasmGcd } from '../wasm/gcd';
import { wasmDivmodScalar } from '../wasm/divmod';

/**
 * Helper: Check if two arrays can use the fast path
 * (both C-contiguous with same shape, no broadcasting needed)
 */
function canUseFastPath(a: ArrayStorage, b: ArrayStorage): boolean {
  return (
    a.isCContiguous &&
    b.isCContiguous &&
    a.shape.length === b.shape.length &&
    a.shape.every((dim, i) => dim === b.shape[i])
  );
}

/**
 * Convert a bool ArrayStorage to int8 (NumPy promotes bool → int8 for arithmetic).
 */
function boolToInt8(a: ArrayStorage): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(a.shape), 'int8');
  const src = a.data as Uint8Array;
  const dst = result.data as Int8Array;
  const off = a.offset;
  for (let i = 0; i < a.size; i++) dst[i] = src[off + i]!;
  return result;
}

/**
 * Convert a bool ArrayStorage to its math result type (float16 or float32).
 * Used for binary math ops like copysign, hypot, arctan2 where NumPy promotes bool → float16.
 */
function boolToMathFloat(a: ArrayStorage): ArrayStorage {
  const dt = mathResultDtype('bool');
  const result = ArrayStorage.empty(Array.from(a.shape), dt);
  const src = a.data as Uint8Array;
  const dst = result.data;
  const off = a.offset;
  for (let i = 0; i < a.size; i++) dst[i] = src[off + i]!;
  return result;
}

// ============================================================
// Complex arithmetic helpers
// ============================================================

/**
 * Get real and imaginary parts from a complex array at index
 */
function getComplexAt(data: Float64Array | Float32Array, i: number): [number, number] {
  return [data[i * 2]!, data[i * 2 + 1]!];
}

/**
 * Set real and imaginary parts in a complex array at index
 */
function setComplexAt(data: Float64Array | Float32Array, i: number, re: number, im: number): void {
  data[i * 2] = re;
  data[i * 2 + 1] = im;
}

/**
 * Add two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function add(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return addScalar(a, b);
  }

  // Optimize single-element non-complex arrays as scalars (only when same dtype to preserve promotion)
  // Skip BigInt dtypes — WASM scalar kernels can't accept i64 scalar params from JS
  if (b.size === 1 && !isComplexDType(b.dtype) && a.dtype === b.dtype) {
    const scalarVal = Number(b.iget(0));
    return addScalar(a, scalarVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    // WASM acceleration for large contiguous arrays
    const wasmResult = wasmAdd(a, b);
    if (wasmResult) return wasmResult;

    return addArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x + y, 'add');
}

/**
 * Fast path for adding two contiguous arrays
 * @private
 */
function addArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.empty(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;
  const aOff = a.offset;
  const bOff = b.offset;

  // Handle complex result dtype
  if (isComplexDType(dtype)) {
    const resultTyped = resultData as Float64Array | Float32Array;
    const aIsComplex = isComplexDType(a.dtype);
    const bIsComplex = isComplexDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const [aRe, aIm] = aIsComplex
        ? getComplexAt(aData as Float64Array | Float32Array, aOff + i)
        : [Number(aData[aOff + i]), 0];
      const [bRe, bIm] = bIsComplex
        ? getComplexAt(bData as Float64Array | Float32Array, bOff + i)
        : [Number(bData[bOff + i]), 0];
      // (a+bi) + (c+di) = (a+c) + (b+d)i
      setComplexAt(resultTyped, i, aRe + bRe, aIm + bIm);
    }
    return result;
  }

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) + (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! + bTyped[bOff + i]!;
      }
    }
  } else if (dtype === 'bool') {
    // Bool addition is logical OR in NumPy (any nonzero → 1)
    for (let i = 0; i < size; i++) {
      resultData[i] = (aData[aOff + i] as number) + (bData[bOff + i] as number) ? 1 : 0;
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? Number(aData[aOff + i])
            : (aData[aOff + i] as number);
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? Number(bData[bOff + i])
            : (bData[bOff + i] as number);
        resultData[i] = aVal + bVal;
      }
    } else {
      // Pure numeric operations - fully optimizable
      if (aOff === 0 && bOff === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[i] as number) + (bData[i] as number);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[aOff + i] as number) + (bData[bOff + i] as number);
        }
      }
    }
  }

  return result;
}

/**
 * Subtract two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function subtract(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  throwIfBool(
    a.dtype,
    'subtract',
    'The `-` operator is not supported for booleans, use `bitwise_xor` instead.'
  );
  if (typeof b === 'number') {
    return subtractScalar(a, b);
  }

  // Optimize single-element non-complex arrays as scalars (only when same dtype to preserve promotion)
  if (b.size === 1 && !isComplexDType(b.dtype) && a.dtype === b.dtype) {
    const scalarVal = Number(b.iget(0));
    return subtractScalar(a, scalarVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    const wasmResult = wasmSub(a, b);
    if (wasmResult) return wasmResult;

    return subtractArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x - y, 'subtract');
}

/**
 * Fast path for subtracting two contiguous arrays
 * @private
 */
function subtractArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.empty(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;
  const aOff = a.offset;
  const bOff = b.offset;

  // Handle complex result dtype
  if (isComplexDType(dtype)) {
    const resultTyped = resultData as Float64Array | Float32Array;
    const aIsComplex = isComplexDType(a.dtype);
    const bIsComplex = isComplexDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const [aRe, aIm] = aIsComplex
        ? getComplexAt(aData as Float64Array | Float32Array, aOff + i)
        : [Number(aData[aOff + i]), 0];
      const [bRe, bIm] = bIsComplex
        ? getComplexAt(bData as Float64Array | Float32Array, bOff + i)
        : [Number(bData[bOff + i]), 0];
      // (a+bi) - (c+di) = (a-c) + (b-d)i
      setComplexAt(resultTyped, i, aRe - bRe, aIm - bIm);
    }
    return result;
  }

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) - (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! - bTyped[bOff + i]!;
      }
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? Number(aData[aOff + i])
            : (aData[aOff + i] as number);
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? Number(bData[bOff + i])
            : (bData[bOff + i] as number);
        resultData[i] = aVal - bVal;
      }
    } else {
      if (aOff === 0 && bOff === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[i] as number) - (bData[i] as number);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[aOff + i] as number) - (bData[bOff + i] as number);
        }
      }
    }
  }

  return result;
}

/**
 * Multiply two arrays or array and scalar
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage
 */
export function multiply(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return multiplyScalar(a, b);
  }

  // Optimize single-element non-complex arrays as scalars (only when same dtype to preserve promotion)
  if (b.size === 1 && !isComplexDType(b.dtype) && a.dtype === b.dtype) {
    const scalarVal = Number(b.iget(0));
    return multiplyScalar(a, scalarVal);
  }

  // Fast path: both contiguous, same shape
  if (canUseFastPath(a, b)) {
    const wasmResult = wasmMul(a, b);
    if (wasmResult) return wasmResult;

    return multiplyArraysFast(a, b);
  }

  // Slow path: broadcasting or non-contiguous
  return elementwiseBinaryOp(a, b, (x, y) => x * y, 'multiply');
}

/**
 * Fast path for multiplying two contiguous arrays
 * @private
 */
function multiplyArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const dtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.empty(Array.from(a.shape), dtype);
  const size = a.size;
  const aData = a.data;
  const bData = b.data;
  const resultData = result.data;
  const aOff = a.offset;
  const bOff = b.offset;

  // Handle complex result dtype
  if (isComplexDType(dtype)) {
    const resultTyped = resultData as Float64Array | Float32Array;
    const aIsComplex = isComplexDType(a.dtype);
    const bIsComplex = isComplexDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const [aRe, aIm] = aIsComplex
        ? getComplexAt(aData as Float64Array | Float32Array, aOff + i)
        : [Number(aData[aOff + i]), 0];
      const [bRe, bIm] = bIsComplex
        ? getComplexAt(bData as Float64Array | Float32Array, bOff + i)
        : [Number(bData[bOff + i]), 0];
      // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      const re = aRe * bRe - aIm * bIm;
      const im = aRe * bIm + aIm * bRe;
      setComplexAt(resultTyped, i, re, im);
    }
    return result;
  }

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const needsConversion = !isBigIntDType(a.dtype) || !isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? aData[aOff + i]
            : BigInt(Math.round(Number(aData[aOff + i])));
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? bData[bOff + i]
            : BigInt(Math.round(Number(bData[bOff + i])));
        resultTyped[i] = (aVal as bigint) * (bVal as bigint);
      }
    } else {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const bTyped = bData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = aTyped[aOff + i]! * bTyped[bOff + i]!;
      }
    }
  } else {
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    if (needsConversion) {
      for (let i = 0; i < size; i++) {
        const aVal =
          typeof aData[aOff + i] === 'bigint'
            ? Number(aData[aOff + i])
            : (aData[aOff + i] as number);
        const bVal =
          typeof bData[bOff + i] === 'bigint'
            ? Number(bData[bOff + i])
            : (bData[bOff + i] as number);
        resultData[i] = aVal * bVal;
      }
    } else {
      if (aOff === 0 && bOff === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[i] as number) * (bData[i] as number);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = (aData[aOff + i] as number) * (bData[bOff + i] as number);
        }
      }
    }
  }

  return result;
}

/**
 * Divide two arrays or array and scalar
 *
 * NumPy behavior: Integer division always promotes to float
 * Type promotion rules:
 * - complex128 + anything → complex128
 * - complex64 + float32/int → complex64
 * - float64 + anything → float64
 * - float32 + integer → float32
 * - integer + integer → float64
 *
 * @param a - First array storage
 * @param b - Second array storage or scalar
 * @returns Result storage with promoted float dtype
 */
export function divide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  if (typeof b === 'number') {
    return divideScalar(a, b);
  }

  // Extract scalar from size-1 array for scalar WASM path
  if (b.size === 1 && !isComplexDType(b.dtype) && !isComplexDType(a.dtype)) {
    const scalarVal = Number(b.iget(0));
    return divideScalar(a, scalarVal);
  }

  // Handle complex numbers
  const aIsComplex = isComplexDType(a.dtype);
  const bIsComplex = isComplexDType(b.dtype);

  if (aIsComplex || bIsComplex) {
    const dtype = promoteDTypes(a.dtype, b.dtype);

    // Try WASM for same-dtype complex divide
    if (aIsComplex && bIsComplex && a.dtype === b.dtype && canUseFastPath(a, b)) {
      const wasmResult = wasmDiv(a, b);
      if (wasmResult) return wasmResult;
    }

    const result = ArrayStorage.empty(Array.from(a.shape), dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const size = a.size;
    const aData = a.data;
    const bData = b.data;
    const aOff = a.offset;
    const bOff = b.offset;

    for (let i = 0; i < size; i++) {
      const [aRe, aIm] = aIsComplex
        ? getComplexAt(aData as Float64Array | Float32Array, aOff + i)
        : [Number(aData[aOff + i]), 0];
      const [bRe, bIm] = bIsComplex
        ? getComplexAt(bData as Float64Array | Float32Array, bOff + i)
        : [Number(bData[bOff + i]), 0];
      const denom = bRe * bRe + bIm * bIm;
      const re = (aRe * bRe + aIm * bIm) / denom;
      const im = (aIm * bRe - aRe * bIm) / denom;
      setComplexAt(resultData, i, re, im);
    }
    return result;
  }

  // Try WASM integer divide directly (avoids slow JS int→float conversion)
  if (a.dtype === b.dtype && canUseFastPath(a, b)) {
    const wasmResult = wasmDiv(a, b);
    if (wasmResult) return wasmResult;
  }

  // Determine result float dtype (NumPy: integer division always promotes to float)
  const aIsFloat64 = a.dtype === 'float64';
  const bIsFloat64 = b.dtype === 'float64';
  const aIsFloat32 = a.dtype === 'float32';
  const bIsFloat32 = b.dtype === 'float32';

  let targetDtype: 'float64' | 'float32';
  if (aIsFloat32 && bIsFloat32) {
    targetDtype = 'float32';
  } else if ((aIsFloat32 || bIsFloat32) && !aIsFloat64 && !bIsFloat64) {
    targetDtype = 'float32';
  } else {
    targetDtype = 'float64';
  }

  // Promote to target float dtype, then try WASM
  const aFloat = a.dtype === targetDtype ? a : convertToFloatDType(a, targetDtype);
  const bFloat = b.dtype === targetDtype ? b : convertToFloatDType(b, targetDtype);

  try {
    if (canUseFastPath(aFloat, bFloat)) {
      const wasmResult = wasmDiv(aFloat, bFloat);
      if (wasmResult) return wasmResult;
    }

    return elementwiseBinaryOp(aFloat, bFloat, (x, y) => x / y, 'divide');
  } finally {
    if (aFloat !== a) aFloat.dispose();
    if (bFloat !== b) bFloat.dispose();
  }
}

/**
 * Convert ArrayStorage to float dtype
 * @private
 */
function convertToFloatDType(
  storage: ArrayStorage,
  targetDtype: 'float32' | 'float64'
): ArrayStorage {
  const result = ArrayStorage.empty(Array.from(storage.shape), targetDtype);
  const size = storage.size;
  const dstData = result.data;

  if (storage.isCContiguous) {
    const srcData = storage.data;
    const off = storage.offset;
    for (let i = 0; i < size; i++) {
      dstData[i] = Number(srcData[off + i]!);
    }
  } else {
    for (let i = 0; i < size; i++) {
      dstData[i] = Number(storage.iget(i));
    }
  }

  return result;
}

/**
 * Add scalar to array (optimized path)
 * @private
 */
function addScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  // WASM acceleration for large contiguous arrays
  const wasmResult = wasmAddScalar(storage, scalar);
  if (wasmResult) return wasmResult;

  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const contiguous = storage.isCContiguous;

  // Create result with same dtype
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    // Complex: add scalar to real part only
    const srcData = data as Float64Array | Float32Array;
    const dstData = resultData as Float64Array | Float32Array;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        setComplexAt(dstData, i, re + scalar, im);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        setComplexAt(dstData, i, re + scalar, im);
      }
    }
  } else if (isBigIntDType(dtype)) {
    // BigInt arithmetic - no precision loss
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[off + i]! + scalarBig;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) + scalarBig;
      }
    }
  } else {
    // Regular numeric types
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(data[i]!) + scalar;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(data[off + i]!) + scalar;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) + scalar;
      }
    }
  }

  return result;
}

/**
 * Subtract scalar from array (optimized path)
 * @private
 */
function subtractScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const wasmResult = wasmSubScalar(storage, scalar);
  if (wasmResult) return wasmResult;

  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const contiguous = storage.isCContiguous;

  // Create result with same dtype
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    const srcData = data as Float64Array | Float32Array;
    const dstData = resultData as Float64Array | Float32Array;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        setComplexAt(dstData, i, re - scalar, im);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        setComplexAt(dstData, i, re - scalar, im);
      }
    }
  } else if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[off + i]! - scalarBig;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) - scalarBig;
      }
    }
  } else {
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(data[off + i]!) - scalar;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) - scalar;
      }
    }
  }

  return result;
}

/**
 * Multiply array by scalar (optimized path)
 * @private
 */
function multiplyScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const wasmResult = wasmMulScalar(storage, scalar);
  if (wasmResult) return wasmResult;

  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const contiguous = storage.isCContiguous;

  // Create result with same dtype
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    const srcData = data as Float64Array | Float32Array;
    const dstData = resultData as Float64Array | Float32Array;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        setComplexAt(dstData, i, re * scalar, im * scalar);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        setComplexAt(dstData, i, re * scalar, im * scalar);
      }
    }
  } else if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const scalarBig = BigInt(Math.round(scalar));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[off + i]! * scalarBig;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) * scalarBig;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(data[i]!) * scalar;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(data[off + i]!) * scalar;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(storage.iget(i)) * scalar;
      }
    }
  }

  return result;
}

/**
 * Divide array by scalar (optimized path)
 * NumPy behavior: Integer division promotes to float64
 * @private
 */
function divideScalar(storage: ArrayStorage, scalar: number): ArrayStorage {
  const dtype = storage.dtype;

  // Handle complex types (no WASM for complex divide)
  if (isComplexDType(dtype)) {
    const shape = Array.from(storage.shape);
    const data = storage.data;
    const size = storage.size;
    const off = storage.offset;
    const contiguous = storage.isCContiguous;

    const result = ArrayStorage.empty(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;
    if (contiguous) {
      const srcData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        setComplexAt(dstData, i, re / scalar, im / scalar);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        setComplexAt(dstData, i, re / scalar, im / scalar);
      }
    }
    return result;
  }

  // Try WASM directly (handles both float and integer dtypes)
  const wasmResult = wasmDivScalar(storage, scalar);
  if (wasmResult) return wasmResult;

  // Promote integer types to float64 for JS fallback
  const isFloat = dtype === 'float16' || dtype === 'float32' || dtype === 'float64';
  const promoted = isFloat ? storage : convertToFloatDType(storage, 'float64');

  try {
    // JS fallback
    const shape = Array.from(promoted.shape);
    const size = promoted.size;
    const result = ArrayStorage.empty(shape, promoted.dtype);
    const resultData = result.data;
    const data = promoted.data;
    const off = promoted.offset;

    if (promoted.isCContiguous) {
      for (let i = 0; i < size; i++) {
        resultData[i] = (data[off + i] as number) / scalar;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Number(promoted.iget(i)) / scalar;
      }
    }

    return result;
  } finally {
    if (promoted !== storage) promoted.dispose();
  }
}

/**
 * Absolute value of each element
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with absolute values
 */
export function absolute(a: ArrayStorage): ArrayStorage {
  // WASM acceleration (non-complex only — complex magnitude handled in JS)
  const wasmResult = wasmAbs(a);
  if (wasmResult) return wasmResult;

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;
  const contiguous = a.isCContiguous;

  // For complex types, result is the component dtype (magnitude is real)
  if (isComplexDType(dtype)) {
    const resultDtype = getComplexComponentDType(dtype);
    const result = ArrayStorage.empty(shape, resultDtype);
    const resultData = result.data as Float64Array | Float32Array;

    if (contiguous) {
      const srcData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        const re = srcData[(off + i) * 2]!;
        const im = srcData[(off + i) * 2 + 1]!;
        resultData[i] = Math.sqrt(re * re + im * im);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        resultData[i] = Math.sqrt(re * re + im * im);
      }
    }

    return result;
  }

  // Create result with same dtype for non-complex
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        const val = thisTyped[off + i]!;
        resultTyped[i] = val < 0n ? -val : val;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i) as bigint;
        resultTyped[i] = val < 0n ? -val : val;
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = Math.abs(Number(data[i]!));
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = Math.abs(Number(data[off + i]!));
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.abs(Number(a.iget(i)));
      }
    }
  }

  return result;
}

/**
 * Numerical negative (element-wise negation)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with negated values
 */
export function negative(a: ArrayStorage): ArrayStorage {
  throwIfBool(
    a.dtype,
    'negative',
    'The `-` operator is not supported for booleans, use `~` instead.'
  );

  const wasmResult = wasmNeg(a);
  if (wasmResult) return wasmResult;

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;
  const contiguous = a.isCContiguous;

  // Create result with same dtype
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    const dstData = resultData as Float64Array | Float32Array;
    if (contiguous) {
      const srcData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        const [re, im] = getComplexAt(srcData, off + i);
        setComplexAt(dstData, i, -re, -im);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        setComplexAt(dstData, i, -re, -im);
      }
    }
  } else if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = -thisTyped[off + i]!;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = -(a.iget(i) as bigint);
      }
    }
  } else {
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = -Number(data[i]!);
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = -Number(data[off + i]!);
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = -Number(a.iget(i));
      }
    }
  }

  return result;
}

/**
 * Sign of each element (-1, 0, or 1)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage with signs
 */
export function sign(a: ArrayStorage): ArrayStorage {
  throwIfComplex(a.dtype, 'sign', 'Sign is not defined for complex numbers.');
  throwIfBool(a.dtype, 'sign', 'Sign is not defined for boolean dtype.');

  const wasmResult = wasmSign(a);
  if (wasmResult) return wasmResult;

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;
  const contiguous = a.isCContiguous;

  // Create result with same dtype
  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        const val = thisTyped[off + i]!;
        resultTyped[i] = val > 0n ? 1n : val < 0n ? -1n : 0n;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i) as bigint;
        resultTyped[i] = val > 0n ? 1n : val < 0n ? -1n : 0n;
      }
    }
  } else {
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const val = Number(data[off + i]!);
        resultData[i] = val > 0 ? 1 : val < 0 ? -1 : 0;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(a.iget(i));
        resultData[i] = val > 0 ? 1 : val < 0 ? -1 : 0;
      }
    }
  }

  return result;
}

/**
 * Modulo operation (remainder after division)
 * NumPy behavior: Uses floor modulo (sign follows divisor), not JavaScript's truncate modulo
 * Preserves dtype for integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with modulo values
 */
export function mod(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  throwIfComplex(a.dtype, 'mod', 'Modulo is not defined for complex numbers.');
  if (typeof b !== 'number') {
    throwIfComplex(b.dtype, 'mod', 'Modulo is not defined for complex numbers.');
  }
  // NumPy promotes bool → int8 for mod
  if (a.dtype === 'bool')
    return mod(boolToInt8(a), typeof b === 'number' ? b : b.dtype === 'bool' ? boolToInt8(b) : b);
  if (typeof b !== 'number' && b.dtype === 'bool') return mod(a, boolToInt8(b));
  if (typeof b === 'number') {
    return modScalar(a, b);
  }
  // NumPy uses floor modulo: ((x % y) + y) % y for proper sign handling
  return elementwiseBinaryOp(a, b, (x, y) => ((x % y) + y) % y, 'mod');
}

/**
 * Modulo with scalar divisor (optimized path)
 * NumPy uses floor modulo: result has same sign as divisor
 * @private
 */
function modScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        const val = thisTyped[off + i]!;
        resultTyped[i] = ((val % divisorBig) + divisorBig) % divisorBig;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as bigint;
        resultTyped[i] = ((val % divisorBig) + divisorBig) % divisorBig;
      }
    }
  } else {
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const val = Number(data[off + i]!);
        resultData[i] = ((val % divisor) + divisor) % divisor;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(storage.iget(i));
        resultData[i] = ((val % divisor) + divisor) % divisor;
      }
    }
  }

  return result;
}

/**
 * Floor division (division with result rounded down)
 * NumPy behavior: Preserves integer types
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with floor division values
 */
export function floorDivide(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  throwIfComplex(a.dtype, 'floor_divide', 'Floor division is not defined for complex numbers.');
  if (typeof b !== 'number') {
    throwIfComplex(b.dtype, 'floor_divide', 'Floor division is not defined for complex numbers.');
  }
  if (a.dtype === 'bool')
    return floorDivide(
      boolToInt8(a),
      typeof b === 'number' ? b : b.dtype === 'bool' ? boolToInt8(b) : b
    );
  if (typeof b !== 'number' && b.dtype === 'bool') return floorDivide(a, boolToInt8(b));
  if (typeof b === 'number') {
    return floorDivideScalar(a, b);
  }
  return elementwiseBinaryOp(a, b, (x, y) => Math.floor(x / y), 'floor_divide');
}

/**
 * Floor division with scalar divisor (optimized path)
 * @private
 */
function floorDivideScalar(storage: ArrayStorage, divisor: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;
  const off = storage.offset;
  const contiguous = storage.isCContiguous;

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    const divisorBig = BigInt(Math.round(divisor));
    if (contiguous) {
      const thisTyped = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        resultTyped[i] = thisTyped[off + i]! / divisorBig;
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultTyped[i] = (storage.iget(i) as bigint) / divisorBig;
      }
    }
  } else {
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.floor(Number(data[off + i]!) / divisor);
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.floor(Number(storage.iget(i)) / divisor);
      }
    }
  }

  return result;
}

/**
 * Unary positive (returns a copy of the array)
 * Preserves dtype
 *
 * @param a - Input array storage
 * @returns Result storage (copy of input)
 */
export function positive(a: ArrayStorage): ArrayStorage {
  throwIfBool(a.dtype, 'positive', 'The `+` operator is not supported for booleans.');

  // Positive is essentially a no-op that returns a copy
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;
  const contiguous = a.isCContiguous;

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    const dstData = resultData as Float64Array | Float32Array;
    if (contiguous) {
      const srcData = data as Float64Array | Float32Array;
      dstData.set(srcData.subarray(off * 2, (off + size) * 2));
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i);
        dstData[i * 2] = (val as { re: number }).re ?? Number(val);
        dstData[i * 2 + 1] = (val as { im: number }).im ?? 0;
      }
    }
  } else {
    if (contiguous) {
      // Use bulk copy — TypedArray.set() is much faster than element-by-element
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (resultData as any).set((data as any).subarray(off, off + size));
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = a.iget(i) as number;
      }
    }
  }

  return result;
}

/**
 * Reciprocal (1/x) of each element
 * NumPy behavior: Always promotes to float64 for integer types
 * For complex: 1/z = z̄/|z|² = (re - i*im) / (re² + im²)
 *
 * @param a - Input array storage
 * @returns Result storage with reciprocal values
 */
export function reciprocal(a: ArrayStorage): ArrayStorage {
  // NumPy promotes bool → int8 for reciprocal (integer division: 1/True=1, 1/False=UB→0)
  if (a.dtype === 'bool') {
    const shape = Array.from(a.shape);
    const promoted = ArrayStorage.empty(shape, 'int8');
    const promData = promoted.data as Int8Array;
    const srcData = a.data;
    const off = a.offset;
    for (let i = 0; i < a.size; i++) {
      const val = a.isCContiguous ? (srcData[off + i] as number) : Number(a.iget(i));
      // Integer reciprocal: 1/1=1, 1/0 is UB in C — use -1 matching NumPy int8 behavior
      promData[i] = val !== 0 ? (1 / val) | 0 : -1;
    }
    return promoted;
  }

  // WASM acceleration for float types
  const wasmResult = wasmReciprocal(a);
  if (wasmResult) return wasmResult;

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;
  const contiguous = a.isCContiguous;

  // Handle complex types
  if (isComplexDType(dtype)) {
    const result = ArrayStorage.empty(shape, dtype);
    const dstData = result.data as Float64Array | Float32Array;

    if (contiguous) {
      const srcData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        const re = srcData[(off + i) * 2]!;
        const im = srcData[(off + i) * 2 + 1]!;
        const magSq = re * re + im * im;
        dstData[i * 2] = re / magSq;
        dstData[i * 2 + 1] = -im / magSq;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = a.iget(i);
        const re = (val as { re: number }).re ?? Number(val);
        const im = (val as { im: number }).im ?? 0;
        const magSq = re * re + im * im;
        dstData[i * 2] = re / magSq;
        dstData[i * 2 + 1] = -im / magSq;
      }
    }

    return result;
  }

  // NumPy keeps dtype for reciprocal: float→float division, int→integer division (truncated)
  const isIntegerType = isIntegerDType(dtype);

  if (isIntegerType) {
    // Integer reciprocal: Math.trunc(1/val), matching NumPy behavior
    const result = ArrayStorage.empty(shape, dtype);
    const resultData = result.data;
    if (isBigIntDType(dtype)) {
      const rd = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        const val = contiguous ? (data[off + i] as bigint) : (a.iget(i) as bigint);
        rd[i] = val !== 0n ? 1n / val : -1n;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = contiguous ? (data[off + i] as number) : Number(a.iget(i));
        resultData[i] = val !== 0 ? Math.trunc(1 / val) : 0;
      }
    }
    return result;
  }

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  if (contiguous) {
    if (off === 0) {
      for (let i = 0; i < size; i++) {
        resultData[i] = 1.0 / Number(data[i]!);
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = 1.0 / Number(data[off + i]!);
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = 1.0 / Number(a.iget(i));
    }
  }

  return result;
}

/**
 * Cube root of each element
 * NumPy behavior: Promotes integer types to float64
 *
 * @param a - Input array storage
 * @returns Result storage with cube root values
 */
export function cbrt(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  throwIfComplex(dtype, 'cbrt', 'cbrt is not supported for complex numbers.');
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;

  const resultDtype = mathResultDtype(dtype);

  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.cbrt(Number(data[off + i]!));
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.cbrt(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Absolute value of each element, returning float
 * NumPy behavior: fabs always returns floating point
 *
 * @param a - Input array storage
 * @returns Result storage with absolute values as float
 */
export function fabs(a: ArrayStorage): ArrayStorage {
  const dtype = a.dtype;
  throwIfComplex(dtype, 'fabs', 'fabs is only for real numbers. Use absolute() for complex.');
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;
  const off = a.offset;

  const resultDtype = mathResultDtype(dtype);

  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;

  if (a.isCContiguous) {
    if (off === 0) {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.abs(Number(data[i]!));
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.abs(Number(data[off + i]!));
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.abs(Number(a.iget(i)));
    }
  }

  return result;
}

/**
 * Returns both quotient and remainder (floor divide and modulo)
 * NumPy behavior: divmod(a, b) = (floor_divide(a, b), mod(a, b))
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Tuple of [quotient, remainder] storages
 */
export function divmod(a: ArrayStorage, b: ArrayStorage | number): [ArrayStorage, ArrayStorage] {
  // Fused WASM path for scalar divisor (single pass over data)
  if (typeof b === 'number') {
    const wasm = wasmDivmodScalar(a, b);
    if (wasm) return wasm;
  }
  const quotient = floorDivide(a, b);
  const remainder = mod(a, b);
  return [quotient, remainder];
}

/**
 * Element-wise square of each element
 * NumPy behavior: x**2
 * For complex: z² = (re + i*im)² = (re² - im²) + i*(2*re*im)
 *
 * @param a - Input array storage
 * @returns Result storage with squared values
 */
export function square(a: ArrayStorage): ArrayStorage {
  // WASM acceleration for non-complex types
  const wasmResult = wasmSquare(a);
  if (wasmResult) return wasmResult;

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const data = a.data;
  const size = a.size;

  // NumPy promotes bool → int8 for square
  const resultDtype = boolArithmeticDtype(dtype);
  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;

  if (isComplexDType(dtype)) {
    // Complex: z² = (re + i*im)² = (re² - im²) + i*(2*re*im)
    const srcData = data as Float64Array | Float32Array;
    const dstData = resultData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const re = srcData[i * 2]!;
      const im = srcData[i * 2 + 1]!;
      dstData[i * 2] = re * re - im * im;
      dstData[i * 2 + 1] = 2 * re * im;
    }
  } else if (isBigIntDType(dtype)) {
    const bigData = data as BigInt64Array | BigUint64Array;
    const bigResultData = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      bigResultData[i] = bigData[i]! * bigData[i]!;
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(data[i]!);
      resultData[i] = val * val;
    }
  }

  return result;
}

/**
 * Remainder of division (same as mod)
 * NumPy behavior: Same as mod, alias for compatibility
 *
 * @param a - Dividend array storage
 * @param b - Divisor (array storage or scalar)
 * @returns Result storage with remainder values
 */
export function remainder(a: ArrayStorage, b: ArrayStorage | number): ArrayStorage {
  return mod(a, b);
}

/**
 * Heaviside step function
 * NumPy behavior:
 *   heaviside(x1, x2) = 0 if x1 < 0
 *                     = x2 if x1 == 0
 *                     = 1 if x1 > 0
 *
 * @param x1 - Input array storage
 * @param x2 - Value to use when x1 == 0 (array storage or scalar)
 * @returns Result storage with heaviside values
 */
export function heaviside(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  // NumPy promotes bool → float16 for heaviside
  if (x1.dtype === 'bool')
    return heaviside(
      boolToMathFloat(x1),
      typeof x2 === 'number' ? x2 : x2.dtype === 'bool' ? boolToMathFloat(x2) : x2
    );
  if (typeof x2 !== 'number' && x2.dtype === 'bool') return heaviside(x1, boolToMathFloat(x2));
  throwIfComplex(
    x1.dtype,
    'heaviside',
    'Heaviside step function is not defined for complex numbers.'
  );
  if (typeof x2 !== 'number') {
    throwIfComplex(
      x2.dtype,
      'heaviside',
      'Heaviside step function is not defined for complex numbers.'
    );
  }
  const dtype = x1.dtype;
  const shape = Array.from(x1.shape);
  const size = x1.size;

  const resultDtype = mathResultDtype(dtype);

  // Promote input to result dtype for WASM and fast paths
  const x1Float =
    x1.dtype === resultDtype
      ? x1
      : resultDtype === 'float16'
        ? x1
        : convertToFloatDType(x1, resultDtype as 'float32' | 'float64');

  // Extract scalar from size-1 array
  if (typeof x2 !== 'number' && x2.size === 1) {
    if (x1Float !== x1) x1Float.dispose();
    return heaviside(x1, Number(x2.iget(0)));
  }

  try {
    if (typeof x2 === 'number') {
      // Try WASM scalar path
      const wasmResult = wasmHeavisideScalar(
        x1Float,
        x2,
        resultDtype as 'float64' | 'float32' | 'float16'
      );
      if (wasmResult) return wasmResult;

      // Scalar x2 — use direct data access for contiguous
      const result = ArrayStorage.empty(shape, resultDtype);
      const resultData = result.data;
      if (x1Float.isCContiguous) {
        const srcData = x1Float.data;
        const off = x1Float.offset;
        for (let i = 0; i < size; i++) {
          const val = srcData[off + i] as number;
          resultData[i] = val < 0 ? 0 : val === 0 ? x2 : 1;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = Number(x1.iget(i));
          resultData[i] = val < 0 ? 0 : val === 0 ? x2 : 1;
        }
      }
      return result;
    } else {
      // Array x2 - needs to broadcast
      const x2Shape = x2.shape;
      const x2Float =
        x2.dtype === resultDtype
          ? x2
          : resultDtype === 'float16'
            ? x2
            : convertToFloatDType(x2, resultDtype as 'float32' | 'float64');

      try {
        // Simple case: same shape
        if (shape.every((d, i) => d === x2Shape[i])) {
          // Try WASM binary path
          const wasmResult = wasmHeaviside(
            x1Float,
            x2Float,
            resultDtype as 'float64' | 'float32' | 'float16'
          );
          if (wasmResult) return wasmResult;

          const result = ArrayStorage.empty(shape, resultDtype);
          const resultData = result.data;
          if (x1Float.isCContiguous && x2Float.isCContiguous) {
            const x1Data = x1Float.data;
            const x1Off = x1Float.offset;
            const x2Data = x2Float.data;
            const x2Off = x2Float.offset;
            for (let i = 0; i < size; i++) {
              const val = x1Data[x1Off + i] as number;
              resultData[i] = val < 0 ? 0 : val === 0 ? (x2Data[x2Off + i] as number) : 1;
            }
          } else {
            for (let i = 0; i < size; i++) {
              const val = Number(x1.iget(i));
              resultData[i] = val < 0 ? 0 : val === 0 ? Number(x2.iget(i)) : 1;
            }
          }
          return result;
        } else {
          // Broadcasting case
          const result = ArrayStorage.empty(shape, resultDtype);
          const resultData = result.data;
          for (let i = 0; i < size; i++) {
            const val = Number(x1.iget(i));
            const x2Idx = i % x2.size;
            resultData[i] = val < 0 ? 0 : val === 0 ? Number(x2.iget(x2Idx)) : 1;
          }
          return result;
        }
      } finally {
        if (x2Float !== x2) x2Float.dispose();
      }
    }
  } finally {
    if (x1Float !== x1) x1Float.dispose();
  }
}

/**
 * First array raised to power of second, always promoting to float
 * @param x1 - Base values
 * @param x2 - Exponent values
 * @returns Result in float64
 */
export function float_power(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  const dtype1 = x1.dtype;

  // Complex float_power: z1^z2 = exp(z2 * log(z1))
  if (isComplexDType(dtype1)) {
    const size = x1.size;
    const result = ArrayStorage.empty(Array.from(x1.shape), dtype1);
    const resultData = result.data as Float64Array | Float32Array;

    if (typeof x2 === 'number') {
      if (x1.isCContiguous) {
        const x1Data = x1.data as Float64Array | Float32Array;
        const x1Off = x1.offset;
        for (let i = 0; i < size; i++) {
          const re = x1Data[(x1Off + i) * 2]!;
          const im = x1Data[(x1Off + i) * 2 + 1]!;

          // z^n = |z|^n * exp(i * n * arg(z))
          const mag = Math.hypot(re, im);
          const arg = Math.atan2(im, re);
          const newMag = Math.pow(mag, x2);
          const newArg = arg * x2;

          resultData[i * 2] = newMag * Math.cos(newArg);
          resultData[i * 2 + 1] = newMag * Math.sin(newArg);
        }
      } else {
        for (let i = 0; i < size; i++) {
          const v = x1.iget(i) as Complex;

          const mag = Math.hypot(v.re, v.im);
          const arg = Math.atan2(v.im, v.re);
          const newMag = Math.pow(mag, x2);
          const newArg = arg * x2;

          resultData[i * 2] = newMag * Math.cos(newArg);
          resultData[i * 2 + 1] = newMag * Math.sin(newArg);
        }
      }
    } else {
      // Complex base with array exponent
      const x2Complex = isComplexDType(x2.dtype);

      if (x1.isCContiguous && x2.isCContiguous) {
        const x1Data = x1.data as Float64Array | Float32Array;
        const x1Off = x1.offset;
        const x2Data = x2.data;
        const x2Off = x2.offset;
        for (let i = 0; i < size; i++) {
          const re1 = x1Data[(x1Off + i) * 2]!;
          const im1 = x1Data[(x1Off + i) * 2 + 1]!;

          let re2: number, im2: number;
          if (x2Complex) {
            const x2Typed = x2Data as Float64Array | Float32Array;
            re2 = x2Typed[(x2Off + i) * 2]!;
            im2 = x2Typed[(x2Off + i) * 2 + 1]!;
          } else {
            re2 = Number(x2Data[x2Off + i]!);
            im2 = 0;
          }

          // z1^z2 = exp(z2 * log(z1))
          // log(z1) = ln|z1| + i*arg(z1)
          const mag1 = Math.hypot(re1, im1);
          const arg1 = Math.atan2(im1, re1);
          const logRe = Math.log(mag1);
          const logIm = arg1;

          // z2 * log(z1)
          const prodRe = re2 * logRe - im2 * logIm;
          const prodIm = re2 * logIm + im2 * logRe;

          // exp(prodRe + i*prodIm)
          const expMag = Math.exp(prodRe);
          resultData[i * 2] = expMag * Math.cos(prodIm);
          resultData[i * 2 + 1] = expMag * Math.sin(prodIm);
        }
      } else {
        for (let i = 0; i < size; i++) {
          const v1 = x1.iget(i) as Complex;

          let re2: number, im2: number;
          if (x2Complex) {
            const v2 = x2.iget(i) as Complex;
            re2 = v2.re;
            im2 = v2.im;
          } else {
            re2 = Number(x2.iget(i));
            im2 = 0;
          }

          const mag1 = Math.hypot(v1.re, v1.im);
          const arg1 = Math.atan2(v1.im, v1.re);
          const logRe = Math.log(mag1);
          const logIm = arg1;

          const prodRe = re2 * logRe - im2 * logIm;
          const prodIm = re2 * logIm + im2 * logRe;

          const expMag = Math.exp(prodRe);
          resultData[i * 2] = expMag * Math.cos(prodIm);
          resultData[i * 2 + 1] = expMag * Math.sin(prodIm);
        }
      }
    }

    return result;
  }

  if (typeof x2 === 'number') {
    const result = ArrayStorage.empty(Array.from(x1.shape), 'float64');
    const resultData = result.data as Float64Array;
    const size = x1.size;

    if (x1.isCContiguous) {
      const x1Data = x1.data;
      const x1Off = x1.offset;
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.pow(Number(x1Data[x1Off + i]!), x2);
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = Math.pow(Number(x1.iget(i)), x2);
      }
    }

    return result;
  }

  // float_power always returns float64 (unlike power which preserves dtype)
  const result = ArrayStorage.empty(Array.from(x1.shape), 'float64');
  const resultData = result.data as Float64Array;
  const size = x1.size;
  for (let i = 0; i < size; i++) {
    resultData[i] = Math.pow(Number(x1.iget(i)), Number(x2.iget(i)));
  }
  return result;
}

/**
 * Element-wise remainder of division (fmod)
 * Unlike mod/remainder, fmod matches C fmod behavior
 * @param x1 - Dividend
 * @param x2 - Divisor
 * @returns Remainder
 */
export function fmod(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'fmod', 'fmod is not defined for complex numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'fmod', 'fmod is not defined for complex numbers.');
  }
  if (x1.dtype === 'bool')
    return fmod(
      boolToInt8(x1),
      typeof x2 === 'number' ? x2 : x2.dtype === 'bool' ? boolToInt8(x2) : x2
    );
  if (typeof x2 !== 'number' && x2.dtype === 'bool') return fmod(x1, boolToInt8(x2));
  if (typeof x2 === 'number') {
    const result = x1.copy();
    const resultData = result.data;
    const size = x1.size;

    for (let i = 0; i < size; i++) {
      const val = Number(resultData[i]!);
      resultData[i] = val - Math.trunc(val / x2) * x2;
    }

    return result;
  }

  return elementwiseBinaryOp(x1, x2, (a, b) => a - Math.trunc(a / b) * b, 'fmod');
}

/**
 * Decompose floating point numbers into mantissa and exponent
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent
 * @param x - Input array
 * @returns Tuple of [mantissa, exponent] arrays
 */
export function frexp(x: ArrayStorage): [ArrayStorage, ArrayStorage] {
  throwIfComplex(x.dtype, 'frexp', 'frexp is not defined for complex numbers.');

  // NumPy promotes bool → float16 for frexp
  if (x.dtype === 'bool') {
    const converted = boolToMathFloat(x);
    const [m, e] = frexp(converted);
    converted.dispose();
    return [m, e];
  }

  // Try WASM path
  const wasmResult = wasmFrexp(x);
  if (wasmResult) return wasmResult;

  const mantissaDtype = mathResultDtype(x.dtype);
  const mantissa = ArrayStorage.empty(Array.from(x.shape), mantissaDtype);
  const exponent = ArrayStorage.empty(Array.from(x.shape), 'int32');
  const mantissaData = mantissa.data;
  const exponentData = exponent.data as Int32Array;
  const size = x.size;

  // Use direct data access for contiguous arrays
  if (x.isCContiguous) {
    const data = x.data;
    const off = x.offset;
    for (let i = 0; i < size; i++) {
      const val = Number(data[off + i]!);
      if (val === 0 || !isFinite(val)) {
        mantissaData[i] = val;
        exponentData[i] = 0;
      } else {
        const exp = Math.floor(Math.log2(Math.abs(val))) + 1;
        mantissaData[i] = val / Math.pow(2, exp);
        exponentData[i] = exp;
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(x.iget(i));
      if (val === 0 || !isFinite(val)) {
        mantissaData[i] = val;
        exponentData[i] = 0;
      } else {
        const exp = Math.floor(Math.log2(Math.abs(val))) + 1;
        mantissaData[i] = val / Math.pow(2, exp);
        exponentData[i] = exp;
      }
    }
  }

  return [mantissa, exponent];
}

/**
 * Greatest common divisor
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns GCD
 */
export function gcd(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'gcd', 'GCD is only defined for integers.');
  throwIfBool(x1.dtype, 'gcd', 'GCD is only defined for integer dtypes.');
  if (isFloatDType(x1.dtype))
    throw new TypeError(
      `ufunc 'gcd' not supported for float dtype '${x1.dtype}'. GCD is only defined for integer dtypes.`
    );
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'gcd', 'GCD is only defined for integers.');
    throwIfBool(x2.dtype, 'gcd', 'GCD is only defined for integer dtypes.');
    if (isFloatDType(x2.dtype))
      throw new TypeError(
        `ufunc 'gcd' not supported for float dtype '${x2.dtype}'. GCD is only defined for integer dtypes.`
      );
  }
  const gcdSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    while (b !== 0) {
      const temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  };

  // NumPy preserves the promoted integer dtype
  const outDtype = typeof x2 === 'number' ? x1.dtype : promoteDTypes(x1.dtype, x2.dtype);

  if (typeof x2 === 'number') {
    // Try WASM scalar path
    const wasmResult = wasmGcdScalar(x1, x2);
    if (wasmResult) return wasmResult;

    const result = ArrayStorage.empty(Array.from(x1.shape), outDtype);
    const resultData = result.data;
    const size = x1.size;
    const x2Int = Math.abs(Math.trunc(x2));

    if (x1.isCContiguous) {
      const x1Data = x1.data;
      const x1Off = x1.offset;
      for (let i = 0; i < size; i++) {
        resultData[i] = gcdSingle(Number(x1Data[x1Off + i]!), x2Int);
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = gcdSingle(Number(x1.iget(i)), x2Int);
      }
    }

    return result;
  }

  // Check size-1 scalar extraction
  if (typeof x2 !== 'number' && x2.size === 1) {
    const scalarVal = Number(x2.iget(0));
    const wasmResult = wasmGcdScalar(x1, scalarVal);
    if (wasmResult) return wasmResult;
  }

  // Try WASM binary path
  if (typeof x2 !== 'number' && canUseFastPath(x1, x2)) {
    const wasmResult = wasmGcd(x1, x2);
    if (wasmResult) return wasmResult;
  }

  // Array case — compute gcd preserving promoted dtype
  const result = ArrayStorage.empty(Array.from(x1.shape), outDtype);
  const resultData = result.data;
  const size = x1.size;

  if (isBigIntDType(outDtype)) {
    const gcdBig = (a: bigint, b: bigint): bigint => {
      a = a < 0n ? -a : a;
      b = b < 0n ? -b : b;
      while (b !== 0n) {
        const t = b;
        b = a % b;
        a = t;
      }
      return a;
    };
    const x1Data = x1.data as BigInt64Array | BigUint64Array;
    const x2Data = (x2 as ArrayStorage).data as BigInt64Array | BigUint64Array;
    const x1Off = x1.offset;
    const x2Off = (x2 as ArrayStorage).offset;
    for (let i = 0; i < size; i++) {
      (resultData as BigInt64Array | BigUint64Array)[i] = gcdBig(
        x1Data[x1Off + i]!,
        x2Data[x2Off + i]!
      );
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = gcdSingle(Number(x1.iget(i)), Number((x2 as ArrayStorage).iget(i)));
    }
  }

  return result;
}

/**
 * Least common multiple
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns LCM
 */
export function lcm(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'lcm', 'LCM is only defined for integers.');
  throwIfBool(x1.dtype, 'lcm', 'LCM is only defined for integer dtypes.');
  if (isFloatDType(x1.dtype))
    throw new TypeError(
      `ufunc 'lcm' not supported for float dtype '${x1.dtype}'. LCM is only defined for integer dtypes.`
    );
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'lcm', 'LCM is only defined for integers.');
    throwIfBool(x2.dtype, 'lcm', 'LCM is only defined for integer dtypes.');
    if (isFloatDType(x2.dtype))
      throw new TypeError(
        `ufunc 'lcm' not supported for float dtype '${x2.dtype}'. LCM is only defined for integer dtypes.`
      );
  }
  const gcdSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    while (b !== 0) {
      const temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  };

  const lcmSingle = (a: number, b: number): number => {
    a = Math.abs(Math.trunc(a));
    b = Math.abs(Math.trunc(b));
    if (a === 0 || b === 0) return 0;
    return (a * b) / gcdSingle(a, b);
  };

  // NumPy preserves the promoted integer dtype
  const outDtype = typeof x2 === 'number' ? x1.dtype : promoteDTypes(x1.dtype, x2.dtype);

  if (typeof x2 === 'number') {
    const result = ArrayStorage.empty(Array.from(x1.shape), outDtype);
    const resultData = result.data;
    const size = x1.size;
    const x2Int = Math.abs(Math.trunc(x2));

    for (let i = 0; i < size; i++) {
      resultData[i] = lcmSingle(Number(x1.iget(i)), x2Int);
    }

    return result;
  }

  // Array case — compute lcm preserving promoted dtype
  const result = ArrayStorage.empty(Array.from(x1.shape), outDtype);
  const resultData = result.data;
  const size = x1.size;

  if (isBigIntDType(outDtype)) {
    const gcdBig = (a: bigint, b: bigint): bigint => {
      a = a < 0n ? -a : a;
      b = b < 0n ? -b : b;
      while (b !== 0n) {
        const t = b;
        b = a % b;
        a = t;
      }
      return a;
    };
    const lcmBig = (a: bigint, b: bigint): bigint => {
      a = a < 0n ? -a : a;
      b = b < 0n ? -b : b;
      if (a === 0n || b === 0n) return 0n;
      return (a * b) / gcdBig(a, b);
    };
    const x1Data = x1.data as BigInt64Array | BigUint64Array;
    const x2Data = x2.data as BigInt64Array | BigUint64Array;
    const x1Off = x1.offset;
    const x2Off = x2.offset;
    for (let i = 0; i < size; i++) {
      (resultData as BigInt64Array | BigUint64Array)[i] = lcmBig(
        x1Data[x1Off + i]!,
        x2Data[x2Off + i]!
      );
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = lcmSingle(Number(x1.iget(i)), Number(x2.iget(i)));
    }
  }

  return result;
}

/**
 * Returns x1 * 2^x2, element-wise
 * @param x1 - Mantissa
 * @param x2 - Exponent
 * @returns Result
 */
export function ldexp(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'ldexp', 'ldexp is not defined for complex numbers.');

  // NumPy promotes bool → float16 for ldexp
  if (x1.dtype === 'bool') {
    const converted = boolToMathFloat(x1);
    const result = ldexp(converted, x2);
    converted.dispose();
    return result;
  }

  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'ldexp', 'ldexp is not defined for complex numbers.');
    // Extract scalar from size-1 array
    if (x2.size === 1) {
      return ldexp(x1, Number(x2.iget(0)));
    }
  }
  if (typeof x2 === 'number') {
    const resultDtype = mathResultDtype(x1.dtype);
    const x1Float =
      x1.dtype === resultDtype
        ? x1
        : resultDtype === 'float16'
          ? x1
          : convertToFloatDType(x1, resultDtype as 'float32' | 'float64');

    try {
      // Try WASM scalar path
      const wasmResult = wasmLdexpScalar(x1Float, x2);
      if (wasmResult) return wasmResult;

      const result = ArrayStorage.empty(Array.from(x1.shape), resultDtype);
      const resultData = result.data;
      const size = x1.size;
      const multiplier = Math.pow(2, x2);

      if (x1.isCContiguous) {
        const data = x1.data;
        const off = x1.offset;
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(data[off + i]!) * multiplier;
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = Number(x1.iget(i)) * multiplier;
        }
      }

      return result;
    } finally {
      if (x1Float !== x1) x1Float.dispose();
    }
  }

  // Array x2: output dtype follows x1 only (not standard binary promotion)
  const resultDtype = mathResultDtype(x1.dtype);
  const result = ArrayStorage.empty(Array.from(x1.shape), resultDtype);
  const resultData = result.data;
  const size = x1.size;
  for (let i = 0; i < size; i++) {
    resultData[i] = Number(x1.iget(i)) * Math.pow(2, Number(x2.iget(i)));
  }
  return result;
}

/**
 * Return fractional and integral parts of array
 * @param x - Input array
 * @returns Tuple of [fractional, integral] arrays
 */
export function modf(x: ArrayStorage): [ArrayStorage, ArrayStorage] {
  throwIfComplex(x.dtype, 'modf', 'modf is not defined for complex numbers.');
  const fractional = ArrayStorage.empty(Array.from(x.shape), 'float64');
  const integral = ArrayStorage.empty(Array.from(x.shape), 'float64');
  const fractionalData = fractional.data as Float64Array;
  const integralData = integral.data as Float64Array;
  const size = x.size;

  for (let i = 0; i < size; i++) {
    const val = Number(x.iget(i));
    const intPart = Math.trunc(val);
    integralData[i] = intPart;
    fractionalData[i] = val - intPart;
  }

  return [fractional, integral];
}

/**
 * Clip (limit) the values in an array
 * Given an interval, values outside the interval are clipped to the interval edges.
 *
 * @param a - Input array storage
 * @param a_min - Minimum value (null to not clip minimum)
 * @param a_max - Maximum value (null to not clip maximum)
 * @returns Clipped array storage
 */
export function clip(
  a: ArrayStorage,
  a_min: number | ArrayStorage | null,
  a_max: number | ArrayStorage | null
): ArrayStorage {
  throwIfComplex(a.dtype, 'clip', 'clip is not supported for complex numbers.');

  // NumPy promotes bool → int64 for clip (Python int bounds are int64)
  if (a.dtype === 'bool') {
    const converted = ArrayStorage.empty(Array.from(a.shape), 'int64');
    const src = a.data as Uint8Array;
    const dst = converted.data as BigInt64Array;
    const off = a.offset;
    for (let i = 0; i < a.size; i++) dst[i] = BigInt(src[off + i]!);
    const result = clip(converted, a_min, a_max);
    converted.dispose();
    return result;
  }

  // WASM acceleration for scalar bounds
  if (
    (a_min === null || typeof a_min === 'number') &&
    (a_max === null || typeof a_max === 'number')
  ) {
    const lo = a_min === null ? -Infinity : a_min;
    const hi = a_max === null ? Infinity : a_max;
    const wasmResult = wasmClip(a, lo, hi);
    if (wasmResult) return wasmResult;
  }

  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;

  // Handle scalar min/max values
  const minIsScalar = a_min === null || typeof a_min === 'number';
  const maxIsScalar = a_max === null || typeof a_max === 'number';

  const minScalar = a_min === null ? -Infinity : typeof a_min === 'number' ? a_min : null;
  const maxScalar = a_max === null ? Infinity : typeof a_max === 'number' ? a_max : null;

  // Check contiguity of all array inputs
  const aContiguous = a.isCContiguous;
  const aData = a.data;
  const aOff = a.offset;
  const minContiguous = !minIsScalar && (a_min as ArrayStorage).isCContiguous;
  const minData = !minIsScalar ? (a_min as ArrayStorage).data : null;
  const minOff = !minIsScalar ? (a_min as ArrayStorage).offset : 0;
  const minSize = !minIsScalar ? (a_min as ArrayStorage).size : 0;
  const maxContiguous = !maxIsScalar && (a_max as ArrayStorage).isCContiguous;
  const maxData = !maxIsScalar ? (a_max as ArrayStorage).data : null;
  const maxOff = !maxIsScalar ? (a_max as ArrayStorage).offset : 0;
  const maxSize = !maxIsScalar ? (a_max as ArrayStorage).size : 0;

  if (isBigIntDType(dtype)) {
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    if (aContiguous && (minIsScalar || minContiguous) && (maxIsScalar || maxContiguous)) {
      const aTyped = aData as BigInt64Array | BigUint64Array;
      const minTyped = minData as BigInt64Array | BigUint64Array | null;
      const maxTyped = maxData as BigInt64Array | BigUint64Array | null;
      for (let i = 0; i < size; i++) {
        let val = aTyped[aOff + i]!;
        const minVal = minIsScalar
          ? minScalar === -Infinity
            ? val
            : BigInt(Math.round(minScalar as number))
          : minTyped![minOff + (i % minSize)]!;
        const maxVal = maxIsScalar
          ? maxScalar === Infinity
            ? val
            : BigInt(Math.round(maxScalar as number))
          : maxTyped![maxOff + (i % maxSize)]!;

        if (val < minVal) val = minVal;
        if (val > maxVal) val = maxVal;
        resultTyped[i] = val;
      }
    } else {
      for (let i = 0; i < size; i++) {
        let val = a.iget(i) as bigint;
        const minVal = minIsScalar
          ? minScalar === -Infinity
            ? val
            : BigInt(Math.round(minScalar as number))
          : ((a_min as ArrayStorage).iget(i % (a_min as ArrayStorage).size) as bigint);
        const maxVal = maxIsScalar
          ? maxScalar === Infinity
            ? val
            : BigInt(Math.round(maxScalar as number))
          : ((a_max as ArrayStorage).iget(i % (a_max as ArrayStorage).size) as bigint);

        if (val < minVal) val = minVal;
        if (val > maxVal) val = maxVal;
        resultTyped[i] = val;
      }
    }
  } else {
    if (aContiguous && (minIsScalar || minContiguous) && (maxIsScalar || maxContiguous)) {
      for (let i = 0; i < size; i++) {
        let val = Number(aData[aOff + i]!);
        const minVal = minIsScalar
          ? (minScalar as number)
          : Number(minData![minOff + (i % minSize)]!);
        const maxVal = maxIsScalar
          ? (maxScalar as number)
          : Number(maxData![maxOff + (i % maxSize)]!);

        if (val < minVal) val = minVal;
        if (val > maxVal) val = maxVal;
        resultData[i] = val;
      }
    } else {
      for (let i = 0; i < size; i++) {
        let val = Number(a.iget(i));
        const minVal = minIsScalar
          ? (minScalar as number)
          : Number((a_min as ArrayStorage).iget(i % (a_min as ArrayStorage).size));
        const maxVal = maxIsScalar
          ? (maxScalar as number)
          : Number((a_max as ArrayStorage).iget(i % (a_max as ArrayStorage).size));

        if (val < minVal) val = minVal;
        if (val > maxVal) val = maxVal;
        resultData[i] = val;
      }
    }
  }

  return result;
}

/**
 * Element-wise maximum of array elements
 *
 * @param x1 - First array storage
 * @param x2 - Second array storage or scalar
 * @returns Element-wise maximum
 */
export function maximum(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'maximum', 'maximum is not supported for complex numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'maximum', 'maximum is not supported for complex numbers.');
  }

  // WASM acceleration
  if (typeof x2 === 'number') {
    const wasmResult = wasmMaxScalar(x1, x2);
    if (wasmResult) return wasmResult;
  } else if (x2.size === 1) {
    const wasmResult = wasmMaxScalar(x1, Number(x2.iget(0)));
    if (wasmResult) return wasmResult;
  } else if (canUseFastPath(x1, x2)) {
    const wasmResult = wasmMax(x1, x2);
    if (wasmResult) return wasmResult;
  }

  if (typeof x2 === 'number') {
    const dtype = x1.dtype;
    const shape = Array.from(x1.shape);
    const size = x1.size;
    const result = ArrayStorage.empty(shape, dtype);
    const resultData = result.data;
    const x1Data = x1.data;

    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const x1Typed = x1Data as BigInt64Array | BigUint64Array;
      const x2Big = BigInt(Math.round(x2));
      for (let i = 0; i < size; i++) {
        resultTyped[i] = x1Typed[i]! > x2Big ? x1Typed[i]! : x2Big;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(x1Data[i]!);
        // NaN propagation: Math.max doesn't handle NaN correctly
        resultData[i] = isNaN(val) || isNaN(x2) ? NaN : Math.max(val, x2);
      }
    }
    return result;
  }

  return elementwiseBinaryOp(
    x1,
    x2,
    (a, b) => (isNaN(a) || isNaN(b) ? NaN : Math.max(a, b)),
    'maximum'
  );
}

/**
 * Element-wise minimum of array elements
 *
 * @param x1 - First array storage
 * @param x2 - Second array storage or scalar
 * @returns Element-wise minimum
 */
export function minimum(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'minimum', 'minimum is not supported for complex numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'minimum', 'minimum is not supported for complex numbers.');
  }

  // WASM acceleration
  if (typeof x2 === 'number') {
    const wasmResult = wasmMinScalar(x1, x2);
    if (wasmResult) return wasmResult;
  } else if (x2.size === 1) {
    const wasmResult = wasmMinScalar(x1, Number(x2.iget(0)));
    if (wasmResult) return wasmResult;
  } else if (canUseFastPath(x1, x2)) {
    const wasmResult = wasmMin(x1, x2);
    if (wasmResult) return wasmResult;
  }

  if (typeof x2 === 'number') {
    const dtype = x1.dtype;
    const shape = Array.from(x1.shape);
    const size = x1.size;
    const result = ArrayStorage.empty(shape, dtype);
    const resultData = result.data;
    const x1Data = x1.data;

    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const x1Typed = x1Data as BigInt64Array | BigUint64Array;
      const x2Big = BigInt(Math.round(x2));
      for (let i = 0; i < size; i++) {
        resultTyped[i] = x1Typed[i]! < x2Big ? x1Typed[i]! : x2Big;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(x1Data[i]!);
        // NaN propagation
        resultData[i] = isNaN(val) || isNaN(x2) ? NaN : Math.min(val, x2);
      }
    }
    return result;
  }

  return elementwiseBinaryOp(
    x1,
    x2,
    (a, b) => (isNaN(a) || isNaN(b) ? NaN : Math.min(a, b)),
    'minimum'
  );
}

/**
 * Element-wise maximum of array elements, ignoring NaNs
 *
 * @param x1 - First array storage
 * @param x2 - Second array storage or scalar
 * @returns Element-wise maximum, NaN-aware
 */
export function fmax(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'fmax', 'fmax is not supported for complex numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'fmax', 'fmax is not supported for complex numbers.');
  }

  // WASM acceleration (same kernels as maximum — NaN handling differs only for float edge cases)
  if (typeof x2 === 'number') {
    const wasmResult = wasmMaxScalar(x1, x2);
    if (wasmResult) return wasmResult;
  } else if (x2.size === 1) {
    const wasmResult = wasmMaxScalar(x1, Number(x2.iget(0)));
    if (wasmResult) return wasmResult;
  } else if (canUseFastPath(x1, x2)) {
    const wasmResult = wasmMax(x1, x2);
    if (wasmResult) return wasmResult;
  }

  if (typeof x2 === 'number') {
    const dtype = x1.dtype;
    const shape = Array.from(x1.shape);
    const size = x1.size;
    const result = ArrayStorage.empty(shape, dtype);
    const resultData = result.data;
    const x1Data = x1.data;

    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const x1Typed = x1Data as BigInt64Array | BigUint64Array;
      const x2Big = BigInt(Math.round(x2));
      for (let i = 0; i < size; i++) {
        resultTyped[i] = x1Typed[i]! > x2Big ? x1Typed[i]! : x2Big;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(x1Data[i]!);
        // Ignore NaN: return the non-NaN value, or NaN if both are NaN
        if (isNaN(val)) {
          resultData[i] = x2;
        } else if (isNaN(x2)) {
          resultData[i] = val;
        } else {
          resultData[i] = Math.max(val, x2);
        }
      }
    }
    return result;
  }

  return elementwiseBinaryOp(
    x1,
    x2,
    (a, b) => {
      if (isNaN(a)) return b;
      if (isNaN(b)) return a;
      return Math.max(a, b);
    },
    'fmax'
  );
}

/**
 * Element-wise minimum of array elements, ignoring NaNs
 *
 * @param x1 - First array storage
 * @param x2 - Second array storage or scalar
 * @returns Element-wise minimum, NaN-aware
 */
export function fmin(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplex(x1.dtype, 'fmin', 'fmin is not supported for complex numbers.');
  if (typeof x2 !== 'number') {
    throwIfComplex(x2.dtype, 'fmin', 'fmin is not supported for complex numbers.');
  }

  // WASM acceleration (same kernels as minimum — NaN handling differs only for float edge cases)
  if (typeof x2 === 'number') {
    const wasmResult = wasmMinScalar(x1, x2);
    if (wasmResult) return wasmResult;
  } else if (x2.size === 1) {
    const wasmResult = wasmMinScalar(x1, Number(x2.iget(0)));
    if (wasmResult) return wasmResult;
  } else if (canUseFastPath(x1, x2)) {
    const wasmResult = wasmMin(x1, x2);
    if (wasmResult) return wasmResult;
  }

  if (typeof x2 === 'number') {
    const dtype = x1.dtype;
    const shape = Array.from(x1.shape);
    const size = x1.size;
    const result = ArrayStorage.empty(shape, dtype);
    const resultData = result.data;
    const x1Data = x1.data;

    if (isBigIntDType(dtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      const x1Typed = x1Data as BigInt64Array | BigUint64Array;
      const x2Big = BigInt(Math.round(x2));
      for (let i = 0; i < size; i++) {
        resultTyped[i] = x1Typed[i]! < x2Big ? x1Typed[i]! : x2Big;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(x1Data[i]!);
        // Ignore NaN: return the non-NaN value, or NaN if both are NaN
        if (isNaN(val)) {
          resultData[i] = x2;
        } else if (isNaN(x2)) {
          resultData[i] = val;
        } else {
          resultData[i] = Math.min(val, x2);
        }
      }
    }
    return result;
  }

  return elementwiseBinaryOp(
    x1,
    x2,
    (a, b) => {
      if (isNaN(a)) return b;
      if (isNaN(b)) return a;
      return Math.min(a, b);
    },
    'fmin'
  );
}

/**
 * Replace NaN with zero and Inf with large finite numbers
 *
 * @param x - Input array storage
 * @param nan - Value to replace NaN (default: 0.0)
 * @param posinf - Value to replace positive infinity (default: largest finite)
 * @param neginf - Value to replace negative infinity (default: most negative finite)
 * @returns Array with replacements
 */
export function nan_to_num(
  x: ArrayStorage,
  nan: number = 0.0,
  posinf?: number,
  neginf?: number
): ArrayStorage {
  throwIfComplex(x.dtype, 'nan_to_num', 'nan_to_num is not supported for complex numbers.');
  const dtype = x.dtype;
  const shape = Array.from(x.shape);
  const size = x.size;

  // Default to dtype max/min values
  const posinfVal = posinf !== undefined ? posinf : Number.MAX_VALUE;
  const neginfVal = neginf !== undefined ? neginf : -Number.MAX_VALUE;

  const result = ArrayStorage.empty(shape, dtype);
  const resultData = result.data;
  const xData = x.data;

  if (isIntegerDType(dtype)) {
    // Integer types can never contain NaN or Inf, just bulk copy
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (resultData as any).set((xData as any).subarray(x.offset, x.offset + size));
  } else {
    for (let i = 0; i < size; i++) {
      const val = Number(xData[i]!);
      if (isNaN(val)) {
        resultData[i] = nan;
      } else if (val === Infinity) {
        resultData[i] = posinfVal;
      } else if (val === -Infinity) {
        resultData[i] = neginfVal;
      } else {
        resultData[i] = val;
      }
    }
  }

  return result;
}

/**
 * One-dimensional linear interpolation
 *
 * Returns the one-dimensional piecewise linear interpolant to a function
 * with given discrete data points (xp, fp), evaluated at x.
 *
 * @param x - The x-coordinates at which to evaluate the interpolated values
 * @param xp - The x-coordinates of the data points (must be increasing)
 * @param fp - The y-coordinates of the data points
 * @param left - Value for x < xp[0] (default: fp[0])
 * @param right - Value for x > xp[-1] (default: fp[-1])
 * @returns Interpolated values
 */
export function interp(
  x: ArrayStorage,
  xp: ArrayStorage,
  fp: ArrayStorage,
  left?: number,
  right?: number
): ArrayStorage {
  throwIfComplex(x.dtype, 'interp', 'interp is not supported for complex numbers.');
  throwIfComplex(xp.dtype, 'interp', 'interp is not supported for complex numbers.');
  throwIfComplex(fp.dtype, 'interp', 'interp is not supported for complex numbers.');

  const shape = Array.from(x.shape);
  const size = x.size;
  const result = ArrayStorage.empty(shape, 'float64');
  const resultData = result.data as Float64Array;
  const xData = x.data;
  const xpData = xp.data;
  const fpData = fp.data;
  const xpSize = xp.size;

  // Default left/right values
  const leftVal = left !== undefined ? left : Number(fpData[0]!);
  const rightVal = right !== undefined ? right : Number(fpData[xpSize - 1]!);

  for (let i = 0; i < size; i++) {
    const xi = Number(xData[i]!);

    // Handle out of bounds
    if (xi <= Number(xpData[0]!)) {
      resultData[i] = leftVal;
      continue;
    }
    if (xi >= Number(xpData[xpSize - 1]!)) {
      resultData[i] = rightVal;
      continue;
    }

    // Binary search for the interval
    let lo = 0;
    let hi = xpSize - 1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (Number(xpData[mid]!) <= xi) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Linear interpolation
    const x0 = Number(xpData[lo]!);
    const x1 = Number(xpData[hi]!);
    const y0 = Number(fpData[lo]!);
    const y1 = Number(fpData[hi]!);
    const t = (xi - x0) / (x1 - x0);
    resultData[i] = y0 + t * (y1 - y0);
  }

  return result;
}

/**
 * Unwrap by changing deltas between values to 2*pi complement
 *
 * Unwrap radian phase p by changing absolute jumps greater than
 * discont to their 2*pi complement along the given axis.
 *
 * @param p - Input array of phase angles in radians
 * @param discont - Maximum discontinuity between values (default: pi)
 * @param axis - Axis along which to unwrap (default: -1, last axis)
 * @param period - Size of the range over which the input wraps (default: 2*pi)
 * @returns Unwrapped array
 */
export function unwrap(
  p: ArrayStorage,
  discont: number = Math.PI,
  axis: number = -1,
  period: number = 2 * Math.PI
): ArrayStorage {
  throwIfComplex(p.dtype, 'unwrap', 'unwrap is not supported for complex numbers.');

  const shape = Array.from(p.shape);
  const ndim = shape.length;

  // Normalize axis
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // For 1D arrays, simple implementation
  if (ndim === 1) {
    const size = p.size;
    const result = ArrayStorage.empty(shape, 'float64');
    const resultData = result.data as Float64Array;
    const pData = p.data;

    if (size === 0) return result;

    resultData[0] = Number(pData[0]!);
    let offset = 0;

    for (let i = 1; i < size; i++) {
      const prev = Number(pData[i - 1]!);
      const curr = Number(pData[i]!);
      let diff = curr - prev;

      // Wrap the difference
      diff = ((diff + period / 2) % period) - period / 2;
      if (diff === -period / 2 && curr - prev > 0) {
        diff = period / 2;
      }

      // Check for discontinuity
      if (Math.abs(diff) > discont) {
        offset -= Math.round((curr - prev - diff) / period) * period;
      }

      resultData[i] = curr + offset;
    }

    return result;
  }

  // For multi-dimensional arrays, we need to process along the axis
  // This is a simplified implementation for 2D case
  const result = ArrayStorage.empty(shape, 'float64');
  const resultData = result.data as Float64Array;
  const pData = p.data;

  // Copy all data first
  for (let i = 0; i < p.size; i++) {
    resultData[i] = Number(pData[i]!);
  }

  // Unwrap along axis
  if (ndim === 2) {
    const [rows, cols] = shape;
    if (axis === 0) {
      // Unwrap along rows (for each column)
      for (let c = 0; c < cols!; c++) {
        let offset = 0;
        for (let r = 1; r < rows!; r++) {
          const prevIdx = (r - 1) * cols! + c;
          const currIdx = r * cols! + c;
          const prev = resultData[prevIdx]!;
          const curr = resultData[currIdx]!;
          let diff = curr - prev;

          diff = ((diff + period / 2) % period) - period / 2;
          if (diff === -period / 2 && curr - prev > 0) {
            diff = period / 2;
          }

          if (Math.abs(diff) > discont) {
            offset -= Math.round((curr - prev - diff) / period) * period;
          }

          resultData[currIdx] = curr + offset;
        }
      }
    } else {
      // Unwrap along columns (for each row)
      for (let r = 0; r < rows!; r++) {
        let offset = 0;
        for (let c = 1; c < cols!; c++) {
          const prevIdx = r * cols! + (c - 1);
          const currIdx = r * cols! + c;
          const prev = resultData[prevIdx]!;
          const curr = resultData[currIdx]!;
          let diff = curr - prev;

          diff = ((diff + period / 2) % period) - period / 2;
          if (diff === -period / 2 && curr - prev > 0) {
            diff = period / 2;
          }

          if (Math.abs(diff) > discont) {
            offset -= Math.round((curr - prev - diff) / period) * period;
          }

          resultData[currIdx] = curr + offset;
        }
      }
    }
  }

  return result;
}

/**
 * Return the normalized sinc function
 *
 * sinc(x) = sin(pi*x) / (pi*x)
 *
 * The sinc function is 1 at x = 0, and sin(pi*x)/(pi*x) otherwise.
 *
 * @param x - Input array
 * @returns Array of sinc values
 */
export function sinc(x: ArrayStorage): ArrayStorage {
  throwIfComplex(x.dtype, 'sinc', 'sinc is not supported for complex numbers.');

  const shape = Array.from(x.shape);
  const size = x.size;
  // NumPy: sinc preserves float32, all other types → float64
  const resultDtype = x.dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;
  const xData = x.data;

  for (let i = 0; i < size; i++) {
    const val = Number(xData[i]!);
    if (val === 0) {
      resultData[i] = 1.0;
    } else {
      const pix = Math.PI * val;
      resultData[i] = Math.sin(pix) / pix;
    }
  }

  return result;
}

/**
 * Modified Bessel function of the first kind, order 0
 *
 * Uses polynomial approximation.
 *
 * @param x - Input array
 * @returns Array of I0 values
 */
export function i0(x: ArrayStorage): ArrayStorage {
  throwIfComplex(x.dtype, 'i0', 'i0 is not supported for complex numbers.');

  const shape = Array.from(x.shape);
  const size = x.size;
  // NumPy: i0 preserves float32, all other types → float64
  const resultDtype = x.dtype === 'float32' ? 'float32' : 'float64';
  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;
  const xData = x.data;

  for (let i = 0; i < size; i++) {
    const val = Math.abs(Number(xData[i]!));
    resultData[i] = besselI0(val);
  }

  return result;
}

/**
 * Polynomial approximation for modified Bessel function I0
 * Based on Abramowitz and Stegun approximations
 */
function besselI0(x: number): number {
  const ax = Math.abs(x);

  if (ax < 3.75) {
    // Polynomial approximation for small x
    const t = x / 3.75;
    const t2 = t * t;
    return (
      1.0 +
      t2 *
        (3.5156229 +
          t2 *
            (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    );
  } else {
    // Polynomial approximation for large x
    const t = 3.75 / ax;
    return (
      (Math.exp(ax) / Math.sqrt(ax)) *
      (0.39894228 +
        t *
          (0.01328592 +
            t *
              (0.00225319 +
                t *
                  (-0.00157565 +
                    t *
                      (0.00916281 +
                        t *
                          (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    );
  }
}
