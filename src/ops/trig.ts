/**
 * Trigonometric operations
 *
 * Pure functions for element-wise trigonometric operations:
 * sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot, degrees, radians
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { elementwiseUnaryOp } from '../internal/compute';
import { isBigIntDType, throwIfComplexNotImplemented } from '../core/dtype';

/**
 * Sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with sin applied
 */
export function sin(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'sin');
  return elementwiseUnaryOp(a, Math.sin, false);
}

/**
 * Cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with cos applied
 */
export function cos(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'cos');
  return elementwiseUnaryOp(a, Math.cos, false);
}

/**
 * Tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Result storage with tan applied
 */
export function tan(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'tan');
  return elementwiseUnaryOp(a, Math.tan, false);
}

/**
 * Inverse sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (values in range [-1, 1])
 * @returns Result storage with arcsin applied (radians)
 */
export function arcsin(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arcsin');
  return elementwiseUnaryOp(a, Math.asin, false);
}

/**
 * Inverse cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (values in range [-1, 1])
 * @returns Result storage with arccos applied (radians)
 */
export function arccos(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arccos');
  return elementwiseUnaryOp(a, Math.acos, false);
}

/**
 * Inverse tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with arctan applied (radians)
 */
export function arctan(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arctan');
  return elementwiseUnaryOp(a, Math.atan, false);
}

/**
 * Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - y-coordinates
 * @param x2 - x-coordinates (array storage or scalar)
 * @returns Angle in radians between -π and π
 */
export function arctan2(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplexNotImplemented(x1.dtype, 'arctan2');
  if (typeof x2 !== 'number') {
    throwIfComplexNotImplemented(x2.dtype, 'arctan2');
  }
  if (typeof x2 === 'number') {
    return arctan2Scalar(x1, x2);
  }
  return arctan2Array(x1, x2);
}

/**
 * arctan2 with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function arctan2Array(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const shape = Array.from(x1.shape);
  const size = x1.size;
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for arctan2 (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.data[i]!) : Number(x1.data[i]!);
    const val2 = isBigIntDType(dtype2) ? Number(x2.data[i]!) : Number(x2.data[i]!);
    resultData[i] = Math.atan2(val1, val2);
  }

  return result;
}

/**
 * arctan2 with scalar x2 (optimized path)
 * @private
 */
function arctan2Scalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Always promote to float64 for trig operations
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.atan2(Number(data[i]!), x2);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.atan2(Number(data[i]!), x2);
    }
  }

  return result;
}

/**
 * Given the "legs" of a right triangle, return its hypotenuse.
 * Equivalent to sqrt(x1**2 + x2**2), element-wise.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param x1 - First leg
 * @param x2 - Second leg (array storage or scalar)
 * @returns Hypotenuse
 */
export function hypot(x1: ArrayStorage, x2: ArrayStorage | number): ArrayStorage {
  throwIfComplexNotImplemented(x1.dtype, 'hypot');
  if (typeof x2 !== 'number') {
    throwIfComplexNotImplemented(x2.dtype, 'hypot');
  }
  if (typeof x2 === 'number') {
    return hypotScalar(x1, x2);
  }
  return hypotArray(x1, x2);
}

/**
 * hypot with array x2 (always returns float64 for non-float32 inputs)
 * @private
 */
function hypotArray(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  const shape = Array.from(x1.shape);
  const size = x1.size;
  const dtype1 = x1.dtype;
  const dtype2 = x2.dtype;

  // Always promote to float64 for hypot (matching NumPy behavior)
  // Only preserve float32 if both inputs are float32
  const resultDtype = dtype1 === 'float32' && dtype2 === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  for (let i = 0; i < size; i++) {
    const val1 = isBigIntDType(dtype1) ? Number(x1.data[i]!) : Number(x1.data[i]!);
    const val2 = isBigIntDType(dtype2) ? Number(x2.data[i]!) : Number(x2.data[i]!);
    resultData[i] = Math.hypot(val1, val2);
  }

  return result;
}

/**
 * hypot with scalar x2 (optimized path)
 * @private
 */
function hypotScalar(storage: ArrayStorage, x2: number): ArrayStorage {
  const dtype = storage.dtype;
  const shape = Array.from(storage.shape);
  const data = storage.data;
  const size = storage.size;

  // Always promote to float64 for trig operations
  const resultDtype = dtype === 'float32' ? 'float32' : 'float64';

  const result = ArrayStorage.zeros(shape, resultDtype);
  const resultData = result.data;

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.hypot(Number(data[i]!), x2);
    }
  } else {
    for (let i = 0; i < size; i++) {
      resultData[i] = Math.hypot(Number(data[i]!), x2);
    }
  }

  return result;
}

/**
 * Convert angles from radians to degrees.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Angles in degrees
 */
export function degrees(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'degrees');
  const factor = 180 / Math.PI;
  return elementwiseUnaryOp(a, (x) => x * factor, false);
}

/**
 * Convert angles from degrees to radians.
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in degrees)
 * @returns Angles in radians
 */
export function radians(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'radians');
  const factor = Math.PI / 180;
  return elementwiseUnaryOp(a, (x) => x * factor, false);
}

/**
 * Convert angles from degrees to radians (alias for radians).
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in degrees)
 * @returns Angles in radians
 */
export function deg2rad(a: ArrayStorage): ArrayStorage {
  return radians(a);
}

/**
 * Convert angles from radians to degrees (alias for degrees).
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (angles in radians)
 * @returns Angles in degrees
 */
export function rad2deg(a: ArrayStorage): ArrayStorage {
  return degrees(a);
}
