/**
 * Hyperbolic operations
 *
 * Pure functions for element-wise hyperbolic operations:
 * sinh, cosh, tanh, arcsinh, arccosh, arctanh
 *
 * These functions are used by NDArray methods but are separated
 * to keep the codebase modular and testable.
 */

import { ArrayStorage } from '../core/storage';
import { elementwiseUnaryOp } from '../internal/compute';
import { throwIfComplexNotImplemented } from '../core/dtype';

/**
 * Hyperbolic sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with sinh applied
 */
export function sinh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'sinh');
  return elementwiseUnaryOp(a, Math.sinh, false);
}

/**
 * Hyperbolic cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with cosh applied
 */
export function cosh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'cosh');
  return elementwiseUnaryOp(a, Math.cosh, false);
}

/**
 * Hyperbolic tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with tanh applied
 */
export function tanh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'tanh');
  return elementwiseUnaryOp(a, Math.tanh, false);
}

/**
 * Inverse hyperbolic sine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage
 * @returns Result storage with arcsinh applied
 */
export function arcsinh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arcsinh');
  return elementwiseUnaryOp(a, Math.asinh, false);
}

/**
 * Inverse hyperbolic cosine of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (values >= 1)
 * @returns Result storage with arccosh applied
 */
export function arccosh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arccosh');
  return elementwiseUnaryOp(a, Math.acosh, false);
}

/**
 * Inverse hyperbolic tangent of each element (element-wise)
 * NumPy behavior: Always promotes to float64 for integer types
 *
 * @param a - Input array storage (values in range (-1, 1))
 * @returns Result storage with arctanh applied
 */
export function arctanh(a: ArrayStorage): ArrayStorage {
  throwIfComplexNotImplemented(a.dtype, 'arctanh');
  return elementwiseUnaryOp(a, Math.atanh, false);
}
