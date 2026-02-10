/**
 * Logic Functions - Tree-shakeable standalone functions
 *
 * This module provides logical and type-checking functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as logicOps from '../common/ops/logic';
import * as comparisonOps from '../common/ops/comparison';
import { NDArrayCore, toStorage, fromStorage, type DType } from './types';

// ============================================================
// Logical Operations
// ============================================================

/** Element-wise logical AND */
export function logical_and(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(logicOps.logical_and(toStorage(x1), x2Storage));
}

/** Element-wise logical OR */
export function logical_or(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(logicOps.logical_or(toStorage(x1), x2Storage));
}

/** Element-wise logical NOT */
export function logical_not(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.logical_not(toStorage(x)));
}

/** Element-wise logical XOR */
export function logical_xor(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(logicOps.logical_xor(toStorage(x1), x2Storage));
}

// ============================================================
// Value Testing
// ============================================================

/** Test for finite values */
export function isfinite(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isfinite(toStorage(x)));
}

/** Test for infinity */
export function isinf(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isinf(toStorage(x)));
}

/** Test for NaN */
export function isnan(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isnan(toStorage(x)));
}

/** Test for NaT (not a time) - returns all False for numeric arrays */
export function isnat(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isnat(toStorage(x)));
}

/** Test for negative infinity */
export function isneginf(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isneginf(toStorage(x)));
}

/** Test for positive infinity */
export function isposinf(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isposinf(toStorage(x)));
}

// ============================================================
// Complex Number Testing
// ============================================================

/** Element-wise test for complex values */
export function iscomplex(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.iscomplex(toStorage(x)));
}

/** Check if array has complex dtype */
export function iscomplexobj(x: NDArrayCore): boolean {
  return logicOps.iscomplexobj(toStorage(x));
}

/** Element-wise test for real values */
export function isreal(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.isreal(toStorage(x)));
}

/** Check if array has real dtype */
export function isrealobj(x: NDArrayCore): boolean {
  return logicOps.isrealobj(toStorage(x));
}

/** Return real array if imaginary part is negligible */
export function real_if_close(x: NDArrayCore, tol?: number): NDArrayCore {
  return fromStorage(logicOps.real_if_close(toStorage(x), tol));
}

// ============================================================
// Memory Layout Testing
// ============================================================

/** Check if array is Fortran contiguous */
export function isfortran(x: NDArrayCore): boolean {
  return logicOps.isfortran(toStorage(x));
}

// ============================================================
// Type Testing
// ============================================================

/** Check if value is a scalar */
export function isscalar(val: unknown): boolean {
  return logicOps.isscalar(val);
}

/** Check if object is iterable */
export function iterable(obj: unknown): boolean {
  return logicOps.iterable(obj);
}

/** Check dtype against a kind string */
export function isdtype(dtype: DType, kind: string): boolean {
  return logicOps.isdtype(dtype, kind);
}

/** Promote dtypes to a common type */
export function promote_types(dtype1: DType, dtype2: DType): DType {
  return logicOps.promote_types(dtype1, dtype2);
}

// ============================================================
// Floating-Point Operations
// ============================================================

/** Return sign of x1 with magnitude of x2 */
export function copysign(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(logicOps.copysign(toStorage(x1), x2Storage));
}

/** Test for negative sign bit */
export function signbit(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.signbit(toStorage(x)));
}

/** Next floating-point value toward x2 */
export function nextafter(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  const x2Storage = typeof x2 === 'number' ? x2 : toStorage(x2);
  return fromStorage(logicOps.nextafter(toStorage(x1), x2Storage));
}

/** Spacing between x and nearest adjacent number */
export function spacing(x: NDArrayCore): NDArrayCore {
  return fromStorage(logicOps.spacing(toStorage(x)));
}

// ============================================================
// Element-wise Comparison Operations
// ============================================================

/** Element-wise greater than comparison */
export function greater(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.greater(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise greater than or equal comparison */
export function greater_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.greaterEqual(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise less than comparison */
export function less(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.less(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise less than or equal comparison */
export function less_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.lessEqual(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise equality comparison */
export function equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.equal(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise inequality comparison */
export function not_equal(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  return fromStorage(
    comparisonOps.notEqual(toStorage(x1), typeof x2 === 'number' ? x2 : toStorage(x2))
  );
}

/** Element-wise close comparison with tolerance */
export function isclose(
  a: NDArrayCore,
  b: NDArrayCore | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): NDArrayCore {
  return fromStorage(
    comparisonOps.isclose(toStorage(a), typeof b === 'number' ? b : toStorage(b), rtol, atol)
  );
}

/** Check if all elements are close */
export function allclose(
  a: NDArrayCore,
  b: NDArrayCore | number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): boolean {
  return comparisonOps.allclose(toStorage(a), typeof b === 'number' ? b : toStorage(b), rtol, atol);
}
