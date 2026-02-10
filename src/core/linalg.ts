/**
 * Linear Algebra Functions - Tree-shakeable standalone functions
 *
 * This module provides linear algebra functions that can be
 * imported independently for optimal tree-shaking.
 */

import * as linalgOps from '../common/ops/linalg';
import {
  NDArrayCore,
  toStorage,
  fromStorage,
  fromStorageView,
  Complex,
  ArrayStorage,
} from './types';

// ============================================================
// Top-Level Linear Algebra Functions
// ============================================================

/** Dot product of two arrays */
export function dot(a: NDArrayCore, b: NDArrayCore): NDArrayCore | number | bigint | Complex {
  const result = linalgOps.dot(toStorage(a), toStorage(b));
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Matrix trace (sum of diagonal elements) */
export function trace(a: NDArrayCore): number | bigint | Complex {
  return linalgOps.trace(toStorage(a));
}

/** Return diagonal or construct diagonal array */
export function diagonal(
  a: NDArrayCore,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): NDArrayCore {
  return fromStorage(linalgOps.diagonal(toStorage(a), offset, axis1, axis2));
}

/** Kronecker product */
export function kron(a: NDArrayCore, b: NDArrayCore): NDArrayCore {
  return fromStorage(linalgOps.kron(toStorage(a), toStorage(b)));
}

/** Transpose array - returns a view */
export function transpose(a: NDArrayCore, axes?: number[]): NDArrayCore {
  return fromStorageView(linalgOps.transpose(toStorage(a), axes), a);
}

/** Inner product of two arrays */
export function inner(a: NDArrayCore, b: NDArrayCore): NDArrayCore | number | bigint | Complex {
  const result = linalgOps.inner(toStorage(a), toStorage(b));
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Outer product of two arrays */
export function outer(a: NDArrayCore, b: NDArrayCore): NDArrayCore {
  return fromStorage(linalgOps.outer(toStorage(a), toStorage(b)));
}

/** Tensor dot product */
export function tensordot(
  a: NDArrayCore,
  b: NDArrayCore,
  axes: number | [number[], number[]] = 2
): NDArrayCore | number | bigint | Complex {
  const result = linalgOps.tensordot(toStorage(a), toStorage(b), axes);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Evaluates Einstein summation convention */
export function einsum(
  subscripts: string,
  ...operands: NDArrayCore[]
): NDArrayCore | number | bigint | Complex {
  const storageOperands = operands.map((op) => toStorage(op));
  const result = linalgOps.einsum(subscripts, ...storageOperands);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Compute the optimal contraction path for einsum */
export function einsum_path(
  subscripts: string,
  ...operands: NDArrayCore[]
): [([number, number] | number[])[], string] {
  const storageOperands = operands.map((op) => toStorage(op));
  return linalgOps.einsum_path(subscripts, ...storageOperands);
}

/** Vector dot product */
export function vdot(a: NDArrayCore, b: NDArrayCore): number | bigint | Complex {
  return linalgOps.vdot(toStorage(a), toStorage(b));
}

/** Vector dot product along specified axis */
export function vecdot(
  x1: NDArrayCore,
  x2: NDArrayCore,
  axis: number = -1
): NDArrayCore | number | bigint | Complex {
  const result = linalgOps.vecdot(toStorage(x1), toStorage(x2), axis);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Matrix transpose (swap last two axes) */
/** Matrix transpose - returns a view */
export function matrix_transpose(a: NDArrayCore): NDArrayCore {
  return fromStorageView(linalgOps.matrix_transpose(toStorage(a)), a);
}

/** Permute array dimensions - returns a view */
export function permute_dims(a: NDArrayCore, axes?: number[]): NDArrayCore {
  return fromStorageView(linalgOps.permute_dims(toStorage(a), axes), a);
}

/** Matrix-vector product */
export function matvec(x1: NDArrayCore, x2: NDArrayCore): NDArrayCore {
  return fromStorage(linalgOps.matvec(toStorage(x1), toStorage(x2)));
}

/** Vector-matrix product */
export function vecmat(x1: NDArrayCore, x2: NDArrayCore): NDArrayCore {
  return fromStorage(linalgOps.vecmat(toStorage(x1), toStorage(x2)));
}

/** Cross product */
export function cross(
  a: NDArrayCore,
  b: NDArrayCore,
  axisa: number = -1,
  axisb: number = -1,
  axisc: number = -1,
  axis?: number
): NDArrayCore | number | Complex {
  const result = linalgOps.cross(toStorage(a), toStorage(b), axisa, axisb, axisc, axis);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return fromStorage(result);
}

/** Matrix multiplication */
export function matmul(a: NDArrayCore, b: NDArrayCore): NDArrayCore {
  return fromStorage(linalgOps.matmul(toStorage(a), toStorage(b)));
}

// ============================================================
// Linear Algebra Namespace (numpy.linalg)
// ============================================================

export const linalg = {
  /** Matrix multiplication */
  matmul: (a: NDArrayCore, b: NDArrayCore): NDArrayCore => {
    return fromStorage(linalgOps.matmul(toStorage(a), toStorage(b)));
  },

  /** Dot product */
  dot: (a: NDArrayCore, b: NDArrayCore): NDArrayCore | number | bigint | Complex => {
    return dot(a, b);
  },

  /** Matrix determinant */
  det: (a: NDArrayCore): number => {
    return linalgOps.det(toStorage(a));
  },

  /** Matrix inverse */
  inv: (a: NDArrayCore): NDArrayCore => {
    return fromStorage(linalgOps.inv(toStorage(a)));
  },

  /** Solve linear system Ax = b */
  solve: (a: NDArrayCore, b: NDArrayCore): NDArrayCore => {
    return fromStorage(linalgOps.solve(toStorage(a), toStorage(b)));
  },

  /** Least-squares solution */
  lstsq: (
    a: NDArrayCore,
    b: NDArrayCore,
    rcond?: number | null
  ): { x: NDArrayCore; residuals: NDArrayCore; rank: number; s: NDArrayCore } => {
    const result = linalgOps.lstsq(toStorage(a), toStorage(b), rcond);
    return {
      x: fromStorage(result.x),
      residuals: fromStorage(result.residuals),
      rank: result.rank,
      s: fromStorage(result.s),
    };
  },

  /** Matrix or vector norm */
  norm: (
    x: NDArrayCore,
    ord?: number | 'fro' | 'nuc' | null,
    axis?: number | [number, number] | null,
    keepdims?: boolean
  ): NDArrayCore | number => {
    const result = linalgOps.norm(toStorage(x), ord, axis, keepdims);
    if (typeof result === 'number') return result;
    return fromStorage(result);
  },

  /** Condition number */
  cond: (a: NDArrayCore, p?: number | 'fro' | 'nuc'): number => {
    return linalgOps.cond(toStorage(a), p);
  },

  /** Matrix rank */
  matrix_rank: (a: NDArrayCore, tol?: number): number => {
    return linalgOps.matrix_rank(toStorage(a), tol);
  },

  /** Raise matrix to a power */
  matrix_power: (a: NDArrayCore, n: number): NDArrayCore => {
    return fromStorage(linalgOps.matrix_power(toStorage(a), n));
  },

  /** Pseudo-inverse */
  pinv: (a: NDArrayCore, rcond?: number): NDArrayCore => {
    return fromStorage(linalgOps.pinv(toStorage(a), rcond));
  },

  /** QR decomposition */
  qr: (
    a: NDArrayCore,
    mode?: 'reduced' | 'complete' | 'r' | 'raw'
  ): { q: NDArrayCore; r: NDArrayCore } | NDArrayCore | { h: NDArrayCore; tau: NDArrayCore } => {
    const result = linalgOps.qr(toStorage(a), mode);
    if ('h' in result && 'tau' in result) {
      return { h: fromStorage(result.h), tau: fromStorage(result.tau) };
    }
    if ('q' in result && 'r' in result) {
      return { q: fromStorage(result.q), r: fromStorage(result.r) };
    }
    return fromStorage(result as ArrayStorage);
  },

  /** Cholesky decomposition */
  cholesky: (a: NDArrayCore, upper?: boolean): NDArrayCore => {
    return fromStorage(linalgOps.cholesky(toStorage(a), upper));
  },

  /** Singular value decomposition */
  svd: (
    a: NDArrayCore,
    full_matrices?: boolean,
    compute_uv?: boolean
  ): { u: NDArrayCore; s: NDArrayCore; vt: NDArrayCore } | NDArrayCore => {
    const result = linalgOps.svd(toStorage(a), full_matrices, compute_uv);
    if ('u' in result && 's' in result && 'vt' in result) {
      return { u: fromStorage(result.u), s: fromStorage(result.s), vt: fromStorage(result.vt) };
    }
    return fromStorage(result as ArrayStorage);
  },

  /** Eigenvalues and eigenvectors */
  eig: (a: NDArrayCore): { w: NDArrayCore; v: NDArrayCore } => {
    const result = linalgOps.eig(toStorage(a));
    return { w: fromStorage(result.w), v: fromStorage(result.v) };
  },

  /** Eigenvalues and eigenvectors of Hermitian matrix */
  eigh: (a: NDArrayCore, UPLO?: 'L' | 'U'): { w: NDArrayCore; v: NDArrayCore } => {
    const result = linalgOps.eigh(toStorage(a), UPLO);
    return { w: fromStorage(result.w), v: fromStorage(result.v) };
  },

  /** Eigenvalues */
  eigvals: (a: NDArrayCore): NDArrayCore => {
    return fromStorage(linalgOps.eigvals(toStorage(a)));
  },

  /** Eigenvalues of Hermitian matrix */
  eigvalsh: (a: NDArrayCore, UPLO?: 'L' | 'U'): NDArrayCore => {
    return fromStorage(linalgOps.eigvalsh(toStorage(a), UPLO));
  },

  /** Sign and log of determinant */
  slogdet: (a: NDArrayCore): { sign: number; logabsdet: number } => {
    return linalgOps.slogdet(toStorage(a));
  },

  /** Singular values */
  svdvals: (a: NDArrayCore): NDArrayCore => {
    return fromStorage(linalgOps.svdvals(toStorage(a)));
  },

  /** Chained dot product */
  multi_dot: (arrays: NDArrayCore[]): NDArrayCore => {
    return fromStorage(linalgOps.multi_dot(arrays.map((a) => toStorage(a))));
  },

  /** Inverse of tensorization */
  tensorinv: (a: NDArrayCore, ind?: number): NDArrayCore => {
    return fromStorage(linalgOps.tensorinv(toStorage(a), ind));
  },

  /** Tensor equation solution */
  tensorsolve: (a: NDArrayCore, b: NDArrayCore, axes?: number[]): NDArrayCore => {
    return fromStorage(linalgOps.tensorsolve(toStorage(a), toStorage(b), axes));
  },

  /** Vector norm */
  vector_norm: (
    x: NDArrayCore,
    ord?: number | 'fro' | 'nuc',
    axis?: number | null,
    keepdims?: boolean
  ): NDArrayCore | number => {
    const result = linalgOps.vector_norm(toStorage(x), ord, axis, keepdims);
    if (typeof result === 'number') return result;
    return fromStorage(result);
  },

  /** Matrix norm */
  matrix_norm: (
    x: NDArrayCore,
    ord?: number | 'fro' | 'nuc',
    keepdims?: boolean
  ): NDArrayCore | number => {
    const result = linalgOps.matrix_norm(toStorage(x), ord, keepdims);
    if (typeof result === 'number') return result;
    return fromStorage(result);
  },

  /** Cross product */
  cross: (
    a: NDArrayCore,
    b: NDArrayCore,
    axisa?: number,
    axisb?: number,
    axisc?: number,
    axis?: number
  ): NDArrayCore | number | Complex => {
    return cross(a, b, axisa, axisb, axisc, axis);
  },

  /** Matrix transpose (transposes last two axes) - returns a view */
  matrix_transpose: (a: NDArrayCore): NDArrayCore => {
    return matrix_transpose(a);
  },

  /** Permute array dimensions - returns a view */
  permute_dims: (a: NDArrayCore, axes?: number[]): NDArrayCore => {
    return permute_dims(a, axes);
  },

  /** Matrix trace (sum of diagonal elements) */
  trace: (a: NDArrayCore): number | bigint | Complex => {
    return trace(a);
  },

  /** Extract diagonal */
  diagonal: (a: NDArrayCore, offset?: number, axis1?: number, axis2?: number): NDArrayCore => {
    return diagonal(a, offset, axis1, axis2);
  },

  /** Outer product */
  outer: (a: NDArrayCore, b: NDArrayCore): NDArrayCore => {
    return outer(a, b);
  },

  /** Inner product */
  inner: (a: NDArrayCore, b: NDArrayCore): NDArrayCore | number | bigint | Complex => {
    return inner(a, b);
  },

  /** Tensor dot product */
  tensordot: (
    a: NDArrayCore,
    b: NDArrayCore,
    axes?: number | [number[], number[]]
  ): NDArrayCore | number | bigint | Complex => {
    return tensordot(a, b, axes);
  },

  /** Vector dot product */
  vecdot: (
    a: NDArrayCore,
    b: NDArrayCore,
    axis?: number
  ): NDArrayCore | number | bigint | Complex => {
    return vecdot(a, b, axis);
  },

  /** Transpose array - returns a view */
  transpose: (a: NDArrayCore, axes?: number[]): NDArrayCore => {
    return transpose(a, axes);
  },
};
