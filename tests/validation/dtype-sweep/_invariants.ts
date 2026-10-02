/**
 * Structural checks for decompositions: the properties that hold whatever
 * convention the implementation picked.
 *
 * Comparing a decomposition against NumPy element by element does not work,
 * because the column signs of Q, the phase of a singular vector and the order
 * of eigenvalues are all free. What is pinned is the algebra, so these assert
 * the algebra instead: Q is unitary, R is upper triangular, the factors
 * multiply back to the input, and each eigenvector really belongs to its
 * eigenvalue.
 *
 * The arithmetic here is deliberately plain JavaScript over `toArray()` rather
 * than numpy-ts operations. Verifying `qr` with this library's own `matmul`
 * would let a matmul bug hide a qr bug, or invent one that is not there.
 */

import { expect } from 'vitest';

/** One element as it comes back from `toArray()`. */
type Cell = number | bigint | { re: number; im: number };

/** A complex value with both parts always present. */
export interface C {
  re: number;
  im: number;
}

/**
 * Normalise one element to a complex pair.
 *
 * @param v - Element from `toArray()`
 * @returns The value with an explicit imaginary part
 */
export function cell(v: Cell): C {
  if (typeof v === 'object' && v !== null && 're' in v) {
    return { re: Number(v.re), im: Number(v.im) };
  }
  return { re: Number(v), im: 0 };
}

/**
 * Normalise a 2D array to complex pairs.
 *
 * @param m - Nested array from `toArray()`
 * @returns The same shape with explicit imaginary parts
 */
export function mat(m: Cell[][]): C[][] {
  return m.map((row) => row.map(cell));
}

/**
 * Normalise a 1D array to complex pairs.
 *
 * @param v - Array from `toArray()`
 * @returns The same length with explicit imaginary parts
 */
export function vec(v: Cell[]): C[] {
  return v.map(cell);
}

/**
 * Matrix product of two complex matrices.
 *
 * @param a - Left operand, m by k
 * @param b - Right operand, k by n
 * @returns The m by n product
 */
export function matmulC(a: C[][], b: C[][]): C[][] {
  const m = a.length;
  const k = b.length;
  const n = k === 0 ? 0 : b[0]!.length;
  const out: C[][] = [];
  for (let i = 0; i < m; i++) {
    const row: C[] = [];
    for (let j = 0; j < n; j++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < k; t++) {
        const x = a[i]![t]!;
        const y = b[t]![j]!;
        re += x.re * y.re - x.im * y.im;
        im += x.re * y.im + x.im * y.re;
      }
      row.push({ re, im });
    }
    out.push(row);
  }
  return out;
}

/**
 * Conjugate transpose. The conjugation is the point: a plain transpose would
 * make every one of these checks pass for a matrix that is not unitary.
 *
 * @param a - Input matrix
 * @returns Its conjugate transpose
 */
export function conjT(a: C[][]): C[][] {
  const m = a.length;
  const n = m === 0 ? 0 : a[0]!.length;
  const out: C[][] = [];
  for (let j = 0; j < n; j++) {
    const row: C[] = [];
    for (let i = 0; i < m; i++) row.push({ re: a[i]![j]!.re, im: -a[i]![j]!.im });
    out.push(row);
  }
  return out;
}

/** Largest elementwise distance between two matrices of the same shape. */
function maxDiff(a: C[][], b: C[][]): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i]!.length; j++) {
      worst = Math.max(worst, Math.hypot(a[i]![j]!.re - b[i]![j]!.re, a[i]![j]!.im - b[i]![j]!.im));
    }
  }
  return worst;
}

/**
 * Tolerance for a dtype. The 32-bit types carry about seven decimal digits, and
 * a decomposition loses several of them, so they get a much looser bound than
 * the 64-bit ones rather than a single tolerance that is wrong for both.
 *
 * @param dtype - Input dtype of the decomposition
 * @returns Absolute tolerance for the invariant checks
 */
export function tolFor(dtype: string): number {
  return dtype === 'float32' || dtype === 'complex64' || dtype === 'float16' ? 1e-3 : 1e-9;
}

/**
 * Assert that a product of factors reproduces the original matrix.
 *
 * @param product - Factors multiplied back together
 * @param original - The matrix that was decomposed
 * @param label - Test identifier used in the failure message
 * @param tol - Absolute tolerance
 */
export function expectReconstructs(
  product: C[][],
  original: C[][],
  label: string,
  tol: number,
): void {
  expect(maxDiff(product, original), `${label}: factors do not reconstruct the input`).toBeLessThan(
    tol,
  );
}

/**
 * Assert that a matrix has orthonormal columns, so Q^H Q is the identity.
 *
 * @param q - Matrix to check, m by k with k <= m
 * @param label - Test identifier used in the failure message
 * @param tol - Absolute tolerance
 */
export function expectUnitaryColumns(q: C[][], label: string, tol: number): void {
  const g = matmulC(conjT(q), q);
  const k = g.length;
  const id: C[][] = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => ({ re: i === j ? 1 : 0, im: 0 })),
  );
  expect(maxDiff(g, id), `${label}: columns are not orthonormal (Q^H Q != I)`).toBeLessThan(tol);
}

/**
 * Assert that everything below the main diagonal is zero.
 *
 * @param r - Matrix to check
 * @param label - Test identifier used in the failure message
 * @param tol - Absolute tolerance
 */
export function expectUpperTriangular(r: C[][], label: string, tol: number): void {
  let worst = 0;
  for (let i = 0; i < r.length; i++) {
    for (let j = 0; j < Math.min(i, r[i]!.length); j++) {
      worst = Math.max(worst, Math.hypot(r[i]![j]!.re, r[i]![j]!.im));
    }
  }
  expect(worst, `${label}: entries below the diagonal are not zero`).toBeLessThan(tol);
}

/**
 * Assert that each column of v is an eigenvector for the matching entry of w,
 * by checking A v = lambda v directly.
 *
 * This is the check that issue #161 needed: comparing only the eigenvalues, or
 * only their magnitudes, passes just as happily when the eigenvectors have been
 * paired with the wrong eigenvalues.
 *
 * @param a - The matrix that was decomposed
 * @param w - Eigenvalues
 * @param v - Eigenvectors, one per column
 * @param label - Test identifier used in the failure message
 * @param tol - Absolute tolerance
 */
export function expectEigenpairs(a: C[][], w: C[], v: C[][], label: string, tol: number): void {
  const n = a.length;
  let worst = 0;
  for (let k = 0; k < w.length; k++) {
    for (let i = 0; i < n; i++) {
      let re = 0;
      let im = 0;
      for (let j = 0; j < n; j++) {
        re += a[i]![j]!.re * v[j]![k]!.re - a[i]![j]!.im * v[j]![k]!.im;
        im += a[i]![j]!.re * v[j]![k]!.im + a[i]![j]!.im * v[j]![k]!.re;
      }
      const lr = w[k]!.re;
      const li = w[k]!.im;
      const vr = v[i]![k]!.re;
      const vi = v[i]![k]!.im;
      worst = Math.max(worst, Math.hypot(re - (lr * vr - li * vi), im - (lr * vi + li * vr)));
    }
  }
  expect(
    worst,
    `${label}: A v != lambda v, eigenvectors do not match their eigenvalues`,
  ).toBeLessThan(tol);
}

/**
 * Assert the result dtype. Comparing values against NumPy cannot catch a result
 * that silently dropped to float64, because the comparison promotes a real to a
 * complex with a zero imaginary part.
 *
 * @param arr - Result array
 * @param expected - The dtype NumPy returns for this input
 * @param label - Test identifier used in the failure message
 */
export function expectDType(arr: { dtype: string }, expected: string, label: string): void {
  expect(arr.dtype, `${label}: expected dtype ${expected}, got ${arr.dtype}`).toBe(expected);
}

/**
 * The dtype NumPy gives a decomposition of this input. Everything narrower than
 * float64 widens to it; float32 and the complex widths are kept.
 *
 * @param dtype - Input dtype
 * @returns Expected dtype of a matrix-valued factor
 */
export function linalgDType(dtype: string): string {
  if (dtype === 'float32') return 'float32';
  if (dtype === 'complex64') return 'complex64';
  if (dtype === 'complex128') return 'complex128';
  return 'float64';
}

/**
 * The dtype NumPy gives a real-valued result of a decomposition, such as
 * singular values or the eigenvalues of a Hermitian matrix. Complex input
 * narrows to the matching real width rather than staying complex.
 *
 * @param dtype - Input dtype
 * @returns Expected dtype of a real-valued factor
 */
export function linalgRealDType(dtype: string): string {
  return dtype === 'float32' || dtype === 'complex64' ? 'float32' : 'float64';
}
