/**
 * Self-tests for the decomposition invariant helpers.
 *
 * The helpers exist because comparing a decomposition elementwise against NumPy
 * cannot work, and the failure they guard against is a check that passes on a
 * wrong answer. So each one is fed a deliberately broken input here and has to
 * reject it. Without this, a helper that silently accepted everything would
 * make the whole sweep read green.
 */

import { describe, expect, it } from 'vitest';
import {
  type C,
  conjT,
  expectDType,
  expectEigenpairs,
  expectReconstructs,
  expectUnitaryColumns,
  expectUpperTriangular,
  mat,
  matmulC,
  vec,
} from '../validation/dtype-sweep/_invariants';

/** A = [[1-0.5i, 2-0.9i], [3-1.3i, 4-1.7i]] with its true eigenpairs. */
const A = mat([
  [
    { re: 1, im: -0.5 },
    { re: 2, im: -0.9 },
  ],
  [
    { re: 3, im: -1.3 },
    { re: 4, im: -1.7 },
  ],
]);
const W = vec([
  { re: -0.37192524, im: 0.1361046 },
  { re: 5.37192524, im: -2.3361046 },
]);
const V = mat([
  [
    { re: 0.82326758, im: 0 },
    { re: 0.41965508, im: -0.01059693 },
  ],
  [
    { re: -0.56761726, im: 0.00641438 },
    { re: 0.90762179, im: 0 },
  ],
]);

/** Whether the assertion inside fn failed. */
function rejects(fn: () => void): boolean {
  try {
    fn();
    return false;
  } catch {
    return true;
  }
}

describe('decomposition invariant helpers', () => {
  it('expectEigenpairs accepts true eigenpairs', () => {
    expectEigenpairs(A, W, V, 'true pairs', 1e-6);
  });

  it('expectEigenpairs rejects eigenvectors paired with the wrong eigenvalues', () => {
    // Issue #161: the values were right and the vectors were right, but the
    // pairing was not. A comparison of eigenvalues alone cannot see this.
    const swapped = V.map((row) => [row[1]!, row[0]!]);
    expect(rejects(() => expectEigenpairs(A, W, swapped, 'swapped', 1e-6))).toBe(true);
  });

  it('expectEigenpairs rejects eigenvectors with the imaginary half dropped', () => {
    const realOnly = V.map((row) => row.map((c) => ({ re: c.re, im: 0 })));
    expect(rejects(() => expectEigenpairs(A, W, realOnly, 'real only', 1e-6))).toBe(true);
  });

  it('expectUnitaryColumns rejects a column that is not unit length', () => {
    const q: C[][] = [
      [
        { re: 1, im: 0 },
        { re: 0, im: 0 },
      ],
      [
        { re: 0, im: 0 },
        { re: 2, im: 0 },
      ],
    ];
    expect(rejects(() => expectUnitaryColumns(q, 'scaled', 1e-9))).toBe(true);
  });

  it('expectUnitaryColumns rejects non-orthogonal columns and accepts orthonormal ones', () => {
    const k = 1 / Math.SQRT2;
    const good: C[][] = [
      [
        { re: k, im: k },
        { re: 0, im: 0 },
      ],
      [
        { re: 0, im: 0 },
        { re: 1, im: 0 },
      ],
    ];
    const bad: C[][] = [
      [
        { re: k, im: k },
        { re: 0, im: 0 },
      ],
      [
        { re: k, im: -k },
        { re: 1, im: 0 },
      ],
    ];
    expect(rejects(() => expectUnitaryColumns(good, 'orthonormal', 1e-9))).toBe(false);
    expect(rejects(() => expectUnitaryColumns(bad, 'not orthogonal', 1e-9))).toBe(true);
  });

  it('expectUpperTriangular rejects a nonzero below the diagonal', () => {
    const r: C[][] = [
      [
        { re: 1, im: 0 },
        { re: 2, im: 0 },
      ],
      [
        { re: 1e-3, im: 0 },
        { re: 3, im: 0 },
      ],
    ];
    expect(rejects(() => expectUpperTriangular(r, 'has lower part', 1e-9))).toBe(true);
  });

  it('expectReconstructs tells a conjugate transpose from a plain one', () => {
    const L = mat([
      [
        { re: 2, im: 0 },
        { re: 0, im: 0 },
      ],
      [
        { re: 1, im: 1 },
        { re: 2, im: 0 },
      ],
    ]);
    const hermitian = matmulC(L, conjT(L));
    const plainTranspose = L[0]!.map((_, j) => L.map((row) => row[j]!));
    expect(rejects(() => expectReconstructs(hermitian, hermitian, 'conjugate', 1e-9))).toBe(false);
    expect(
      rejects(() => expectReconstructs(matmulC(L, plainTranspose), hermitian, 'plain', 1e-9)),
    ).toBe(true);
  });

  it('expectDType rejects a result that dropped to float64', () => {
    expect(rejects(() => expectDType({ dtype: 'complex128' }, 'complex128', 'kept'))).toBe(false);
    expect(rejects(() => expectDType({ dtype: 'float64' }, 'complex128', 'dropped'))).toBe(true);
  });
});
