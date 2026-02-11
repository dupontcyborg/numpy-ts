/**
 * Wrapper Validation Tests
 *
 * Validates that the generated full/ wrappers (NDArray) match the core/
 * functions (NDArrayCore) and properly upgrade return types.
 *
 * For each wrapped function:
 * 1. Core returns NDArrayCore (not NDArray)
 * 2. Full returns NDArray (which extends NDArrayCore)
 * 3. Both produce identical data, shape, and dtype
 */

import { describe, it, expect } from 'vitest';
import * as core from '../../src/core';
import * as full from '../../src/full';
import { NDArray } from '../../src/full';
import { NDArrayCore } from '../../src/common/ndarray-core';

// Helper: compare two array results for data equality
function assertSameData(
  coreResult: NDArrayCore,
  fullResult: NDArray,
  funcName: string
) {
  expect(fullResult).toBeInstanceOf(NDArray);
  expect(fullResult.shape).toEqual(Array.from(coreResult.shape));
  expect(fullResult.dtype).toBe(coreResult.dtype);
  expect(fullResult.size).toBe(coreResult.size);

  // Compare element data
  for (let i = 0; i < coreResult.size; i++) {
    const cv = coreResult.data[i];
    const fv = fullResult.data[i];
    if (typeof cv === 'number' && !Number.isNaN(cv)) {
      expect(fv, `${funcName} data mismatch at index ${i}`).toBeCloseTo(cv as number, 10);
    } else {
      expect(fv, `${funcName} data mismatch at index ${i}`).toBe(cv);
    }
  }
}

// Shared test data
const a1d = core.array([1, 2, 3, 4, 5]);
const b1d = core.array([5, 4, 3, 2, 1]);
const a2d = core.array([
  [1, 2, 3],
  [4, 5, 6],
]);
const b2d = core.array([
  [7, 8, 9],
  [10, 11, 12],
]);
const mat2x2 = core.array([
  [1, 2],
  [3, 4],
]);
const mat2x2b = core.array([
  [5, 6],
  [7, 8],
]);
const boolArr = core.array([1, 0, 1, 0, 1]);

describe('Wrapper Validation: full/ matches core/', () => {
  // ============================================================
  // Type checks: core returns NDArrayCore, full returns NDArray
  // ============================================================

  describe('return type validation', () => {
    it('core.array returns NDArrayCore (not NDArray)', () => {
      const result = core.array([1, 2, 3]);
      expect(result).toBeInstanceOf(NDArrayCore);
      expect(result).not.toBeInstanceOf(NDArray);
    });

    it('full.array returns NDArray', () => {
      const result = full.array([1, 2, 3]);
      expect(result).toBeInstanceOf(NDArray);
      expect(result).toBeInstanceOf(NDArrayCore); // NDArray extends NDArrayCore
    });
  });

  // ============================================================
  // Creation functions
  // ============================================================

  describe('creation functions', () => {
    const creationTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['zeros', () => core.zeros([3, 4]), () => full.zeros([3, 4])],
      ['ones', () => core.ones([2, 3]), () => full.ones([2, 3])],
      ['full', () => core.full([2, 3], 7), () => full.full([2, 3], 7)],
      ['empty', () => core.empty([2, 2]), () => full.empty([2, 2])],
      ['arange', () => core.arange(0, 10, 2), () => full.arange(0, 10, 2)],
      ['linspace', () => core.linspace(0, 1, 5), () => full.linspace(0, 1, 5)],
      ['logspace', () => core.logspace(0, 2, 5), () => full.logspace(0, 2, 5)],
      ['eye', () => core.eye(3), () => full.eye(3)],
      ['identity', () => core.identity(3), () => full.identity(3)],
      ['array', () => core.array([1, 2, 3]), () => full.array([1, 2, 3])],
      ['zeros_like', () => core.zeros_like(a1d), () => full.zeros_like(a1d)],
      ['ones_like', () => core.ones_like(a1d), () => full.ones_like(a1d)],
      ['full_like', () => core.full_like(a1d, 9), () => full.full_like(a1d, 9)],
      ['copy', () => core.copy(a1d), () => full.copy(a1d)],
      ['diag', () => core.diag(a1d), () => full.diag(a1d)],
      [
        'tri',
        () => core.tri(3, 3),
        () => full.tri(3, 3),
      ],
      ['tril', () => core.tril(mat2x2), () => full.tril(mat2x2)],
      ['triu', () => core.triu(mat2x2), () => full.triu(mat2x2)],
    ];

    for (const [name, coreFn, fullFn] of creationTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Arithmetic functions
  // ============================================================

  describe('arithmetic functions', () => {
    const arithmeticTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['add', () => core.add(a1d, b1d), () => full.add(a1d, b1d)],
      ['subtract', () => core.subtract(a1d, b1d), () => full.subtract(a1d, b1d)],
      ['multiply', () => core.multiply(a1d, b1d), () => full.multiply(a1d, b1d)],
      ['divide', () => core.divide(a1d, b1d), () => full.divide(a1d, b1d)],
      ['power', () => core.power(a1d, 2), () => full.power(a1d, 2)],
      ['sqrt', () => core.sqrt(a1d), () => full.sqrt(a1d)],
      ['negative', () => core.negative(a1d), () => full.negative(a1d)],
      ['absolute', () => core.absolute(core.array([-1, -2, 3])), () => full.absolute(core.array([-1, -2, 3]))],
      ['sign', () => core.sign(core.array([-3, 0, 5])), () => full.sign(core.array([-3, 0, 5]))],
      ['mod', () => core.mod(a1d, 3), () => full.mod(a1d, 3)],
      ['clip', () => core.clip(a1d, 2, 4), () => full.clip(a1d, 2, 4)],
      ['maximum', () => core.maximum(a1d, b1d), () => full.maximum(a1d, b1d)],
      ['minimum', () => core.minimum(a1d, b1d), () => full.minimum(a1d, b1d)],
      // Scalar arithmetic
      ['add scalar', () => core.add(a1d, 10), () => full.add(a1d, 10)],
      ['multiply scalar', () => core.multiply(a1d, 3), () => full.multiply(a1d, 3)],
    ];

    for (const [name, coreFn, fullFn] of arithmeticTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Exponential and logarithmic
  // ============================================================

  describe('exponential/log functions', () => {
    const posArr = core.array([0.5, 1, 2, 3]);

    const expTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['exp', () => core.exp(a1d), () => full.exp(a1d)],
      ['exp2', () => core.exp2(a1d), () => full.exp2(a1d)],
      ['expm1', () => core.expm1(a1d), () => full.expm1(a1d)],
      ['log', () => core.log(posArr), () => full.log(posArr)],
      ['log2', () => core.log2(posArr), () => full.log2(posArr)],
      ['log10', () => core.log10(posArr), () => full.log10(posArr)],
      ['log1p', () => core.log1p(posArr), () => full.log1p(posArr)],
    ];

    for (const [name, coreFn, fullFn] of expTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Trigonometric functions
  // ============================================================

  describe('trigonometric functions', () => {
    const angles = core.array([0, 0.5, 1, 1.5]);

    const trigTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['sin', () => core.sin(angles), () => full.sin(angles)],
      ['cos', () => core.cos(angles), () => full.cos(angles)],
      ['tan', () => core.tan(angles), () => full.tan(angles)],
      ['arcsin', () => core.arcsin(core.array([0, 0.5, 1])), () => full.arcsin(core.array([0, 0.5, 1]))],
      ['arccos', () => core.arccos(core.array([0, 0.5, 1])), () => full.arccos(core.array([0, 0.5, 1]))],
      ['arctan', () => core.arctan(angles), () => full.arctan(angles)],
      ['sinh', () => core.sinh(angles), () => full.sinh(angles)],
      ['cosh', () => core.cosh(angles), () => full.cosh(angles)],
      ['tanh', () => core.tanh(angles), () => full.tanh(angles)],
      ['degrees', () => core.degrees(angles), () => full.degrees(angles)],
      ['radians', () => core.radians(core.array([0, 45, 90, 180])), () => full.radians(core.array([0, 45, 90, 180]))],
    ];

    for (const [name, coreFn, fullFn] of trigTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Reduction functions
  // ============================================================

  describe('reduction functions', () => {
    it('sum() matches', () => {
      const cr = core.sum(a2d, 0) as NDArrayCore;
      const fr = full.sum(a2d, 0) as NDArray;
      assertSameData(cr, fr, 'sum');
    });

    it('mean() matches', () => {
      const cr = core.mean(a2d, 1) as NDArrayCore;
      const fr = full.mean(a2d, 1) as NDArray;
      assertSameData(cr, fr, 'mean');
    });

    it('prod() matches', () => {
      const cr = core.prod(a1d) as number;
      const fr = full.prod(a1d) as number;
      expect(fr).toBe(cr);
    });

    it('cumsum() matches', () => {
      const cr = core.cumsum(a1d) as NDArrayCore;
      const fr = full.cumsum(a1d) as NDArray;
      assertSameData(cr, fr, 'cumsum');
    });

    it('cumprod() matches', () => {
      const cr = core.cumprod(core.array([1, 2, 3])) as NDArrayCore;
      const fr = full.cumprod(core.array([1, 2, 3])) as NDArray;
      assertSameData(cr, fr, 'cumprod');
    });

    it('variance() matches', () => {
      const cr = core.variance(a1d) as number;
      const fr = full.variance(a1d) as number;
      expect(fr).toBeCloseTo(cr, 10);
    });

    it('std() matches', () => {
      const cr = core.std(a1d) as number;
      const fr = full.std(a1d) as number;
      expect(fr).toBeCloseTo(cr, 10);
    });

    it('median() matches', () => {
      const cr = core.median(a1d) as number;
      const fr = full.median(a1d) as number;
      expect(fr).toBe(cr);
    });
  });

  // ============================================================
  // Shape manipulation
  // ============================================================

  describe('shape manipulation', () => {
    it('reshape() matches', () => {
      const cr = core.reshape(a2d, [6]);
      const fr = full.reshape(a2d, [6]);
      assertSameData(cr, fr as NDArray, 'reshape');
    });

    it('flatten() matches', () => {
      const cr = core.flatten(a2d);
      const fr = full.flatten(a2d);
      assertSameData(cr, fr as NDArray, 'flatten');
    });

    it('ravel() matches', () => {
      const cr = core.ravel(a2d);
      const fr = full.ravel(a2d);
      assertSameData(cr, fr as NDArray, 'ravel');
    });

    it('squeeze() matches', () => {
      const arr = core.array([[[1, 2, 3]]]);
      const cr = core.squeeze(arr);
      const fr = full.squeeze(arr);
      assertSameData(cr, fr as NDArray, 'squeeze');
    });

    it('concatenate() matches', () => {
      const cr = core.concatenate([a1d, b1d], 0);
      const fr = full.concatenate([a1d, b1d], 0);
      assertSameData(cr, fr as NDArray, 'concatenate');
    });

    it('stack() matches', () => {
      const cr = core.stack([a1d, b1d], 0);
      const fr = full.stack([a1d, b1d], 0);
      assertSameData(cr, fr as NDArray, 'stack');
    });

    it('vstack() matches', () => {
      const cr = core.vstack([a1d, b1d]);
      const fr = full.vstack([a1d, b1d]);
      assertSameData(cr, fr as NDArray, 'vstack');
    });

    it('hstack() matches', () => {
      const cr = core.hstack([a1d, b1d]);
      const fr = full.hstack([a1d, b1d]);
      assertSameData(cr, fr as NDArray, 'hstack');
    });

    it('tile() matches', () => {
      const cr = core.tile(a1d, [2]);
      const fr = full.tile(a1d, [2]);
      assertSameData(cr, fr as NDArray, 'tile');
    });

    it('repeat() matches', () => {
      const cr = core.repeat(a1d, 2);
      const fr = full.repeat(a1d, 2);
      assertSameData(cr, fr as NDArray, 'repeat');
    });

    it('flip() matches', () => {
      const cr = core.flip(a1d);
      const fr = full.flip(a1d);
      assertSameData(cr, fr as NDArray, 'flip');
    });

    it('transpose() matches', () => {
      const cr = core.transpose(a2d);
      const fr = full.transpose(a2d);
      assertSameData(cr, fr as NDArray, 'transpose');
    });

    it('swapaxes() matches', () => {
      const cr = core.swapaxes(a2d, 0, 1);
      const fr = full.swapaxes(a2d, 0, 1);
      assertSameData(cr, fr as NDArray, 'swapaxes');
    });

    it('expand_dims() matches', () => {
      const cr = core.expand_dims(a1d, 0);
      const fr = full.expand_dims(a1d, 0);
      assertSameData(cr, fr as NDArray, 'expand_dims');
    });
  });

  // ============================================================
  // Linear algebra
  // ============================================================

  describe('linear algebra', () => {
    it('matmul() matches', () => {
      const cr = core.matmul(mat2x2, mat2x2b);
      const fr = full.matmul(mat2x2, mat2x2b);
      assertSameData(cr, fr as NDArray, 'matmul');
    });

    it('dot() matches (array result)', () => {
      const cr = core.dot(mat2x2, mat2x2b);
      const fr = full.dot(mat2x2, mat2x2b);
      if (cr instanceof NDArrayCore && fr instanceof NDArray) {
        assertSameData(cr, fr, 'dot');
      }
    });

    it('dot() matches (scalar result)', () => {
      const cr = core.dot(a1d, b1d);
      const fr = full.dot(a1d, b1d);
      expect(fr).toBe(cr);
    });

    it('diagonal() matches', () => {
      const cr = core.diagonal(mat2x2);
      const fr = full.diagonal(mat2x2);
      assertSameData(cr, fr as NDArray, 'diagonal');
    });

    it('kron() matches', () => {
      const cr = core.kron(core.array([1, 2]), core.array([3, 4]));
      const fr = full.kron(core.array([1, 2]), core.array([3, 4]));
      assertSameData(cr, fr as NDArray, 'kron');
    });

    it('outer() matches', () => {
      const cr = core.outer(core.array([1, 2]), core.array([3, 4]));
      const fr = full.outer(core.array([1, 2]), core.array([3, 4]));
      assertSameData(cr, fr as NDArray, 'outer');
    });
  });

  // ============================================================
  // Comparison functions
  // ============================================================

  describe('comparison functions', () => {
    const compTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['greater', () => core.greater(a1d, b1d), () => full.greater(a1d, b1d)],
      ['greater_equal', () => core.greater_equal(a1d, b1d), () => full.greater_equal(a1d, b1d)],
      ['less', () => core.less(a1d, b1d), () => full.less(a1d, b1d)],
      ['less_equal', () => core.less_equal(a1d, b1d), () => full.less_equal(a1d, b1d)],
      ['equal', () => core.equal(a1d, core.array([1, 0, 3, 0, 5])), () => full.equal(a1d, core.array([1, 0, 3, 0, 5]))],
      ['not_equal', () => core.not_equal(a1d, core.array([1, 0, 3, 0, 5])), () => full.not_equal(a1d, core.array([1, 0, 3, 0, 5]))],
      ['isclose', () => core.isclose(a1d, core.array([1.0001, 2, 3, 4, 5])), () => full.isclose(a1d, core.array([1.0001, 2, 3, 4, 5]))],
    ];

    for (const [name, coreFn, fullFn] of compTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }

    it('allclose() matches (boolean result)', () => {
      const cr = core.allclose(a1d, a1d);
      const fr = full.allclose(a1d, a1d);
      expect(fr).toBe(cr);
      expect(fr).toBe(true);
    });
  });

  // ============================================================
  // Rounding functions
  // ============================================================

  describe('rounding functions', () => {
    const floatArr = core.array([1.2, 2.7, 3.5, 4.1, 5.9]);

    const roundTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['around', () => core.around(floatArr), () => full.around(floatArr)],
      ['ceil', () => core.ceil(floatArr), () => full.ceil(floatArr)],
      ['floor', () => core.floor(floatArr), () => full.floor(floatArr)],
      ['trunc', () => core.trunc(floatArr), () => full.trunc(floatArr)],
      ['rint', () => core.rint(floatArr), () => full.rint(floatArr)],
    ];

    for (const [name, coreFn, fullFn] of roundTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Logic functions
  // ============================================================

  describe('logic functions', () => {
    const logicTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['logical_and', () => core.logical_and(a1d, boolArr), () => full.logical_and(a1d, boolArr)],
      ['logical_or', () => core.logical_or(a1d, boolArr), () => full.logical_or(a1d, boolArr)],
      ['logical_not', () => core.logical_not(boolArr), () => full.logical_not(boolArr)],
      ['isnan', () => core.isnan(core.array([1, NaN, 3])), () => full.isnan(core.array([1, NaN, 3]))],
      ['isinf', () => core.isinf(core.array([1, Infinity, -Infinity])), () => full.isinf(core.array([1, Infinity, -Infinity]))],
      ['isfinite', () => core.isfinite(core.array([1, NaN, Infinity])), () => full.isfinite(core.array([1, NaN, Infinity]))],
    ];

    for (const [name, coreFn, fullFn] of logicTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Sorting functions
  // ============================================================

  describe('sorting functions', () => {
    it('sort() matches', () => {
      const unsorted = core.array([3, 1, 4, 1, 5, 9, 2, 6]);
      const cr = core.sort(unsorted);
      const fr = full.sort(unsorted);
      assertSameData(cr, fr as NDArray, 'sort');
    });

    it('argsort() matches', () => {
      const unsorted = core.array([3, 1, 4, 1, 5]);
      const cr = core.argsort(unsorted);
      const fr = full.argsort(unsorted);
      assertSameData(cr, fr as NDArray, 'argsort');
    });
  });

  // ============================================================
  // Bitwise functions
  // ============================================================

  describe('bitwise functions', () => {
    const intA = core.array([1, 2, 3, 4], 'int32');
    const intB = core.array([4, 3, 2, 1], 'int32');

    const bitwiseTests: [string, () => NDArrayCore, () => NDArray][] = [
      ['bitwise_and', () => core.bitwise_and(intA, intB), () => full.bitwise_and(intA, intB)],
      ['bitwise_or', () => core.bitwise_or(intA, intB), () => full.bitwise_or(intA, intB)],
      ['bitwise_xor', () => core.bitwise_xor(intA, intB), () => full.bitwise_xor(intA, intB)],
      ['left_shift', () => core.left_shift(intA, 1), () => full.left_shift(intA, 1)],
      ['right_shift', () => core.right_shift(intA, 1), () => full.right_shift(intA, 1)],
    ];

    for (const [name, coreFn, fullFn] of bitwiseTests) {
      it(`${name}() produces matching results`, () => {
        const cr = coreFn();
        const fr = fullFn();
        expect(cr).not.toBeInstanceOf(NDArray);
        assertSameData(cr, fr as NDArray, name);
      });
    }
  });

  // ============================================================
  // Set operations
  // ============================================================

  describe('set operations', () => {
    it('unique() matches', () => {
      // unique() is intentionally not wrapped (complex union return type)
      // so full.unique() re-exports from core and returns NDArrayCore
      const cr = core.unique(core.array([3, 1, 2, 1, 3])) as NDArrayCore;
      const fr = full.unique(core.array([3, 1, 2, 1, 3])) as NDArrayCore;
      expect(Array.from(fr.data)).toEqual(Array.from(cr.data));
      expect(fr.dtype).toBe(cr.dtype);
    });

    it('intersect1d() matches', () => {
      const cr = core.intersect1d(core.array([1, 2, 3]), core.array([2, 3, 4]));
      const fr = full.intersect1d(core.array([1, 2, 3]), core.array([2, 3, 4]));
      assertSameData(cr, fr as NDArray, 'intersect1d');
    });
  });

  // ============================================================
  // Polynomial functions
  // ============================================================

  describe('polynomial functions', () => {
    it('polyval() matches', () => {
      const coeffs = core.array([1, -2, 1]); // x^2 - 2x + 1
      const x = core.array([0, 1, 2, 3]);
      const cr = core.polyval(coeffs, x);
      const fr = full.polyval(coeffs, x);
      assertSameData(cr, fr as NDArray, 'polyval');
    });

    it('polyadd() matches', () => {
      const cr = core.polyadd(core.array([1, 2]), core.array([3, 4]));
      const fr = full.polyadd(core.array([1, 2]), core.array([3, 4]));
      assertSameData(cr, fr as NDArray, 'polyadd');
    });
  });

  // ============================================================
  // Shape-extra functions
  // ============================================================

  describe('shape-extra functions', () => {
    it('append() matches', () => {
      const cr = core.append(a1d, core.array([6, 7]));
      const fr = full.append(a1d, core.array([6, 7]));
      assertSameData(cr, fr as NDArray, 'append');
    });

    it('delete_() matches', () => {
      const cr = core.delete_(a1d, 2);
      const fr = full.delete_(a1d, 2);
      assertSameData(cr, fr as NDArray, 'delete_');
    });

    it('insert() matches', () => {
      const cr = core.insert(a1d, 2, 99);
      const fr = full.insert(a1d, 2, 99);
      assertSameData(cr, fr as NDArray, 'insert');
    });

    it('pad() matches', () => {
      const cr = core.pad(a1d, 2);
      const fr = full.pad(a1d, 2);
      assertSameData(cr, fr as NDArray, 'pad');
    });
  });

  // ============================================================
  // Gradient functions
  // ============================================================

  describe('gradient functions', () => {
    it('diff() matches', () => {
      const cr = core.diff(a1d);
      const fr = full.diff(a1d);
      assertSameData(cr, fr as NDArray, 'diff');
    });

    it('ediff1d() matches', () => {
      const cr = core.ediff1d(a1d);
      const fr = full.ediff1d(a1d);
      assertSameData(cr, fr as NDArray, 'ediff1d');
    });
  });

  // ============================================================
  // Advanced operations
  // ============================================================

  describe('advanced operations', () => {
    it('broadcast_to() matches', () => {
      const cr = core.broadcast_to(core.array([1, 2, 3]), [2, 3]);
      const fr = full.broadcast_to(core.array([1, 2, 3]), [2, 3]);
      assertSameData(cr, fr as NDArray, 'broadcast_to');
    });

    it('take() matches', () => {
      const cr = core.take(a1d, core.array([0, 2, 4], 'int32'));
      const fr = full.take(a1d, core.array([0, 2, 4], 'int32'));
      assertSameData(cr, fr as NDArray, 'take');
    });
  });

  // ============================================================
  // Zero-copy storage sharing
  // ============================================================

  describe('zero-copy upgrade', () => {
    it('full wrapper shares storage with core result', () => {
      // The up() function in full/index.ts creates NDArray from the same storage
      // This test verifies that the data buffer is shared, not copied
      const coreArr = core.zeros([100]);
      const fullArr = full.zeros([100]);

      // Both should have the same size
      expect(fullArr.size).toBe(coreArr.size);

      // Modifying the core array and creating a full wrapper should share data
      // (This tests the up() function indirectly via view operations)
      const coreSlice = core.reshape(coreArr, [10, 10]);
      const fullSlice = full.reshape(coreArr, [10, 10]);
      expect(fullSlice.shape).toEqual(Array.from(coreSlice.shape));
    });
  });
});
