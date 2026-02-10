/**
 * Python NumPy validation tests for polynomial functions
 *
 * These tests run the same operations in both TypeScript and Python NumPy
 * and verify the results match exactly.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array } from '../../src';
import {
  poly,
  polyadd,
  polyder,
  polydiv,
  polyfit,
  polyint,
  polymul,
  polysub,
  polyval,
  roots,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: Polynomial Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  describe('poly()', () => {
    it('validates poly with single root', () => {
      const result = poly([2]);

      const npResult = runNumPy(`
result = np.poly([2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates poly with multiple roots', () => {
      const result = poly([1, 2, 3]);

      const npResult = runNumPy(`
result = np.poly([1, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates poly with zero roots', () => {
      const result = poly([0, 0]);

      const npResult = runNumPy(`
result = np.poly([0, 0])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polyadd()', () => {
    it('validates polyadd with same degree polynomials', () => {
      const result = polyadd([1, 2, 3], [4, 5, 6]);

      const npResult = runNumPy(`
result = np.polyadd([1, 2, 3], [4, 5, 6])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates polyadd with different degree polynomials', () => {
      const result = polyadd([1, 2, 3], [4, 5]);

      const npResult = runNumPy(`
result = np.polyadd([1, 2, 3], [4, 5])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polysub()', () => {
    it('validates polysub with polynomials', () => {
      const result = polysub([1, 2, 3], [0, 1, 1]);

      const npResult = runNumPy(`
result = np.polysub([1, 2, 3], [0, 1, 1])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polymul()', () => {
    it('validates polymul with linear polynomials', () => {
      const result = polymul([1, 1], [1, 2]);

      const npResult = runNumPy(`
result = np.polymul([1, 1], [1, 2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates polymul with higher degree', () => {
      const result = polymul([1, 0, 1], [1, 1]);

      const npResult = runNumPy(`
result = np.polymul([1, 0, 1], [1, 1])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polydiv()', () => {
    it('validates polydiv with exact division', () => {
      const [quotient, remainder] = polydiv([1, -3, 2], [1, -1]);

      const npResult = runNumPy(`
q, r = np.polydiv([1, -3, 2], [1, -1])
result = np.concatenate([q, r])
`);

      const combined = [...(quotient.toArray() as number[]), ...(remainder.toArray() as number[])];
      expect(arraysClose(combined, npResult.value, 1e-10)).toBe(true);
    });

    it('validates polydiv with remainder', () => {
      const [quotient, remainder] = polydiv([1, 0, 1], [1, 1]);

      const npResult = runNumPy(`
q, r = np.polydiv([1, 0, 1], [1, 1])
result = np.array([q[0], q[1], r[0]])
`);

      const qArr = quotient.toArray() as number[];
      const rArr = remainder.toArray() as number[];
      expect(arraysClose([qArr[0], qArr[1], rArr[0]], npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polyder()', () => {
    it('validates polyder for quadratic', () => {
      const result = polyder([3, 2, 1]);

      const npResult = runNumPy(`
result = np.polyder([3, 2, 1])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates polyder for second derivative', () => {
      const result = polyder([1, 1, 1, 1], 2);

      const npResult = runNumPy(`
result = np.polyder([1, 1, 1, 1], 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polyint()', () => {
    it('validates polyint for basic integration', () => {
      const result = polyint([6, 2]);

      const npResult = runNumPy(`
result = np.polyint([6, 2])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });

    it('validates polyint with integration constant', () => {
      const result = polyint([2], 1, 5);

      const npResult = runNumPy(`
result = np.polyint([2], 1, 5)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polyval()', () => {
    it('validates polyval at single value', () => {
      const result = polyval([1, -2, 1], 3);

      const npResult = runNumPy(`
result = np.polyval([1, -2, 1], 3)
`);

      expect(result).toBeCloseTo(npResult.value, 10);
    });

    it('validates polyval at multiple values', () => {
      const result = polyval([1, 0, 0], [1, 2, 3]) as any;

      const npResult = runNumPy(`
result = np.polyval([1, 0, 0], [1, 2, 3])
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });

  describe('polyfit()', () => {
    it('validates polyfit for linear fit', () => {
      const x = array([0, 1, 2, 3, 4]);
      const y = array([1, 3, 5, 7, 9]);
      const result = polyfit(x, y, 1);

      const npResult = runNumPy(`
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 5, 7, 9])
result = np.polyfit(x, y, 1)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-5)).toBe(true);
    });

    it('validates polyfit for quadratic fit', () => {
      const x = array([0, 1, 2, 3, 4]);
      const y = array([0, 1, 4, 9, 16]);
      const result = polyfit(x, y, 2);

      const npResult = runNumPy(`
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])
result = np.polyfit(x, y, 2)
`);

      expect(result.shape).toEqual(npResult.shape);
      expect(arraysClose(result.toArray(), npResult.value, 1e-5)).toBe(true);
    });
  });

  describe('roots()', () => {
    it('validates roots for quadratic', () => {
      const result = roots([1, -3, 2]);
      const resultArr = (result.toArray() as number[]).sort((a, b) => a - b);

      const npResult = runNumPy(`
result = np.sort(np.roots([1, -3, 2]))
`);

      expect(arraysClose(resultArr, npResult.value, 1e-5)).toBe(true);
    });

    it('validates roots for x^2 - 1', () => {
      const result = roots([1, 0, -1]);
      const resultArr = (result.toArray() as number[]).sort((a, b) => a - b);

      const npResult = runNumPy(`
result = np.sort(np.roots([1, 0, -1]).real)
`);

      expect(arraysClose(resultArr, npResult.value, 1e-5)).toBe(true);
    });

    it('validates roots for linear polynomial', () => {
      const result = roots([1, -5]);

      const npResult = runNumPy(`
result = np.roots([1, -5])
`);

      expect(arraysClose(result.toArray(), npResult.value, 1e-10)).toBe(true);
    });
  });
});
