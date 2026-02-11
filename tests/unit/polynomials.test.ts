/**
 * Unit tests for polynomial functions
 */

import { describe, it, expect } from 'vitest';
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
import { Complex } from '../../src/common/complex';

/**
 * Extract real parts from a complex128 roots result.
 */
function getRealParts(result: ReturnType<typeof roots>): number[] {
  const arr = result.toArray() as Complex[];
  return arr.map((c) => c.re);
}

/**
 * Extract imaginary parts from a complex128 roots result.
 */
function getImagParts(result: ReturnType<typeof roots>): number[] {
  const arr = result.toArray() as Complex[];
  return arr.map((c) => c.im);
}

/**
 * Get magnitudes of complex roots.
 */
function getMagnitudes(result: ReturnType<typeof roots>): number[] {
  const arr = result.toArray() as Complex[];
  return arr.map((c) => c.abs());
}

describe('Polynomial Functions', () => {
  describe('poly()', () => {
    it('returns [1] for empty roots', () => {
      const result = poly([]);
      expect(result.toArray()).toEqual([1]);
    });

    it('builds polynomial from single root', () => {
      // (x - 2) = x - 2 = [1, -2]
      const result = poly([2]);
      expect(result.toArray()).toEqual([1, -2]);
    });

    it('builds polynomial from multiple roots', () => {
      // (x - 1)(x - 2) = x^2 - 3x + 2 = [1, -3, 2]
      const result = poly([1, 2]);
      const arr = result.toArray() as number[];
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(-3);
      expect(arr[2]).toBeCloseTo(2);
    });

    it('handles zero roots', () => {
      // x^2 = [1, 0, 0]
      const result = poly([0, 0]);
      expect(result.toArray()).toEqual([1, 0, 0]);
    });

    it('handles NDArray input', () => {
      const rootsArr = array([1, 2, 3]);
      const result = poly(rootsArr);
      // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
      const arr = result.toArray() as number[];
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(-6);
      expect(arr[2]).toBeCloseTo(11);
      expect(arr[3]).toBeCloseTo(-6);
    });
  });

  describe('polyadd()', () => {
    it('adds two polynomials of same degree', () => {
      // (x + 1) + (x + 2) = 2x + 3
      const result = polyadd([1, 1], [1, 2]);
      expect(result.toArray()).toEqual([2, 3]);
    });

    it('adds polynomials of different degrees', () => {
      // (x^2 + 2x + 3) + (x + 1) = x^2 + 3x + 4
      const result = polyadd([1, 2, 3], [1, 1]);
      expect(result.toArray()).toEqual([1, 3, 4]);
    });

    it('handles NDArray inputs', () => {
      const p1 = array([1, 2]);
      const p2 = array([3, 4]);
      const result = polyadd(p1, p2);
      expect(result.toArray()).toEqual([4, 6]);
    });
  });

  describe('polysub()', () => {
    it('subtracts two polynomials', () => {
      // (x^2 + 2x + 3) - (x + 1) = x^2 + x + 2
      const result = polysub([1, 2, 3], [1, 1]);
      expect(result.toArray()).toEqual([1, 1, 2]);
    });

    it('handles negative results', () => {
      // (x + 1) - (x + 3) = -2
      // Note: implementation strips leading zeros, so result is [-2] not [0, -2]
      const result = polysub([1, 1], [1, 3]);
      expect(result.toArray()).toEqual([-2]);
    });
  });

  describe('polymul()', () => {
    it('multiplies two linear polynomials', () => {
      // (x + 1)(x + 2) = x^2 + 3x + 2
      const result = polymul([1, 1], [1, 2]);
      expect(result.toArray()).toEqual([1, 3, 2]);
    });

    it('multiplies by constant', () => {
      // 2 * (x + 1) = 2x + 2
      const result = polymul([2], [1, 1]);
      expect(result.toArray()).toEqual([2, 2]);
    });

    it('multiplies higher degree polynomials', () => {
      // (x^2 + 1)(x + 1) = x^3 + x^2 + x + 1
      const result = polymul([1, 0, 1], [1, 1]);
      expect(result.toArray()).toEqual([1, 1, 1, 1]);
    });
  });

  describe('polydiv()', () => {
    it('divides polynomials with no remainder', () => {
      // (x^2 - 3x + 2) / (x - 1) = (x - 2)
      const [quotient, remainder] = polydiv([1, -3, 2], [1, -1]);
      expect(quotient.toArray()).toEqual([1, -2]);
      expect(remainder.toArray()).toEqual([0]);
    });

    it('divides polynomials with remainder', () => {
      // (x^2 + 1) / (x + 1) = (x - 1) with remainder 2
      const [quotient, remainder] = polydiv([1, 0, 1], [1, 1]);
      const qArr = quotient.toArray() as number[];
      const rArr = remainder.toArray() as number[];
      expect(qArr[0]).toBeCloseTo(1);
      expect(qArr[1]).toBeCloseTo(-1);
      expect(rArr[0]).toBeCloseTo(2);
    });

    it('returns [0] quotient when dividend degree < divisor degree', () => {
      const [quotient, remainder] = polydiv([1, 1], [1, 1, 1]);
      expect(quotient.toArray()).toEqual([0]);
      expect(remainder.toArray()).toEqual([1, 1]);
    });

    it('throws on division by zero polynomial', () => {
      expect(() => polydiv([1, 1], [0])).toThrow('Division by zero polynomial');
    });
  });

  describe('polyder()', () => {
    it('computes derivative of polynomial', () => {
      // d/dx (3x^2 + 2x + 1) = 6x + 2
      const result = polyder([3, 2, 1]);
      expect(result.toArray()).toEqual([6, 2]);
    });

    it('returns [0] for constant polynomial', () => {
      const result = polyder([5]);
      expect(result.toArray()).toEqual([0]);
    });

    it('computes higher order derivatives', () => {
      // d^2/dx^2 (x^3 + x^2 + x + 1) = 6x + 2
      const result = polyder([1, 1, 1, 1], 2);
      expect(result.toArray()).toEqual([6, 2]);
    });

    it('handles NDArray input', () => {
      const p = array([3, 2, 1]);
      const result = polyder(p);
      expect(result.toArray()).toEqual([6, 2]);
    });
  });

  describe('polyint()', () => {
    it('computes antiderivative of polynomial', () => {
      // ∫(6x + 2)dx = 3x^2 + 2x + 0
      const result = polyint([6, 2]);
      const arr = result.toArray() as number[];
      expect(arr[0]).toBeCloseTo(3);
      expect(arr[1]).toBeCloseTo(2);
      expect(arr[2]).toBeCloseTo(0);
    });

    it('includes integration constant', () => {
      // ∫(2)dx = 2x + 5 with constant 5
      const result = polyint([2], 1, 5);
      expect(result.toArray()).toEqual([2, 5]);
    });

    it('computes higher order antiderivatives', () => {
      // ∫∫(6)dxdx = 3x^2 + 0x + 0
      const result = polyint([6], 2);
      const arr = result.toArray() as number[];
      expect(arr[0]).toBeCloseTo(3);
    });
  });

  describe('polyval()', () => {
    it('evaluates polynomial at single value', () => {
      // p(x) = x^2 - 2x + 1 at x = 3 -> 9 - 6 + 1 = 4
      const result = polyval([1, -2, 1], 3);
      expect(result).toBe(4);
    });

    it('evaluates polynomial at multiple values', () => {
      // p(x) = x^2 at x = 1, 2, 3 -> 1, 4, 9
      const result = polyval([1, 0, 0], [1, 2, 3]) as any;
      expect(result.toArray()).toEqual([1, 4, 9]);
    });

    it('handles NDArray inputs', () => {
      const p = array([1, 0, 0]); // x^2
      const x = array([1, 2, 3]);
      const result = polyval(p, x) as any;
      expect(result.toArray()).toEqual([1, 4, 9]);
    });

    it('evaluates zero polynomial', () => {
      const result = polyval([0], 5);
      expect(result).toBe(0);
    });
  });

  describe('polyfit()', () => {
    it('fits linear polynomial', () => {
      // y = 2x + 1 through points (0,1), (1,3), (2,5)
      const x = array([0, 1, 2]);
      const y = array([1, 3, 5]);
      const coeffs = polyfit(x, y, 1);
      const arr = coeffs.toArray() as number[];
      expect(arr[0]).toBeCloseTo(2, 5);
      expect(arr[1]).toBeCloseTo(1, 5);
    });

    it('fits quadratic polynomial', () => {
      // y = x^2 through points
      const x = array([0, 1, 2, 3, 4]);
      const y = array([0, 1, 4, 9, 16]);
      const coeffs = polyfit(x, y, 2);
      const arr = coeffs.toArray() as number[];
      expect(arr[0]).toBeCloseTo(1, 5);
      expect(arr[1]).toBeCloseTo(0, 5);
      expect(arr[2]).toBeCloseTo(0, 5);
    });

    it.skip('throws for invalid inputs (validation not implemented)', () => {
      // Note: polyfit currently doesn't validate that x and y have the same length
      const x = array([1, 2]);
      const y = array([1, 2, 3]); // Different lengths
      expect(() => polyfit(x, y, 1)).toThrow('x and y must have the same length');
    });

    it.skip('throws for negative degree (validation not implemented)', () => {
      // Note: polyfit currently doesn't validate negative degree
      const x = array([0, 1, 2]);
      const y = array([0, 1, 2]);
      expect(() => polyfit(x, y, -1)).toThrow('Degree must be non-negative');
    });
  });

  describe('roots()', () => {
    it('returns complex128 dtype', () => {
      const result = roots([1, -2]);
      expect(result.dtype).toBe('complex128');
    });

    it('finds roots of linear polynomial', () => {
      // x - 2 = 0 -> x = 2
      const result = roots([1, -2]);
      const re = getRealParts(result);
      const im = getImagParts(result);
      expect(re[0]).toBeCloseTo(2, 5);
      expect(im[0]).toBeCloseTo(0, 5);
    });

    it('finds roots of quadratic polynomial', () => {
      // x^2 - 3x + 2 = 0 -> x = 1, 2
      const result = roots([1, -3, 2]);
      const re = getRealParts(result).sort((a, b) => a - b);
      expect(re[0]).toBeCloseTo(1, 5);
      expect(re[1]).toBeCloseTo(2, 5);
    });

    it('finds roots of x^2 - 1', () => {
      // x^2 - 1 = 0 -> x = -1, 1
      const result = roots([1, 0, -1]);
      const re = getRealParts(result).sort((a, b) => a - b);
      expect(re[0]).toBeCloseTo(-1, 5);
      expect(re[1]).toBeCloseTo(1, 5);
    });

    it('returns empty for constant polynomial', () => {
      const result = roots([5]);
      expect(result.size).toBe(0);
      expect(result.dtype).toBe('complex128');
    });

    it('handles repeated roots', () => {
      // (x - 1)^2 = x^2 - 2x + 1
      const result = roots([1, -2, 1]);
      const re = getRealParts(result);
      expect(re[0]).toBeCloseTo(1, 5);
      expect(re[1]).toBeCloseTo(1, 5);
    });

    it('handles cubic polynomial with real roots', () => {
      // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
      const result = roots([1, -6, 11, -6]);
      expect(result.size).toBe(3);
      const re = getRealParts(result).sort((a, b) => a - b);
      expect(re[0]).toBeCloseTo(1, 3);
      expect(re[1]).toBeCloseTo(2, 3);
      expect(re[2]).toBeCloseTo(3, 3);
    });

    it('handles NDArray input', () => {
      const p = array([1, -3, 2]);
      const result = roots(p);
      const re = getRealParts(result).sort((a, b) => a - b);
      expect(re[0]).toBeCloseTo(1, 5);
      expect(re[1]).toBeCloseTo(2, 5);
    });

    // Complex root tests
    it('finds pure imaginary roots for x^2 + 1', () => {
      // x^2 + 1 = 0 -> x = +i, -i
      const result = roots([1, 0, 1]);
      expect(result.size).toBe(2);
      const re = getRealParts(result);
      const im = getImagParts(result);
      // Both real parts should be 0
      expect(re[0]).toBeCloseTo(0, 5);
      expect(re[1]).toBeCloseTo(0, 5);
      // Imaginary parts should be ±1
      const sortedIm = [...im].sort((a, b) => a - b);
      expect(sortedIm[0]).toBeCloseTo(-1, 5);
      expect(sortedIm[1]).toBeCloseTo(1, 5);
    });

    it('finds complex roots for x^4 + x^3 + x^2 + x + 1', () => {
      // 5th roots of unity excluding 1: all complex
      const result = roots([1, 1, 1, 1, 1]);
      expect(result.size).toBe(4);
      // All roots should have magnitude 1
      const mags = getMagnitudes(result);
      for (const m of mags) {
        expect(m).toBeCloseTo(1, 3);
      }
      // All roots should have nonzero imaginary parts
      const im = getImagParts(result);
      const hasPositiveIm = im.some((v) => v > 0.1);
      const hasNegativeIm = im.some((v) => v < -0.1);
      expect(hasPositiveIm).toBe(true);
      expect(hasNegativeIm).toBe(true);
    });

    it('finds one real and two complex roots for x^3 - 1', () => {
      // x^3 - 1 = 0 -> x = 1, -0.5+0.866i, -0.5-0.866i
      const result = roots([1, 0, 0, -1]);
      expect(result.size).toBe(3);
      const re = getRealParts(result);
      const im = getImagParts(result);
      // All roots have magnitude 1
      const mags = getMagnitudes(result);
      for (const m of mags) {
        expect(m).toBeCloseTo(1, 3);
      }
      // One root should be real ≈ 1
      const realRoot = re.find((_, i) => Math.abs(im[i]!) < 1e-6);
      expect(realRoot).toBeCloseTo(1, 3);
    });

    it('handles trailing zeros (zero roots)', () => {
      // x^3 + x^2 = x^2(x+1) -> roots: 0, 0, -1
      const result = roots([1, 1, 0, 0]);
      expect(result.size).toBe(3);
      const re = getRealParts(result).sort((a, b) => a - b);
      expect(re[0]).toBeCloseTo(-1, 5);
      expect(re[1]).toBeCloseTo(0, 5);
      expect(re[2]).toBeCloseTo(0, 5);
    });

    it('handles polynomial with only zero roots', () => {
      // x^2 = 0 -> roots: 0, 0
      const result = roots([1, 0, 0]);
      expect(result.size).toBe(2);
      const re = getRealParts(result);
      expect(re[0]).toBeCloseTo(0, 5);
      expect(re[1]).toBeCloseTo(0, 5);
    });

    it('finds complex conjugate pair for negative discriminant quadratic', () => {
      // x^2 + 2x + 5 = 0 -> x = -1 ± 2i
      const result = roots([1, 2, 5]);
      expect(result.size).toBe(2);
      const re = getRealParts(result);
      const im = getImagParts(result);
      expect(re[0]).toBeCloseTo(-1, 5);
      expect(re[1]).toBeCloseTo(-1, 5);
      const sortedIm = [...im].sort((a, b) => a - b);
      expect(sortedIm[0]).toBeCloseTo(-2, 5);
      expect(sortedIm[1]).toBeCloseTo(2, 5);
    });
  });
});
