/**
 * Python NumPy validation tests for Formatting functions
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { base_repr, binary_repr } from '../../src';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Formatting Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n'
      );
    }
  });

  describe('base_repr', () => {
    it('matches NumPy for decimal to binary', () => {
      const jsResult = base_repr(10, 2);
      const pyResult = runNumPy(`result = np.base_repr(10, 2)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for decimal to octal', () => {
      const jsResult = base_repr(64, 8);
      const pyResult = runNumPy(`result = np.base_repr(64, 8)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for decimal to hex', () => {
      const jsResult = base_repr(255, 16);
      const pyResult = runNumPy(`result = np.base_repr(255, 16)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for decimal to base 3', () => {
      const jsResult = base_repr(9, 3);
      const pyResult = runNumPy(`result = np.base_repr(9, 3)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for zero', () => {
      const jsResult = base_repr(0, 2);
      const pyResult = runNumPy(`result = np.base_repr(0, 2)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for positive with padding', () => {
      const jsResult = base_repr(5, 2, 8);
      const pyResult = runNumPy(`result = np.base_repr(5, 2, 8)`);

      expect(jsResult).toBe(pyResult.value);
    });
  });

  describe('binary_repr', () => {
    it('matches NumPy for positive number', () => {
      const jsResult = binary_repr(5);
      const pyResult = runNumPy(`result = np.binary_repr(5)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy for zero', () => {
      const jsResult = binary_repr(0);
      const pyResult = runNumPy(`result = np.binary_repr(0)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it('matches NumPy with specified width', () => {
      const jsResult = binary_repr(5, 8);
      const pyResult = runNumPy(`result = np.binary_repr(5, 8)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it("matches NumPy for negative with two's complement", () => {
      const jsResult = binary_repr(-1, 8);
      const pyResult = runNumPy(`result = np.binary_repr(-1, 8)`);

      expect(jsResult).toBe(pyResult.value);
    });

    it("matches NumPy for -5 in 8-bit two's complement", () => {
      const jsResult = binary_repr(-5, 8);
      const pyResult = runNumPy(`result = np.binary_repr(-5, 8)`);

      expect(jsResult).toBe(pyResult.value);
    });
  });
});
