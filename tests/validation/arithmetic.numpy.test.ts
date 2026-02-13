/**
 * Python NumPy validation tests for arithmetic operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { array, float_power, fmod, frexp, gcd, lcm, ldexp, modf } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Arithmetic Operations', () => {
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

  describe('add', () => {
    it('matches NumPy for scalar addition', () => {
      const jsResult = array([1, 2, 3, 4, 5]).add(10);
      const pyResult = runNumPy(`
result = np.array([1, 2, 3, 4, 5]) + 10
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array addition', () => {
      const jsResult = array([1, 2, 3]).add(array([4, 5, 6]));
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) + np.array([4, 5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array addition', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).add(
        array([
          [5, 6],
          [7, 8],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) + np.array([[5, 6], [7, 8]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('subtract', () => {
    it('matches NumPy for scalar subtraction', () => {
      const jsResult = array([10, 20, 30]).subtract(5);
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) - 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array subtraction', () => {
      const jsResult = array([10, 20, 30]).subtract(array([1, 2, 3]));
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) - np.array([1, 2, 3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('multiply', () => {
    it('matches NumPy for scalar multiplication', () => {
      const jsResult = array([1, 2, 3]).multiply(5);
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) * 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for element-wise multiplication', () => {
      const jsResult = array([1, 2, 3]).multiply(array([4, 5, 6]));
      const pyResult = runNumPy(`
result = np.array([1, 2, 3]) * np.array([4, 5, 6])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D element-wise multiplication', () => {
      const jsResult = array([
        [1, 2],
        [3, 4],
      ]).multiply(
        array([
          [2, 3],
          [4, 5],
        ])
      );
      const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) * np.array([[2, 3], [4, 5]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('divide', () => {
    it('matches NumPy for scalar division', () => {
      const jsResult = array([10, 20, 30]).divide(10);
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) / 10
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for element-wise division', () => {
      const jsResult = array([10, 20, 30]).divide(array([2, 4, 5]));
      const pyResult = runNumPy(`
result = np.array([10, 20, 30]) / np.array([2, 4, 5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('chained operations', () => {
    it('matches NumPy for complex expression', () => {
      const jsResult = array([1, 2, 3]).add(10).multiply(2).subtract(5);
      const pyResult = runNumPy(`
result = (np.array([1, 2, 3]) + 10) * 2 - 5
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cbrt', () => {
    it('matches NumPy for positive numbers', () => {
      const jsResult = array([1, 8, 27, 64, 125]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([1, 8, 27, 64, 125])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-1, -8, -27]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([-1, -8, -27])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = array([
        [8, 27],
        [64, 125],
      ]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([[8, 27], [64, 125]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for fractional cubes', () => {
      const jsResult = array([0.001, 0.125, 1.728]).cbrt();
      const pyResult = runNumPy(`
result = np.cbrt([0.001, 0.125, 1.728])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('fabs', () => {
    it('matches NumPy for positive numbers', () => {
      const jsResult = array([1, 2, 3]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([1, 2, 3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative numbers', () => {
      const jsResult = array([-1, -2, -3]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([-1, -2, -3])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for mixed numbers', () => {
      const jsResult = array([-1.5, 0, 2.5, -3.5]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([-1.5, 0, 2.5, -3.5])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = array([
        [-1, 2],
        [-3, 4],
      ]).fabs();
      const pyResult = runNumPy(`
result = np.fabs([[-1, 2], [-3, 4]])
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('divmod', () => {
    it('matches NumPy for positive integers with scalar (quotient)', () => {
      const [jsQuotient] = array([10, 20, 30]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([10, 20, 30], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for positive integers with scalar (remainder)', () => {
      const [, jsRemainder] = array([10, 20, 30]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([10, 20, 30], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array divisor (quotient)', () => {
      const [jsQuotient] = array([10, 20, 30]).divmod(array([3, 4, 7]));
      const pyResult = runNumPy(`
result = np.floor_divide([10, 20, 30], [3, 4, 7])
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array divisor (remainder)', () => {
      const [, jsRemainder] = array([10, 20, 30]).divmod(array([3, 4, 7]));
      const pyResult = runNumPy(`
result = np.mod([10, 20, 30], [3, 4, 7])
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative dividend (quotient)', () => {
      const [jsQuotient] = array([-10, -20, -30]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([-10, -20, -30], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for negative dividend (remainder)', () => {
      const [, jsRemainder] = array([-10, -20, -30]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([-10, -20, -30], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays (quotient)', () => {
      const [jsQuotient] = array([
        [10, 20],
        [15, 25],
      ]).divmod(7);
      const pyResult = runNumPy(`
result = np.floor_divide([[10, 20], [15, 25]], 7)
      `);

      expect(jsQuotient.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsQuotient.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays (remainder)', () => {
      const [, jsRemainder] = array([
        [10, 20],
        [15, 25],
      ]).divmod(7);
      const pyResult = runNumPy(`
result = np.mod([[10, 20], [15, 25]], 7)
      `);

      expect(jsRemainder.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsRemainder.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('float_power()', () => {
    it('validates float_power() with scalar', () => {
      const arr = array([2, 3, 4]);
      const result = float_power(arr, 2);

      const npResult = runNumPy(`
arr = np.array([2, 3, 4])
result = np.float_power(arr, 2)
`);

      expect(result.dtype).toBe('float64');
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates float_power() with array', () => {
      const base = array([2, 3, 4]);
      const exp = array([1, 2, 3]);
      const result = float_power(base, exp);

      const npResult = runNumPy(`
base = np.array([2, 3, 4])
exp = np.array([1, 2, 3])
result = np.float_power(base, exp)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('fmod()', () => {
    it('validates fmod() with scalar', () => {
      const arr = array([7, 8, 9, -7, -8]);
      const result = fmod(arr, 3);

      const npResult = runNumPy(`
arr = np.array([7, 8, 9, -7, -8])
result = np.fmod(arr, 3)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('frexp()', () => {
    it('validates frexp()', () => {
      const arr = array([8, 16, 32]);
      const [mantissa, exponent] = frexp(arr);

      const npResult = runNumPy(`
arr = np.array([8, 16, 32])
mantissa, exponent = np.frexp(arr)
result = {'mantissa': mantissa.tolist(), 'exponent': exponent.tolist()}
`);

      expect(mantissa.dtype).toBe('float64');
      expect(exponent.dtype).toBe('int32');
      expect(arraysClose(mantissa.toArray(), npResult.value.mantissa)).toBe(true);
      expect(arraysClose(exponent.toArray(), npResult.value.exponent)).toBe(true);
    });
  });

  describe('gcd()', () => {
    it('validates gcd() with arrays', () => {
      const a = array([12, 18, 24]);
      const b = array([8, 12, 16]);
      const result = gcd(a, b);

      const npResult = runNumPy(`
a = np.array([12, 18, 24])
b = np.array([8, 12, 16])
result = np.gcd(a, b)
`);

      expect(result.dtype).toBe('int32');
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates gcd() with scalar', () => {
      const arr = array([12, 18, 24]);
      const result = gcd(arr, 6);

      const npResult = runNumPy(`
arr = np.array([12, 18, 24])
result = np.gcd(arr, 6)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('lcm()', () => {
    it('validates lcm() with arrays', () => {
      const a = array([12, 18, 24]);
      const b = array([8, 12, 16]);
      const result = lcm(a, b);

      const npResult = runNumPy(`
a = np.array([12, 18, 24])
b = np.array([8, 12, 16])
result = np.lcm(a, b)
`);

      expect(result.dtype).toBe('int32');
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates lcm() with scalar', () => {
      const arr = array([4, 6, 8]);
      const result = lcm(arr, 3);

      const npResult = runNumPy(`
arr = np.array([4, 6, 8])
result = np.lcm(arr, 3)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('ldexp()', () => {
    it('validates ldexp() with scalar', () => {
      const arr = array([1, 2, 3]);
      const result = ldexp(arr, 3);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3])
result = np.ldexp(arr, 3)
`);

      expect(result.dtype).toBe('float64');
      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });

    it('validates ldexp() with array', () => {
      const base = array([1, 1, 1]);
      const exp = array([1, 2, 3]);
      const result = ldexp(base, exp);

      const npResult = runNumPy(`
base = np.array([1, 1, 1])
exp = np.array([1, 2, 3])
result = np.ldexp(base, exp)
`);

      expect(arraysClose(result.toArray(), npResult.value)).toBe(true);
    });
  });

  describe('modf()', () => {
    it('validates modf()', () => {
      const arr = array([1.5, 2.7, -3.2]);
      const [fractional, integral] = modf(arr);

      const npResult = runNumPy(`
arr = np.array([1.5, 2.7, -3.2])
fractional, integral = np.modf(arr)
result = {'fractional': fractional.tolist(), 'integral': integral.tolist()}
`);

      expect(fractional.dtype).toBe('float64');
      expect(integral.dtype).toBe('float64');
      expect(arraysClose(fractional.toArray(), npResult.value.fractional)).toBe(true);
      expect(arraysClose(integral.toArray(), npResult.value.integral)).toBe(true);
    });
  });
});
