import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  float_power,
  fmod,
  frexp,
  gcd,
  lcm,
  ldexp,
  modf,
} from '../../src/core/ndarray';

describe('Arithmetic Operations', () => {
  describe('add', () => {
    it('adds scalar to array', () => {
      const arr = array([1, 2, 3]);
      const result = arr.add(10);
      expect(result.toArray()).toEqual([11, 12, 13]);
    });

    it('adds two arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = a.add(b);
      expect(result.toArray()).toEqual([5, 7, 9]);
    });

    it('adds 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = a.add(b);
      expect(result.toArray()).toEqual([
        [6, 8],
        [10, 12],
      ]);
    });

    it('adds mixed int32 and int64 arrays', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([4n, 5n, 6n], 'int64');
      const result = a.add(b);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([5n, 7n, 9n]);
    });

    it('adds int64 with float64 (converts to float)', () => {
      const a = array([1.5, 2.5, 3.5], 'float64');
      const b = array([1n, 1n, 1n], 'int64');
      const result = a.add(b);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2.5, 3.5, 4.5]);
    });
  });

  describe('subtract', () => {
    it('subtracts scalar from array', () => {
      const arr = array([10, 20, 30]);
      const result = arr.subtract(5);
      expect(result.toArray()).toEqual([5, 15, 25]);
    });

    it('subtracts two arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([1, 2, 3]);
      const result = a.subtract(b);
      expect(result.toArray()).toEqual([9, 18, 27]);
    });

    it('subtracts mixed int32 and int64 arrays', () => {
      const a = array([10, 20, 30], 'int32');
      const b = array([1n, 2n, 3n], 'int64');
      const result = a.subtract(b);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([9n, 18n, 27n]);
    });

    it('subtracts int64 from float64 (converts to float)', () => {
      const a = array([10.5, 20.5, 30.5], 'float64');
      const b = array([1n, 2n, 3n], 'int64');
      const result = a.subtract(b);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([9.5, 18.5, 27.5]);
    });
  });

  describe('multiply', () => {
    it('multiplies array by scalar', () => {
      const arr = array([1, 2, 3]);
      const result = arr.multiply(3);
      expect(result.toArray()).toEqual([3, 6, 9]);
    });

    it('multiplies two arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = a.multiply(b);
      expect(result.toArray()).toEqual([4, 10, 18]);
    });

    it('multiplies mixed int32 and int64 arrays', () => {
      const a = array([2, 3, 4], 'int32');
      const b = array([3n, 4n, 5n], 'int64');
      const result = a.multiply(b);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([6n, 12n, 20n]);
    });

    it('multiplies int64 with float64 (converts to float)', () => {
      const a = array([2.5, 3.5, 4.5], 'float64');
      const b = array([2n, 2n, 2n], 'int64');
      const result = a.multiply(b);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([5, 7, 9]);
    });
  });

  describe('divide', () => {
    it('divides array by scalar', () => {
      const arr = array([10, 20, 30]);
      const result = arr.divide(10);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('divides two arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([2, 4, 5]);
      const result = a.divide(b);
      expect(result.toArray()).toEqual([5, 5, 6]);
    });

    it('divides int64 by int32 (returns float64)', () => {
      const a = array([10n, 20n, 30n], 'int64');
      const b = array([2, 4, 5], 'int32');
      const result = a.divide(b);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([5, 5, 6]);
    });
  });

  describe('chained operations', () => {
    it('chains multiple operations', () => {
      const arr = array([1, 2, 3]);
      const result = arr.add(10).multiply(2).subtract(5);
      expect(result.toArray()).toEqual([17, 19, 21]);
    });
  });
});

describe('cbrt', () => {
  it('computes cube root of positive numbers', () => {
    const arr = array([1, 8, 27, 64]);
    const result = arr.cbrt();
    expect(result.toArray()).toEqual([1, 2, 3, 4]);
  });

  it('computes cube root of negative numbers', () => {
    const arr = array([-1, -8, -27]);
    const result = arr.cbrt();
    expect(result.toArray()[0]).toBeCloseTo(-1, 10);
    expect(result.toArray()[1]).toBeCloseTo(-2, 10);
    expect(result.toArray()[2]).toBeCloseTo(-3, 10);
  });

  it('computes cube root of 2D arrays', () => {
    const arr = array([
      [8, 27],
      [64, 125],
    ]);
    const result = arr.cbrt();
    expect((result.toArray() as number[][])[0]![0]).toBeCloseTo(2, 10);
    expect((result.toArray() as number[][])[0]![1]).toBeCloseTo(3, 10);
    expect((result.toArray() as number[][])[1]![0]).toBeCloseTo(4, 10);
    expect((result.toArray() as number[][])[1]![1]).toBeCloseTo(5, 10);
  });

  it('promotes integers to float64', () => {
    const arr = array([8], 'int32');
    const result = arr.cbrt();
    expect(result.dtype).toBe('float64');
  });
});

describe('fabs', () => {
  it('computes absolute value of positive numbers', () => {
    const arr = array([1, 2, 3]);
    const result = arr.fabs();
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('computes absolute value of negative numbers', () => {
    const arr = array([-1, -2, -3]);
    const result = arr.fabs();
    expect(result.toArray()).toEqual([1, 2, 3]);
  });

  it('computes absolute value of mixed numbers', () => {
    const arr = array([-1.5, 0, 2.5, -3.5]);
    const result = arr.fabs();
    expect(result.toArray()).toEqual([1.5, 0, 2.5, 3.5]);
  });

  it('returns float dtype', () => {
    const arr = array([-1, 2, -3], 'int32');
    const result = arr.fabs();
    expect(result.dtype).toBe('float64');
  });

  it('preserves float32', () => {
    const arr = array([-1, 2, -3], 'float32');
    const result = arr.fabs();
    expect(result.dtype).toBe('float32');
  });
});

describe('divmod', () => {
  it('computes quotient and remainder with scalar', () => {
    const arr = array([10, 20, 30]);
    const [quotient, remainder] = arr.divmod(7);
    expect(quotient.toArray()).toEqual([1, 2, 4]);
    expect(remainder.toArray()).toEqual([3, 6, 2]);
  });

  it('computes quotient and remainder with array', () => {
    const a = array([10, 20, 30]);
    const b = array([3, 4, 7]);
    const [quotient, remainder] = a.divmod(b);
    expect(quotient.toArray()).toEqual([3, 5, 4]);
    expect(remainder.toArray()).toEqual([1, 0, 2]);
  });

  it('handles negative dividend', () => {
    const arr = array([-10, -20, -30]);
    const [quotient, remainder] = arr.divmod(7);
    // Floor division: -10 // 7 = -2, -10 % 7 = 4 (NumPy behavior)
    expect(quotient.toArray()).toEqual([-2, -3, -5]);
    expect(remainder.toArray()).toEqual([4, 1, 5]);
  });

  it('works with 2D arrays', () => {
    const arr = array([
      [10, 20],
      [15, 25],
    ]);
    const [quotient, remainder] = arr.divmod(7);
    expect(quotient.toArray()).toEqual([
      [1, 2],
      [2, 3],
    ]);
    expect(remainder.toArray()).toEqual([
      [3, 6],
      [1, 4],
    ]);
  });
});

describe('square', () => {
  it('computes square of positive numbers', () => {
    const arr = array([1, 2, 3, 4]);
    const result = arr.square();
    expect(result.toArray()).toEqual([1, 4, 9, 16]);
  });

  it('computes square of negative numbers', () => {
    const arr = array([-1, -2, -3]);
    const result = arr.square();
    expect(result.toArray()).toEqual([1, 4, 9]);
  });

  it('computes square of floats', () => {
    const arr = array([1.5, 2.5]);
    const result = arr.square();
    expect(result.toArray()[0]).toBeCloseTo(2.25, 10);
    expect(result.toArray()[1]).toBeCloseTo(6.25, 10);
  });

  it('preserves dtype', () => {
    const arr = array([2, 3], 'int32');
    const result = arr.square();
    expect(result.dtype).toBe('int32');
    expect(result.toArray()).toEqual([4, 9]);
  });

  it('works with 2D arrays', () => {
    const arr = array([
      [1, 2],
      [3, 4],
    ]);
    const result = arr.square();
    expect(result.toArray()).toEqual([
      [1, 4],
      [9, 16],
    ]);
  });
});

describe('remainder', () => {
  it('computes remainder with scalar (same as mod)', () => {
    const arr = array([10, 20, 30]);
    const result = arr.remainder(7);
    expect(result.toArray()).toEqual([3, 6, 2]);
  });

  it('computes remainder with array', () => {
    const a = array([10, 20, 30]);
    const b = array([3, 4, 7]);
    const result = a.remainder(b);
    expect(result.toArray()).toEqual([1, 0, 2]);
  });

  it('handles negative dividend (floor modulo)', () => {
    const arr = array([-10, -20, -30]);
    const result = arr.remainder(7);
    expect(result.toArray()).toEqual([4, 1, 5]);
  });
});

describe('heaviside', () => {
  it('computes heaviside with scalar x2', () => {
    const arr = array([-2, -1, 0, 1, 2]);
    const result = arr.heaviside(0.5);
    expect(result.toArray()).toEqual([0, 0, 0.5, 1, 1]);
  });

  it('handles all negative values', () => {
    const arr = array([-3, -2, -1]);
    const result = arr.heaviside(0.5);
    expect(result.toArray()).toEqual([0, 0, 0]);
  });

  it('handles all positive values', () => {
    const arr = array([1, 2, 3]);
    const result = arr.heaviside(0.5);
    expect(result.toArray()).toEqual([1, 1, 1]);
  });

  it('handles zero values with different x2', () => {
    const arr = array([0, 0, 0]);
    const result = arr.heaviside(0.75);
    expect(result.toArray()).toEqual([0.75, 0.75, 0.75]);
  });

  it('works with array x2 (same shape)', () => {
    const arr = array([-1, 0, 1]);
    const x2 = array([0.1, 0.5, 0.9]);
    const result = arr.heaviside(x2);
    expect(result.toArray()).toEqual([0, 0.5, 1]);
  });

  it('returns float dtype', () => {
    const arr = array([-1, 0, 1], 'int32');
    const result = arr.heaviside(0.5);
    expect(result.dtype).toBe('float64');
  });
});

describe('Reduction Operations', () => {
  describe('sum', () => {
    it('sums all elements in 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.sum()).toBe(15);
    });

    it('sums all elements in 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.sum()).toBe(21);
    });

    it('sums empty array', () => {
      const arr = zeros([0]);
      expect(arr.sum()).toBe(0);
    });
  });

  describe('mean', () => {
    it('computes mean of 1D array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.mean()).toBe(3);
    });

    it('computes mean of 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.mean()).toBe(3.5);
    });
  });

  describe('max', () => {
    it('finds maximum in 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(arr.max()).toBe(9);
    });

    it('finds maximum in 2D array', () => {
      const arr = array([
        [1, 5, 3],
        [8, 2, 7],
      ]);
      expect(arr.max()).toBe(8);
    });

    it('handles negative numbers', () => {
      const arr = array([-5, -1, -10, -3]);
      expect(arr.max()).toBe(-1);
    });
  });

  describe('min', () => {
    it('finds minimum in 1D array', () => {
      const arr = array([3, 1, 4, 1, 5, 9, 2, 6]);
      expect(arr.min()).toBe(1);
    });

    it('finds minimum in 2D array', () => {
      const arr = array([
        [5, 3, 8],
        [2, 7, 1],
      ]);
      expect(arr.min()).toBe(1);
    });

    it('handles negative numbers', () => {
      const arr = array([-5, -1, -10, -3]);
      expect(arr.min()).toBe(-10);
    });
  });

  describe('float_power', () => {
    it('raises array to power with float result', () => {
      const arr = array([2, 3, 4]);
      const result = float_power(arr, 2);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([4, 9, 16]);
    });

    it('works with array exponent', () => {
      const base = array([2, 3, 4]);
      const exp = array([1, 2, 3]);
      const result = float_power(base, exp);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2, 9, 64]);
    });
  });

  describe('fmod', () => {
    it('computes remainder with fmod semantics', () => {
      const arr = array([7, 8, 9]);
      const result = fmod(arr, 3);
      expect(result.toArray()).toEqual([1, 2, 0]);
    });

    it('handles negative values', () => {
      const arr = array([-7, 7]);
      const result = fmod(arr, 3);
      expect(result.toArray()).toEqual([-1, 1]);
    });
  });

  describe('frexp', () => {
    it('decomposes floats into mantissa and exponent', () => {
      const arr = array([8, 16, 32]);
      const [mantissa, exponent] = frexp(arr);
      expect(mantissa.dtype).toBe('float64');
      expect(exponent.dtype).toBe('int32');
      // 8 = 0.5 * 2^4, 16 = 0.5 * 2^5, 32 = 0.5 * 2^6
      expect(mantissa.shape).toEqual([3]);
      expect(exponent.shape).toEqual([3]);
    });

    it('handles zero', () => {
      const arr = array([0]);
      const [mantissa, exponent] = frexp(arr);
      expect(mantissa.toArray()).toEqual([0]);
      expect(exponent.toArray()).toEqual([0]);
    });
  });

  describe('gcd', () => {
    it('computes greatest common divisor', () => {
      const a = array([12, 18, 24]);
      const b = array([8, 12, 16]);
      const result = gcd(a, b);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([4, 6, 8]);
    });

    it('works with scalar', () => {
      const arr = array([12, 18, 24]);
      const result = gcd(arr, 6);
      expect(result.toArray()).toEqual([6, 6, 6]);
    });
  });

  describe('lcm', () => {
    it('computes least common multiple', () => {
      const a = array([12, 18, 24]);
      const b = array([8, 12, 16]);
      const result = lcm(a, b);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([24, 36, 48]);
    });

    it('works with scalar', () => {
      const arr = array([4, 6, 8]);
      const result = lcm(arr, 3);
      expect(result.toArray()).toEqual([12, 6, 24]);
    });

    it('handles zero', () => {
      const arr = array([0, 5, 10]);
      const result = lcm(arr, 5);
      expect(result.toArray()).toEqual([0, 5, 10]);
    });
  });

  describe('ldexp', () => {
    it('computes x * 2^y', () => {
      const arr = array([1, 2, 3]);
      const result = ldexp(arr, 3);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([8, 16, 24]);
    });

    it('works with array exponent', () => {
      const base = array([1, 1, 1]);
      const exp = array([1, 2, 3]);
      const result = ldexp(base, exp);
      expect(result.toArray()).toEqual([2, 4, 8]);
    });
  });

  describe('modf', () => {
    it('splits into fractional and integral parts', () => {
      const arr = array([1.5, 2.7, -3.2]);
      const [fractional, integral] = modf(arr);
      expect(fractional.dtype).toBe('float64');
      expect(integral.dtype).toBe('float64');
      expect(fractional.shape).toEqual([3]);
      expect(integral.shape).toEqual([3]);
      const frac = fractional.toArray() as number[];
      const integ = integral.toArray() as number[];
      expect(integ).toEqual([1, 2, -3]);
      expect(frac[0]).toBeCloseTo(0.5);
      expect(frac[1]).toBeCloseTo(0.7);
      expect(frac[2]).toBeCloseTo(-0.2);
    });
  });
});
