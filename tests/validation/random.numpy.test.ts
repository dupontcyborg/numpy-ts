/**
 * Python NumPy validation tests for random operations
 *
 * Note: Random functions produce different sequences in JS vs Python,
 * so we validate shapes, dtypes, ranges, and statistical properties
 * rather than exact values.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { random } from '../../src/index';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Random Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        'Python NumPy not available!\n\n' +
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

  describe('random.random', () => {
    it('matches NumPy output shape for 1D', () => {
      const jsResult = random.random(10) as any;
      const pyResult = runNumPy(`
result = np.random.random(10)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('matches NumPy output shape for 2D', () => {
      const jsResult = random.random([3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.random((3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('produces values in [0, 1) like NumPy', () => {
      random.seed(42);
      const jsResult = random.random(1000) as any;
      const data = jsResult.toArray() as number[];

      // Verify range [0, 1)
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }

      // NumPy also produces values in [0, 1)
      const pyResult = runNumPy(`
result = np.array([np.random.random(1000).min() >= 0, np.random.random(1000).max() < 1])
      `);
      expect(pyResult.value).toEqual([true, true]);
    });
  });

  describe('random.rand', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.rand(2, 3, 4) as any;
      const pyResult = runNumPy(`
result = np.random.rand(2, 3, 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });
  });

  describe('random.randn', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.randn(2, 3) as any;
      const pyResult = runNumPy(`
result = np.random.randn(2, 3)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('produces standard normal distribution like NumPy', () => {
      random.seed(42);
      const jsResult = random.randn(10000) as any;
      const data = jsResult.toArray() as number[];

      // Calculate JS mean and std
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const jsMean = sum / data.length;

      let variance = 0;
      for (const val of data) {
        variance += (val - jsMean) ** 2;
      }
      const jsStd = Math.sqrt(variance / data.length);

      // Both should be close to mean=0, std=1
      expect(Math.abs(jsMean)).toBeLessThan(0.1);
      expect(Math.abs(jsStd - 1)).toBeLessThan(0.1);

      // Verify NumPy also produces standard normal
      const pyResult = runNumPy(`
np.random.seed(42)
samples = np.random.randn(10000)
result = np.array([samples.mean(), samples.std()])
      `);
      expect(Math.abs(pyResult.value[0])).toBeLessThan(0.1);
      expect(Math.abs(pyResult.value[1] - 1)).toBeLessThan(0.1);
    });
  });

  describe('random.randint', () => {
    it('matches NumPy output shape and dtype', () => {
      const jsResult = random.randint(0, 100, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.randint(0, 100, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      // Note: NumPy default is int64, we also use int64
    });

    it('produces integers in correct range like NumPy', () => {
      random.seed(42);
      const jsResult = random.randint(10, 20, 1000) as any;
      const data = jsResult.toArray() as number[];

      // All values should be in [10, 20)
      for (const val of data) {
        expect(Number(val)).toBeGreaterThanOrEqual(10);
        expect(Number(val)).toBeLessThan(20);
      }

      // Verify NumPy behavior
      const pyResult = runNumPy(`
np.random.seed(42)
samples = np.random.randint(10, 20, 1000)
result = np.array([samples.min() >= 10, samples.max() < 20])
      `);
      expect(pyResult.value).toEqual([true, true]);
    });
  });

  describe('random.uniform', () => {
    it('matches NumPy output shape and dtype', () => {
      const jsResult = random.uniform(0, 1, [2, 3]) as any;
      const pyResult = runNumPy(`
result = np.random.uniform(0, 1, (2, 3))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('produces values in correct range like NumPy', () => {
      random.seed(42);
      const jsResult = random.uniform(-5, 5, 1000) as any;
      const data = jsResult.toArray() as number[];

      // All values should be in [-5, 5)
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(-5);
        expect(val).toBeLessThan(5);
      }
    });
  });

  describe('random.normal', () => {
    it('matches NumPy output shape and dtype', () => {
      const jsResult = random.normal(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.normal(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('respects loc and scale parameters like NumPy', () => {
      random.seed(42);
      const loc = 10;
      const scale = 2;
      const jsResult = random.normal(loc, scale, 10000) as any;
      const data = jsResult.toArray() as number[];

      // Calculate mean
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const jsMean = sum / data.length;

      // Calculate std
      let variance = 0;
      for (const val of data) {
        variance += (val - jsMean) ** 2;
      }
      const jsStd = Math.sqrt(variance / data.length);

      // Mean should be close to loc, std close to scale
      expect(Math.abs(jsMean - loc)).toBeLessThan(0.2);
      expect(Math.abs(jsStd - scale)).toBeLessThan(0.2);

      // Verify NumPy behavior
      const pyResult = runNumPy(`
np.random.seed(42)
samples = np.random.normal(${loc}, ${scale}, 10000)
result = np.array([samples.mean(), samples.std()])
      `);
      expect(Math.abs(pyResult.value[0] - loc)).toBeLessThan(0.2);
      expect(Math.abs(pyResult.value[1] - scale)).toBeLessThan(0.2);
    });
  });

  describe('random.standard_normal', () => {
    it('matches NumPy output shape and dtype', () => {
      const jsResult = random.standard_normal([2, 3]) as any;
      const pyResult = runNumPy(`
result = np.random.standard_normal((2, 3))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });
  });

  describe('random.exponential', () => {
    it('matches NumPy output shape and dtype', () => {
      const jsResult = random.exponential(1, [2, 3]) as any;
      const pyResult = runNumPy(`
result = np.random.exponential(1, (2, 3))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });

    it('produces correct mean like NumPy', () => {
      random.seed(42);
      const scale = 2;
      const jsResult = random.exponential(scale, 10000) as any;
      const data = jsResult.toArray() as number[];

      // Calculate mean
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const jsMean = sum / data.length;

      // Mean of exponential equals scale
      expect(Math.abs(jsMean - scale)).toBeLessThan(0.2);
    });
  });

  describe('random.poisson', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.poisson(5, [2, 3]) as any;
      const pyResult = runNumPy(`
result = np.random.poisson(5, (2, 3))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });

    it('produces correct mean like NumPy', () => {
      random.seed(42);
      const lam = 5;
      const jsResult = random.poisson(lam, 10000) as any;
      const data = jsResult.toArray() as number[];

      // Calculate mean
      let sum = 0;
      for (const val of data) {
        sum += Number(val);
      }
      const jsMean = sum / data.length;

      // Mean of Poisson equals lambda
      expect(Math.abs(jsMean - lam)).toBeLessThan(0.2);
    });
  });

  describe('random.binomial', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.binomial(10, 0.5, [2, 3]) as any;
      const pyResult = runNumPy(`
result = np.random.binomial(10, 0.5, (2, 3))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });

    it('produces correct mean like NumPy', () => {
      random.seed(42);
      const n = 20;
      const p = 0.3;
      const jsResult = random.binomial(n, p, 10000) as any;
      const data = jsResult.toArray() as number[];

      // Calculate mean
      let sum = 0;
      for (const val of data) {
        sum += Number(val);
      }
      const jsMean = sum / data.length;

      // Mean of binomial is n*p
      expect(Math.abs(jsMean - n * p)).toBeLessThan(0.3);
    });
  });

  describe('random.choice', () => {
    it('produces values from correct range', () => {
      random.seed(42);
      const jsResult = random.choice(5, 100) as any;
      const data = jsResult.toArray() as number[];

      // All values should be in [0, 5)
      for (const val of data) {
        expect([0, 1, 2, 3, 4]).toContain(val);
      }
    });

    it('without replacement produces unique values', () => {
      random.seed(42);
      const jsResult = random.choice(10, 10, false) as any;
      const data = jsResult.toArray() as number[];

      // Should have all unique values
      const values = new Set<number>(data);
      expect(values.size).toBe(10);
    });
  });

  describe('random.permutation', () => {
    it('produces permutation containing all values', () => {
      random.seed(42);
      const jsResult = random.permutation(10) as any;
      const data = jsResult.toArray() as number[];

      // Should contain all values 0-9
      const values = new Set<number>(data);
      expect(values.size).toBe(10);
      expect([...values].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });
  });

  describe('random.default_rng', () => {
    it('Generator produces correct shapes', () => {
      const rng = random.default_rng(42);

      const r = rng.random([2, 3]) as any;
      expect(r.shape).toEqual([2, 3]);

      const n = rng.normal(0, 1, [3, 4]) as any;
      expect(n.shape).toEqual([3, 4]);

      const u = rng.uniform(0, 1, [4, 5]) as any;
      expect(u.shape).toEqual([4, 5]);
    });

    it('Generator with same seed produces same sequence', () => {
      const rng1 = random.default_rng(12345);
      const rng2 = random.default_rng(12345);

      const r1 = rng1.random(5) as any;
      const r2 = rng2.random(5) as any;

      expect(Array.from(r1.data)).toEqual(Array.from(r2.data));
    });
  });

  describe('random.seed', () => {
    it('produces reproducible results', () => {
      random.seed(99);
      const r1 = random.random(5) as any;
      const arr1 = r1.toArray();

      random.seed(99);
      const r2 = random.random(5) as any;
      const arr2 = r2.toArray();

      expect(arr1).toEqual(arr2);
    });
  });

  describe('random.get_state and set_state', () => {
    it('allows state restoration', () => {
      random.seed(42);
      random.random(10); // Advance state

      const state = random.get_state();

      const r1 = random.random(5) as any;
      const arr1 = r1.toArray();

      random.set_state(state);

      const r2 = random.random(5) as any;
      const arr2 = r2.toArray();

      expect(arr1).toEqual(arr2);
    });
  });
});
