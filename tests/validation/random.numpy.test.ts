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

  // ============================================================
  // EXACT MATCH TESTS - NumPy compatibility validation
  // These tests verify that seeded random functions produce
  // identical output to NumPy's implementations
  // ============================================================

  describe('EXACT MATCH: MT19937 Legacy Functions', () => {
    it('random.random() matches NumPy exactly with seed', () => {
      random.seed(42);
      const jsResult = random.random(5) as any;
      const jsValues = jsResult.toArray() as number[];

      const pyResult = runNumPy(`
np.random.seed(42)
result = np.random.random(5)
      `);

      // Exact match within floating point precision
      for (let i = 0; i < 5; i++) {
        expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 14);
      }
    });

    it('random.rand() matches NumPy exactly with seed', () => {
      random.seed(123);
      const jsResult = random.rand(3, 2) as any;
      const jsFlat = jsResult.flatten().toArray() as number[];

      const pyResult = runNumPy(`
np.random.seed(123)
result = np.random.rand(3, 2).flatten()
      `);

      for (let i = 0; i < jsFlat.length; i++) {
        expect(jsFlat[i]).toBeCloseTo(pyResult.value[i], 14);
      }
    });

    it('random.uniform() matches NumPy exactly with seed', () => {
      random.seed(999);
      const jsResult = random.uniform(-10, 10, 5) as any;
      const jsValues = jsResult.toArray() as number[];

      const pyResult = runNumPy(`
np.random.seed(999)
result = np.random.uniform(-10, 10, 5)
      `);

      for (let i = 0; i < 5; i++) {
        expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 13);
      }
    });
  });

  describe('EXACT MATCH: PCG64 Generator (NumPy 2.0+)', () => {
    it('default_rng().random() matches NumPy exactly', () => {
      const rng = random.default_rng(42);
      const jsResult = rng.random(5) as any;
      const jsValues = Array.from(jsResult.data as Float64Array);

      const pyResult = runNumPy(`
rng = np.random.default_rng(42)
result = rng.random(5)
      `);

      // Exact match within floating point precision
      for (let i = 0; i < 5; i++) {
        expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 14);
      }
    });

    it('default_rng().random() with different seeds matches NumPy', () => {
      for (const seed of [0, 1, 12345, 99999]) {
        const rng = random.default_rng(seed);
        const jsResult = rng.random(3) as any;
        const jsValues = Array.from(jsResult.data as Float64Array);

        const pyResult = runNumPy(`
rng = np.random.default_rng(${seed})
result = rng.random(3)
        `);

        for (let i = 0; i < 3; i++) {
          expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 14);
        }
      }
    });

    it('default_rng().integers() produces values in correct range', () => {
      // Note: integers() uses bounded random which may differ from NumPy's
      // exact algorithm, so we test range and statistical properties
      const rng = random.default_rng(42);
      const jsResult = rng.integers(0, 100, 1000) as any;
      const jsValues = Array.from(jsResult.data as BigInt64Array).map(Number);

      // All values should be in [0, 100)
      for (const val of jsValues) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(100);
      }

      // Mean should be close to 49.5
      const mean = jsValues.reduce((a, b) => a + b, 0) / jsValues.length;
      expect(Math.abs(mean - 49.5)).toBeLessThan(5);
    });

    it('default_rng().uniform() matches NumPy exactly', () => {
      const rng = random.default_rng(42);
      const jsResult = rng.uniform(-5, 5, 5) as any;
      const jsValues = Array.from(jsResult.data as Float64Array);

      const pyResult = runNumPy(`
rng = np.random.default_rng(42)
result = rng.uniform(-5, 5, 5)
      `);

      for (let i = 0; i < 5; i++) {
        expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 13);
      }
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

  // ============================================================
  // NumPy Validation tests for new distributions
  // These tests validate statistical properties match NumPy
  // ============================================================

  describe('random.gamma', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.gamma(2, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.gamma(2, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);

      // Verify statistical properties
      random.seed(42);
      const jsSamples = random.gamma(3, 2, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      // Mean of gamma(shape, scale) = shape * scale = 6
      expect(Math.abs(mean - 6)).toBeLessThan(0.3);
    });
  });

  describe('random.beta', () => {
    it('matches NumPy output shape and produces values in [0, 1]', () => {
      const jsResult = random.beta(2, 5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.beta(2, 5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);

      // Verify range
      random.seed(42);
      const jsSamples = random.beta(2, 5, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }

      // Mean of beta(a, b) = a / (a + b)
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 2 / 7)).toBeLessThan(0.02);
    });
  });

  describe('random.chisquare', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.chisquare(5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.chisquare(5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of chi-square = df
      random.seed(42);
      const jsSamples = random.chisquare(5, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 5)).toBeLessThan(0.2);
    });
  });

  describe('random.f', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.f(5, 10, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.f(5, 10, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of F(d1, d2) when d2 > 2 is d2 / (d2 - 2)
      random.seed(42);
      const jsSamples = random.f(5, 10, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 10 / 8)).toBeLessThan(0.2);
    });
  });

  describe('random.laplace', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.laplace(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.laplace(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);

      // Mean of laplace(loc, scale) = loc
      random.seed(42);
      const jsSamples = random.laplace(5, 2, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 5)).toBeLessThan(0.2);
    });
  });

  describe('random.logistic', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.logistic(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.logistic(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);

      // Mean of logistic(loc, scale) = loc
      random.seed(42);
      const jsSamples = random.logistic(5, 2, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 5)).toBeLessThan(0.2);
    });
  });

  describe('random.lognormal', () => {
    it('matches NumPy output shape and produces positive values', () => {
      const jsResult = random.lognormal(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.lognormal(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);

      // All values should be positive
      random.seed(42);
      const jsSamples = random.lognormal(0, 1, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThan(0);
      }
    });
  });

  describe('random.gumbel', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.gumbel(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.gumbel(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(jsResult.dtype).toBe(pyResult.dtype);
    });
  });

  describe('random.pareto', () => {
    it('matches NumPy output shape and produces values >= 0', () => {
      const jsResult = random.pareto(2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.pareto(2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.pareto(2, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('random.power', () => {
    it('matches NumPy output shape and produces values in [0, 1]', () => {
      const jsResult = random.power(2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.power(2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.power(2, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('random.rayleigh', () => {
    it('matches NumPy output shape and produces positive values', () => {
      const jsResult = random.rayleigh(1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.rayleigh(1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.rayleigh(1, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThan(0);
      }
    });
  });

  describe('random.triangular', () => {
    it('matches NumPy output shape and produces values in range', () => {
      const jsResult = random.triangular(0, 0.5, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.triangular(0, 0.5, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.triangular(0, 0.5, 1, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }

      // Mean = (left + mode + right) / 3
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 0.5)).toBeLessThan(0.02);
    });
  });

  describe('random.wald', () => {
    it('matches NumPy output shape and produces positive values', () => {
      const jsResult = random.wald(1, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.wald(1, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.wald(1, 1, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThan(0);
      }
    });
  });

  describe('random.weibull', () => {
    it('matches NumPy output shape and produces non-negative values', () => {
      const jsResult = random.weibull(2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.weibull(2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.weibull(2, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('random.geometric', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.geometric(0.5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.geometric(0.5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of geometric(p) = 1/p
      random.seed(42);
      const jsSamples = random.geometric(0.3, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      expect(Math.abs(mean - 1 / 0.3)).toBeLessThan(0.3);
    });
  });

  describe('random.hypergeometric', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.hypergeometric(10, 10, 5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.hypergeometric(10, 10, 5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of hypergeometric(ngood, nbad, nsample) = nsample * ngood / (ngood + nbad)
      random.seed(42);
      const jsSamples = random.hypergeometric(20, 30, 10, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      expect(Math.abs(mean - (10 * 20) / 50)).toBeLessThan(0.2);
    });
  });

  describe('random.logseries', () => {
    it('matches NumPy output shape and produces positive integers', () => {
      const jsResult = random.logseries(0.5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.logseries(0.5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.logseries(0.5, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(Number(val)).toBeGreaterThanOrEqual(1);
      }
    });
  });

  describe('random.negative_binomial', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const jsResult = random.negative_binomial(5, 0.5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.negative_binomial(5, 0.5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of negative_binomial(n, p) = n * (1-p) / p
      random.seed(42);
      const jsSamples = random.negative_binomial(5, 0.3, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      expect(Math.abs(mean - (5 * 0.7) / 0.3)).toBeLessThan(1);
    });
  });

  describe('random.zipf', () => {
    it('matches NumPy output shape and produces positive integers', () => {
      const jsResult = random.zipf(2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.zipf(2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.zipf(2, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(Number(val)).toBeGreaterThanOrEqual(1);
      }
    });
  });

  describe('random.multinomial', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.multinomial(10, [0.2, 0.3, 0.5], 5) as any;
      const pyResult = runNumPy(`
result = np.random.multinomial(10, [0.2, 0.3, 0.5], 5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Each row should sum to n
      random.seed(42);
      const jsSamples = random.multinomial(10, [0.2, 0.3, 0.5]) as any;
      const data = jsSamples.toArray() as number[];
      const sum = data.reduce((a, b) => a + Number(b), 0);
      expect(sum).toBe(10);
    });
  });

  describe('random.multivariate_normal', () => {
    it('matches NumPy output shape and produces correct mean', () => {
      const mean = [0, 0];
      const cov = [
        [1, 0],
        [0, 1],
      ];
      const jsResult = random.multivariate_normal(mean, cov, 5) as any;
      const pyResult = runNumPy(`
result = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Check means
      random.seed(42);
      const jsSamples = random.multivariate_normal(
        [5, 10],
        [
          [1, 0],
          [0, 1],
        ],
        1000
      ) as any;
      const data = jsSamples.data as Float64Array;
      let sum0 = 0,
        sum1 = 0;
      for (let i = 0; i < 1000; i++) {
        sum0 += data[i * 2]!;
        sum1 += data[i * 2 + 1]!;
      }
      expect(Math.abs(sum0 / 1000 - 5)).toBeLessThan(0.2);
      expect(Math.abs(sum1 / 1000 - 10)).toBeLessThan(0.2);
    });
  });

  describe('random.dirichlet', () => {
    it('matches NumPy output shape and sums to 1', () => {
      const jsResult = random.dirichlet([1, 2, 3], 5) as any;
      const pyResult = runNumPy(`
result = np.random.dirichlet([1, 2, 3], 5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Each sample should sum to 1
      random.seed(42);
      const jsSamples = random.dirichlet([1, 2, 3]) as any;
      const data = jsSamples.toArray() as number[];
      const sum = data.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-10);
    });
  });

  describe('random.vonmises', () => {
    it('matches NumPy output shape and produces values in [-pi, pi]', () => {
      const jsResult = random.vonmises(0, 1, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.vonmises(0, 1, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      random.seed(42);
      const jsSamples = random.vonmises(0, 1, 1000) as any;
      const data = jsSamples.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(-Math.PI);
        expect(val).toBeLessThanOrEqual(Math.PI);
      }
    });
  });

  describe('random.standard_cauchy', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.standard_cauchy([3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.standard_cauchy((3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });
  });

  describe('random.standard_t', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.standard_t(5, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.standard_t(5, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });
  });

  describe('random.standard_exponential', () => {
    it('matches NumPy output shape and mean', () => {
      const jsResult = random.standard_exponential([3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.standard_exponential((3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean of standard exponential = 1
      random.seed(42);
      const jsSamples = random.standard_exponential(10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 1)).toBeLessThan(0.1);
    });
  });

  describe('random.standard_gamma', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.standard_gamma(2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.standard_gamma(2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });
  });

  describe('random.noncentral_chisquare', () => {
    it('matches NumPy output shape and mean', () => {
      const jsResult = random.noncentral_chisquare(5, 2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.noncentral_chisquare(5, 2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);

      // Mean = df + nonc
      random.seed(42);
      const jsSamples = random.noncentral_chisquare(5, 2, 10000) as any;
      const data = jsSamples.toArray() as number[];
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 7)).toBeLessThan(0.3);
    });
  });

  describe('random.noncentral_f', () => {
    it('matches NumPy output shape', () => {
      const jsResult = random.noncentral_f(5, 10, 2, [3, 4]) as any;
      const pyResult = runNumPy(`
result = np.random.noncentral_f(5, 10, 2, (3, 4))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
    });
  });
});
