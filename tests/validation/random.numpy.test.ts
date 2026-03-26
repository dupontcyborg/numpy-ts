/**
 * Python NumPy validation tests for random operations
 *
 * Note: Random functions produce different sequences in JS vs Python,
 * so we validate shapes, dtypes, ranges, and statistical properties
 * rather than exact values.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { random, arange } from '../../src/index';
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
      const data = Array.from(jsResult.data as BigInt64Array).map(Number);

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
  // bit-identical output to NumPy's implementations.
  //
  // Core generators (MT19937 uniform, PCG64 uniform) and
  // distribution algorithms (polar/Ziggurat) all run in WASM
  // and match NumPy exactly.
  // ============================================================

  /** Helper: compare JS float64 array against NumPy result, 14-digit precision */
  function expectExactMatch(
    jsValues: number[],
    pyResult: { value: any },
    n: number = jsValues.length
  ) {
    for (let i = 0; i < n; i++) {
      expect(jsValues[i]).toBeCloseTo(pyResult.value[i], 14);
    }
  }

  describe('EXACT MATCH: MT19937 Legacy — Core', () => {
    it('random.random() matches NumPy', () => {
      random.seed(42);
      const js = (random.random(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.random(10)
      `);
      expectExactMatch(js, py);
    });

    it('random.rand() matches NumPy', () => {
      random.seed(123);
      const js = (random.rand(3, 2) as any).flatten().toArray() as number[];
      const py = runNumPy(`
np.random.seed(123)
result = np.random.rand(3, 2).flatten()
      `);
      expectExactMatch(js, py);
    });

    it('random.uniform() matches NumPy', () => {
      random.seed(999);
      const js = (random.uniform(-10, 10, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(999)
result = np.random.uniform(-10, 10, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.randn() matches NumPy', () => {
      random.seed(42);
      const js = (random.randn(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.randn(10)
      `);
      expectExactMatch(js, py);
    });

    it('random.standard_normal() matches NumPy', () => {
      random.seed(42);
      const js = (random.standard_normal(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.standard_normal(10)
      `);
      expectExactMatch(js, py);
    });

    it('random.normal() matches NumPy', () => {
      random.seed(42);
      const js = (random.normal(2, 3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.normal(2, 3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.exponential() matches NumPy', () => {
      random.seed(42);
      const js = (random.exponential(2, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.exponential(2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.standard_exponential() matches NumPy', () => {
      random.seed(42);
      const js = (random.standard_exponential(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.standard_exponential(10)
      `);
      expectExactMatch(js, py);
    });

    it('random.randint() matches NumPy', () => {
      random.seed(42);
      const js = (random.randint(0, 100, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.randint(0, 100, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    for (const dt of ['int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'] as const) {
      it(`random.randint() matches NumPy with dtype=${dt}`, () => {
        random.seed(42);
        const js = (random.randint(0, 50, 10, dt) as any).toArray().map(Number) as number[];
        const py = runNumPy(`
np.random.seed(42)
result = np.random.randint(0, 50, 10, dtype=np.${dt})
        `);
        for (let i = 0; i < 10; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      });
    }

    it('random.bytes() matches NumPy', () => {
      random.seed(42);
      const jsBytes = Array.from(random.bytes(20));
      const py = runNumPy(`
np.random.seed(42)
result = np.array(list(np.random.bytes(20)))
      `);
      for (let i = 0; i < 20; i++) {
        expect(jsBytes[i]).toBe(py.value[i]);
      }
    });

    it('matches NumPy across multiple seeds', () => {
      for (const s of [0, 1, 12345, 99999, 2147483647]) {
        random.seed(s);
        const js = (random.random(5) as any).toArray() as number[];
        const py = runNumPy(`
np.random.seed(${s})
result = np.random.random(5)
        `);
        expectExactMatch(js, py);
      }
    });
  });

  describe('EXACT MATCH: MT19937 Legacy — Distributions', () => {
    it('random.laplace() matches NumPy', () => {
      random.seed(42);
      const js = (random.laplace(2, 3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.laplace(2, 3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.logistic() matches NumPy', () => {
      random.seed(42);
      const js = (random.logistic(2, 3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.logistic(2, 3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.pareto() matches NumPy', () => {
      random.seed(42);
      const js = (random.pareto(3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.pareto(3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.power() matches NumPy', () => {
      random.seed(42);
      const js = (random.power(3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.power(3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.weibull() matches NumPy', () => {
      random.seed(42);
      const js = (random.weibull(3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.weibull(3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.triangular() matches NumPy', () => {
      random.seed(42);
      const js = (random.triangular(0, 0.5, 1, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.triangular(0, 0.5, 1, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.standard_cauchy() matches NumPy', () => {
      random.seed(42);
      const js = (random.standard_cauchy(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.standard_cauchy(10)
      `);
      expectExactMatch(js, py);
    });

    it('random.gumbel() matches NumPy', () => {
      random.seed(42);
      const js = (random.gumbel(2, 3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.gumbel(2, 3, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.rayleigh() matches NumPy', () => {
      random.seed(42);
      const js = (random.rayleigh(2, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.rayleigh(2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.lognormal() matches NumPy', () => {
      random.seed(42);
      const js = (random.lognormal(0, 1, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.lognormal(0, 1, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.geometric() matches NumPy', () => {
      random.seed(42);
      const js = (random.geometric(0.3, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.geometric(0.3, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.geometric() matches NumPy for high p (search algorithm)', () => {
      random.seed(42);
      const js = (random.geometric(0.7, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.geometric(0.7, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.standard_gamma() matches NumPy', () => {
      random.seed(42);
      const js = (random.standard_gamma(2, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.standard_gamma(2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.gamma() matches NumPy', () => {
      random.seed(42);
      const js = (random.gamma(2, 3, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.gamma(2, 3, 10)
      `);
      // 13-digit precision: scale multiply can introduce ~1 ULP difference
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBeCloseTo(py.value[i], 13);
      }
    });

    it('random.chisquare() matches NumPy', () => {
      random.seed(42);
      const js = (random.chisquare(5, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.chisquare(5, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.beta() matches NumPy', () => {
      random.seed(42);
      const js = (random.beta(2, 5, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.beta(2, 5, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.beta() matches NumPy (Johnk, a<=1 b<=1)', () => {
      random.seed(42);
      const js = (random.beta(0.5, 0.5, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.beta(0.5, 0.5, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.standard_t() matches NumPy', () => {
      random.seed(42);
      const js = (random.standard_t(5, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.standard_t(5, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.wald() matches NumPy', () => {
      random.seed(42);
      const js = (random.wald(1, 1, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.wald(1, 1, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.f() matches NumPy', () => {
      random.seed(42);
      const js = (random.f(5, 10, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.f(5, 10, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.noncentral_chisquare() matches NumPy', () => {
      random.seed(42);
      const js = (random.noncentral_chisquare(5, 2, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.noncentral_chisquare(5, 2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.noncentral_f() matches NumPy', () => {
      random.seed(42);
      const js = (random.noncentral_f(5, 10, 2, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.noncentral_f(5, 10, 2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.binomial() matches NumPy', () => {
      random.seed(42);
      const js = (random.binomial(10, 0.5, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.binomial(10, 0.5, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.poisson() matches NumPy', () => {
      random.seed(42);
      const js = (random.poisson(5, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.poisson(5, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.negative_binomial() matches NumPy', () => {
      random.seed(42);
      const js = (random.negative_binomial(5, 0.5, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.negative_binomial(5, 0.5, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.hypergeometric() matches NumPy', () => {
      random.seed(42);
      const js = (random.hypergeometric(20, 30, 10, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.hypergeometric(20, 30, 10, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.logseries() matches NumPy', () => {
      random.seed(42);
      const js = (random.logseries(0.5, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.logseries(0.5, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.zipf() matches NumPy', () => {
      random.seed(42);
      const js = (random.zipf(2, 10) as any).toArray().map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.zipf(2, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.vonmises() matches NumPy', () => {
      random.seed(42);
      const js = (random.vonmises(0, 1, 10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.vonmises(0, 1, 10)
      `);
      expectExactMatch(js, py);
    });

    it('random.multinomial() matches NumPy', () => {
      random.seed(42);
      const js = (random.multinomial(20, [0.2, 0.3, 0.5], 5) as any)
        .flatten()
        .toArray()
        .map(Number) as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.multinomial(20, [0.2, 0.3, 0.5], 5).flatten()
      `);
      for (let i = 0; i < js.length; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('random.dirichlet() matches NumPy', () => {
      random.seed(42);
      const js = (random.dirichlet([1, 2, 3], 5) as any).flatten().toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.dirichlet([1, 2, 3], 5).flatten()
      `);
      expectExactMatch(js, py);
    });

    it('random.multivariate_normal() produces correct statistics', () => {
      // NumPy uses SVD-based decomposition, we use Cholesky.
      // Both are correct but produce different streams. Validate statistically.
      random.seed(42);
      const js = random.multivariate_normal(
        [5, 10],
        [
          [1, 0],
          [0, 1],
        ],
        1000
      ) as any;
      const data = js.data as Float64Array;
      let sum0 = 0;
      let sum1 = 0;
      for (let i = 0; i < 1000; i++) {
        sum0 += data[i * 2]!;
        sum1 += data[i * 2 + 1]!;
      }
      expect(Math.abs(sum0 / 1000 - 5)).toBeLessThan(0.2);
      expect(Math.abs(sum1 / 1000 - 10)).toBeLessThan(0.2);
    });
  });

  describe('EXACT MATCH: PCG64 Generator — Core', () => {
    it('default_rng().random() matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.random(10) as any).data as Float64Array);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.random(10)
      `);
      expectExactMatch(js, py);
    });

    it('default_rng().random() matches across multiple seeds', () => {
      for (const s of [0, 1, 12345, 99999, 2147483647]) {
        const rng = random.default_rng(s);
        const js = Array.from((rng.random(5) as any).data as Float64Array);
        const py = runNumPy(`
rng = np.random.default_rng(${s})
result = rng.random(5)
        `);
        expectExactMatch(js, py);
      }
    });

    it('default_rng().uniform() matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.uniform(-5, 5, 10) as any).data as Float64Array);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.uniform(-5, 5, 10)
      `);
      expectExactMatch(js, py);
    });

    it('default_rng().standard_normal() matches NumPy (Ziggurat)', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.standard_normal(10) as any).data as Float64Array);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.standard_normal(10)
      `);
      expectExactMatch(js, py);
    });

    it('default_rng().normal() matches NumPy (Ziggurat)', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.normal(5, 2, 10) as any).data as Float64Array);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.normal(5, 2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('default_rng().exponential() matches NumPy (Ziggurat)', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.exponential(2, 10) as any).data as Float64Array);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.exponential(2, 10)
      `);
      expectExactMatch(js, py);
    });

    it('default_rng().integers() matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.integers(0, 100, 10) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.integers(0, 100, 10)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });
  });

  describe('EXACT MATCH: Independent Generator instances', () => {
    it('two generators with same seed produce identical output', () => {
      const rng1 = random.default_rng(42);
      const rng2 = random.default_rng(42);
      const js1 = Array.from((rng1.random(10) as any).data as Float64Array);
      const js2 = Array.from((rng2.random(10) as any).data as Float64Array);
      expect(js1).toEqual(js2);
    });

    it('interleaved generators maintain independent state', () => {
      const rng1 = random.default_rng(42);
      const rng2 = random.default_rng(99);

      // Interleave calls
      const a1 = rng1.random() as number;
      const b1 = rng2.random() as number;
      const a2 = rng1.random() as number;
      const b2 = rng2.random() as number;

      // Verify against sequential generation
      const rng1b = random.default_rng(42);
      const rng2b = random.default_rng(99);
      expect(a1).toBe(rng1b.random() as number);
      expect(a2).toBe(rng1b.random() as number);
      expect(b1).toBe(rng2b.random() as number);
      expect(b2).toBe(rng2b.random() as number);
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

  describe('EXACT MATCH: Legacy permutation', () => {
    it('permutation(int) matches NumPy exactly', () => {
      random.seed(42);
      const js = (random.permutation(10) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.permutation(10).astype(float)
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBeCloseTo(py.value[i], 14);
      }
    });

    it('permutation(int) matches NumPy across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        random.seed(s);
        const js = (random.permutation(20) as any).toArray() as number[];
        const py = runNumPy(`
np.random.seed(${s})
result = np.random.permutation(20).astype(float)
        `);
        for (let i = 0; i < 20; i++) {
          expect(js[i]).toBeCloseTo(py.value[i], 14);
        }
      }
    });
  });

  // ============================================================
  // EXACT MATCH: PCG64 Generator — Extended
  // Covers every Generator method not tested above.
  // ============================================================

  describe('EXACT MATCH: PCG64 Generator — Extended', () => {
    it('default_rng().permutation(int) matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.permutation(20) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.permutation(20)
      `);
      for (let i = 0; i < 20; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().permutation(int) matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        const rng = random.default_rng(s);
        const js = Array.from((rng.permutation(15) as any).data as BigInt64Array).map(Number);
        const py = runNumPy(`
rng = np.random.default_rng(${s})
result = rng.permutation(15)
        `);
        for (let i = 0; i < 15; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      }
    });

    it('default_rng().poisson() matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.poisson(5, 20) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.poisson(5, 20)
      `);
      for (let i = 0; i < 20; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().binomial() matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.binomial(10, 0.3, 20) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.binomial(10, 0.3, 20)
      `);
      for (let i = 0; i < 20; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().choice(int, size, replace=True) matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.choice(10, 5) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.choice(10, 5)
      `);
      for (let i = 0; i < 5; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().choice(int, size, replace=True) matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        const rng = random.default_rng(s);
        const js = Array.from((rng.choice(20, 10) as any).data as BigInt64Array).map(Number);
        const py = runNumPy(`
rng = np.random.default_rng(${s})
result = rng.choice(20, 10)
        `);
        for (let i = 0; i < 10; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      }
    });

    it('default_rng().shuffle() matches NumPy', () => {
      const rng = random.default_rng(42);
      const arr = arange(10);
      rng.shuffle(arr);
      const js = Array.from((arr as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
arr = np.arange(10)
rng.shuffle(arr)
result = arr
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().shuffle() matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        const rng = random.default_rng(s);
        const arr = arange(15);
        rng.shuffle(arr);
        const js = Array.from((arr as any).data as BigInt64Array).map(Number);
        const py = runNumPy(`
rng = np.random.default_rng(${s})
arr = np.arange(15)
rng.shuffle(arr)
result = arr
        `);
        for (let i = 0; i < 15; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      }
    });
  });

  describe('EXACT MATCH: Legacy choice & shuffle', () => {
    it('choice(int, size, replace=True) matches NumPy', () => {
      random.seed(42);
      const js = Array.from((random.choice(10, 5) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
np.random.seed(42)
result = np.random.choice(10, 5)
      `);
      for (let i = 0; i < 5; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('choice(int, size, replace=True) matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        random.seed(s);
        const js = Array.from((random.choice(20, 10) as any).data as BigInt64Array).map(Number);
        const py = runNumPy(`
np.random.seed(${s})
result = np.random.choice(20, 10)
        `);
        for (let i = 0; i < 10; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      }
    });

    it('choice(int, size, replace=False) matches NumPy', () => {
      random.seed(42);
      const js = (random.choice(10, 5, false) as any).toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
result = np.random.choice(10, 5, replace=False).astype(float)
      `);
      for (let i = 0; i < 5; i++) {
        expect(js[i]).toBeCloseTo(py.value[i], 14);
      }
    });

    it('shuffle() matches NumPy', () => {
      random.seed(42);
      const arr = arange(10);
      random.shuffle(arr);
      const js = arr.toArray() as number[];
      const py = runNumPy(`
np.random.seed(42)
arr = np.arange(10, dtype=float)
np.random.shuffle(arr)
result = arr
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBeCloseTo(py.value[i], 14);
      }
    });

    it('shuffle() matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        random.seed(s);
        const arr = arange(15);
        random.shuffle(arr);
        const js = arr.toArray() as number[];
        const py = runNumPy(`
np.random.seed(${s})
arr = np.arange(15, dtype=float)
np.random.shuffle(arr)
result = arr
        `);
        for (let i = 0; i < 15; i++) {
          expect(js[i]).toBeCloseTo(py.value[i], 14);
        }
      }
    });
  });

  describe('multivariate_normal statistical validation', () => {
    // Note: multivariate_normal cannot be bit-exact because NumPy uses LAPACK's
    // dgesdd for SVD, whose sign convention for singular vectors is
    // implementation-dependent. Our WASM SVD produces valid but differently-signed
    // singular vectors. Both produce correct samples from the same distribution,
    // but the specific sample values differ. We validate statistical properties.

    it('multivariate_normal produces correct shape and mean', () => {
      random.seed(42);
      const result = random.multivariate_normal(
        [5, 10],
        [
          [1, 0.5],
          [0.5, 1],
        ],
        5000
      ) as any;
      expect(result.shape).toEqual([5000, 2]);
      const data = result.data as Float64Array;
      let sum0 = 0,
        sum1 = 0;
      for (let i = 0; i < 5000; i++) {
        sum0 += data[i * 2]!;
        sum1 += data[i * 2 + 1]!;
      }
      expect(Math.abs(sum0 / 5000 - 5)).toBeLessThan(0.1);
      expect(Math.abs(sum1 / 5000 - 10)).toBeLessThan(0.1);
    });

    it('multivariate_normal produces correct covariance structure', () => {
      random.seed(42);
      const result = random.multivariate_normal(
        [0, 0],
        [
          [1, 0.8],
          [0.8, 1],
        ],
        10000
      ) as any;
      const data = result.data as Float64Array;
      // Compute sample correlation
      let sumXY = 0,
        sumX2 = 0,
        sumY2 = 0;
      for (let i = 0; i < 10000; i++) {
        const x = data[i * 2]!,
          y = data[i * 2 + 1]!;
        sumXY += x * y;
        sumX2 += x * x;
        sumY2 += y * y;
      }
      const corr = sumXY / Math.sqrt(sumX2 * sumY2);
      expect(Math.abs(corr - 0.8)).toBeLessThan(0.05);
    });

    it('multivariate_normal is deterministic with same seed', () => {
      random.seed(42);
      const r1 = Array.from(
        (
          random.multivariate_normal(
            [0, 0],
            [
              [1, 0.5],
              [0.5, 1],
            ],
            3
          ) as any
        ).data as Float64Array
      );
      random.seed(42);
      const r2 = Array.from(
        (
          random.multivariate_normal(
            [0, 0],
            [
              [1, 0.5],
              [0.5, 1],
            ],
            3
          ) as any
        ).data as Float64Array
      );
      for (let i = 0; i < 6; i++) {
        expect(r1[i]).toBe(r2[i]);
      }
    });
  });

  describe('EXACT MATCH: Generator choice (replace=False)', () => {
    it('default_rng().choice(int, size, replace=False) matches NumPy', () => {
      const rng = random.default_rng(42);
      const js = Array.from((rng.choice(10, 5, false) as any).data as BigInt64Array).map(Number);
      const py = runNumPy(`
rng = np.random.default_rng(42)
result = rng.choice(10, 5, replace=False)
      `);
      for (let i = 0; i < 5; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });

    it('default_rng().choice(int, size, replace=False) matches across seeds', () => {
      for (const s of [0, 1, 12345, 99999]) {
        const rng = random.default_rng(s);
        const js = Array.from((rng.choice(20, 8, false) as any).data as BigInt64Array).map(Number);
        const py = runNumPy(`
rng = np.random.default_rng(${s})
result = rng.choice(20, 8, replace=False)
        `);
        for (let i = 0; i < 8; i++) {
          expect(js[i]).toBe(py.value[i]);
        }
      }
    });
  });

  describe('EXACT MATCH: choice with probabilities', () => {
    it('choice(int, size, p=) matches NumPy (replace=True)', () => {
      random.seed(42);
      const js = Array.from(
        (random.choice(5, 10, true, [0.1, 0.2, 0.3, 0.2, 0.2]) as any).data
      ).map(Number);
      const py = runNumPy(`
np.random.seed(42)
result = np.random.choice(5, 10, p=[0.1, 0.2, 0.3, 0.2, 0.2])
      `);
      for (let i = 0; i < 10; i++) {
        expect(js[i]).toBe(py.value[i]);
      }
    });
  });
});
