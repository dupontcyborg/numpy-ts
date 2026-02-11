import { describe, it, expect, beforeEach } from 'vitest';
import { random, array } from '../../src/index';

describe('Random Module', () => {
  beforeEach(() => {
    // Reset seed for reproducibility
    random.seed(42);
  });

  describe('seed', () => {
    it('produces reproducible results with same seed', () => {
      random.seed(12345);
      const result1 = random.random(5) as any;
      const arr1 = result1.toArray();

      random.seed(12345);
      const result2 = random.random(5) as any;
      const arr2 = result2.toArray();

      expect(arr1).toEqual(arr2);
    });

    it('produces different results with different seeds', () => {
      random.seed(12345);
      const result1 = random.random(5) as any;
      const arr1 = result1.toArray();

      random.seed(54321);
      const result2 = random.random(5) as any;
      const arr2 = result2.toArray();

      expect(arr1).not.toEqual(arr2);
    });

    it('handles null seed (non-deterministic)', () => {
      random.seed(null);
      const result = random.random(3) as any;
      expect(result.size).toBe(3);
    });
  });

  describe('random', () => {
    it('returns a single float when no size specified', () => {
      const result = random.random();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(1);
    });

    it('returns array of specified size', () => {
      const result = random.random(10) as any;
      expect(result.size).toBe(10);
      expect(result.shape).toEqual([10]);
    });

    it('returns array of specified shape', () => {
      const result = random.random([3, 4]) as any;
      expect(result.shape).toEqual([3, 4]);
      expect(result.size).toBe(12);
    });

    it('produces values in [0, 1)', () => {
      const result = random.random(1000) as any;
      const data = result.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });
  });

  describe('rand', () => {
    it('returns a single float when no arguments', () => {
      const result = random.rand();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(1);
    });

    it('returns 1D array with single dimension', () => {
      const result = random.rand(5) as any;
      expect(result.shape).toEqual([5]);
    });

    it('returns multi-dimensional array', () => {
      const result = random.rand(2, 3, 4) as any;
      expect(result.shape).toEqual([2, 3, 4]);
      expect(result.size).toBe(24);
    });
  });

  describe('randn', () => {
    it('returns a single float when no arguments', () => {
      const result = random.randn();
      expect(typeof result).toBe('number');
    });

    it('returns array with standard normal distribution', () => {
      random.seed(42);
      const result = random.randn(10000) as any;
      const data = result.toArray() as number[];

      // Calculate mean and std
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const mean = sum / data.length;

      let variance = 0;
      for (const val of data) {
        variance += (val - mean) ** 2;
      }
      const std = Math.sqrt(variance / data.length);

      // Mean should be close to 0, std close to 1
      expect(Math.abs(mean)).toBeLessThan(0.1);
      expect(Math.abs(std - 1)).toBeLessThan(0.1);
    });

    it('returns multi-dimensional array', () => {
      const result = random.randn(2, 3) as any;
      expect(result.shape).toEqual([2, 3]);
    });
  });

  describe('randint', () => {
    it('returns integer with single argument (0 to high)', () => {
      const result = random.randint(10);
      expect(typeof result).toBe('number');
      expect(Number.isInteger(result as number)).toBe(true);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(10);
    });

    it('returns integer in range [low, high)', () => {
      random.seed(42);
      for (let i = 0; i < 100; i++) {
        const result = random.randint(5, 10) as number;
        expect(result).toBeGreaterThanOrEqual(5);
        expect(result).toBeLessThan(10);
      }
    });

    it('returns array of specified size', () => {
      const result = random.randint(0, 100, [3, 4]) as any;
      expect(result.shape).toEqual([3, 4]);
    });

    it('returns integers in valid range for array', () => {
      const result = random.randint(10, 20, 100) as any;
      const data = result.toArray() as number[];
      for (const val of data) {
        expect(Number(val)).toBeGreaterThanOrEqual(10);
        expect(Number(val)).toBeLessThan(20);
      }
    });
  });

  describe('uniform', () => {
    it('returns single float in [low, high)', () => {
      for (let i = 0; i < 100; i++) {
        const result = random.uniform(2, 5) as number;
        expect(result).toBeGreaterThanOrEqual(2);
        expect(result).toBeLessThan(5);
      }
    });

    it('defaults to [0, 1)', () => {
      const result = random.uniform();
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(1);
    });

    it('returns array of specified size', () => {
      const result = random.uniform(0, 1, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces values in specified range', () => {
      const result = random.uniform(-5, 5, 1000) as any;
      const data = result.toArray() as number[];
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(-5);
        expect(val).toBeLessThan(5);
      }
    });
  });

  describe('normal', () => {
    it('returns single float', () => {
      const result = random.normal();
      expect(typeof result).toBe('number');
    });

    it('respects loc and scale parameters', () => {
      random.seed(42);
      const result = random.normal(10, 2, 10000) as any;
      const data = result.toArray() as number[];

      // Calculate mean and std
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const mean = sum / data.length;

      let variance = 0;
      for (const val of data) {
        variance += (val - mean) ** 2;
      }
      const std = Math.sqrt(variance / data.length);

      // Mean should be close to 10, std close to 2
      expect(Math.abs(mean - 10)).toBeLessThan(0.2);
      expect(Math.abs(std - 2)).toBeLessThan(0.2);
    });

    it('returns array of specified shape', () => {
      const result = random.normal(0, 1, [3, 4]) as any;
      expect(result.shape).toEqual([3, 4]);
    });
  });

  describe('standard_normal', () => {
    it('returns single float', () => {
      const result = random.standard_normal();
      expect(typeof result).toBe('number');
    });

    it('returns array of specified size', () => {
      const result = random.standard_normal([2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces standard normal distribution', () => {
      random.seed(42);
      const result = random.standard_normal(10000) as any;
      const data = result.toArray() as number[];

      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const mean = sum / data.length;

      let variance = 0;
      for (const val of data) {
        variance += (val - mean) ** 2;
      }
      const std = Math.sqrt(variance / data.length);

      expect(Math.abs(mean)).toBeLessThan(0.1);
      expect(Math.abs(std - 1)).toBeLessThan(0.1);
    });
  });

  describe('exponential', () => {
    it('returns single positive float', () => {
      const result = random.exponential() as number;
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns array of specified size', () => {
      const result = random.exponential(1, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces exponential distribution with correct scale', () => {
      random.seed(42);
      const scale = 2;
      const result = random.exponential(scale, 10000) as any;
      const data = result.toArray() as number[];

      // Mean of exponential distribution equals scale
      let sum = 0;
      for (const val of data) {
        sum += val;
      }
      const mean = sum / data.length;

      expect(Math.abs(mean - scale)).toBeLessThan(0.2);
    });
  });

  describe('poisson', () => {
    it('returns single non-negative integer', () => {
      const result = random.poisson(5) as number;
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(result)).toBe(true);
    });

    it('returns array of specified size', () => {
      const result = random.poisson(5, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces Poisson distribution with correct mean', () => {
      random.seed(42);
      const lam = 5;
      const result = random.poisson(lam, 10000) as any;
      const data = result.toArray() as number[];

      let sum = 0;
      for (const val of data) {
        sum += Number(val);
      }
      const mean = sum / data.length;

      // Mean of Poisson equals lambda
      expect(Math.abs(mean - lam)).toBeLessThan(0.2);
    });
  });

  describe('binomial', () => {
    it('returns single integer in [0, n]', () => {
      const result = random.binomial(10, 0.5) as number;
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThanOrEqual(10);
    });

    it('returns array of specified size', () => {
      const result = random.binomial(10, 0.5, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces binomial distribution with correct mean', () => {
      random.seed(42);
      const n = 20;
      const p = 0.3;
      const result = random.binomial(n, p, 10000) as any;
      const data = result.toArray() as number[];

      let sum = 0;
      for (const val of data) {
        sum += Number(val);
      }
      const mean = sum / data.length;

      // Mean of binomial is n*p
      expect(Math.abs(mean - n * p)).toBeLessThan(0.3);
    });
  });

  describe('choice', () => {
    it('returns single element from range', () => {
      const result = random.choice(5) as number;
      expect([0, 1, 2, 3, 4]).toContain(result);
    });

    it('returns array of specified size with replacement', () => {
      const result = random.choice(10, [3, 4]) as any;
      expect(result.shape).toEqual([3, 4]);
    });

    it('returns array without replacement', () => {
      random.seed(42);
      const result = random.choice(5, 5, false) as any;
      const data = result.toArray() as number[];

      // Should contain all values 0-4 exactly once
      const values = new Set<number>(data);
      expect(values.size).toBe(5);
      expect([...values].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
    });

    it('throws error for invalid size without replacement', () => {
      expect(() => random.choice(3, 5, false)).toThrow();
    });

    it('respects probability weights', () => {
      random.seed(42);
      // Heavily favor index 0
      const p = [0.9, 0.05, 0.05];
      const result = random.choice(3, 1000, true, p) as any;
      const data = result.toArray() as number[];

      let count0 = 0;
      for (const val of data) {
        if (val === 0) count0++;
      }

      // Most samples should be 0
      expect(count0).toBeGreaterThan(800);
    });
  });

  describe('permutation', () => {
    it('returns permuted range for integer input', () => {
      random.seed(42);
      const result = random.permutation(5) as any;
      const data = result.toArray() as number[];

      // Should contain all values 0-4
      const values = new Set<number>(data);
      expect(values.size).toBe(5);
      expect([...values].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4]);
    });
  });

  describe('get_state and set_state', () => {
    it('allows state to be saved and restored', () => {
      random.seed(42);
      random.random(10); // Advance state

      const state = random.get_state();

      const result1 = random.random(5) as any;
      const arr1 = result1.toArray();

      random.set_state(state);

      const result2 = random.random(5) as any;
      const arr2 = result2.toArray();

      expect(arr1).toEqual(arr2);
    });
  });

  describe('default_rng', () => {
    it('creates independent Generator instances', () => {
      const rng1 = random.default_rng(42);
      const rng2 = random.default_rng(42);

      const result1 = rng1.random(5) as any;
      const result2 = rng2.random(5) as any;

      expect(result1.data).toEqual(result2.data);
    });

    it('Generator has all methods', () => {
      const rng = random.default_rng(42);

      expect(typeof rng.random).toBe('function');
      expect(typeof rng.integers).toBe('function');
      expect(typeof rng.normal).toBe('function');
      expect(typeof rng.standard_normal).toBe('function');
      expect(typeof rng.uniform).toBe('function');
      expect(typeof rng.choice).toBe('function');
      expect(typeof rng.permutation).toBe('function');
      expect(typeof rng.shuffle).toBe('function');
      expect(typeof rng.exponential).toBe('function');
      expect(typeof rng.poisson).toBe('function');
      expect(typeof rng.binomial).toBe('function');
    });

    it('Generator produces correct output types', () => {
      const rng = random.default_rng(42);

      const r = rng.random(5) as any;
      expect(r.shape).toEqual([5]);

      const n = rng.normal(0, 1, [2, 3]) as any;
      expect(n.shape).toEqual([2, 3]);

      const i = rng.integers(0, 10, [2, 2]) as any;
      expect(i.shape).toEqual([2, 2]);
    });
  });

  // ============================================================
  // Tests for new random functions
  // ============================================================

  describe('random_sample (alias)', () => {
    it('is an alias for random()', () => {
      random.seed(42);
      const r1 = random.random(5) as any;

      random.seed(42);
      const r2 = random.random_sample(5) as any;

      expect(r1.toArray()).toEqual(r2.toArray());
    });
  });

  describe('ranf (alias)', () => {
    it('is an alias for random()', () => {
      random.seed(42);
      const r1 = random.random(5) as any;

      random.seed(42);
      const r2 = random.ranf(5) as any;

      expect(r1.toArray()).toEqual(r2.toArray());
    });
  });

  describe('sample (alias)', () => {
    it('is an alias for random()', () => {
      random.seed(42);
      const r1 = random.random(5) as any;

      random.seed(42);
      const r2 = random.sample(5) as any;

      expect(r1.toArray()).toEqual(r2.toArray());
    });
  });

  describe('random_integers', () => {
    it('returns integers in inclusive range', () => {
      random.seed(42);
      // random_integers(1, 5) should return integers 1, 2, 3, 4, 5
      for (let i = 0; i < 100; i++) {
        const result = random.random_integers(1, 5) as number;
        expect(result).toBeGreaterThanOrEqual(1);
        expect(result).toBeLessThanOrEqual(5);
        expect(Number.isInteger(result)).toBe(true);
      }
    });

    it('with single argument returns integers from 1 to high', () => {
      random.seed(42);
      for (let i = 0; i < 100; i++) {
        const result = random.random_integers(5) as number;
        expect(result).toBeGreaterThanOrEqual(1);
        expect(result).toBeLessThanOrEqual(5);
      }
    });
  });

  describe('bytes', () => {
    it('returns Uint8Array of specified length', () => {
      random.seed(42);
      const result = random.bytes(10);
      expect(result).toBeInstanceOf(Uint8Array);
      expect(result.length).toBe(10);
    });

    it('produces values in byte range', () => {
      random.seed(42);
      const result = random.bytes(1000);
      for (const byte of result) {
        expect(byte).toBeGreaterThanOrEqual(0);
        expect(byte).toBeLessThanOrEqual(255);
      }
    });
  });

  describe('get_bit_generator and set_bit_generator', () => {
    it('can get and set bit generator', () => {
      const bg = random.get_bit_generator();
      expect(bg).toBeDefined();
      expect(bg.name).toBe('MT19937');

      const newBg = { name: 'Custom', state: {} };
      random.set_bit_generator(newBg);
      expect(random.get_bit_generator()).toBe(newBg);

      // Restore original
      random.set_bit_generator(bg);
    });
  });

  describe('standard_exponential', () => {
    it('is equivalent to exponential with scale=1', () => {
      random.seed(42);
      const r1 = random.exponential(1, 5) as any;

      random.seed(42);
      const r2 = random.standard_exponential(5) as any;

      expect(r1.toArray()).toEqual(r2.toArray());
    });

    it('produces positive values', () => {
      random.seed(42);
      const result = random.standard_exponential(100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });
  });

  describe('gamma', () => {
    it('returns single positive float', () => {
      random.seed(42);
      const result = random.gamma(2, 1) as number;
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns array of specified shape', () => {
      const result = random.gamma(2, 1, [3, 4]) as any;
      expect(result.shape).toEqual([3, 4]);
    });

    it('produces correct mean (shape * scale)', () => {
      random.seed(42);
      const shape = 3;
      const scale = 2;
      const result = random.gamma(shape, scale, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      // Mean of gamma is shape * scale
      expect(Math.abs(mean - shape * scale)).toBeLessThan(0.3);
    });

    it('throws for non-positive shape', () => {
      expect(() => random.gamma(0, 1)).toThrow();
      expect(() => random.gamma(-1, 1)).toThrow();
    });
  });

  describe('standard_gamma', () => {
    it('is equivalent to gamma with scale=1', () => {
      random.seed(42);
      const r1 = random.gamma(2, 1, 5) as any;

      random.seed(42);
      const r2 = random.standard_gamma(2, 5) as any;

      expect(r1.toArray()).toEqual(r2.toArray());
    });
  });

  describe('beta', () => {
    it('returns values in [0, 1]', () => {
      random.seed(42);
      const result = random.beta(2, 5, 1000) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });

    it('produces correct mean (a / (a + b))', () => {
      random.seed(42);
      const a = 2;
      const b = 5;
      const result = random.beta(a, b, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((acc, val) => acc + val, 0) / data.length;
      expect(Math.abs(mean - a / (a + b))).toBeLessThan(0.02);
    });

    it('throws for non-positive parameters', () => {
      expect(() => random.beta(0, 1)).toThrow();
      expect(() => random.beta(1, 0)).toThrow();
    });
  });

  describe('chisquare', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.chisquare(5, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mean (df)', () => {
      random.seed(42);
      const df = 5;
      const result = random.chisquare(df, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - df)).toBeLessThan(0.3);
    });
  });

  describe('noncentral_chisquare', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.noncentral_chisquare(5, 2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mean (df + nonc)', () => {
      random.seed(42);
      const df = 5;
      const nonc = 2;
      const result = random.noncentral_chisquare(df, nonc, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - (df + nonc))).toBeLessThan(0.5);
    });
  });

  describe('f', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.f(5, 10, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mean when dfden > 2', () => {
      random.seed(42);
      const dfnum = 5;
      const dfden = 10;
      const result = random.f(dfnum, dfden, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      // Mean of F is dfden / (dfden - 2) when dfden > 2
      const expectedMean = dfden / (dfden - 2);
      expect(Math.abs(mean - expectedMean)).toBeLessThan(0.3);
    });
  });

  describe('noncentral_f', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.noncentral_f(5, 10, 2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });
  });

  describe('standard_cauchy', () => {
    it('returns array of specified shape', () => {
      const result = random.standard_cauchy([2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces values (Cauchy has no mean/variance)', () => {
      random.seed(42);
      const result = random.standard_cauchy(100) as any;
      // Just check we get numbers
      for (const val of result.toArray() as number[]) {
        expect(typeof val).toBe('number');
        expect(isFinite(val)).toBe(true);
      }
    });
  });

  describe('standard_t', () => {
    it('returns array of specified shape', () => {
      const result = random.standard_t(5, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('with high df approaches normal distribution', () => {
      random.seed(42);
      const result = random.standard_t(100, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean)).toBeLessThan(0.1);
    });

    it('throws for non-positive df', () => {
      expect(() => random.standard_t(0)).toThrow();
      expect(() => random.standard_t(-1)).toThrow();
    });
  });

  describe('laplace', () => {
    it('returns array of specified shape', () => {
      const result = random.laplace(0, 1, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces correct mean', () => {
      random.seed(42);
      const loc = 5;
      const result = random.laplace(loc, 1, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - loc)).toBeLessThan(0.1);
    });
  });

  describe('logistic', () => {
    it('returns array of specified shape', () => {
      const result = random.logistic(0, 1, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('produces correct mean', () => {
      random.seed(42);
      const loc = 5;
      const result = random.logistic(loc, 1, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - loc)).toBeLessThan(0.1);
    });
  });

  describe('lognormal', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.lognormal(0, 1, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mean exp(mu + sigma^2/2)', () => {
      random.seed(42);
      const mu = 0;
      const sigma = 0.5;
      const result = random.lognormal(mu, sigma, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const expectedMean = Math.exp(mu + (sigma * sigma) / 2);
      expect(Math.abs(mean - expectedMean)).toBeLessThan(0.1);
    });
  });

  describe('gumbel', () => {
    it('returns array of specified shape', () => {
      const result = random.gumbel(0, 1, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });
  });

  describe('pareto', () => {
    it('returns values >= 0', () => {
      random.seed(42);
      const result = random.pareto(2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('throws for non-positive a', () => {
      expect(() => random.pareto(0)).toThrow();
      expect(() => random.pareto(-1)).toThrow();
    });
  });

  describe('power', () => {
    it('returns values in [0, 1]', () => {
      random.seed(42);
      const result = random.power(2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });

    it('throws for non-positive a', () => {
      expect(() => random.power(0)).toThrow();
      expect(() => random.power(-1)).toThrow();
    });
  });

  describe('rayleigh', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.rayleigh(1, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mode (scale)', () => {
      random.seed(42);
      const scale = 2;
      const result = random.rayleigh(scale, 10000) as any;
      const data = result.toArray() as number[];

      // Mean of Rayleigh is scale * sqrt(pi/2)
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const expectedMean = scale * Math.sqrt(Math.PI / 2);
      expect(Math.abs(mean - expectedMean)).toBeLessThan(0.1);
    });
  });

  describe('triangular', () => {
    it('returns values in [left, right]', () => {
      random.seed(42);
      const result = random.triangular(0, 0.5, 1, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      }
    });

    it('produces correct mean ((left + mode + right) / 3)', () => {
      random.seed(42);
      const left = 0;
      const mode = 0.5;
      const right = 1;
      const result = random.triangular(left, mode, right, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const expectedMean = (left + mode + right) / 3;
      expect(Math.abs(mean - expectedMean)).toBeLessThan(0.02);
    });

    it('throws for invalid parameters', () => {
      expect(() => random.triangular(1, 0, 2)).toThrow(); // mode < left
      expect(() => random.triangular(0, 3, 2)).toThrow(); // mode > right
      expect(() => random.triangular(1, 1, 1)).toThrow(); // left == right
    });
  });

  describe('wald', () => {
    it('returns positive values', () => {
      random.seed(42);
      const result = random.wald(1, 1, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('produces correct mean', () => {
      random.seed(42);
      const mean = 2;
      const scale = 3;
      const result = random.wald(mean, scale, 10000) as any;
      const data = result.toArray() as number[];

      const sampleMean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(sampleMean - mean)).toBeLessThan(0.2);
    });
  });

  describe('weibull', () => {
    it('returns non-negative values', () => {
      random.seed(42);
      const result = random.weibull(2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('throws for non-positive a', () => {
      expect(() => random.weibull(0)).toThrow();
      expect(() => random.weibull(-1)).toThrow();
    });
  });

  describe('geometric', () => {
    it('returns positive integers', () => {
      random.seed(42);
      const result = random.geometric(0.5, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(Number(val)).toBeGreaterThanOrEqual(1);
        expect(Number.isInteger(Number(val))).toBe(true);
      }
    });

    it('produces correct mean (1/p)', () => {
      random.seed(42);
      const p = 0.3;
      const result = random.geometric(p, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      expect(Math.abs(mean - 1 / p)).toBeLessThan(0.3);
    });

    it('throws for invalid p', () => {
      expect(() => random.geometric(0)).toThrow();
      expect(() => random.geometric(1.5)).toThrow();
      expect(() => random.geometric(-0.5)).toThrow();
    });
  });

  describe('hypergeometric', () => {
    it('returns integers in valid range', () => {
      random.seed(42);
      const ngood = 10;
      const nbad = 10;
      const nsample = 5;
      const result = random.hypergeometric(ngood, nbad, nsample, 100) as any;
      for (const val of result.toArray() as number[]) {
        const v = Number(val);
        expect(v).toBeGreaterThanOrEqual(Math.max(0, nsample - nbad));
        expect(v).toBeLessThanOrEqual(Math.min(nsample, ngood));
        expect(Number.isInteger(v)).toBe(true);
      }
    });

    it('produces correct mean', () => {
      random.seed(42);
      const ngood = 20;
      const nbad = 30;
      const nsample = 10;
      const result = random.hypergeometric(ngood, nbad, nsample, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      const expectedMean = (nsample * ngood) / (ngood + nbad);
      expect(Math.abs(mean - expectedMean)).toBeLessThan(0.2);
    });
  });

  describe('logseries', () => {
    it('returns positive integers', () => {
      random.seed(42);
      const result = random.logseries(0.5, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(Number(val)).toBeGreaterThanOrEqual(1);
        expect(Number.isInteger(Number(val))).toBe(true);
      }
    });

    it('throws for invalid p', () => {
      expect(() => random.logseries(0)).toThrow();
      expect(() => random.logseries(1)).toThrow();
      expect(() => random.logseries(1.5)).toThrow();
    });
  });

  describe('negative_binomial', () => {
    it('returns non-negative integers', () => {
      random.seed(42);
      const result = random.negative_binomial(5, 0.5, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(Number(val)).toBeGreaterThanOrEqual(0);
        expect(Number.isInteger(Number(val))).toBe(true);
      }
    });

    it('produces correct mean (n * (1-p) / p)', () => {
      random.seed(42);
      const n = 5;
      const p = 0.3;
      const result = random.negative_binomial(n, p, 10000) as any;
      const data = result.toArray() as number[];

      const mean = data.reduce((a, b) => a + Number(b), 0) / data.length;
      const expectedMean = (n * (1 - p)) / p;
      expect(Math.abs(mean - expectedMean)).toBeLessThan(1);
    });
  });

  describe('zipf', () => {
    it('returns positive integers', () => {
      random.seed(42);
      const result = random.zipf(2, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(Number(val)).toBeGreaterThanOrEqual(1);
        expect(Number.isInteger(Number(val))).toBe(true);
      }
    });

    it('throws for a <= 1', () => {
      expect(() => random.zipf(1)).toThrow();
      expect(() => random.zipf(0.5)).toThrow();
    });
  });

  describe('multinomial', () => {
    it('returns array summing to n', () => {
      random.seed(42);
      const n = 10;
      const pvals = [0.2, 0.3, 0.5];
      const result = random.multinomial(n, pvals) as any;

      const data = result.toArray() as number[];
      const sum = data.reduce((a, b) => a + Number(b), 0);
      expect(sum).toBe(n);
    });

    it('returns correct shape for single sample', () => {
      const result = random.multinomial(10, [0.2, 0.3, 0.5]) as any;
      expect(result.shape).toEqual([3]);
    });

    it('returns correct shape for multiple samples', () => {
      const result = random.multinomial(10, [0.2, 0.3, 0.5], 5) as any;
      expect(result.shape).toEqual([5, 3]);
    });

    it('returns correct shape for multi-dimensional size', () => {
      const result = random.multinomial(10, [0.2, 0.3, 0.5], [2, 3]) as any;
      expect(result.shape).toEqual([2, 3, 3]);
    });
  });

  describe('multivariate_normal', () => {
    it('returns array of correct shape for single sample', () => {
      const mean = [0, 0];
      const cov = [
        [1, 0],
        [0, 1],
      ];
      const result = random.multivariate_normal(mean, cov) as any;
      expect(result.shape).toEqual([2]);
    });

    it('returns array of correct shape for multiple samples', () => {
      const mean = [0, 0, 0];
      const cov = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
      const result = random.multivariate_normal(mean, cov, 5) as any;
      expect(result.shape).toEqual([5, 3]);
    });

    it('produces correct mean', () => {
      random.seed(42);
      const mean = [5, 10];
      const cov = [
        [1, 0],
        [0, 1],
      ];
      const result = random.multivariate_normal(mean, cov, 1000) as any;
      const data = result.data as Float64Array;

      let sum0 = 0;
      let sum1 = 0;
      for (let i = 0; i < 1000; i++) {
        sum0 += data[i * 2]!;
        sum1 += data[i * 2 + 1]!;
      }
      expect(Math.abs(sum0 / 1000 - mean[0]!)).toBeLessThan(0.2);
      expect(Math.abs(sum1 / 1000 - mean[1]!)).toBeLessThan(0.2);
    });
  });

  describe('dirichlet', () => {
    it('returns array summing to 1', () => {
      random.seed(42);
      const alpha = [1, 2, 3];
      const result = random.dirichlet(alpha) as any;

      const data = result.toArray() as number[];
      const sum = data.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-10);
    });

    it('returns values in [0, 1]', () => {
      random.seed(42);
      const result = random.dirichlet([1, 1, 1], 100) as any;
      const data = result.data as Float64Array;
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0);
        expect(data[i]).toBeLessThanOrEqual(1);
      }
    });

    it('returns correct shape', () => {
      const result = random.dirichlet([1, 2, 3, 4], 5) as any;
      expect(result.shape).toEqual([5, 4]);
    });

    it('throws for invalid alpha', () => {
      expect(() => random.dirichlet([1])).toThrow(); // needs at least 2
      expect(() => random.dirichlet([1, 0])).toThrow(); // all must be positive
      expect(() => random.dirichlet([1, -1])).toThrow();
    });
  });

  describe('vonmises', () => {
    it('returns values in [-pi, pi]', () => {
      random.seed(42);
      const result = random.vonmises(0, 1, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThanOrEqual(-Math.PI);
        expect(val).toBeLessThanOrEqual(Math.PI);
      }
    });

    it('with kappa=0 is uniform on circle', () => {
      random.seed(42);
      const result = random.vonmises(0, 0, 10000) as any;
      const data = result.toArray() as number[];

      // Mean should be close to 0 for uniform
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean)).toBeLessThan(0.1);
    });

    it('throws for negative kappa', () => {
      expect(() => random.vonmises(0, -1)).toThrow();
    });

    it('returns single value when size not specified', () => {
      random.seed(42);
      const result = random.vonmises(0, 1);
      expect(typeof result).toBe('number');
    });
  });

  // ============================================================
  // Additional coverage tests for uncovered paths
  // ============================================================

  describe('shuffle', () => {
    it('shuffles array in-place', () => {
      random.seed(42);
      const a = array([1, 2, 3, 4, 5]);
      random.shuffle(a.storage);
      // After shuffle, should have same elements but likely different order
      const sorted = (a.toArray() as number[]).slice().sort((x, y) => x - y);
      expect(sorted).toEqual([1, 2, 3, 4, 5]);
    });

    it('preserves array size after shuffle', () => {
      random.seed(42);
      const a = array([10, 20, 30, 40, 50, 60]);
      random.shuffle(a.storage);
      expect(a.size).toBe(6);
    });
  });

  describe('Generator method execution', () => {
    it('Generator.shuffle modifies array in place', () => {
      const rng = random.default_rng(42);
      const a = array([1, 2, 3, 4, 5]);
      rng.shuffle(a.storage);
      const sorted = (a.toArray() as number[]).slice().sort((x, y) => x - y);
      expect(sorted).toEqual([1, 2, 3, 4, 5]);
    });

    it('Generator.choice returns valid elements', () => {
      const rng = random.default_rng(42);
      const result = rng.choice(5) as number;
      expect([0, 1, 2, 3, 4]).toContain(result);
    });

    it('Generator.choice returns array of specified size', () => {
      const rng = random.default_rng(42);
      const result = rng.choice(10, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });

    it('Generator.permutation returns permuted range', () => {
      const rng = random.default_rng(42);
      const result = rng.permutation(5) as any;
      expect(result.shape).toEqual([5]);
      const data = Array.from(result.data as Float64Array);
      const sorted = data.slice().sort((a: number, b: number) => a - b);
      expect(sorted).toEqual([0, 1, 2, 3, 4]);
    });

    it('Generator.exponential returns positive values', () => {
      const rng = random.default_rng(42);
      const result = rng.exponential(1, 10) as any;
      const data = result.data as Float64Array;
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0);
      }
    });

    it('Generator.exponential returns single value', () => {
      const rng = random.default_rng(42);
      const result = rng.exponential();
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThan(0);
    });

    it('Generator.poisson returns non-negative integers', () => {
      const rng = random.default_rng(42);
      const result = rng.poisson(5, 10) as any;
      const data = result.data as BigInt64Array;
      for (let i = 0; i < data.length; i++) {
        expect(Number(data[i])).toBeGreaterThanOrEqual(0);
      }
    });

    it('Generator.poisson returns single value', () => {
      const rng = random.default_rng(42);
      const result = rng.poisson(5);
      expect(typeof result).toBe('number');
    });

    it('Generator.binomial returns integers in valid range', () => {
      const rng = random.default_rng(42);
      const result = rng.binomial(10, 0.5, 10) as any;
      const data = result.data as BigInt64Array;
      for (let i = 0; i < data.length; i++) {
        const v = Number(data[i]);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(10);
      }
    });

    it('Generator.binomial returns single value', () => {
      const rng = random.default_rng(42);
      const result = rng.binomial(10, 0.5);
      expect(typeof result).toBe('number');
    });
  });

  describe('scalar return paths', () => {
    it('standard_cauchy returns single number when no size', () => {
      random.seed(42);
      const result = random.standard_cauchy();
      expect(typeof result).toBe('number');
      expect(isFinite(result as number)).toBe(true);
    });

    it('standard_t returns single number when no size', () => {
      random.seed(42);
      const result = random.standard_t(5);
      expect(typeof result).toBe('number');
    });

    it('standard_gamma returns single number when no size', () => {
      random.seed(42);
      const result = random.standard_gamma(2);
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThan(0);
    });

    it('standard_gamma throws for non-positive shape', () => {
      expect(() => random.standard_gamma(0)).toThrow();
      expect(() => random.standard_gamma(-1)).toThrow();
    });

    it('noncentral_chisquare returns single number when no size', () => {
      random.seed(42);
      const result = random.noncentral_chisquare(5, 2);
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThan(0);
    });

    it('noncentral_chisquare with nonc=0 reduces to chisquare', () => {
      random.seed(42);
      const result = random.noncentral_chisquare(5, 0, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('noncentral_chisquare throws for invalid params', () => {
      expect(() => random.noncentral_chisquare(0, 2)).toThrow();
      expect(() => random.noncentral_chisquare(5, -1)).toThrow();
    });

    it('noncentral_f returns single number when no size', () => {
      random.seed(42);
      const result = random.noncentral_f(5, 10, 2);
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThan(0);
    });

    it('noncentral_f with nonc=0 produces positive values', () => {
      random.seed(42);
      const result = random.noncentral_f(5, 10, 0, 100) as any;
      for (const val of result.toArray() as number[]) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it('noncentral_f throws for invalid params', () => {
      expect(() => random.noncentral_f(0, 10, 2)).toThrow();
      expect(() => random.noncentral_f(5, 0, 2)).toThrow();
      expect(() => random.noncentral_f(5, 10, -1)).toThrow();
    });

    it('f returns single number when no size', () => {
      random.seed(42);
      const result = random.f(5, 10);
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThan(0);
    });

    it('f throws for invalid params', () => {
      expect(() => random.f(0, 10)).toThrow();
      expect(() => random.f(5, 0)).toThrow();
    });

    it('logseries returns single number when no size', () => {
      random.seed(42);
      const result = random.logseries(0.5);
      expect(typeof result).toBe('number');
      expect(result as number).toBeGreaterThanOrEqual(1);
    });

    it('vonmises returns correct shape array', () => {
      random.seed(42);
      const result = random.vonmises(0, 5, [2, 3]) as any;
      expect(result.shape).toEqual([2, 3]);
    });
  });
});
