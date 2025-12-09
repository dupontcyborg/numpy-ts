import { describe, it, expect, beforeEach } from 'vitest';
import { random } from '../../src/index';

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
});
