/**
 * NumPy validation tests for accumulation dtype promotion
 *
 * NumPy automatically promotes narrow integer types (int8, int16, uint8, uint16)
 * to wider types for accumulation operations like sum, prod, cumsum, etc.
 * This test systematically maps out where numpy-ts differs from NumPy.
 *
 * Key NumPy behaviors:
 * - int8/int16 → int64 for sum/prod/cumsum/cumprod
 * - uint8/uint16 → uint64 for sum/prod/cumsum/cumprod
 * - int32 → int64 for sum/prod (on most platforms)
 * - uint32 → uint64 for sum/prod
 * - mean always returns float64 regardless of input dtype
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  sum,
  mean,
  prod,
  cumsum,
  cumprod,
  nansum,
  nanmean,
  nanprod,
  ptp,
  average,
} from '../../src';
import { checkNumPyAvailable, runNumPy, arraysClose } from './numpy-oracle';
import type { DType } from '../../src/common/dtype';

// All integer dtypes to test
const SIGNED_NARROW: DType[] = ['int8', 'int16'];
const UNSIGNED_NARROW: DType[] = ['uint8', 'uint16'];
const SIGNED_WIDE: DType[] = ['int32'];
const UNSIGNED_WIDE: DType[] = ['uint32'];
const ALL_INT_DTYPES: DType[] = [
  ...SIGNED_NARROW,
  ...UNSIGNED_NARROW,
  ...SIGNED_WIDE,
  ...UNSIGNED_WIDE,
];

// Map TS dtype names to NumPy dtype names
const npDtype: Record<string, string> = {
  int8: 'np.int8',
  int16: 'np.int16',
  int32: 'np.int32',
  uint8: 'np.uint8',
  uint16: 'np.uint16',
  uint32: 'np.uint32',
};

// Small values that won't overflow even in int8 element-wise,
// but will overflow narrow accumulators when summed/multiplied
const SMALL_DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // sum=55, prod=3628800
const MEDIUM_DATA = [10, 20, 30, 40, 50]; // sum=150 (overflows int8)
const LARGE_DATA = Array.from({ length: 100 }, (_, i) => (i % 10) + 1); // sum=550

// 2D data for axis tests
const DATA_2D = [
  [1, 2, 3, 4, 5],
  [6, 7, 8, 9, 10],
];

describe('NumPy Validation: Accumulation Dtype Promotion', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  // ==================== RESULT DTYPE TESTS ====================
  // These test what dtype NumPy returns vs what numpy-ts returns

  describe('Result dtype mapping', () => {
    for (const dtype of ALL_INT_DTYPES) {
      describe(`${dtype}`, () => {
        it(`sum() result dtype`, () => {
          const np = runNumPy(`
result = np.array([1, 2, 3], dtype=${npDtype[dtype]}).sum()
          `);

          const arr = array([1, 2, 3], dtype);
          const result = sum(arr);
          const tsDtype = typeof result === 'number' ? 'number' : 'bigint';

          // Record what NumPy returns
          const npResultDtype = np.dtype;

          // Log the mapping for analysis
          console.log(
            `  sum(${dtype}): NumPy→${npResultDtype}(${np.value}), TS→${tsDtype}(${result})`
          );
        });

        it(`prod() result dtype`, () => {
          const np = runNumPy(`
result = np.array([1, 2, 3], dtype=${npDtype[dtype]}).prod()
          `);

          const arr = array([1, 2, 3], dtype);
          const result = prod(arr);
          const tsDtype = typeof result === 'number' ? 'number' : 'bigint';

          console.log(`  prod(${dtype}): NumPy→${np.dtype}(${np.value}), TS→${tsDtype}(${result})`);
        });

        it(`mean() result dtype`, () => {
          const np = runNumPy(`
result = np.array([1, 2, 3], dtype=${npDtype[dtype]}).mean()
          `);

          const arr = array([1, 2, 3], dtype);
          const result = mean(arr);
          const tsDtype = typeof result === 'number' ? 'number' : 'bigint';

          console.log(`  mean(${dtype}): NumPy→${np.dtype}(${np.value}), TS→${tsDtype}(${result})`);
        });

        it(`cumsum() result dtype`, () => {
          const np = runNumPy(`
result = np.cumsum(np.array([1, 2, 3], dtype=${npDtype[dtype]}))
          `);

          const arr = array([1, 2, 3], dtype);
          const result = cumsum(arr);

          console.log(
            `  cumsum(${dtype}): NumPy→${np.dtype}(${JSON.stringify(np.value)}), TS→${result.dtype}(${JSON.stringify(result.toArray(), (_, v) => typeof v === 'bigint' ? v.toString() : v)})`
          );
        });

        it(`cumprod() result dtype`, () => {
          const np = runNumPy(`
result = np.cumprod(np.array([1, 2, 3], dtype=${npDtype[dtype]}))
          `);

          const arr = array([1, 2, 3], dtype);
          const result = cumprod(arr);

          console.log(
            `  cumprod(${dtype}): NumPy→${np.dtype}(${JSON.stringify(np.value)}), TS→${result.dtype}(${JSON.stringify(result.toArray(), (_, v) => typeof v === 'bigint' ? v.toString() : v)})`
          );
        });
      });
    }
  });

  // ==================== VALUE CORRECTNESS TESTS ====================
  // These test whether the actual computed values match NumPy

  describe('sum() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`sum([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = arr.sum()
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(sum(arr));

        expect(result).toBe(npResult.value);
      });

      it(`sum([10,20,30,40,50]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(MEDIUM_DATA)}, dtype=${npDtype[dtype]})
result = arr.sum()
        `);

        const arr = array(MEDIUM_DATA, dtype);
        const result = Number(sum(arr));

        // This is where int8 diverges — 150 overflows int8 (max 127)
        // NumPy promotes to int64 (result: 150), numpy-ts may wrap
        if (result !== npResult.value) {
          console.log(
            `  MISMATCH sum(${dtype}): NumPy=${npResult.value} (dtype=${npResult.dtype}), TS=${result}`
          );
        }
        expect(result).toBe(npResult.value);
      });

      it(`sum(100 elements) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(LARGE_DATA)}, dtype=${npDtype[dtype]})
result = arr.sum()
        `);

        const arr = array(LARGE_DATA, dtype);
        const result = Number(sum(arr));

        if (result !== npResult.value) {
          console.log(`  MISMATCH sum(${dtype}, 100 elems): NumPy=${npResult.value}, TS=${result}`);
        }
        expect(result).toBe(npResult.value);
      });
    }
  });

  describe('sum() with axis matches NumPy', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`sum(axis=0) with ${dtype}`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(DATA_2D)}, dtype=${npDtype[dtype]})
result = arr.sum(axis=0)
        `);

        const arr = array(DATA_2D, dtype);
        const result = sum(arr, 0);

        const tsValues = (result as any).toArray();
        if (!arraysClose(tsValues, npResult.value, 0, 0)) {
          console.log(
            `  MISMATCH sum(${dtype}, axis=0): NumPy=${JSON.stringify(npResult.value)} (${npResult.dtype}), TS=${JSON.stringify(tsValues)} (${(result as any).dtype})`
          );
        }
        expect(arraysClose(tsValues, npResult.value, 0, 0)).toBe(true);
      });

      it(`sum(axis=1) with ${dtype}`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(DATA_2D)}, dtype=${npDtype[dtype]})
result = arr.sum(axis=1)
        `);

        const arr = array(DATA_2D, dtype);
        const result = sum(arr, 1);

        const tsValues = (result as any).toArray();
        if (!arraysClose(tsValues, npResult.value, 0, 0)) {
          console.log(
            `  MISMATCH sum(${dtype}, axis=1): NumPy=${JSON.stringify(npResult.value)} (${npResult.dtype}), TS=${JSON.stringify(tsValues)} (${(result as any).dtype})`
          );
        }
        expect(arraysClose(tsValues, npResult.value, 0, 0)).toBe(true);
      });
    }
  });

  describe('prod() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`prod([1,2,3,4,5]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5], dtype=${npDtype[dtype]})
result = arr.prod()
        `);

        const arr = array([1, 2, 3, 4, 5], dtype);
        const result = Number(prod(arr));

        // prod([1..5]) = 120, overflows int8 (max 127 for signed)
        if (result !== npResult.value) {
          console.log(
            `  MISMATCH prod(${dtype}): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
          );
        }
        expect(result).toBe(npResult.value);
      });

      it(`prod([1,2,3]) with ${dtype} matches NumPy (no overflow)`, () => {
        const npResult = runNumPy(`
arr = np.array([1, 2, 3], dtype=${npDtype[dtype]})
result = arr.prod()
        `);

        const arr = array([1, 2, 3], dtype);
        const result = Number(prod(arr));

        // prod([1,2,3]) = 6, should never overflow
        expect(result).toBe(npResult.value);
      });
    }
  });

  describe('mean() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`mean([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = arr.mean()
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(mean(arr));

        // mean uses float accumulation, but if sum overflows first...
        if (Math.abs(result - npResult.value) > 1e-10) {
          console.log(
            `  MISMATCH mean(${dtype}): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
          );
        }
        expect(Math.abs(result - npResult.value)).toBeLessThan(1e-10);
      });

      it(`mean([10,20,30,40,50]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(MEDIUM_DATA)}, dtype=${npDtype[dtype]})
result = arr.mean()
        `);

        const arr = array(MEDIUM_DATA, dtype);
        const result = Number(mean(arr));

        if (Math.abs(result - npResult.value) > 1e-10) {
          console.log(`  MISMATCH mean(${dtype}): NumPy=${npResult.value}, TS=${result}`);
        }
        expect(Math.abs(result - npResult.value)).toBeLessThan(1e-10);
      });
    }
  });

  describe('cumsum() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`cumsum([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = np.cumsum(arr)
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = cumsum(arr);
        const tsValues = result.toArray();

        if (!arraysClose(tsValues, npResult.value, 0, 0)) {
          console.log(
            `  MISMATCH cumsum(${dtype}): NumPy=${JSON.stringify(npResult.value)} (${npResult.dtype}), TS=${JSON.stringify(tsValues)} (${result.dtype})`
          );
        }
        expect(arraysClose(tsValues, npResult.value, 0, 0)).toBe(true);
      });
    }
  });

  describe('cumprod() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`cumprod([1,2,3,4,5]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5], dtype=${npDtype[dtype]})
result = np.cumprod(arr)
        `);

        const arr = array([1, 2, 3, 4, 5], dtype);
        const result = cumprod(arr);
        const tsValues = result.toArray();

        if (!arraysClose(tsValues, npResult.value, 0, 0)) {
          console.log(
            `  MISMATCH cumprod(${dtype}): NumPy=${JSON.stringify(npResult.value)} (${npResult.dtype}), TS=${JSON.stringify(tsValues)} (${result.dtype})`
          );
        }
        expect(arraysClose(tsValues, npResult.value, 0, 0)).toBe(true);
      });
    }
  });

  describe('nansum() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`nansum([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = np.nansum(arr)
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(nansum(arr));

        if (result !== npResult.value) {
          console.log(
            `  MISMATCH nansum(${dtype}): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
          );
        }
        expect(result).toBe(npResult.value);
      });
    }
  });

  describe('nanmean() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`nanmean([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = np.nanmean(arr)
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(nanmean(arr));

        if (Math.abs(result - npResult.value) > 1e-10) {
          console.log(`  MISMATCH nanmean(${dtype}): NumPy=${npResult.value}, TS=${result}`);
        }
        expect(Math.abs(result - npResult.value)).toBeLessThan(1e-10);
      });
    }
  });

  describe('nanprod() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`nanprod([1,2,3,4,5]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5], dtype=${npDtype[dtype]})
result = np.nanprod(arr)
        `);

        const arr = array([1, 2, 3, 4, 5], dtype);
        const result = Number(nanprod(arr));

        if (result !== npResult.value) {
          console.log(
            `  MISMATCH nanprod(${dtype}): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
          );
        }
        expect(result).toBe(npResult.value);
      });
    }
  });

  describe('ptp() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`ptp([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = np.ptp(arr)
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(ptp(arr));

        if (result !== npResult.value) {
          console.log(
            `  MISMATCH ptp(${dtype}): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
          );
        }
        expect(result).toBe(npResult.value);
      });
    }
  });

  describe('average() value correctness', () => {
    for (const dtype of ALL_INT_DTYPES) {
      it(`average([1..10]) with ${dtype} matches NumPy`, () => {
        const npResult = runNumPy(`
arr = np.array(${JSON.stringify(SMALL_DATA)}, dtype=${npDtype[dtype]})
result = np.average(arr)
        `);

        const arr = array(SMALL_DATA, dtype);
        const result = Number(average(arr));

        if (Math.abs(result - npResult.value) > 1e-10) {
          console.log(`  MISMATCH average(${dtype}): NumPy=${npResult.value}, TS=${result}`);
        }
        expect(Math.abs(result - npResult.value)).toBeLessThan(1e-10);
      });
    }
  });

  // ==================== OVERFLOW BOUNDARY TESTS ====================
  // Specifically test values near the boundaries where promotion matters

  describe('Overflow boundary: int8 accumulation', () => {
    it('sum of values totaling 200 (above int8 max 127)', () => {
      const data = [50, 50, 50, 50]; // sum=200
      const npResult = runNumPy(`
arr = np.array([50, 50, 50, 50], dtype=np.int8)
result = arr.sum()
      `);

      const arr = array(data, 'int8');
      const result = Number(sum(arr));

      console.log(
        `  int8 sum([50,50,50,50]): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
      );
      expect(result).toBe(npResult.value);
    });

    it('cumsum crossing int8 boundary', () => {
      const data = [100, 50, 50]; // cumsum=[100, 150, 200] — all above int8 max after first
      const npResult = runNumPy(`
arr = np.array([100, 50, 50], dtype=np.int8)
result = np.cumsum(arr)
      `);

      const arr = array(data, 'int8');
      const result = cumsum(arr);
      const tsValues = result.toArray();

      const stringify = (v: unknown) => JSON.stringify(v, (_, x) => typeof x === 'bigint' ? Number(x) : x);
      console.log(
        `  int8 cumsum([100,50,50]): NumPy=${JSON.stringify(npResult.value)} (${npResult.dtype}), TS=${stringify(tsValues)} (${result.dtype})`
      );
      // cumsum promotes int8→int64 (BigInt); compare as numbers
      const tsNumbers = tsValues.map(Number);
      expect(arraysClose(tsNumbers, npResult.value, 0, 0)).toBe(true);
    });

    it('prod crossing int8 boundary', () => {
      const data = [10, 15]; // prod=150 > 127
      const npResult = runNumPy(`
arr = np.array([10, 15], dtype=np.int8)
result = arr.prod()
      `);

      const arr = array(data, 'int8');
      const result = Number(prod(arr));

      console.log(
        `  int8 prod([10,15]): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
      );
      expect(result).toBe(npResult.value);
    });
  });

  describe('Overflow boundary: uint8 accumulation', () => {
    it('sum of values totaling 300 (above uint8 max 255)', () => {
      const data = [100, 100, 100]; // sum=300
      const npResult = runNumPy(`
arr = np.array([100, 100, 100], dtype=np.uint8)
result = arr.sum()
      `);

      const arr = array(data, 'uint8');
      const result = Number(sum(arr));

      console.log(
        `  uint8 sum([100,100,100]): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
      );
      expect(result).toBe(npResult.value);
    });
  });

  describe('Overflow boundary: int16 accumulation', () => {
    it('sum of values totaling 40000 (above int16 max 32767)', () => {
      const data = [10000, 10000, 10000, 10000]; // sum=40000
      const npResult = runNumPy(`
arr = np.array([10000, 10000, 10000, 10000], dtype=np.int16)
result = arr.sum()
      `);

      const arr = array(data, 'int16');
      const result = Number(sum(arr));

      console.log(
        `  int16 sum([10000x4]): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
      );
      expect(result).toBe(npResult.value);
    });
  });

  describe('Overflow boundary: uint16 accumulation', () => {
    it('sum of values totaling 70000 (above uint16 max 65535)', () => {
      const data = [20000, 20000, 15000, 15000]; // sum=70000
      const npResult = runNumPy(`
arr = np.array([20000, 20000, 15000, 15000], dtype=np.uint16)
result = arr.sum()
      `);

      const arr = array(data, 'uint16');
      const result = Number(sum(arr));

      console.log(
        `  uint16 sum([20000,20000,15000,15000]): NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`
      );
      expect(result).toBe(npResult.value);
    });
  });

  // ==================== INT32/UINT32 PROMOTION TESTS ====================
  // NumPy also promotes int32→int64 for sum/prod on most platforms

  describe('int32/uint32 accumulation promotion', () => {
    it('int32 sum result dtype matches NumPy', () => {
      const npResult = runNumPy(`
arr = np.array([1000000, 2000000, 3000000], dtype=np.int32)
result = arr.sum()
      `);

      const arr = array([1000000, 2000000, 3000000], 'int32');
      const result = Number(sum(arr));

      console.log(`  int32 sum: NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`);
      expect(result).toBe(npResult.value);
    });

    it('uint32 sum result dtype matches NumPy', () => {
      const npResult = runNumPy(`
arr = np.array([1000000, 2000000, 3000000], dtype=np.uint32)
result = arr.sum()
      `);

      const arr = array([1000000, 2000000, 3000000], 'uint32');
      const result = Number(sum(arr));

      console.log(`  uint32 sum: NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`);
      expect(result).toBe(npResult.value);
    });

    it('int32 large sum (near overflow boundary)', () => {
      // int32 max is ~2.1 billion — this sum would overflow int32
      const data = [2000000000, 1000000000]; // sum=3 billion > int32 max
      const npResult = runNumPy(`
arr = np.array([2000000000, 1000000000], dtype=np.int32)
result = arr.sum()
      `);

      const arr = array(data, 'int32');
      const result = Number(sum(arr));

      console.log(`  int32 large sum: NumPy=${npResult.value} (${npResult.dtype}), TS=${result}`);
      expect(result).toBe(npResult.value);
    });
  });
});
