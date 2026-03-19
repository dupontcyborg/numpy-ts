/**
 * Python NumPy validation tests for sorting and searching operations
 *
 * Tests sorting functions across:
 * - Both WASM modes (default thresholds + forced WASM threshold=0)
 * - All numeric dtypes (float64, float32, int64..int8, uint64..uint8)
 * - Multiple axis combinations
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as np from '../../src/full/index';
import { wasmConfig } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

/** NumPy dtype string for a given numpy-ts dtype */
const NP_DTYPE: Record<string, string> = {
  float64: 'np.float64',
  float32: 'np.float32',
  int64: 'np.int64',
  int32: 'np.int32',
  int16: 'np.int16',
  int8: 'np.int8',
  uint64: 'np.uint64',
  uint32: 'np.uint32',
  uint16: 'np.uint16',
  uint8: 'np.uint8',
};

const ALL_DTYPES = Object.keys(NP_DTYPE);

const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1 },
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
] as const;

/** Small 1D test data with values that fit in all dtypes (positive, no overflow) */
const SORT_DATA_1D = [3, 1, 4, 1, 5, 2, 6, 3];

/** Small 2x4 test data for axis tests */
const SORT_DATA_2D = [
  [3, 1, 4, 2],
  [6, 5, 7, 8],
];

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: Sorting [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      if (!checkNumPyAvailable()) {
        throw new Error('Python NumPy not available');
      }
      if (mode.multiplier === 1) {
        const info = getPythonInfo();
        console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
      }
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    // ============================================================
    // sort() — all dtypes, 1D and 2D with axis
    // ============================================================
    describe('sort()', () => {
      for (const dtype of ALL_DTYPES) {
        it(`1D ${dtype}`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(SORT_DATA_1D, dtype as any);
          const result = np.sort(a);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_1D)}, dtype=${npDtype})
result = np.sort(a)
`);
          expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
        });
      }

      for (const dtype of ALL_DTYPES) {
        for (const axis of [0, 1, -1] as const) {
          it(`2D ${dtype} axis=${axis}`, () => {
            const npDtype = NP_DTYPE[dtype]!;
            const a = np.array(SORT_DATA_2D, dtype as any);
            const result = np.sort(a, axis);

            const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_2D)}, dtype=${npDtype})
result = np.sort(a, axis=${axis})
`);
            expect((result as any).shape).toEqual(pyResult.shape);
            expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
          });
        }
      }
    });

    // ============================================================
    // argsort() — all dtypes, 1D and 2D with axis
    // ============================================================
    describe('argsort()', () => {
      for (const dtype of ALL_DTYPES) {
        it(`1D ${dtype}`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(SORT_DATA_1D, dtype as any);
          const result = np.argsort(a);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_1D)}, dtype=${npDtype})
result = np.argsort(a, kind='stable')
`);
          expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
        });
      }

      for (const dtype of ALL_DTYPES) {
        for (const axis of [0, -1] as const) {
          it(`2D ${dtype} axis=${axis}`, () => {
            const npDtype = NP_DTYPE[dtype]!;
            const a = np.array(SORT_DATA_2D, dtype as any);
            const result = np.argsort(a, axis);

            const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_2D)}, dtype=${npDtype})
result = np.argsort(a, axis=${axis}, kind='stable')
`);
            expect((result as any).shape).toEqual(pyResult.shape);
            expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
          });
        }
      }
    });

    // ============================================================
    // partition() — all dtypes, kth element correctness
    // ============================================================
    describe('partition()', () => {
      for (const dtype of ALL_DTYPES) {
        it(`1D ${dtype} kth=2`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(SORT_DATA_1D, dtype as any);
          const result = np.partition(a, 2);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_1D)}, dtype=${npDtype})
result = np.partition(a, 2)
`);
          // kth element must match
          const jsArr = (result as any).toArray() as number[];
          const npArr = pyResult.value as number[];
          expect(Number(jsArr[2])).toBe(Number(npArr[2]));
        });

        it(`1D ${dtype} kth=0 (minimum)`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(SORT_DATA_1D, dtype as any);
          const result = np.partition(a, 0);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_1D)}, dtype=${npDtype})
result = np.partition(a, 0)
`);
          const jsArr = (result as any).toArray() as number[];
          const npArr = pyResult.value as number[];
          expect(Number(jsArr[0])).toBe(Number(npArr[0]));
        });
      }

      for (const dtype of ALL_DTYPES) {
        it(`2D ${dtype} axis=-1 kth=2`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(SORT_DATA_2D, dtype as any);
          const result = np.partition(a, 2, -1);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(SORT_DATA_2D)}, dtype=${npDtype})
result = np.partition(a, 2, axis=-1)
`);
          // kth element in each row must match
          const jsArr = (result as any).toArray() as number[][];
          const npArr = pyResult.value as number[][];
          expect(Number(jsArr[0]![2])).toBe(Number(npArr[0]![2]));
          expect(Number(jsArr[1]![2])).toBe(Number(npArr[1]![2]));
        });
      }
    });

    // ============================================================
    // argpartition() — all dtypes, kth correctness
    // ============================================================
    describe('argpartition()', () => {
      for (const dtype of ALL_DTYPES) {
        it(`1D ${dtype} kth=2`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const data = SORT_DATA_1D;
          const a = np.array(data, dtype as any);
          const result = np.argpartition(a, 2);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype})
result = np.argpartition(a, 2)
`);
          // Value at kth index must match
          const jsArr = (result as any).toArray() as number[];
          const npArr = pyResult.value as number[];
          expect(Number(data[jsArr[2]!])).toBe(Number(data[npArr[2]!]));
        });

        it(`1D ${dtype} kth=0 (minimum)`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const data = SORT_DATA_1D;
          const a = np.array(data, dtype as any);
          const result = np.argpartition(a, 0);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype})
result = np.argpartition(a, 0)
`);
          const jsArr = (result as any).toArray() as number[];
          const npArr = pyResult.value as number[];
          expect(Number(data[jsArr[0]!])).toBe(Number(data[npArr[0]!]));
        });
      }
    });

    // ============================================================
    // lexsort() — all dtypes, 2 keys
    // ============================================================
    describe('lexsort()', () => {
      const key1Data = [1, 2, 1, 2, 1];
      const key2Data = [3, 1, 2, 4, 5];

      for (const dtype of ALL_DTYPES) {
        it(`2 keys ${dtype}`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const key1 = np.array(key1Data, dtype as any);
          const key2 = np.array(key2Data, dtype as any);
          const result = np.lexsort([key1, key2]);

          const pyResult = runNumPy(`
key1 = np.array(${JSON.stringify(key1Data)}, dtype=${npDtype})
key2 = np.array(${JSON.stringify(key2Data)}, dtype=${npDtype})
result = np.lexsort((key1, key2))
`);
          expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
        });
      }
    });

    // ============================================================
    // searchsorted() — all dtypes, left and right
    // ============================================================
    describe('searchsorted()', () => {
      const sortedData = [1, 2, 3, 4, 5];
      const needles = [2, 3, 6];

      for (const dtype of ALL_DTYPES) {
        it(`left ${dtype}`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(sortedData, dtype as any);
          const v = np.array(needles, dtype as any);
          const result = np.searchsorted(a, v, 'left');

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(sortedData)}, dtype=${npDtype})
v = np.array(${JSON.stringify(needles)}, dtype=${npDtype})
result = np.searchsorted(a, v, side='left')
`);
          expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
        });

        it(`right ${dtype}`, () => {
          const npDtype = NP_DTYPE[dtype]!;
          const a = np.array(sortedData, dtype as any);
          const v = np.array(needles, dtype as any);
          const result = np.searchsorted(a, v, 'right');

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(sortedData)}, dtype=${npDtype})
v = np.array(${JSON.stringify(needles)}, dtype=${npDtype})
result = np.searchsorted(a, v, side='right')
`);
          expect(arraysClose((result as any).toArray(), pyResult.value)).toBe(true);
        });
      }

      it('duplicates left vs right', () => {
        const a = np.array([1, 2, 2, 3]);
        const v = np.array([2]);
        const resultLeft = np.searchsorted(a, v, 'left');
        const resultRight = np.searchsorted(a, v, 'right');

        const npLeft = runNumPy(`
a = np.array([1, 2, 2, 3])
v = np.array([2])
result = np.searchsorted(a, v, side='left')
`);
        const npRight = runNumPy(`
a = np.array([1, 2, 2, 3])
v = np.array([2])
result = np.searchsorted(a, v, side='right')
`);
        expect(arraysClose((resultLeft as any).toArray(), npLeft.value)).toBe(true);
        expect(arraysClose((resultRight as any).toArray(), npRight.value)).toBe(true);
      });
    });
  });
}

// ============================================================
// Non-dtype-sweep tests (complex, edge cases)
// ============================================================
describe('NumPy Validation: Sorting Edge Cases', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  afterEach(() => {
    wasmConfig.thresholdMultiplier = 1;
  });

  describe('sort_complex()', () => {
    it('validates sort_complex() on real array', () => {
      const arr = np.array([3, 1, 4, 2]);
      const result = np.sort_complex(arr);

      const npResult = runNumPy(`
arr = np.array([3, 1, 4, 2])
result = np.sort_complex(arr).real
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});

// ============================================================
// Searching functions (these don't need WASM dtype sweeps)
// ============================================================
describe('NumPy Validation: Searching Functions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('nonzero()', () => {
    it('validates nonzero() on 1D array', () => {
      const arr = np.array([1, 0, 2, 0, 3]);
      const result = np.nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = [x.tolist() for x in np.nonzero(arr)]
`);
      expect(result[0]!.toArray()).toEqual(npResult.value[0]);
    });

    it('validates nonzero() on 2D array', () => {
      const arr = np.array([
        [1, 0],
        [0, 2],
      ]);
      const result = np.nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = [x.tolist() for x in np.nonzero(arr)]
`);
      expect(result[0]!.toArray()).toEqual(npResult.value[0]);
      expect(result[1]!.toArray()).toEqual(npResult.value[1]);
    });
  });

  describe('argwhere()', () => {
    it('validates argwhere() on 1D array', () => {
      const arr = np.array([1, 0, 2, 0, 3]);
      const result = np.argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = np.argwhere(arr)
`);
      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates argwhere() on 2D array', () => {
      const arr = np.array([
        [1, 0],
        [0, 2],
      ]);
      const result = np.argwhere(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = np.argwhere(arr)
`);
      expect((result as any).shape).toEqual(npResult.shape);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('flatnonzero()', () => {
    it('validates flatnonzero() on 2D array', () => {
      const arr = np.array([
        [1, 0],
        [0, 2],
      ]);
      const result = np.flatnonzero(arr);

      const npResult = runNumPy(`
arr = np.array([[1, 0], [0, 2]])
result = np.flatnonzero(arr)
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('where()', () => {
    it('validates where() with condition only (like nonzero)', () => {
      const arr = np.array([1, 0, 2]);
      const result = np.where(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2])
result = [x.tolist() for x in np.where(arr)]
`);
      expect((result as any[])[0].toArray()).toEqual(npResult.value[0]);
    });

    it('validates where() with x and y', () => {
      const condition = np.array([1, 0, 1, 0]);
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([10, 20, 30, 40]);
      const result = np.where(condition, x, y);

      const npResult = runNumPy(`
condition = np.array([1, 0, 1, 0])
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
result = np.where(condition, x, y)
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('extract()', () => {
    it('validates extract() with condition', () => {
      const arr = np.array([1, 2, 3, 4, 5]);
      const condition = np.array([1, 0, 1, 0, 1]);
      const result = np.extract(condition, arr);

      const npResult = runNumPy(`
arr = np.array([1, 2, 3, 4, 5])
condition = np.array([1, 0, 1, 0, 1])
result = np.extract(condition, arr)
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });

  describe('count_nonzero()', () => {
    it('validates count_nonzero() on full array', () => {
      const arr = np.array([1, 0, 2, 0, 3]);
      const result = np.count_nonzero(arr);

      const npResult = runNumPy(`
arr = np.array([1, 0, 2, 0, 3])
result = np.count_nonzero(arr)
`);
      expect(result).toBe(npResult.value);
    });

    it('validates count_nonzero() along axis=0', () => {
      const arr = np.array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = np.count_nonzero(arr, 0);

      const npResult = runNumPy(`
arr = np.array([[1, 0, 2], [0, 3, 0]])
result = np.count_nonzero(arr, axis=0)
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });

    it('validates count_nonzero() along axis=1', () => {
      const arr = np.array([
        [1, 0, 2],
        [0, 3, 0],
      ]);
      const result = np.count_nonzero(arr, 1);

      const npResult = runNumPy(`
arr = np.array([[1, 0, 2], [0, 3, 0]])
result = np.count_nonzero(arr, axis=1)
`);
      expect(arraysClose((result as any).toArray(), npResult.value)).toBe(true);
    });
  });
});
