/**
 * Regressions for binary ufuncs that skipped broadcasting one or both operands.
 * Small cases take the JS paths; the 64-element case is large enough to reach
 * the WASM kernels.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

const { array } = np;

/** Repeat a list until it has n elements, so a case reaches the WASM threshold. */
function tile(values: number[], n: number): number[] {
  return Array.from({ length: n }, (_, i) => values[i % values.length]!);
}

function expectMatches(result: np.NDArray, py: string): void {
  const expected = runNumPy(py);
  expect(result.shape).toEqual(expected.shape);
  expect(result.dtype).toBe(expected.dtype);
  expect(arraysClose(result.toArray(), expected.value)).toBe(true);
}

describe('NumPy Validation: ufunc parity regressions', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('binary ufuncs broadcast both operands', () => {
    const shapes: [string, unknown, unknown][] = [
      ['(4,) with (1,)', [3, 6, 9, 12], [4]],
      ['(1,) with (4,)', [4], [3, 6, 9, 12]],
      ['(2,1) with (3,)', [[1], [2]], [3, 4, 5]],
      ['0-d with (3,)', 2, [1, 2, 3]],
      ['(3,) with (1,1)', [1, 2, 3], [[2]]],
    ];
    const ops = {
      hypot: np.hypot,
      arctan2: np.arctan2,
      logaddexp: np.logaddexp,
      logaddexp2: np.logaddexp2,
      float_power: np.float_power,
    };

    for (const [op, fn] of Object.entries(ops)) {
      for (const [label, a, b] of shapes) {
        it(`${op} ${label}`, () => {
          const result = fn(array(a as number[]), array(b as number[]));
          expectMatches(
            result,
            `result = np.${op}(np.array(${JSON.stringify(a)}, dtype=float), np.array(${JSON.stringify(b)}, dtype=float))`,
          );
        });
      }
    }

    for (const [label, a, b] of shapes) {
      it(`ldexp ${label}`, () => {
        const result = np.ldexp(array(a as number[]), array(b as number[], 'int32'));
        expectMatches(
          result,
          `result = np.ldexp(np.array(${JSON.stringify(a)}, dtype=float), np.array(${JSON.stringify(b)}, dtype=np.int32))`,
        );
      });
    }

    it('heaviside broadcasts a column of fill values across rows', () => {
      const x = [
        [-1, 0, 2],
        [0, 0, -3],
      ];
      const h = [[7], [9]];
      expectMatches(
        np.heaviside(array(x), array(h)),
        `result = np.heaviside(np.array(${JSON.stringify(x)}, dtype=float), np.array(${JSON.stringify(h)}, dtype=float))`,
      );
    });

    it('heaviside broadcasts a smaller first operand', () => {
      expectMatches(
        np.heaviside(array([0]), array([0.5, 2])),
        'result = np.heaviside(np.array([0.0]), np.array([0.5, 2.0]))',
      );
    });

    it('hypot broadcasts at WASM sizes', () => {
      const a = tile([3, 6, 9, 12], 64).map((v) => [v]);
      expectMatches(
        np.hypot(array(a), array([4, 8])),
        `result = np.hypot(np.array(${JSON.stringify(a)}, dtype=float), np.array([4.0, 8.0]))`,
      );
    });

    it('rejects incompatible shapes', () => {
      expect(() => np.hypot(array([1, 2, 3]), array([1, 2]))).toThrow(/could not be broadcast/);
    });
  });
});
