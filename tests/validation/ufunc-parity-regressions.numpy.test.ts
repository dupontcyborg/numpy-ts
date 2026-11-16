/**
 * Regressions for binary ufuncs that skipped broadcasting one or both operands,
 * and where() ignoring y's dtype. Small cases take the JS paths; the 64-element
 * case is large enough to reach the WASM kernels.
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

  describe('where promotes x and y together', () => {
    const cases: [string, string, string, string, string][] = [
      ['int32 / float64', '[1, 2]', 'int32', '[0.5, 2.5]', 'float64'],
      ['uint8 / uint32', '[0, 1]', 'uint8', '[256, 32767]', 'uint32'],
      ['float64 / int64', '[1.5, 2.5]', 'float64', '[2, 3]', 'int64'],
      ['bool / int8', '[1, 0]', 'bool', '[-5, 7]', 'int8'],
    ];
    for (const [label, x, xd, y, yd] of cases) {
      it(label, () => {
        const xs = JSON.parse(x) as number[];
        const ys = JSON.parse(y) as number[];
        const xArr = xd === 'bool' ? array(xs.map(Boolean)) : array(xs, xd as np.DType);
        const yArr = yd === 'int64' ? array(ys.map(BigInt), 'int64') : array(ys, yd as np.DType);
        const result = np.where(array([true, false]), xArr, yArr) as np.NDArray;
        expectMatches(
          result,
          `result = np.where(np.array([True, False]), np.array(${x}, dtype=np.${xd === 'bool' ? 'bool_' : xd}), np.array(${y}, dtype=np.${yd}))`,
        );
      });
    }

    it('promotes while broadcasting a 0-d operand', () => {
      const result = np.where(
        array([true, false, true]),
        array(1, 'int16'),
        array([7.5, 8.5, 9.5], 'float32'),
      ) as np.NDArray;
      expectMatches(
        result,
        'result = np.where(np.array([True, False, True]), np.array(1, dtype=np.int16), np.array([7.5, 8.5, 9.5], dtype=np.float32))',
      );
    });

    // A JS number is a weak scalar: it adopts the array's dtype, wrapping when
    // an integer is out of range, unless it is fractional and the array is not.
    const scalarCases: [string, () => unknown, string][] = [
      [
        'int32 array, integer scalar',
        () => np.where(array([true, false]), array([1, 2], 'int32'), 0),
        'np.array([1, 2], dtype=np.int32), 0',
      ],
      [
        'integer scalar, int32 array',
        () => np.where(array([true, false]), 0, array([1, 2], 'int32')),
        '0, np.array([1, 2], dtype=np.int32)',
      ],
      [
        'float32 array, fractional scalar',
        () => np.where(array([true, false]), array([1, 2], 'float32'), 0.5),
        'np.array([1, 2], dtype=np.float32), 0.5',
      ],
      [
        'int32 array, fractional scalar',
        () => np.where(array([true, false]), array([1, 2], 'int32'), 0.5),
        'np.array([1, 2], dtype=np.int32), 0.5',
      ],
      [
        'int8 array, boolean scalar',
        () => np.where(array([true, false]), array([1, 2], 'int8'), true),
        'np.array([1, 2], dtype=np.int8), True',
      ],
      [
        'uint8 array, out-of-range scalar wraps',
        () => np.where(array([true, false]), array([1, 2], 'uint8'), -1),
        'np.array([1, 2], dtype=np.uint8), -1',
      ],
    ];
    for (const [label, call, args] of scalarCases) {
      it(`treats a JS scalar as weak: ${label}`, () => {
        expectMatches(call() as np.NDArray, `result = np.where(np.array([True, False]), ${args})`);
      });
    }
  });
});
