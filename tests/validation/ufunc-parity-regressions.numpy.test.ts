/**
 * Regressions for binary ufuncs that skipped broadcasting, where() ignoring
 * y's dtype, std/var along the only axis of a 1-D array, floor and truncated
 * remainders with huge or infinite divisors, and nan_to_num on narrow floats.
 * Small cases take the JS paths; the 64-element cases are large enough to reach
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

/** Signed zeros are invisible to arraysClose, so compare signbit separately. */
function expectSignbits(result: np.NDArray, py: string): void {
  const expected = runNumPy(`${py}\nresult = np.signbit(result)`);
  const flat = (result.flatten().toArray() as number[]).map((v) => Object.is(v, -0) || v < 0);
  expect(flat).toEqual((expected.value as boolean[]).flat(Infinity));
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

  describe('std and var along the only axis of a 1-D array', () => {
    for (const axis of [0, -1]) {
      for (const ddof of [0, 1]) {
        it(`axis=${axis} ddof=${ddof}`, () => {
          const expectedStd = runNumPy(
            `result = np.std(np.array([7, 9, 255]), axis=${axis}, ddof=${ddof})`,
          );
          const expectedVar = runNumPy(
            `result = np.var(np.array([7, 9, 255]), axis=${axis}, ddof=${ddof})`,
          );
          expect(np.std(array([7, 9, 255]), axis, ddof)).toBeCloseTo(
            expectedStd.value as number,
            10,
          );
          expect(array([7, 9, 255]).var(axis, ddof)).toBeCloseTo(expectedVar.value as number, 8);
        });
      }
    }

    it('rejects an out-of-bounds axis', () => {
      expect(() => np.std(array([7, 9, 255]), 1)).toThrow(/out of bounds/);
    });
  });

  describe('mod, fmod and divmod with huge or infinite divisors', () => {
    const x = [5, -5, 13, -13, 0, 7];
    const d = [1e17, 1e300, Infinity, -Infinity, -Infinity, -3];

    for (const n of [x.length, 64]) {
      const path = n === x.length ? 'JS' : 'WASM';
      const xs = tile(x, n);
      const ds = tile(d, n);
      // JSON turns +-Infinity into null, so encode the divisors by hand.
      const dsPy = `[${ds.map((v) => (v === Infinity ? 'np.inf' : v === -Infinity ? '-np.inf' : String(v))).join(', ')}]`;

      it(`mod, array divisor (${path})`, () => {
        const result = np.mod(array(xs), array(ds));
        const code = `result = np.mod(np.array(${JSON.stringify(xs)}, dtype=float), np.array(${dsPy}))`;
        expectMatches(result, code);
        expectSignbits(result, code);
      });

      it(`fmod, array divisor (${path})`, () => {
        const result = np.fmod(array(xs), array(ds));
        const code = `result = np.fmod(np.array(${JSON.stringify(xs)}, dtype=float), np.array(${dsPy}))`;
        expectMatches(result, code);
        expectSignbits(result, code);
      });

      it(`divmod remainder, array divisor (${path})`, () => {
        const [, rem] = np.divmod(array(xs), array(ds));
        expectMatches(
          rem,
          `result = np.divmod(np.array(${JSON.stringify(xs)}, dtype=float), np.array(${dsPy}))[1]`,
        );
      });

      for (const s of [Infinity, -Infinity, 1e300]) {
        const sPy = s === Infinity ? 'np.inf' : s === -Infinity ? '-np.inf' : String(s);
        for (const [fn, f] of Object.entries({ mod: np.mod, fmod: np.fmod })) {
          it(`${fn}, scalar divisor ${s} (${path})`, () => {
            const code = `result = np.${fn}(np.array(${JSON.stringify(xs)}, dtype=float), ${sPy})`;
            expectMatches(f(array(xs), s), code);
          });
        }
        it(`mod float32, scalar divisor ${s} (${path})`, () => {
          const code = `result = np.mod(np.array(${JSON.stringify(xs)}, dtype=np.float32), np.float32(${sPy}))`;
          expectMatches(np.mod(array(xs, 'float32'), s), code);
        });
      }
    }

    it('mod gives a zero the divisor sign', () => {
      const code =
        'result = np.mod(np.array([3.0, -3.0, 0.0, -0.0]), np.array([-3.0, 3.0, -3.0, 3.0]))';
      expectSignbits(np.mod(array([3, -3, 0, -0]), array([-3, 3, -3, 3])), code);
    });
  });

  describe('nan_to_num defaults to the dtype maximum', () => {
    for (const dtype of ['float16', 'float32', 'float64'] as const) {
      it(dtype, () => {
        expectMatches(
          np.nan_to_num(array([Infinity, -Infinity, NaN, 1.5], dtype)),
          `result = np.nan_to_num(np.array([np.inf, -np.inf, np.nan, 1.5], dtype=np.${dtype}))`,
        );
      });
    }

    it('complex64', () => {
      const z = array([new np.Complex(Infinity, -Infinity)], 'complex64');
      expectMatches(
        np.nan_to_num(z),
        'result = np.nan_to_num(np.array([complex(np.inf, -np.inf)], dtype=np.complex64))',
      );
    });

    it('reads a sliced view from its offset', () => {
      const a = array([NaN, 1, Infinity, 4, NaN]);
      expectMatches(
        np.nan_to_num(a.slice('1:')),
        'result = np.nan_to_num(np.array([np.nan, 1.0, np.inf, 4.0, np.nan])[1:])',
      );
      expectMatches(
        np.nan_to_num(a.slice('::2')),
        'result = np.nan_to_num(np.array([np.nan, 1.0, np.inf, 4.0, np.nan])[::2])',
      );
    });
  });
});
