/**
 * DType Sweep: Binary element-wise arithmetic functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Binary arithmetic', () => {
  const binaryOps: { name: string; fn: (a: any, b: any) => any }[] = [
    { name: 'add', fn: np.add },
    { name: 'subtract', fn: np.subtract },
    { name: 'multiply', fn: np.multiply },
    { name: 'divide', fn: np.divide },
    { name: 'power', fn: np.power },
    { name: 'maximum', fn: np.maximum },
    { name: 'minimum', fn: np.minimum },
    { name: 'mod', fn: np.mod },
    { name: 'floor_divide', fn: np.floor_divide },
    { name: 'copysign', fn: np.copysign },
    { name: 'hypot', fn: np.hypot },
    { name: 'arctan2', fn: np.arctan2 },
    { name: 'logaddexp', fn: np.logaddexp },
    { name: 'logaddexp2', fn: np.logaddexp2 },
    { name: 'fmax', fn: np.fmax },
    { name: 'fmin', fn: np.fmin },
    { name: 'remainder', fn: np.remainder },
    { name: 'float_power', fn: np.float_power },
    { name: 'heaviside', fn: np.heaviside },
    { name: 'gcd', fn: np.gcd },
    { name: 'lcm', fn: np.lcm },
  ];

  for (const { name, fn } of binaryOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 1, 0, 1] : [6, 7, 8, 9];
          const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
          const a = array(data1, dtype);
          const b = array(data2, dtype);
          const jsResult = fn(a, b);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.${name}(a, b).astype(np.float64)
          `);
          expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-3)).toBe(true);
        });
      }
    });
  }
});

describe('DType Sweep: Divmod', () => {
  for (const dtype of ALL_DTYPES) {
    it(`divmod ${dtype}`, () => {
      const data1 = dtype === 'bool' ? [1, 1, 0] : [7, 8, 9];
      const data2 = dtype === 'bool' ? [1, 1, 1] : [2, 3, 4];
      const [q, r] = np.divmod(array(data1, dtype), array(data2, dtype)) as [any, any];
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
q, r = np.divmod(a, b)
result = q.astype(np.float64)
      `);
      expect(arraysClose(q.toArray(), pyResult.value, 1e-3)).toBe(true);
    });
  }
});
