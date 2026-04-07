/**
 * DType Sweep: Unary element-wise math functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isInt } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Unary math', () => {
  const unaryOps: { name: string; fn: (a: any) => any }[] = [
    { name: 'absolute', fn: np.absolute },
    { name: 'negative', fn: np.negative },
    { name: 'positive', fn: np.positive },
    { name: 'sign', fn: np.sign },
    { name: 'square', fn: np.square },
    { name: 'sqrt', fn: np.sqrt },
    { name: 'cbrt', fn: np.cbrt },
    { name: 'reciprocal', fn: np.reciprocal },
    { name: 'exp', fn: np.exp },
    { name: 'exp2', fn: np.exp2 },
    { name: 'expm1', fn: np.expm1 },
    { name: 'log', fn: np.log },
    { name: 'log2', fn: np.log2 },
    { name: 'log10', fn: np.log10 },
    { name: 'log1p', fn: np.log1p },
    { name: 'sin', fn: np.sin },
    { name: 'cos', fn: np.cos },
    { name: 'tan', fn: np.tan },
    { name: 'arcsin', fn: np.arcsin },
    { name: 'arccos', fn: np.arccos },
    { name: 'arctan', fn: np.arctan },
    { name: 'sinh', fn: np.sinh },
    { name: 'cosh', fn: np.cosh },
    { name: 'tanh', fn: np.tanh },
    { name: 'arcsinh', fn: np.arcsinh },
    { name: 'arccosh', fn: np.arccosh },
    { name: 'arctanh', fn: np.arctanh },
    { name: 'ceil', fn: np.ceil },
    { name: 'floor', fn: np.floor },
    { name: 'rint', fn: np.rint },
    { name: 'trunc', fn: np.trunc },
    { name: 'fix', fn: np.fix },
    { name: 'around', fn: np.around },
    { name: 'degrees', fn: np.degrees },
    { name: 'radians', fn: np.radians },
    { name: 'deg2rad', fn: np.deg2rad },
    { name: 'rad2deg', fn: np.rad2deg },
    { name: 'sinc', fn: np.sinc },
    { name: 'fabs', fn: np.fabs },
    { name: 'signbit', fn: np.signbit },
    { name: 'i0', fn: np.i0 },
    { name: 'spacing', fn: np.spacing },
    { name: 'nan_to_num', fn: np.nan_to_num },
  ];

  for (const { name, fn } of unaryOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const needsDomain = ['arcsin', 'arccos', 'arctanh'].includes(name);
          const needsPositive = ['arccosh', 'log', 'log2', 'log10', 'log1p', 'sqrt', 'i0'].includes(name);
          const data =
            dtype === 'bool'
              ? [1, 0, 1, 0]
              : needsDomain
                ? isInt(dtype) ? [0, 1, 0, 1] : [0.1, 0.5, 0.9, 0.3]
                : needsPositive
                  ? [1, 2, 3, 4]
                  : [1, 2, 3, 4];
          const a = array(data, dtype);
          const jsResult = fn(a);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.${name}(a).astype(np.float64)
          `);
          const rtol = dtype === 'float32' ? 1e-2 : 1e-3;
          const atol = dtype === 'float32' ? 1e-5 : 1e-8;
          expect(arraysClose(jsResult.toArray(), pyResult.value, rtol, atol)).toBe(true);
        });
      }
    });
  }
});

describe('DType Sweep: Binary-like unary', () => {
  for (const dtype of ALL_DTYPES) {
    it(`nextafter ${dtype}`, () => {
      const data1 = dtype === 'bool' ? [1, 0] : [1, 2];
      const data2 = dtype === 'bool' ? [0, 1] : [2, 3];
      const a = array(data1, dtype);
      const b = array(data2, dtype);
      const jsResult = np.nextafter(a, b);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.nextafter(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-3)).toBe(true);
    });
  }
});
