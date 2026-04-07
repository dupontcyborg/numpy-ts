/**
 * DType Sweep: Logical and bitwise operations, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Logical', () => {
  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : isComplex(dtype) ? [1, 0, 1, 0] : [1, 1, 0, 0];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 1, 0, 0] : [1, 0, 1, 0];

    it(`logical_and ${dtype}`, () => {
      const jsResult = np.logical_and(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_and(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_or ${dtype}`, () => {
      const jsResult = np.logical_or(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_or(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_not ${dtype}`, () => {
      const jsResult = np.logical_not(array(data1, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
result = np.logical_not(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_xor ${dtype}`, () => {
      const jsResult = np.logical_xor(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_xor(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isnan ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isnan(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isnan(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isinf(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isinf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isfinite ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isfinite(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isfinite(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isneginf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isneginf(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isneginf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isposinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isposinf(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isposinf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Bitwise', () => {
  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : [15, 7, 3, 1];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [9, 6, 5, 3];

    it(`bitwise_and ${dtype}`, () => {
      const jsResult = np.bitwise_and(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.bitwise_and(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_or ${dtype}`, () => {
      const jsResult = np.bitwise_or(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.bitwise_or(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_xor ${dtype}`, () => {
      const jsResult = np.bitwise_xor(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.bitwise_xor(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_not ${dtype}`, () => {
      const jsResult = np.bitwise_not(array(data1, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
result = np.bitwise_not(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`invert ${dtype}`, () => {
      const jsResult = np.invert(array(data1, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
result = np.invert(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`left_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [1, 1, 0] : [1, 1, 1];
      const jsResult = np.left_shift(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})
result = np.left_shift(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`right_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [8, 16, 32];
      const d2 = dtype === 'bool' ? [1, 0, 0] : [1, 1, 1];
      const jsResult = np.right_shift(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})
result = np.right_shift(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_count ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [7, 15, 3];
      const jsResult = np.bitwise_count(array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.bitwise_count(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Packbits/Unpackbits', () => {
  for (const dtype of ALL_DTYPES) {
    it(`packbits ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0, 1, 0] : [1, 0, 1, 0, 1, 0, 1, 0];
      const jsResult = np.packbits(array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.packbits(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`unpackbits ${dtype}`, () => {
      const data = [170]; // 10101010 in binary
      const jsResult = np.unpackbits(array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.unpackbits(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
