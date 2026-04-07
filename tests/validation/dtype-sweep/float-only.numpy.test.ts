/**
 * DType Sweep: Indexing, set operations, statistics, and misc functions.
 * All tested across ALL dtypes, value-producing tests validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Indexing', () => {
  for (const dtype of ALL_DTYPES) {
    it(`take ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [10, 20, 30, 40, 50];
      const jsResult = np.take(array(data, dtype), [0, 2, 4]);
      const pyResult = runNumPy(
        `result = np.take(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), [0,2,4]).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`take_along_axis ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
      const indices = [2, 0, 1];
      const jsResult = np.take_along_axis(array(data, dtype), array(indices, 'int32'), 0);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
idx = np.array(${JSON.stringify(indices)}, dtype=np.intp)
result = np.take_along_axis(a, idx, axis=0).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`put ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [10, 20, 30];
      const vals = dtype === 'bool' ? [0, 1] : [99, 88];
      const a = array([...data], dtype);
      np.put(a, [0, 2], array(vals, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
np.put(a, [0, 2], np.array(${JSON.stringify(vals)}, dtype=${npDtype(dtype)}))
result = a.astype(np.float64)
      `);
      expect(arraysClose(a.toArray(), pyResult.value)).toBe(true);
    });

    it(`put_along_axis ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0], [0, 1]] : [[10, 20], [30, 40]];
      const vals = dtype === 'bool' ? [[1], [0]] : [[99], [88]];
      const a = array(data, dtype);
      np.put_along_axis(a, array([[0], [1]], 'int32'), array(vals, dtype), 1);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
np.put_along_axis(a, np.array([[0],[1]], dtype=np.intp), np.array(${JSON.stringify(vals)}, dtype=${npDtype(dtype)}), axis=1)
result = a.astype(np.float64)
      `);
      expect(arraysClose(a.toArray(), pyResult.value)).toBe(true);
    });

    it(`choose ${dtype}`, () => {
      const choices = [
        array(dtype === 'bool' ? [0, 0, 0] : [0, 1, 2], dtype),
        array(dtype === 'bool' ? [1, 1, 1] : [10, 11, 12], dtype),
      ];
      const jsResult = np.choose(array([0, 1, 0], 'int32'), choices);
      expect(jsResult.shape).toEqual([3]);
    });

    it(`diag ${dtype}`, () => {
      const jsResult = np.diag(
        array(
          dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]],
          dtype
        )
      );
      const pyResult = runNumPy(
        `result = np.diag(np.array(${dtype === 'bool' ? '[[True,False],[False,True]]' : '[[1,2],[3,4]]'}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`where ${dtype}`, () => {
      const cond = array([1, 0, 1, 0], 'bool');
      const data1 = dtype === 'bool' ? [1, 1, 1, 1] : [1, 2, 3, 4];
      const data2 = dtype === 'bool' ? [0, 0, 0, 0] : [5, 6, 7, 8];
      const jsResult = np.where(cond, array(data1, dtype), array(data2, dtype));
      const castSuffix = isComplex(dtype) ? '.astype(np.complex128)' : '.astype(np.float64)';
      const pyResult = runNumPy(`
cond = np.array([True, False, True, False])
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.where(cond, a, b)${castSuffix}
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`nonzero ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [0, 1, 0, 2];
      const jsResult = np.nonzero(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.nonzero(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))[0]`
      );
      expect(arraysClose(jsResult[0]!.toArray(), pyResult.value)).toBe(true);
    });

    it(`extract ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
      const a = array(data, dtype);
      const cond = array([1, 0, 1, 0, 1], 'bool');
      const jsResult = np.extract(cond, a);
      const pyResult = runNumPy(
        `result = np.extract(np.array([True,False,True,False,True]), np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`flatnonzero ${dtype}`, () => {
      const data = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
      const jsResult = np.flatnonzero(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.flatnonzero(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`argwhere ${dtype}`, () => {
      const data = dtype === 'bool' ? [0, 1, 0, 1, 0] : [0, 1, 0, 2, 0];
      const jsResult = np.argwhere(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.argwhere(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Set operations', () => {
  for (const dtype of ALL_DTYPES) {
    // Skip int64/uint64 — BigInt results can't be compared via arraysClose
    if (dtype === 'int64' || dtype === 'uint64') continue;

    const d1 = dtype === 'bool' ? [1, 0, 1, 0] : [3, 1, 2, 1, 3, 2];
    const d2 = dtype === 'bool' ? [0, 1] : [3, 4, 5, 6];
    const d3 = dtype === 'bool' ? [1, 0] : [1, 2, 3, 4];

    it(`unique ${dtype}`, () => {
      const jsResult = np.unique(array(d1, dtype));
      const pyResult = runNumPy(
        `result = np.unique(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`intersect1d ${dtype}`, () => {
      const jsResult = np.intersect1d(array(d3, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.intersect1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`union1d ${dtype}`, () => {
      const jsResult = np.union1d(array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype), array(dtype === 'bool' ? [0, 1] : [3, 4, 5], dtype));
      const pyResult = runNumPy(
        `result = np.union1d(np.array(${JSON.stringify(dtype === 'bool' ? [1, 0] : [1, 2, 3])}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dtype === 'bool' ? [0, 1] : [3, 4, 5])}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`setdiff1d ${dtype}`, () => {
      const jsResult = np.setdiff1d(array(d3, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.setdiff1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`setxor1d ${dtype}`, () => {
      const jsResult = np.setxor1d(array(d3, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.setxor1d(np.array(${JSON.stringify(d3)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`in1d ${dtype}`, () => {
      const jsResult = np.in1d(array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype), array(dtype === 'bool' ? [1] : [2, 3, 4], dtype));
      const pyResult = runNumPy(
        `result = np.in1d(np.array(${JSON.stringify(dtype === 'bool' ? [1, 0] : [1, 2, 3])}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dtype === 'bool' ? [1] : [2, 3, 4])}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isin ${dtype}`, () => {
      const jsResult = np.isin(array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype), array(dtype === 'bool' ? [1] : [2, 3, 4], dtype));
      const pyResult = runNumPy(
        `result = np.isin(np.array(${JSON.stringify(dtype === 'bool' ? [1, 0] : [1, 2, 3])}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dtype === 'bool' ? [1] : [2, 3, 4])}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Statistics', () => {
  for (const dtype of ALL_DTYPES) {
    it(`histogram ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
      const [hist] = np.histogram(array(data, dtype)) as [any, any];
      const pyResult = runNumPy(`
h, e = np.histogram(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = h.astype(np.float64)
      `);
      expect(arraysClose(hist.toArray(), pyResult.value)).toBe(true);
    });

    it(`corrcoef ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0, 1], [0, 1, 0]] : [[1, 2, 3], [4, 5, 6]];
      const jsResult = np.corrcoef(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.corrcoef(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`cov ${dtype}`, () => {
      const data = dtype === 'bool' ? [[1, 0, 1], [0, 1, 0]] : [[1, 2, 3], [4, 5, 6]];
      const jsResult = np.cov(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.cov(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`bincount ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : [0, 1, 1, 2, 3, 3, 3];
      const jsResult = np.bincount(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.bincount(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`digitize ${dtype}`, () => {
      const data = dtype === 'bool' ? [0, 1, 0, 1] : [1, 3, 5, 7];
      const bins = dtype === 'bool' ? [0, 1] : [2, 4, 6];
      const jsResult = np.digitize(array(data, dtype), array(bins, dtype));
      const pyResult = runNumPy(
        `result = np.digitize(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(bins)}, dtype=${npDtype(dtype)}))`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`correlate ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [0, 1] : [0, 1];
      const jsResult = np.correlate(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.correlate(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`convolve ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [0, 1] : [0, 1];
      const jsResult = np.convolve(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.convolve(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`trapezoid ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
      const jsResult = np.trapezoid(array(data, dtype));
      const pyResult = runNumPy(
        `result = float(np.trapezoid(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`diff ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11];
      const jsResult = np.diff(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.diff(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`gradient ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11];
      const jsResult = np.gradient(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.gradient(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`interp ${dtype}`, () => {
      const jsResult = np.interp(
        array(dtype === 'bool' ? [0, 1] : [1.5, 2.5, 3.5], dtype),
        array(dtype === 'bool' ? [0, 1] : [1, 2, 3, 4], dtype),
        array(dtype === 'bool' ? [0, 1] : [10, 20, 30, 40], dtype)
      );
      const xp = dtype === 'bool' ? [0, 1] : [1, 2, 3, 4];
      const fp = dtype === 'bool' ? [0, 1] : [10, 20, 30, 40];
      const x = dtype === 'bool' ? [0, 1] : [1.5, 2.5, 3.5];
      const pyResult = runNumPy(
        `result = np.interp(${JSON.stringify(x)}, ${JSON.stringify(xp)}, ${JSON.stringify(fp)})`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });
  }
});

describe('DType Sweep: Misc', () => {
  for (const dtype of ALL_DTYPES) {
    it(`clip ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 5, 10];
      const jsResult = np.clip(array(data, dtype), dtype === 'bool' ? 0 : 2, dtype === 'bool' ? 1 : 8);
      const pyResult = runNumPy(
        `result = np.clip(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), ${dtype === 'bool' ? 0 : 2}, ${dtype === 'bool' ? 1 : 8}).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`fmod ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 1, 0] : [6, 7, 8, 9];
      const d2 = dtype === 'bool' ? [1, 1, 1] : [4, 3, 5, 2];
      const jsResult = np.fmod(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(
        `result = np.fmod(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`nan_to_num ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.nan_to_num(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.nan_to_num(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`ldexp ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.ldexp(array(data, dtype), array([1, 2, 3], 'int32'));
      const pyResult = runNumPy(
        `result = np.ldexp(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), np.array([1,2,3], dtype=np.int32)).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`frexp ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 4];
      const [m] = np.frexp(array(data, dtype)) as [any, any];
      const pyResult = runNumPy(`
m, e = np.frexp(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = m.astype(np.float64)
      `);
      expect(arraysClose(m.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    if (!isComplex(dtype)) {
      it(`array_equiv ${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
        const a = array(data, dtype);
        const b = array(data, dtype);
        const jsResult = np.array_equiv(a, b);
        const pyResult = runNumPy(
          `result = bool(np.array_equiv(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`
        );
        expect(Boolean(jsResult)).toBe(Boolean(pyResult.value));
      });
    }
  }
});

describe('DType Sweep: Complex functions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`real ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.real(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.real(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`imag ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.imag(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.imag(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`conj ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.conj(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.conj(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`angle ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.angle(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.angle(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });
  }
});
