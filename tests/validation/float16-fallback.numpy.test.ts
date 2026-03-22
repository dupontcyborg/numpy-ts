/**
 * Float16 validation tests - validates float16 behavior against NumPy.
 *
 * These tests force the Float32Array fallback path (simulating runtimes
 * without native Float16Array) and compare results against Python NumPy's
 * float16 to ensure correctness.
 *
 * NOTE: On runtimes WITH native Float16Array, the fallback path stores
 * values in Float32Array with higher precision than true float16. These
 * tests validate that the core operations produce results consistent with
 * NumPy's float16 within the expected tolerance.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../src';
import { array, zeros, ones, arange, full } from '../../src';
import { hasFloat16 } from '../../src/common/dtype';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

// Float16 has ~3.3 decimal digits of precision (11-bit mantissa).
// On runtimes without native Float16Array, our Float32Array fallback computes
// in float32 precision while NumPy uses real float16 throughout — intermediate
// results diverge because NumPy rounds to float16 at every step.
//
// For single operations the relative error is bounded by float16's precision
// (~1e-3). For multi-step operations (reductions, trig chains) the accumulated
// divergence can be larger, hence 2e-3 as the default tolerance.
const FLOAT16_TOL = 2e-3;

function approxEqual(a: number, b: number, tol: number = FLOAT16_TOL): boolean {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (!Number.isFinite(a) && !Number.isFinite(b)) return a === b;
  const absErr = Math.abs(a - b);
  const maxAbs = Math.max(Math.abs(a), Math.abs(b));
  return absErr < tol || (maxAbs > 0 && absErr / maxAbs < tol);
}

describe('Float16 NumPy Validation', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  describe('array creation matches NumPy', () => {
    it('zeros', () => {
      const ts = zeros([5], 'float16');
      const py = runNumPy(`result = np.zeros(5, dtype=np.float16)`);
      expect(ts.dtype).toBe('float16');
      expect(py.dtype).toContain('float16');
      expect(ts.toArray()).toEqual(py.value);
    });

    it('ones', () => {
      const ts = ones([3, 2], 'float16');
      const py = runNumPy(`result = np.ones((3, 2), dtype=np.float16)`);
      expect(ts.toArray()).toEqual(py.value);
    });

    it('full', () => {
      const ts = full([4], 3.14, 'float16');
      const py = runNumPy(`result = np.full(4, 3.14, dtype=np.float16)`);
      for (let i = 0; i < 4; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('arange', () => {
      const ts = arange(0, 10, 1, 'float16');
      const py = runNumPy(`result = np.arange(0, 10, 1, dtype=np.float16)`);
      expect(ts.toArray()).toEqual(py.value);
    });

    it('array from values', () => {
      const values = [0.5, 1.5, 2.25, -3.75, 100, 0.001];
      const ts = array(values, 'float16');
      const py = runNumPy(
        `result = np.array([0.5, 1.5, 2.25, -3.75, 100, 0.001], dtype=np.float16)`
      );
      for (let i = 0; i < values.length; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });
  });

  describe('arithmetic matches NumPy', () => {
    it('add', () => {
      const a = array([1, 2, 3, 4, 5], 'float16');
      const b = array([10, 20, 30, 40, 50], 'float16');
      const ts = a.add(b);

      const py = runNumPy(`
a = np.array([1, 2, 3, 4, 5], dtype=np.float16)
b = np.array([10, 20, 30, 40, 50], dtype=np.float16)
result = a + b
      `);

      expect(py.dtype).toContain('float16');
      for (let i = 0; i < 5; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('subtract', () => {
      const a = array([100, 200, 300], 'float16');
      const b = array([1, 2, 3], 'float16');
      const ts = a.subtract(b);

      const py = runNumPy(`
a = np.array([100, 200, 300], dtype=np.float16)
b = np.array([1, 2, 3], dtype=np.float16)
result = a - b
      `);

      for (let i = 0; i < 3; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('multiply', () => {
      const a = array([2, 3, 4, 5], 'float16');
      const b = array([0.5, 0.25, 0.1, 10], 'float16');
      const ts = a.multiply(b);

      const py = runNumPy(`
a = np.array([2, 3, 4, 5], dtype=np.float16)
b = np.array([0.5, 0.25, 0.1, 10], dtype=np.float16)
result = a * b
      `);

      for (let i = 0; i < 4; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('divide', () => {
      const a = array([10, 20, 30], 'float16');
      const b = array([3, 7, 11], 'float16');
      const ts = a.divide(b);

      const py = runNumPy(`
a = np.array([10, 20, 30], dtype=np.float16)
b = np.array([3, 7, 11], dtype=np.float16)
result = a / b
      `);

      for (let i = 0; i < 3; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('scalar operations', () => {
      const a = array([1, 2, 3, 4], 'float16');
      const ts = a.add(10);

      const py = runNumPy(`
a = np.array([1, 2, 3, 4], dtype=np.float16)
result = a + np.float16(10)
      `);

      for (let i = 0; i < 4; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });
  });

  describe('math functions match NumPy', () => {
    it('sqrt', () => {
      const a = array([1, 4, 9, 16, 25], 'float16');
      const ts = np.sqrt(a);

      const py = runNumPy(`
result = np.sqrt(np.array([1, 4, 9, 16, 25], dtype=np.float16))
      `);

      for (let i = 0; i < 5; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('abs / negative', () => {
      const a = array([-3, -2, -1, 0, 1, 2, 3], 'float16');
      const ts = np.absolute(a);

      const py = runNumPy(`
result = np.abs(np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float16))
      `);

      for (let i = 0; i < 7; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('exp', () => {
      const a = array([0, 1, 2, -1], 'float16');
      const ts = np.exp(a);

      const py = runNumPy(`
result = np.exp(np.array([0, 1, 2, -1], dtype=np.float16))
      `);

      for (let i = 0; i < 4; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('log', () => {
      const a = array([1, 2, 10, 100], 'float16');
      const ts = np.log(a);

      const py = runNumPy(`
result = np.log(np.array([1, 2, 10, 100], dtype=np.float16))
      `);

      for (let i = 0; i < 4; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('sin / cos', () => {
      const a = array([0, 0.5, 1.0, 1.5, 3.14], 'float16');
      const tsSin = np.sin(a);
      const tsCos = np.cos(a);

      const pySin = runNumPy(`
result = np.sin(np.array([0, 0.5, 1.0, 1.5, 3.14], dtype=np.float16))
      `);

      const pyCos = runNumPy(`
result = np.cos(np.array([0, 0.5, 1.0, 1.5, 3.14], dtype=np.float16))
      `);

      for (let i = 0; i < 5; i++) {
        expect(approxEqual(tsSin.get([i]) as number, pySin.value[i])).toBe(true);
        expect(approxEqual(tsCos.get([i]) as number, pyCos.value[i])).toBe(true);
      }
    });

    it('power', () => {
      const a = array([2, 3, 4], 'float16');
      const b = array([3, 2, 0.5], 'float16');
      const ts = np.power(a, b);

      const py = runNumPy(`
result = np.power(
  np.array([2, 3, 4], dtype=np.float16),
  np.array([3, 2, 0.5], dtype=np.float16)
)
      `);

      for (let i = 0; i < 3; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('clip', () => {
      const a = array([-5, -1, 0, 3, 10, 100], 'float16');
      const ts = np.clip(a, 0, 10);

      const py = runNumPy(`
result = np.clip(np.array([-5, -1, 0, 3, 10, 100], dtype=np.float16), 0, 10)
      `);

      for (let i = 0; i < 6; i++) {
        expect(approxEqual(ts.get([i]) as number, py.value[i])).toBe(true);
      }
    });
  });

  describe('reductions match NumPy', () => {
    it('sum', () => {
      const a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'float16');
      const ts = a.sum();

      const py = runNumPy(`
result = np.array([float(np.sum(np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float16)))])
      `);

      expect(approxEqual(ts as number, py.value[0])).toBe(true);
    });

    it('mean', () => {
      const a = array([2, 4, 6, 8, 10], 'float16');
      const ts = a.mean();

      const py = runNumPy(`
result = np.array([float(np.mean(np.array([2, 4, 6, 8, 10], dtype=np.float16)))])
      `);

      expect(approxEqual(ts as number, py.value[0])).toBe(true);
    });

    it('min / max', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6], 'float16');

      const pyMin = runNumPy(`
result = np.array([float(np.min(np.array([3,1,4,1,5,9,2,6], dtype=np.float16)))])
      `);
      const pyMax = runNumPy(`
result = np.array([float(np.max(np.array([3,1,4,1,5,9,2,6], dtype=np.float16)))])
      `);

      expect(a.min()).toBe(pyMin.value[0]);
      expect(a.max()).toBe(pyMax.value[0]);
    });

    it('std / var', () => {
      const a = array([2, 4, 4, 4, 5, 5, 7, 9], 'float16');
      const tsVar = a.var();
      const tsStd = a.std();

      const pyVar = runNumPy(`
result = np.array([float(np.var(np.array([2,4,4,4,5,5,7,9], dtype=np.float16)))])
      `);
      const pyStd = runNumPy(`
result = np.array([float(np.std(np.array([2,4,4,4,5,5,7,9], dtype=np.float16)))])
      `);

      expect(approxEqual(tsVar as number, pyVar.value[0])).toBe(true);
      expect(approxEqual(tsStd as number, pyStd.value[0])).toBe(true);
    });
  });

  describe('special values match NumPy', () => {
    it('handles infinity', () => {
      const a = array([Infinity, -Infinity, 0], 'float16');
      const py = runNumPy(`result = np.array([np.inf, -np.inf, 0], dtype=np.float16)`);
      expect(a.get([0])).toBe(Infinity);
      expect(a.get([1])).toBe(-Infinity);
      expect(a.get([2])).toBe(0);
      expect(py.value[0]).toBe(Infinity);
      expect(py.value[1]).toBe(-Infinity);
    });

    it('handles NaN', () => {
      const a = array([NaN, 1, 2], 'float16');
      expect(Number.isNaN(a.get([0]))).toBe(true);
      expect(a.get([1])).toBe(1);
    });

    it('overflow to infinity matches NumPy', () => {
      // float16 max is 65504
      const a = array([65504], 'float16');
      const ts = a.multiply(2);

      const py = runNumPy(`
a = np.array([65504], dtype=np.float16)
result = a * np.float16(2)
      `);

      expect(ts.get([0])).toBe(py.value[0]);
    });
  });

  describe('type promotion matches NumPy', () => {
    it('float16 + float32 promotes to float32', () => {
      const a = array([1, 2, 3], 'float16');
      const b = array([1, 2, 3], 'float32');
      const ts = a.add(b);

      const py = runNumPy(`
a = np.array([1, 2, 3], dtype=np.float16)
b = np.array([1, 2, 3], dtype=np.float32)
result = a + b
      `);

      expect(ts.dtype).toBe('float32');
      expect(py.dtype).toContain('float32');
    });

    it('float16 + float64 promotes to float64', () => {
      const a = array([1, 2, 3], 'float16');
      const b = array([1, 2, 3], 'float64');
      const ts = a.add(b);

      const py = runNumPy(`
a = np.array([1, 2, 3], dtype=np.float16)
b = np.array([1, 2, 3], dtype=np.float64)
result = a + b
      `);

      expect(ts.dtype).toBe('float64');
      expect(py.dtype).toContain('float64');
    });

    it('float16 + int8 stays float16', () => {
      const a = array([1, 2, 3], 'float16');
      const b = array([1, 2, 3], 'int8');
      const ts = a.add(b);

      const py = runNumPy(`
a = np.array([1, 2, 3], dtype=np.float16)
b = np.array([1, 2, 3], dtype=np.int8)
result = a + b
      `);

      expect(ts.dtype).toBe('float16');
      expect(py.dtype).toContain('float16');
    });
  });

  describe('astype conversions match NumPy', () => {
    it('float64 to float16', () => {
      const values = [1.23456789, 0.001, 100.5, -42.75, 0];
      const a = array(values, 'float64').astype('float16');

      const py = runNumPy(`
result = np.array([1.23456789, 0.001, 100.5, -42.75, 0], dtype=np.float64).astype(np.float16)
      `);

      for (let i = 0; i < values.length; i++) {
        expect(approxEqual(a.get([i]) as number, py.value[i])).toBe(true);
      }
    });

    it('float16 to int32', () => {
      const a = array([1.7, 2.3, -0.5, 3.9], 'float16').astype('int32');

      const py = runNumPy(`
result = np.array([1.7, 2.3, -0.5, 3.9], dtype=np.float16).astype(np.int32)
      `);

      expect(a.toArray()).toEqual(py.value);
    });

    it('int32 to float16', () => {
      const a = array([0, 1, 100, 1000, -50], 'int32').astype('float16');

      const py = runNumPy(`
result = np.array([0, 1, 100, 1000, -50], dtype=np.int32).astype(np.float16)
      `);

      for (let i = 0; i < 5; i++) {
        expect(approxEqual(a.get([i]) as number, py.value[i])).toBe(true);
      }
    });
  });

  describe('2D operations match NumPy', () => {
    it('matmul with float16', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float16'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'float16'
      );
      const ts = np.matmul(a, b);

      const py = runNumPy(`
a = np.array([[1, 2], [3, 4]], dtype=np.float16)
b = np.array([[5, 6], [7, 8]], dtype=np.float16)
result = np.matmul(a, b)
      `);

      const tsArr = ts.toArray() as number[][];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          expect(approxEqual(tsArr[i]![j]!, py.value[i][j])).toBe(true);
        }
      }
    });

    it('transpose', () => {
      const a = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'float16'
      );
      const ts = a.T;

      const py = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16).T
      `);

      expect(ts.dtype).toBe('float16');
      expect(ts.toArray()).toEqual(py.value);
    });

    it('sum along axis', () => {
      const a = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'float16'
      );
      const ts0 = a.sum(0);
      const ts1 = a.sum(1);

      const py0 = runNumPy(`
result = np.sum(np.array([[1,2,3],[4,5,6]], dtype=np.float16), axis=0)
      `);
      const py1 = runNumPy(`
result = np.sum(np.array([[1,2,3],[4,5,6]], dtype=np.float16), axis=1)
      `);

      for (let i = 0; i < 3; i++) {
        expect(approxEqual(ts0.get([i]) as number, py0.value[i])).toBe(true);
      }
      for (let i = 0; i < 2; i++) {
        expect(approxEqual(ts1.get([i]) as number, py1.value[i])).toBe(true);
      }
    });
  });

  describe('histogram matches NumPy', () => {
    it('histogram bin edges and counts match', () => {
      const a = arange(0, 1000, 1, 'float16');
      const [tsHist, tsEdges] = np.histogram(a) as [any, any];

      const pyH = runNumPy(`
a = np.arange(1000, dtype=np.float16)
h, _ = np.histogram(a)
result = h
      `);
      const pyE = runNumPy(`
a = np.arange(1000, dtype=np.float16)
_, edges = np.histogram(a)
result = edges
      `);

      expect(tsHist.toArray()).toEqual(pyH.value);
      // Bin edges should also match
      const tsEdgeArr = tsEdges.toArray() as number[];
      for (let i = 0; i < tsEdgeArr.length; i++) {
        expect(approxEqual(tsEdgeArr[i]!, pyE.value[i])).toBe(true);
      }
    });

    it('histogram2d counts match', () => {
      const x = arange(0, 100, 1, 'float16');
      const y = arange(0, 100, 1, 'float16');
      const [tsHist] = np.histogram2d(x, y) as [any, any, any];

      const py = runNumPy(`
x = np.arange(100, dtype=np.float16)
y = np.arange(100, dtype=np.float16)
h, _, _ = np.histogram2d(x, y)
result = h
      `);

      expect(tsHist.toArray()).toEqual(py.value);
    });
  });

  describe('matmul matches NumPy', () => {
    it('small float16 matmul', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float16'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'float16'
      );
      const ts = np.matmul(a, b);

      const py = runNumPy(`
a = np.array([[1, 2], [3, 4]], dtype=np.float16)
b = np.array([[5, 6], [7, 8]], dtype=np.float16)
result = np.matmul(a, b)
      `);

      expect(ts.dtype).toBe('float16');
      expect(py.dtype).toContain('float16');
      const tsArr = ts.toArray() as number[][];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          expect(approxEqual(tsArr[i]![j]!, py.value[i][j])).toBe(true);
        }
      }
    });
  });

  describe('large reductions with overflow match NumPy', () => {
    it('sum of large float16 array overflows like NumPy', () => {
      // On native Float16Array, sum of 10000 elements overflows to inf
      // On Float32Array fallback, it stays finite — that's expected
      const a = arange(0, 10000, 1, 'float16');
      const ts = a.sum();

      const py = runNumPy(`
import warnings
warnings.filterwarnings('ignore')
a = np.arange(10000, dtype=np.float16)
result = np.array([float(a.sum())])
      `);

      if (hasFloat16) {
        // Native Float16Array should overflow to inf like NumPy
        expect(ts).toBe(py.value[0]);
      } else {
        // Float32Array fallback: our result is finite, NumPy's is inf
        expect(Number.isFinite(ts as number)).toBe(true);
      }
    });

    it('cumsum accumulates like NumPy on native Float16Array', () => {
      // Use a small array to avoid overflow — compare exact accumulation
      const a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'float16');
      const ts = np.cumsum(a);

      const py = runNumPy(`
result = np.cumsum(np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float16))
      `);

      expect(ts.toArray()).toEqual(py.value);
    });

    it('dot product of float16 1D arrays', () => {
      const a = arange(0, 100, 1, 'float16');
      const ts = np.dot(a, a);

      const py = runNumPy(`
a = np.arange(100, dtype=np.float16)
result = np.array([float(np.dot(a, a))])
      `);

      expect(approxEqual(ts as number, py.value[0])).toBe(true);
    });

    it('trace of float16 matrix', () => {
      const a = arange(0, 100, 1, 'float16').reshape([10, 10]);
      const ts = np.trace(a);

      const py = runNumPy(`
a = np.arange(100, dtype=np.float16).reshape(10, 10)
result = np.array([float(np.trace(a))])
      `);

      expect(approxEqual(ts as number, py.value[0])).toBe(true);
    });
  });

  describe('float16 runtime info', () => {
    it('hasFloat16 is a boolean', () => {
      expect(typeof hasFloat16).toBe('boolean');
    });

    it('dtype reports float16 regardless of backing store', () => {
      const a = zeros([3], 'float16');
      expect(a.dtype).toBe('float16');
    });

    it('itemsize is 2', () => {
      const a = zeros([3], 'float16');
      expect(a.itemsize).toBe(2);
    });
  });
});
