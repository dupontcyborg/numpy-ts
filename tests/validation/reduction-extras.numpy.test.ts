/**
 * NumPy validation tests for: argmin/argmax keepdims, ptp multi-axis,
 * average returned=True.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, argmax, argmin, array, average, ptp } from '../../src';
import type { NDArray } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: reduction extras', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('argmin / argmax keepdims', () => {
    it('argmin axis=1 keepdims', () => {
      const a = array([
        [3, 1, 2],
        [9, 4, 7],
      ]);
      const result = argmin(a, 1, true) as NDArray;
      const np = runNumPy(`
a = np.array([[3, 1, 2], [9, 4, 7]])
result = np.argmin(a, axis=1, keepdims=True)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('argmax full reduction keepdims (shape of all 1s)', () => {
      const a = arange(12).reshape(3, 4);
      const result = argmax(a, undefined, true) as NDArray;
      const np = runNumPy(`
a = np.arange(12).reshape(3, 4)
result = np.argmax(a, keepdims=True)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('argmax axis=0 keepdims preserves rank', () => {
      const a = arange(24).reshape(2, 3, 4);
      const result = argmax(a, 0, true) as NDArray;
      const np = runNumPy(`
a = np.arange(24).reshape(2, 3, 4)
result = np.argmax(a, axis=0, keepdims=True)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });

  describe('ptp with multi-axis', () => {
    it('ptp axis=[0,1] reduces both', () => {
      const a = arange(24).reshape(2, 3, 4);
      const result = ptp(a, [0, 1]) as NDArray;
      const np = runNumPy(`
a = np.arange(24).reshape(2, 3, 4)
result = np.ptp(a, axis=(0, 1))
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });

    it('ptp axis=[1,2] keepdims', () => {
      const a = arange(24).reshape(2, 3, 4);
      const result = ptp(a, [1, 2], true) as NDArray;
      const np = runNumPy(`
a = np.arange(24).reshape(2, 3, 4)
result = np.ptp(a, axis=(1, 2), keepdims=True)
`);
      expect(Array.from(result.shape)).toEqual(np.shape);
      expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
    });
  });

  describe('average returned=True', () => {
    it('returns (avg, count) for unweighted full reduction', () => {
      const a = array([1.0, 2.0, 3.0, 4.0]);
      const [avg, sw] = average(a, undefined, undefined, undefined, true) as [number, number];
      const npAvg = runNumPy(`
a = np.array([1.0, 2.0, 3.0, 4.0])
avg, sw = np.average(a, returned=True)
result = float(avg)
`);
      const npSw = runNumPy(`
a = np.array([1.0, 2.0, 3.0, 4.0])
avg, sw = np.average(a, returned=True)
result = float(sw)
`);
      expect(avg).toBeCloseTo(npAvg.value, 10);
      expect(sw).toBeCloseTo(npSw.value, 10);
    });

    it('returns (avg, sum_of_weights) for weighted', () => {
      const a = array([1.0, 2.0, 3.0, 4.0]);
      const w = array([1, 2, 3, 4]);
      const [avg, sw] = average(a, undefined, w, undefined, true) as [number, number];
      const npAvg = runNumPy(`
a = np.array([1.0, 2.0, 3.0, 4.0])
w = np.array([1, 2, 3, 4])
avg, sw = np.average(a, weights=w, returned=True)
result = float(avg)
`);
      const npSw = runNumPy(`
a = np.array([1.0, 2.0, 3.0, 4.0])
w = np.array([1, 2, 3, 4])
avg, sw = np.average(a, weights=w, returned=True)
result = float(sw)
`);
      expect(avg).toBeCloseTo(npAvg.value, 10);
      expect(sw).toBeCloseTo(npSw.value, 10);
    });

    it('axis-reduce returned along axis', () => {
      const a = array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const [avg, sw] = average(a, 0, undefined, undefined, true) as [NDArray, NDArray];
      const npAvg = runNumPy(`
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
avg, sw = np.average(a, axis=0, returned=True)
result = avg
`);
      const npSw = runNumPy(`
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
avg, sw = np.average(a, axis=0, returned=True)
result = sw
`);
      expect(arraysClose(avg.toArray() as number[], npAvg.value, 1e-10)).toBe(true);
      expect(arraysClose(sw.toArray() as number[], npSw.value, 1e-10)).toBe(true);
    });
  });
});
