/**
 * Python NumPy validation tests for statistics operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  bincount,
  digitize,
  histogram,
  histogram2d,
  histogramdd,
  correlate,
  convolve,
  cov,
  corrcoef,
  histogram_bin_edges,
  trapezoid,
} from '../../src/core/ndarray';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Statistics Operations', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          '   Current Python command: ' +
          (process.env.NUMPY_PYTHON || 'python3') +
          '\n'
      );
    }
  });

  describe('bincount', () => {
    it('matches NumPy for basic case', () => {
      const jsResult = bincount(array([0, 1, 1, 3, 2, 1, 7]));
      const pyResult = runNumPy(`
result = np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with minlength', () => {
      const jsResult = bincount(array([0, 1, 1]), undefined, 5);
      const pyResult = runNumPy(`
result = np.bincount(np.array([0, 1, 1]), minlength=5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with weights', () => {
      const jsResult = bincount(array([0, 1, 1, 2]), array([0.5, 1.0, 0.5, 2.0]));
      const pyResult = runNumPy(`
result = np.bincount(np.array([0, 1, 1, 2]), weights=np.array([0.5, 1.0, 0.5, 2.0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('digitize', () => {
    it('matches NumPy for increasing bins', () => {
      const jsResult = digitize(array([0.2, 6.4, 3.0, 1.6]), array([0.0, 1.0, 2.5, 4.0, 10.0]));
      const pyResult = runNumPy(`
result = np.digitize(np.array([0.2, 6.4, 3.0, 1.6]), np.array([0.0, 1.0, 2.5, 4.0, 10.0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with right=True', () => {
      const jsResult = digitize(array([0.0, 1.0, 2.5]), array([0.0, 1.0, 2.5, 4.0]), true);
      const pyResult = runNumPy(`
result = np.digitize(np.array([0.0, 1.0, 2.5]), np.array([0.0, 1.0, 2.5, 4.0]), right=True)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for decreasing bins', () => {
      const jsResult = digitize(array([0.5, 1.5, 2.5]), array([4.0, 2.0, 1.0, 0.0]));
      const pyResult = runNumPy(`
result = np.digitize(np.array([0.5, 1.5, 2.5]), np.array([4.0, 2.0, 1.0, 0.0]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('histogram', () => {
    it('matches NumPy for basic histogram', () => {
      const [jsHist, jsEdges] = histogram(array([1, 2, 1, 3, 4, 2, 2, 3, 3, 3]), 4);
      const pyResult = runNumPy(`
hist, edges = np.histogram(np.array([1, 2, 1, 3, 4, 2, 2, 3, 3, 3]), bins=4)
result = hist
      `);
      const pyEdges = runNumPy(`
hist, edges = np.histogram(np.array([1, 2, 1, 3, 4, 2, 2, 3, 3, 3]), bins=4)
result = edges
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
      expect(arraysClose(jsEdges.toArray(), pyEdges.value)).toBe(true);
    });

    it('matches NumPy with range', () => {
      const [jsHist] = histogram(array([1, 2, 3, 4, 5]), 5, [0, 5]);
      const pyResult = runNumPy(`
hist, edges = np.histogram(np.array([1, 2, 3, 4, 5]), bins=5, range=(0, 5))
result = hist
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with density', () => {
      const [jsHist] = histogram(array([1, 2, 1, 2, 3, 4]), 4, undefined, true);
      const pyResult = runNumPy(`
hist, edges = np.histogram(np.array([1, 2, 1, 2, 3, 4]), bins=4, density=True)
result = hist
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value, 1e-10)).toBe(true);
    });
  });

  describe('histogram2d', () => {
    it('matches NumPy for basic 2D histogram', () => {
      const [jsHist] = histogram2d(array([0, 0, 1, 1]), array([0, 1, 0, 1]), 2);
      const pyResult = runNumPy(`
hist, xedges, yedges = np.histogram2d(
    np.array([0, 0, 1, 1]),
    np.array([0, 1, 0, 1]),
    bins=2
)
result = hist
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with different bin counts', () => {
      const [jsHist] = histogram2d(
        array([0.5, 1.5, 2.5, 3.5, 4.5]),
        array([0.1, 0.2, 0.3, 0.4, 0.5]),
        [5, 3]
      );
      const pyResult = runNumPy(`
hist, xedges, yedges = np.histogram2d(
    np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
    np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    bins=[5, 3]
)
result = hist
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('histogramdd', () => {
    it('matches NumPy for 1D histogram', () => {
      const [jsHist] = histogramdd(array([1, 2, 3, 4, 5]), 5);
      const pyResult = runNumPy(`
hist, edges = np.histogramdd(np.array([1, 2, 3, 4, 5]).reshape(-1, 1), bins=5)
result = hist.flatten()
      `);

      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D histogram', () => {
      const [jsHist] = histogramdd(
        array([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]),
        2
      );
      const pyResult = runNumPy(`
hist, edges = np.histogramdd(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), bins=2)
result = hist
      `);

      expect(jsHist.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsHist.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('correlate', () => {
    it('matches NumPy for full mode', () => {
      const jsResult = correlate(array([1, 2, 3]), array([0, 1, 0.5]), 'full');
      const pyResult = runNumPy(`
result = np.correlate(np.array([1, 2, 3]), np.array([0, 1, 0.5]), mode='full')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for same mode', () => {
      const jsResult = correlate(array([1, 2, 3, 4, 5]), array([1, 2, 3]), 'same');
      const pyResult = runNumPy(`
result = np.correlate(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]), mode='same')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for valid mode', () => {
      const jsResult = correlate(array([1, 2, 3, 4, 5]), array([1, 2, 3]), 'valid');
      const pyResult = runNumPy(`
result = np.correlate(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]), mode='valid')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('convolve', () => {
    it('matches NumPy for full mode', () => {
      const jsResult = convolve(array([1, 2, 3]), array([0, 1, 0.5]), 'full');
      const pyResult = runNumPy(`
result = np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]), mode='full')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for same mode', () => {
      const jsResult = convolve(array([1, 2, 3, 4, 5]), array([1, 2, 3]), 'same');
      const pyResult = runNumPy(`
result = np.convolve(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]), mode='same')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for valid mode', () => {
      const jsResult = convolve(array([1, 2, 3, 4, 5]), array([1, 2, 3]), 'valid');
      const pyResult = runNumPy(`
result = np.convolve(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]), mode='valid')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cov', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = cov(array([1, 2, 3, 4, 5]));
      const pyResult = runNumPy(`
result = np.cov(np.array([1, 2, 3, 4, 5]))
      `);

      // Both return 0-d scalar for 1D input
      expect(jsResult.shape).toEqual(pyResult.shape);
      const jsValue = jsResult.get([]) as number;
      const pyValue = typeof pyResult.value === 'number' ? pyResult.value : pyResult.value;
      expect(Math.abs(jsValue - pyValue)).toBeLessThan(1e-10);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = cov(
        array([
          [0, 1, 2],
          [2, 1, 0],
        ])
      );
      const pyResult = runNumPy(`
result = np.cov(np.array([[0, 1, 2], [2, 1, 0]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with rowvar=False', () => {
      const jsResult = cov(
        array([
          [0, 2],
          [1, 1],
          [2, 0],
        ]),
        undefined,
        false
      );
      const pyResult = runNumPy(`
result = np.cov(np.array([[0, 2], [1, 1], [2, 0]]), rowvar=False)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with bias=True', () => {
      const jsResult = cov(array([1, 2, 3, 4, 5]), undefined, true, true);
      const pyResult = runNumPy(`
result = np.cov(np.array([1, 2, 3, 4, 5]), bias=True)
      `);

      const jsValue = jsResult.get([]) as number;
      const pyValue = typeof pyResult.value === 'number' ? pyResult.value : pyResult.value;
      expect(Math.abs(jsValue - pyValue)).toBeLessThan(1e-10);
    });

    it('matches NumPy with ddof', () => {
      const jsResult = cov(array([1, 2, 3, 4, 5]), undefined, true, false, 2);
      const pyResult = runNumPy(`
result = np.cov(np.array([1, 2, 3, 4, 5]), ddof=2)
      `);

      const jsValue = jsResult.get([]) as number;
      const pyValue = typeof pyResult.value === 'number' ? pyResult.value : pyResult.value;
      expect(Math.abs(jsValue - pyValue)).toBeLessThan(1e-10);
    });
  });

  describe('corrcoef', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = corrcoef(array([1, 2, 3, 4, 5]));
      const pyResult = runNumPy(`
result = np.corrcoef(np.array([1, 2, 3, 4, 5]))
      `);

      // Both return 0-d scalar for 1D input
      expect(jsResult.shape).toEqual(pyResult.shape);
      const jsValue = jsResult.get([]) as number;
      const pyValue = typeof pyResult.value === 'number' ? pyResult.value : pyResult.value;
      expect(Math.abs(jsValue - pyValue)).toBeLessThan(1e-10);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = corrcoef(
        array([
          [1, 2, 3],
          [3, 2, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.corrcoef(np.array([[1, 2, 3], [3, 2, 1]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with y parameter', () => {
      const jsResult = corrcoef(array([1, 2, 3, 4, 5]), array([2, 4, 6, 8, 10]));
      const pyResult = runNumPy(`
result = np.corrcoef(np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with rowvar=False', () => {
      const jsResult = corrcoef(
        array([
          [0, 2],
          [1, 1],
          [2, 0],
        ]),
        undefined,
        false
      );
      const pyResult = runNumPy(`
result = np.corrcoef(np.array([[0, 2], [1, 1], [2, 0]]), rowvar=False)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('histogram_bin_edges', () => {
    it('matches NumPy for basic bin count', () => {
      const jsResult = histogram_bin_edges(array([1, 2, 3, 4, 5]), 5);
      const pyResult = runNumPy(`
result = np.histogram_bin_edges(np.array([1, 2, 3, 4, 5]), bins=5)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with specified range', () => {
      const jsResult = histogram_bin_edges(array([1, 2, 3, 4, 5]), 4, [0, 8]);
      const pyResult = runNumPy(`
result = np.histogram_bin_edges(np.array([1, 2, 3, 4, 5]), bins=4, range=(0, 8))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy with sturges bin selection', () => {
      const jsResult = histogram_bin_edges(array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'sturges');
      const pyResult = runNumPy(`
result = np.histogram_bin_edges(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), bins='sturges')
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      // Allow for slight differences in edge calculation
      const jsArr = jsResult.toArray() as number[];
      const pyArr = pyResult.value as number[];
      expect(jsArr[0]).toBeCloseTo(pyArr[0]!, 5);
      expect(jsArr[jsArr.length - 1]).toBeCloseTo(pyArr[pyArr.length - 1]!, 5);
    });
  });

  describe('trapezoid', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = trapezoid(array([1, 2, 3, 4, 5]));
      const pyResult = runNumPy(`
result = np.trapezoid(np.array([1, 2, 3, 4, 5]))
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy with custom dx', () => {
      const jsResult = trapezoid(array([1, 2, 3, 4, 5]), undefined, 0.5);
      const pyResult = runNumPy(`
result = np.trapezoid(np.array([1, 2, 3, 4, 5]), dx=0.5)
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy with x values', () => {
      const jsResult = trapezoid(array([0, 1, 4]), array([0, 1, 2]));
      const pyResult = runNumPy(`
result = np.trapezoid(np.array([0, 1, 4]), x=np.array([0, 1, 2]))
      `);

      expect(Math.abs((jsResult as number) - pyResult.value)).toBeLessThan(1e-10);
    });

    it('matches NumPy for 2D array along axis 1', () => {
      const jsResult = trapezoid(
        array([
          [1, 2, 3, 4],
          [2, 4, 6, 8],
        ]),
        undefined,
        1,
        1
      );
      const pyResult = runNumPy(`
result = np.trapezoid(np.array([[1, 2, 3, 4], [2, 4, 6, 8]]), axis=1)
      `);

      expect(arraysClose((jsResult as { toArray: () => number[] }).toArray(), pyResult.value)).toBe(
        true
      );
    });

    it('matches NumPy for 2D array along axis 0', () => {
      const jsResult = trapezoid(
        array([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        undefined,
        1,
        0
      );
      const pyResult = runNumPy(`
result = np.trapezoid(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
      `);

      expect(arraysClose((jsResult as { toArray: () => number[] }).toArray(), pyResult.value)).toBe(
        true
      );
    });
  });
});
