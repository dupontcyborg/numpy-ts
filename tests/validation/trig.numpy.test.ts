/**
 * Python NumPy validation tests for trigonometric operations
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

describe('NumPy Validation: Trigonometric Operations', () => {
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

  describe('sin', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = sin(array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]));
      const pyResult = runNumPy(`
import math
result = np.sin(np.array([0, math.pi/6, math.pi/4, math.pi/2, math.pi]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = sin(
        array([
          [0, Math.PI / 4],
          [Math.PI / 2, Math.PI],
        ])
      );
      const pyResult = runNumPy(`
import math
result = np.sin(np.array([[0, math.pi/4], [math.pi/2, math.pi]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = sin(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.sin(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('cos', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = cos(array([0, Math.PI / 3, Math.PI / 2, Math.PI]));
      const pyResult = runNumPy(`
import math
result = np.cos(np.array([0, math.pi/3, math.pi/2, math.pi]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = cos(
        array([
          [0, Math.PI / 2],
          [Math.PI, 2 * Math.PI],
        ])
      );
      const pyResult = runNumPy(`
import math
result = np.cos(np.array([[0, math.pi/2], [math.pi, 2*math.pi]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = cos(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.cos(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('tan', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = tan(array([0, Math.PI / 6, Math.PI / 4]));
      const pyResult = runNumPy(`
import math
result = np.tan(np.array([0, math.pi/6, math.pi/4]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = tan(array([0, 1], 'int32'));
      const pyResult = runNumPy(`
result = np.tan(np.array([0, 1], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arcsin', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arcsin(array([0, 0.5, 1, -0.5, -1]));
      const pyResult = runNumPy(`
result = np.arcsin(np.array([0, 0.5, 1, -0.5, -1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = arcsin(
        array([
          [0, 0.5],
          [-0.5, 1],
        ])
      );
      const pyResult = runNumPy(`
result = np.arcsin(np.array([[0, 0.5], [-0.5, 1]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arcsin(array([0, 1, -1], 'int32'));
      const pyResult = runNumPy(`
result = np.arcsin(np.array([0, 1, -1], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arccos', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arccos(array([1, 0.5, 0, -0.5, -1]));
      const pyResult = runNumPy(`
result = np.arccos(np.array([1, 0.5, 0, -0.5, -1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arccos(array([1, 0, -1], 'int32'));
      const pyResult = runNumPy(`
result = np.arccos(np.array([1, 0, -1], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arctan', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = arctan(array([-10, -1, 0, 1, 10]));
      const pyResult = runNumPy(`
result = np.arctan(np.array([-10, -1, 0, 1, 10]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arctan(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.arctan(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('arctan2', () => {
    it('matches NumPy for array-array', () => {
      const jsResult = arctan2(array([1, 1, -1, -1]), array([1, -1, 1, -1]));
      const pyResult = runNumPy(`
result = np.arctan2(np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array-scalar', () => {
      const jsResult = arctan2(array([1, 2, 3, 4]), 2);
      const pyResult = runNumPy(`
result = np.arctan2(np.array([1, 2, 3, 4]), 2)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = arctan2(
        array([
          [1, 0],
          [0, 1],
        ]),
        array([
          [0, 1],
          [1, 0],
        ])
      );
      const pyResult = runNumPy(`
result = np.arctan2(np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = arctan2(array([1, 2], 'int32'), array([1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.arctan2(np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('hypot', () => {
    it('matches NumPy for array-array', () => {
      const jsResult = hypot(array([3, 5, 8]), array([4, 12, 15]));
      const pyResult = runNumPy(`
result = np.hypot(np.array([3, 5, 8]), np.array([4, 12, 15]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for array-scalar', () => {
      const jsResult = hypot(array([3, 6, 9]), 4);
      const pyResult = runNumPy(`
result = np.hypot(np.array([3, 6, 9]), 4)
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D arrays', () => {
      const jsResult = hypot(
        array([
          [3, 5],
          [8, 7],
        ]),
        array([
          [4, 12],
          [15, 24],
        ])
      );
      const pyResult = runNumPy(`
result = np.hypot(np.array([[3, 5], [8, 7]]), np.array([[4, 12], [15, 24]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = hypot(array([3, 4], 'int32'), array([4, 3], 'int32'));
      const pyResult = runNumPy(`
result = np.hypot(np.array([3, 4], dtype=np.int32), np.array([4, 3], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('degrees', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = degrees(array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]));
      const pyResult = runNumPy(`
import math
result = np.degrees(np.array([0, math.pi/6, math.pi/4, math.pi/2, math.pi]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = degrees(
        array([
          [0, Math.PI],
          [2 * Math.PI, Math.PI / 2],
        ])
      );
      const pyResult = runNumPy(`
import math
result = np.degrees(np.array([[0, math.pi], [2*math.pi, math.pi/2]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = degrees(array([0, 1, 2], 'int32'));
      const pyResult = runNumPy(`
result = np.degrees(np.array([0, 1, 2], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('radians', () => {
    it('matches NumPy for 1D array', () => {
      const jsResult = radians(array([0, 30, 45, 90, 180]));
      const pyResult = runNumPy(`
result = np.radians(np.array([0, 30, 45, 90, 180]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy for 2D array', () => {
      const jsResult = radians(
        array([
          [0, 90],
          [180, 360],
        ])
      );
      const pyResult = runNumPy(`
result = np.radians(np.array([[0, 90], [180, 360]]))
      `);

      expect(jsResult.shape).toEqual(pyResult.shape);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('matches NumPy dtype promotion for integers', () => {
      const jsResult = radians(array([0, 90, 180], 'int32'));
      const pyResult = runNumPy(`
result = np.radians(np.array([0, 90, 180], dtype=np.int32))
      `);

      expect(jsResult.dtype).toBe(pyResult.dtype);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('roundtrip with degrees matches NumPy', () => {
      const original = array([0, 30, 45, 90, 180]);
      const jsRadians = radians(original);
      const jsBack = degrees(jsRadians);

      const pyResult = runNumPy(`
original = np.array([0, 30, 45, 90, 180])
result = np.degrees(np.radians(original))
      `);

      expect(arraysClose(jsBack.toArray(), pyResult.value)).toBe(true);
    });
  });

  describe('deg2rad', () => {
    it('matches NumPy np.deg2rad', () => {
      const angles = array([0, 30, 45, 90, 180, 270, 360]);
      const jsResult = deg2rad(angles);

      const pyResult = runNumPy(`
angles = np.array([0, 30, 45, 90, 180, 270, 360])
result = np.deg2rad(angles)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('produces same result as radians()', () => {
      const angles = array([0, 45, 90, 180]);
      const result1 = deg2rad(angles);
      const result2 = radians(angles);
      expect(arraysClose(result1.toArray(), result2.toArray())).toBe(true);
    });
  });

  describe('rad2deg', () => {
    it('matches NumPy np.rad2deg', () => {
      const radians_arr = array([0, Math.PI / 6, Math.PI / 4, Math.PI / 2, Math.PI]);
      const jsResult = rad2deg(radians_arr);

      const pyResult = runNumPy(`
radians = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])
result = np.rad2deg(radians)
      `);

      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it('produces same result as degrees()', () => {
      const radians_arr = array([0, Math.PI / 4, Math.PI / 2, Math.PI]);
      const result1 = rad2deg(radians_arr);
      const result2 = degrees(radians_arr);
      expect(arraysClose(result1.toArray(), result2.toArray())).toBe(true);
    });
  });
});
