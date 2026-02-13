import { describe, it, expect } from 'vitest';
import {
  arange,
  linspace,
  logspace,
  geomspace,
  eye,
  empty,
  full,
  identity,
  asarray,
  copy,
  zeros,
  ones,
  zeros_like,
  ones_like,
  empty_like,
  full_like,
  array,
  asanyarray,
  ascontiguousarray,
  asfortranarray,
  asarray_chkfinite,
  diag,
  diagflat,
  frombuffer,
  fromfile,
  fromfunction,
  fromiter,
  fromstring,
  meshgrid,
  tri,
  tril,
  triu,
  vander,
  require,
  genfromtxt,
  fromregex,
  Complex,
} from '../../src';

describe('Array Creation Functions', () => {
  describe('arange', () => {
    it('creates array from 0 to n', () => {
      const arr = arange(5);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('creates array from start to stop', () => {
      const arr = arange(2, 7);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('creates array with custom step', () => {
      const arr = arange(0, 10, 2);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('creates array with negative step', () => {
      const arr = arange(10, 0, -2);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([10, 8, 6, 4, 2]);
    });

    it('creates array with float step', () => {
      const arr = arange(0, 1, 0.25);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()).toEqual([0, 0.25, 0.5, 0.75]);
    });

    it('creates empty array for invalid range', () => {
      const arr = arange(5, 2);
      expect(arr.shape).toEqual([0]);
      expect(arr.toArray()).toEqual([]);
    });
  });

  describe('linspace', () => {
    it('creates 50 points by default', () => {
      const arr = linspace(0, 1);
      expect(arr.shape).toEqual([50]);
      expect(arr.toArray()[0]).toBe(0);
      expect(arr.toArray()[49]).toBeCloseTo(1);
    });

    it('creates specified number of points', () => {
      const arr = linspace(0, 10, 5);
      expect(arr.shape).toEqual([5]);
      expect(arr.toArray()).toEqual([0, 2.5, 5, 7.5, 10]);
    });

    it('creates single point', () => {
      const arr = linspace(5, 10, 1);
      expect(arr.shape).toEqual([1]);
      expect(arr.toArray()).toEqual([5]);
    });

    it('creates empty array for num=0', () => {
      const arr = linspace(0, 1, 0);
      expect(arr.shape).toEqual([0]);
    });

    it('works with negative range', () => {
      const arr = linspace(-5, 5, 11);
      expect(arr.shape).toEqual([11]);
      expect(arr.toArray()[0]).toBe(-5);
      expect(arr.toArray()[5]).toBe(0);
      expect(arr.toArray()[10]).toBe(5);
    });
  });

  describe('logspace', () => {
    it('creates 50 points by default (base 10)', () => {
      const arr = logspace(0, 2);
      expect(arr.shape).toEqual([50]);
      expect(arr.toArray()[0]).toBeCloseTo(1); // 10^0
      expect(arr.toArray()[49]).toBeCloseTo(100); // 10^2
    });

    it('creates logarithmically spaced values', () => {
      const arr = logspace(0, 3, 4);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()[0]).toBeCloseTo(1); // 10^0
      expect(arr.toArray()[1]).toBeCloseTo(10); // 10^1
      expect(arr.toArray()[2]).toBeCloseTo(100); // 10^2
      expect(arr.toArray()[3]).toBeCloseTo(1000); // 10^3
    });

    it('works with custom base', () => {
      const arr = logspace(0, 3, 4, 2);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()[0]).toBeCloseTo(1); // 2^0
      expect(arr.toArray()[1]).toBeCloseTo(2); // 2^1
      expect(arr.toArray()[2]).toBeCloseTo(4); // 2^2
      expect(arr.toArray()[3]).toBeCloseTo(8); // 2^3
    });

    it('creates single point', () => {
      const arr = logspace(2, 3, 1);
      expect(arr.shape).toEqual([1]);
      expect(arr.toArray()[0]).toBeCloseTo(100); // 10^2
    });

    it('creates empty array for num=0', () => {
      const arr = logspace(0, 1, 0);
      expect(arr.shape).toEqual([0]);
    });

    it('works with negative exponents', () => {
      const arr = logspace(-2, 0, 3);
      expect(arr.shape).toEqual([3]);
      expect(arr.toArray()[0]).toBeCloseTo(0.01); // 10^-2
      expect(arr.toArray()[1]).toBeCloseTo(0.1); // 10^-1
      expect(arr.toArray()[2]).toBeCloseTo(1); // 10^0
    });
  });

  describe('geomspace', () => {
    it('creates 50 points by default', () => {
      const arr = geomspace(1, 1000);
      expect(arr.shape).toEqual([50]);
      expect(arr.toArray()[0]).toBeCloseTo(1);
      expect(arr.toArray()[49]).toBeCloseTo(1000);
    });

    it('creates geometrically spaced values', () => {
      const arr = geomspace(1, 1000, 4);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()[0]).toBeCloseTo(1);
      expect(arr.toArray()[1]).toBeCloseTo(10);
      expect(arr.toArray()[2]).toBeCloseTo(100);
      expect(arr.toArray()[3]).toBeCloseTo(1000);
    });

    it('creates single point', () => {
      const arr = geomspace(5, 10, 1);
      expect(arr.shape).toEqual([1]);
      expect(arr.toArray()[0]).toBeCloseTo(5);
    });

    it('creates empty array for num=0', () => {
      const arr = geomspace(1, 10, 0);
      expect(arr.shape).toEqual([0]);
    });

    it('works with small values', () => {
      const arr = geomspace(0.01, 1, 3);
      expect(arr.shape).toEqual([3]);
      expect(arr.toArray()[0]).toBeCloseTo(0.01);
      expect(arr.toArray()[1]).toBeCloseTo(0.1);
      expect(arr.toArray()[2]).toBeCloseTo(1);
    });

    it('works with negative values', () => {
      const arr = geomspace(-1, -1000, 4);
      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()[0]).toBeCloseTo(-1);
      expect(arr.toArray()[1]).toBeCloseTo(-10);
      expect(arr.toArray()[2]).toBeCloseTo(-100);
      expect(arr.toArray()[3]).toBeCloseTo(-1000);
    });

    it('throws error for zero values', () => {
      expect(() => geomspace(0, 100, 10)).toThrow('Geometric sequence cannot include zero');
      expect(() => geomspace(1, 0, 10)).toThrow('Geometric sequence cannot include zero');
    });

    it('throws error for mixed signs', () => {
      expect(() => geomspace(-1, 100, 10)).toThrow(
        'Geometric sequence cannot contain both positive and negative values'
      );
    });
  });

  describe('eye', () => {
    it('creates square identity matrix', () => {
      const arr = eye(3);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    it('creates rectangular identity matrix', () => {
      const arr = eye(2, 3);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });

    it('creates identity matrix with offset diagonal (k=1)', () => {
      const arr = eye(3, 3, 1);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
      ]);
    });

    it('creates identity matrix with offset diagonal (k=-1)', () => {
      const arr = eye(3, 3, -1);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });
  });

  describe('empty', () => {
    it('creates uninitialized array with given shape', () => {
      const arr = empty([2, 3]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.size).toBe(6);
      expect(arr.dtype).toBe('float64');
    });

    it('creates empty array with specified dtype', () => {
      const arr = empty([3, 2], 'int32');
      expect(arr.shape).toEqual([3, 2]);
      expect(arr.dtype).toBe('int32');
    });

    it('creates 1D empty array', () => {
      const arr = empty([5]);
      expect(arr.shape).toEqual([5]);
      expect(arr.size).toBe(5);
    });
  });

  describe('full', () => {
    it('creates array filled with constant value', () => {
      const arr = full([2, 3], 7);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [7, 7, 7],
        [7, 7, 7],
      ]);
    });

    it('creates array with negative fill value', () => {
      const arr = full([2, 2], -3.5);
      expect(arr.toArray()).toEqual([
        [-3.5, -3.5],
        [-3.5, -3.5],
      ]);
    });

    it('creates array with specified dtype', () => {
      const arr = full([2, 2], 5, 'int32');
      expect(arr.dtype).toBe('int32');
      expect(arr.toArray()).toEqual([
        [5, 5],
        [5, 5],
      ]);
    });

    it('creates array with boolean fill value', () => {
      const arr = full([2, 2], true);
      expect(arr.dtype).toBe('bool');
      expect(arr.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    it('creates array with bigint fill value', () => {
      const arr = full([2, 2], BigInt(42));
      expect(arr.dtype).toBe('int64');
    });

    it('infers int32 dtype for integer fill values', () => {
      const arr = full([2, 2], 42);
      expect(arr.dtype).toBe('int32');
    });

    it('infers float64 dtype for float fill values', () => {
      const arr = full([2, 2], 3.14);
      expect(arr.dtype).toBe('float64');
    });
  });

  describe('identity', () => {
    it('creates square identity matrix', () => {
      const arr = identity(3);
      expect(arr.shape).toEqual([3, 3]);
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    it('creates identity matrix with specified dtype', () => {
      const arr = identity(2, 'int32');
      expect(arr.shape).toEqual([2, 2]);
      expect(arr.dtype).toBe('int32');
      expect(arr.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it('creates 1x1 identity matrix', () => {
      const arr = identity(1);
      expect(arr.shape).toEqual([1, 1]);
      expect(arr.toArray()).toEqual([[1]]);
    });
  });

  describe('asarray', () => {
    it('converts nested arrays to NDArray', () => {
      const arr = asarray([
        [1, 2],
        [3, 4],
      ]);
      expect(arr.shape).toEqual([2, 2]);
      expect(arr.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('returns existing NDArray unchanged', () => {
      const original = array([
        [1, 2],
        [3, 4],
      ]);
      const converted = asarray(original);
      expect(converted).toBe(original); // Same object
    });

    it('converts dtype when specified', () => {
      const original = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int32'
      );
      const converted = asarray(original, 'float64');
      expect(converted.dtype).toBe('float64');
      expect(converted).not.toBe(original); // Different object
      expect(converted.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('preserves dtype when not specified', () => {
      const original = array([[1, 2]], 'int16');
      const converted = asarray(original);
      expect(converted.dtype).toBe('int16');
      expect(converted).toBe(original); // Same object
    });
  });

  describe('copy', () => {
    it('creates deep copy of array', () => {
      const original = array([
        [1, 2],
        [3, 4],
      ]);
      const copied = copy(original);

      expect(copied.shape).toEqual(original.shape);
      expect(copied.toArray()).toEqual(original.toArray());
      expect(copied).not.toBe(original);
      expect(copied.data).not.toBe(original.data);
    });

    it('copy is independent of original', () => {
      const original = zeros([2, 2]);
      const copied = copy(original);

      original.set([0, 0], 42);
      expect(original.get([0, 0])).toBe(42);
      expect(copied.get([0, 0])).toBe(0);
    });

    it('copies views correctly', () => {
      const original = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const view = original.slice(':', '1:3');
      const copied = copy(view);

      expect(copied.shape).toEqual([2, 2]);
      expect(copied.toArray()).toEqual([
        [2, 3],
        [5, 6],
      ]);
      expect(copied.flags.OWNDATA).toBe(true);
    });

    it('preserves dtype', () => {
      const original = array([[1, 2]], 'int16');
      const copied = copy(original);
      expect(copied.dtype).toBe('int16');
    });
  });

  describe('zeros_like', () => {
    it('creates zeros array with same shape', () => {
      const original = array([
        [1, 2],
        [3, 4],
      ]);
      const result = zeros_like(original);

      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 0],
        [0, 0],
      ]);
      expect(result.dtype).toBe(original.dtype);
    });

    it('creates zeros array with different dtype', () => {
      const original = array([[1, 2]], 'int32');
      const result = zeros_like(original, 'float64');

      expect(result.shape).toEqual([1, 2]);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([[0, 0]]);
    });

    it('works with multi-dimensional arrays', () => {
      const original = ones([2, 3, 4]);
      const result = zeros_like(original);

      expect(result.shape).toEqual([2, 3, 4]);
      expect(result.size).toBe(24);
    });
  });

  describe('ones_like', () => {
    it('creates ones array with same shape', () => {
      const original = zeros([2, 3]);
      const result = ones_like(original);

      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });

    it('creates ones array with different dtype', () => {
      const original = zeros([2, 2], 'float64');
      const result = ones_like(original, 'int32');

      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });
  });

  describe('empty_like', () => {
    it('creates empty array with same shape', () => {
      const original = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = empty_like(original);

      expect(result.shape).toEqual([2, 3]);
      expect(result.dtype).toBe(original.dtype);
    });

    it('creates empty array with different dtype', () => {
      const original = zeros([3, 3], 'float64');
      const result = empty_like(original, 'int16');

      expect(result.shape).toEqual([3, 3]);
      expect(result.dtype).toBe('int16');
    });
  });

  describe('full_like', () => {
    it('creates filled array with same shape', () => {
      const original = zeros([2, 2]);
      const result = full_like(original, 42);

      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [42, 42],
        [42, 42],
      ]);
    });

    it('creates filled array with different dtype', () => {
      const original = zeros([2, 2], 'float64');
      const result = full_like(original, 7, 'int32');

      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([
        [7, 7],
        [7, 7],
      ]);
    });

    it('works with negative fill values', () => {
      const original = ones([3, 2]);
      const result = full_like(original, -1.5);

      expect(result.toArray()).toEqual([
        [-1.5, -1.5],
        [-1.5, -1.5],
        [-1.5, -1.5],
      ]);
    });
  });

  // ========================================
  // New Creation Functions
  // ========================================

  describe('asanyarray', () => {
    it('returns same array if already NDArray', () => {
      const arr = array([1, 2, 3]);
      const result = asanyarray(arr);
      expect(result).toBe(arr);
    });

    it('converts nested array to NDArray', () => {
      const result = asanyarray([
        [1, 2],
        [3, 4],
      ]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('ascontiguousarray', () => {
    it('returns same array if already contiguous', () => {
      const arr = array([1, 2, 3, 4]);
      const result = ascontiguousarray(arr);
      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('converts nested array to contiguous', () => {
      const result = ascontiguousarray([
        [1, 2],
        [3, 4],
      ]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.flags.C_CONTIGUOUS).toBe(true);
    });
  });

  describe('asfortranarray', () => {
    it('creates array from input', () => {
      const result = asfortranarray([
        [1, 2],
        [3, 4],
      ]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('diag', () => {
    it('creates diagonal matrix from 1D array', () => {
      const arr = array([1, 2, 3]);
      const result = diag(arr);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
    });

    it('creates diagonal matrix with offset', () => {
      const arr = array([1, 2]);
      const result = diag(arr, 1);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0],
      ]);
    });

    it('creates diagonal matrix with negative offset', () => {
      const arr = array([1, 2]);
      const result = diag(arr, -1);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [0, 0, 0],
        [1, 0, 0],
        [0, 2, 0],
      ]);
    });

    it('extracts diagonal from 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diag(arr);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 5, 9]);
    });

    it('extracts diagonal with offset', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diag(arr, 1);
      expect(result.toArray()).toEqual([2, 6]);
    });
  });

  describe('diagflat', () => {
    it('creates diagonal matrix from flat input', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = diagflat(arr);
      expect(result.shape).toEqual([4, 4]);
      expect(result.get([0, 0])).toBe(1);
      expect(result.get([1, 1])).toBe(2);
      expect(result.get([2, 2])).toBe(3);
      expect(result.get([3, 3])).toBe(4);
    });
  });

  describe('fromfunction', () => {
    it('creates array from coordinate function', () => {
      const result = fromfunction((i, j) => i + j, [3, 3]);
      expect(result.shape).toEqual([3, 3]);
      expect(result.toArray()).toEqual([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
      ]);
    });

    it('creates array from single index function', () => {
      const result = fromfunction((i) => i * 2, [5]);
      expect(result.toArray()).toEqual([0, 2, 4, 6, 8]);
    });
  });

  describe('meshgrid', () => {
    it('creates 2D coordinate grids from 1D arrays', () => {
      const x = array([1, 2, 3]);
      const y = array([4, 5]);
      const [X, Y] = meshgrid(x, y);
      expect(X.shape).toEqual([2, 3]);
      expect(Y.shape).toEqual([2, 3]);
      expect(X.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
      ]);
      expect(Y.toArray()).toEqual([
        [4, 4, 4],
        [5, 5, 5],
      ]);
    });

    it('creates meshgrid with ij indexing', () => {
      const x = array([1, 2]);
      const y = array([3, 4, 5]);
      const [X, Y] = meshgrid(x, y, { indexing: 'ij' });
      expect(X.shape).toEqual([2, 3]);
      expect(Y.shape).toEqual([2, 3]);
    });
  });

  describe('tri', () => {
    it('creates lower triangular matrix', () => {
      const result = tri(3);
      expect(result.toArray()).toEqual([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
      ]);
    });

    it('creates rectangular lower triangular matrix', () => {
      const result = tri(3, 4);
      expect(result.toArray()).toEqual([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
      ]);
    });

    it('creates with diagonal offset', () => {
      const result = tri(3, 3, 1);
      expect(result.toArray()).toEqual([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });
  });

  describe('tril', () => {
    it('extracts lower triangle', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = tril(arr);
      expect(result.toArray()).toEqual([
        [1, 0, 0],
        [4, 5, 0],
        [7, 8, 9],
      ]);
    });

    it('extracts lower triangle with offset', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = tril(arr, 1);
      expect(result.toArray()).toEqual([
        [1, 2, 0],
        [4, 5, 6],
        [7, 8, 9],
      ]);
    });
  });

  describe('triu', () => {
    it('extracts upper triangle', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = triu(arr);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [0, 5, 6],
        [0, 0, 9],
      ]);
    });

    it('extracts upper triangle with offset', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = triu(arr, 1);
      expect(result.toArray()).toEqual([
        [0, 2, 3],
        [0, 0, 6],
        [0, 0, 0],
      ]);
    });
  });

  describe('vander', () => {
    it('creates Vandermonde matrix', () => {
      const x = array([1, 2, 3]);
      const result = vander(x);
      expect(result.toArray()).toEqual([
        [1, 1, 1],
        [4, 2, 1],
        [9, 3, 1],
      ]);
    });

    it('creates Vandermonde matrix with N columns', () => {
      const x = array([1, 2, 3]);
      const result = vander(x, 4);
      expect(result.toArray()).toEqual([
        [1, 1, 1, 1],
        [8, 4, 2, 1],
        [27, 9, 3, 1],
      ]);
    });

    it('creates Vandermonde matrix with increasing powers', () => {
      const x = array([1, 2, 3]);
      const result = vander(x, 3, true);
      expect(result.toArray()).toEqual([
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9],
      ]);
    });
  });

  describe('frombuffer', () => {
    it('creates array from ArrayBuffer', () => {
      const buffer = new Float64Array([1, 2, 3, 4]).buffer;
      const result = frombuffer(buffer, 'float64');
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('creates array with count limit', () => {
      const buffer = new Float64Array([1, 2, 3, 4, 5]).buffer;
      const result = frombuffer(buffer, 'float64', 3);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('creates array with offset', () => {
      const buffer = new Float64Array([1, 2, 3, 4]).buffer;
      const result = frombuffer(buffer, 'float64', -1, 16); // Skip first 2 elements (16 bytes)
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([3, 4]);
    });

    it('creates int32 array from buffer', () => {
      const buffer = new Int32Array([10, 20, 30]).buffer;
      const result = frombuffer(buffer, 'int32');
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([10, 20, 30]);
    });
  });

  describe('fromfile', () => {
    // Note: fromfile reads from actual files and requires Node.js fs
    // These tests are skipped in browser/test environments
    it('throws in browser environment (requires Node.js)', () => {
      expect(() => fromfile([1, 2, 3, 4, 5], 'float64')).toThrow(
        'fromfile requires Node.js file system access'
      );
    });
  });

  describe('fromiter', () => {
    it('creates array from array', () => {
      const result = fromiter([1, 2, 3, 4], 'float64');
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('creates array from generator', () => {
      function* gen() {
        yield 1;
        yield 2;
        yield 3;
      }
      const result = fromiter(gen(), 'float64');
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('respects count limit', () => {
      function* gen() {
        for (let i = 0; i < 100; i++) yield i;
      }
      const result = fromiter(gen(), 'int32', 5);
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([0, 1, 2, 3, 4]);
    });
  });

  describe('fromstring', () => {
    it('creates array from whitespace-separated string', () => {
      const result = fromstring('1 2 3 4 5', 'float64');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('creates array with custom separator', () => {
      const result = fromstring('1,2,3,4', 'float64', -1, ',');
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('respects count limit', () => {
      const result = fromstring('1 2 3 4 5', 'float64', 3);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('handles multiple whitespaces', () => {
      const result = fromstring('  1   2   3  ', 'float64');
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('creates int32 array', () => {
      const result = fromstring('10 20 30', 'int32');
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([10, 20, 30]);
    });
  });

  describe('asarray_chkfinite', () => {
    it('passes through valid finite array', () => {
      const a = array([1, 2, 3]);
      expect(asarray_chkfinite(a).toArray()).toEqual([1, 2, 3]);
    });

    it('throws on NaN', () => {
      const a = array([1, NaN, 3]);
      expect(() => asarray_chkfinite(a)).toThrow('infs or NaNs');
    });
  });

  describe('Creation dtype branches', () => {
    it('zeros with int32 dtype', () => {
      expect(zeros([3], 'int32').dtype).toBe('int32');
    });

    it('ones with float32 dtype', () => {
      expect(ones([3], 'float32').dtype).toBe('float32');
    });

    it('full with int32 dtype and value', () => {
      expect(full([3], 7, 'int32').toArray()).toEqual([7, 7, 7]);
    });

    it('empty with float32 dtype', () => {
      const a = empty([3], 'float32');
      expect(a.size).toBe(3);
      expect(a.dtype).toBe('float32');
    });

    it('eye with k offset (non-square)', () => {
      const a = eye(3, 4, 1);
      expect(a.shape).toEqual([3, 4]);
    });

    it('linspace with specified num', () => {
      const a = linspace(0, 10, 5);
      expect(a.size).toBe(5);
    });

    it('arange with float step', () => {
      const a = arange(0, 1, 0.25);
      expect(a.size).toBe(4);
    });

    it('array with bool dtype', () => {
      const a = array([0, 1, 1, 0], 'bool');
      expect(a.dtype).toBe('bool');
    });

    it('array with int16 dtype', () => {
      const a = array([1, 2, 3], 'int16');
      expect(a.dtype).toBe('int16');
    });

    it('array with uint8 dtype', () => {
      const a = array([1, 2, 3], 'uint8');
      expect(a.dtype).toBe('uint8');
    });

    it('copy preserves dtype', () => {
      const a = array([1, 2, 3], 'int32');
      const b = copy(a);
      expect(b.dtype).toBe('int32');
    });
  });

  // ========================================
  // Additional coverage tests
  // ========================================

  describe('fromfunction additional', () => {
    it('creates 3D array from coordinate function', () => {
      const result = fromfunction((i, j, k) => i * 100 + j * 10 + k, [2, 3, 2]);
      expect(result.shape).toEqual([2, 3, 2]);
      expect(result.get([0, 0, 0])).toBe(0);
      expect(result.get([1, 2, 1])).toBe(121);
    });

    it('creates 1D array with custom dtype', () => {
      const result = fromfunction((i) => i * 3, [4], 'int32');
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([0, 3, 6, 9]);
    });
  });

  describe('fromiter additional', () => {
    it('creates array with int32 dtype', () => {
      const result = fromiter([10, 20, 30], 'int32');
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([10, 20, 30]);
    });

    it('creates array from empty iterable', () => {
      const result = fromiter([], 'float64');
      expect(result.shape).toEqual([0]);
    });

    it('creates array with count=0', () => {
      const result = fromiter([1, 2, 3], 'float64', 0);
      expect(result.shape).toEqual([0]);
    });

    it('handles iterable with Set', () => {
      const result = fromiter(new Set([1, 2, 3]), 'float64');
      expect(result.shape).toEqual([3]);
    });
  });

  describe('fromstring additional', () => {
    it('parses floats from string', () => {
      const result = fromstring('1.5 2.5 3.5', 'float64');
      expect(result.toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('parses with semicolon separator', () => {
      const result = fromstring('10;20;30', 'float64', -1, ';');
      expect(result.toArray()).toEqual([10, 20, 30]);
    });

    it('handles empty string', () => {
      const result = fromstring('', 'float64');
      expect(result.shape).toEqual([0]);
    });

    it('creates float32 array', () => {
      const result = fromstring('1 2 3', 'float32');
      expect(result.dtype).toBe('float32');
    });
  });

  describe('frombuffer additional', () => {
    it('creates array from TypedArray directly', () => {
      const source = new Float64Array([10, 20, 30, 40]);
      const result = frombuffer(source, 'float64');
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([10, 20, 30, 40]);
    });

    it('creates array from TypedArray with count', () => {
      const source = new Float64Array([10, 20, 30, 40, 50]);
      const result = frombuffer(source, 'float64', 3);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([10, 20, 30]);
    });

    it('creates array from TypedArray with offset', () => {
      const source = new Float64Array([10, 20, 30, 40, 50]);
      const result = frombuffer(source, 'float64', -1, 2);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([30, 40, 50]);
    });

    it('creates array from Int32Array source', () => {
      const source = new Int32Array([1, 2, 3]);
      const result = frombuffer(source, 'float64');
      expect(result.dtype).toBe('float64');
      expect(result.shape).toEqual([3]);
    });
  });

  describe('geomspace additional', () => {
    it('throws for negative num', () => {
      expect(() => geomspace(1, 10, -1)).toThrow('num must be non-negative');
    });

    it('works with two negative values', () => {
      const result = geomspace(-1000, -1, 4);
      expect(result.shape).toEqual([4]);
      expect(result.toArray()[0]).toBeCloseTo(-1000);
      expect(result.toArray()[3]).toBeCloseTo(-1);
    });
  });

  describe('logspace additional', () => {
    it('throws for negative num', () => {
      expect(() => logspace(0, 1, -1)).toThrow('num must be non-negative');
    });

    it('works with base e', () => {
      const arr = logspace(0, 1, 3, Math.E);
      expect(arr.shape).toEqual([3]);
      expect(arr.toArray()[0]).toBeCloseTo(1); // e^0
      expect(arr.toArray()[2]).toBeCloseTo(Math.E); // e^1
    });
  });

  describe('linspace additional', () => {
    it('throws for negative num', () => {
      expect(() => linspace(0, 1, -1)).toThrow('num must be non-negative');
    });

    it('handles start equal to stop', () => {
      const arr = linspace(5, 5, 3);
      expect(arr.toArray()).toEqual([5, 5, 5]);
    });
  });

  describe('arange additional', () => {
    it('handles single argument (just stop)', () => {
      const arr = arange(3);
      expect(arr.toArray()).toEqual([0, 1, 2]);
    });

    it('creates array with int32 dtype', () => {
      const arr = arange(0, 5, 1, 'int32');
      expect(arr.dtype).toBe('int32');
      expect(arr.toArray()).toEqual([0, 1, 2, 3, 4]);
    });
  });

  describe('meshgrid additional', () => {
    it('returns empty array for no input', () => {
      const result = meshgrid();
      expect(result).toEqual([]);
    });

    it('returns copy for single input', () => {
      const x = array([1, 2, 3]);
      const [X] = meshgrid(x);
      expect(X.toArray()).toEqual([1, 2, 3]);
    });

    it('creates 3D meshgrid', () => {
      const x = array([1, 2]);
      const y = array([3, 4]);
      const z = array([5, 6]);
      const [X, Y, Z] = meshgrid(x, y, z);
      expect(X.shape).toEqual([2, 2, 2]);
      expect(Y.shape).toEqual([2, 2, 2]);
      expect(Z.shape).toEqual([2, 2, 2]);
    });
  });

  describe('require function', () => {
    it('returns same array if no dtype change needed', () => {
      const a = array([1, 2, 3], 'float64');
      const result = require(a);
      expect(result).toBe(a);
    });

    it('converts dtype when specified', () => {
      const a = array([1, 2, 3], 'float64');
      const result = require(a, 'int32');
      expect(result.dtype).toBe('int32');
    });

    it('handles requirements parameter (no-op)', () => {
      const a = array([1, 2, 3]);
      const result = require(a, undefined, ['C', 'A']);
      expect(result).toBe(a);
    });
  });

  describe('asarray_chkfinite additional', () => {
    it('throws on Infinity', () => {
      const a = array([1, Infinity, 3]);
      expect(() => asarray_chkfinite(a)).toThrow('infs or NaNs');
    });

    it('throws on negative Infinity', () => {
      const a = array([1, -Infinity, 3]);
      expect(() => asarray_chkfinite(a)).toThrow('infs or NaNs');
    });

    it('accepts zero values', () => {
      const a = array([0, 0, 0]);
      expect(asarray_chkfinite(a).toArray()).toEqual([0, 0, 0]);
    });
  });

  describe('ascontiguousarray additional', () => {
    it('converts with dtype', () => {
      const result = ascontiguousarray([1, 2, 3], 'int32');
      expect(result.dtype).toBe('int32');
    });
  });

  describe('asfortranarray additional', () => {
    it('accepts NDArray input', () => {
      const a = array([1, 2, 3]);
      const result = asfortranarray(a);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });
  });

  describe('array with Complex input', () => {
    it('creates complex128 array from Complex objects', () => {
      const result = array([new Complex(1, 2), new Complex(3, 4)]);
      expect(result.dtype).toBe('complex128');
      expect(result.shape).toEqual([2]);
    });

    it('creates array from NDArrayCore with dtype conversion', () => {
      const original = array([1, 2, 3], 'float64');
      const result = array(original, 'int32');
      expect(result.dtype).toBe('int32');
    });

    it('creates copy from NDArrayCore without dtype', () => {
      const original = array([1, 2, 3]);
      const result = array(original);
      expect(result.toArray()).toEqual(original.toArray());
      expect(result).not.toBe(original);
    });

    it('creates array with bigint values', () => {
      const result = array([BigInt(1), BigInt(2), BigInt(3)]);
      expect(result.dtype).toBe('int64');
    });
  });

  describe('genfromtxt', () => {
    it('parses CSV text', () => {
      const text = '1,2,3\n4,5,6';
      const result = genfromtxt(text, { delimiter: ',' });
      expect(result.shape).toEqual([2, 3]);
    });

    it('parses with whitespace delimiter', () => {
      const text = '1 2 3\n4 5 6';
      const result = genfromtxt(text);
      expect(result.shape).toEqual([2, 3]);
    });

    it('handles missing values with NaN', () => {
      const text = '1,2,\n4,,6';
      const result = genfromtxt(text, { delimiter: ',' });
      expect(result.shape).toEqual([2, 3]);
      // Missing values should become NaN
      const data = result.toArray() as number[][];
      expect(isNaN(data[0]![2]!)).toBe(true);
    });

    it('handles skiprows', () => {
      const text = 'header\n1,2,3\n4,5,6';
      const result = genfromtxt(text, { delimiter: ',', skiprows: 1 });
      expect(result.shape).toEqual([2, 3]);
    });
  });

  describe('fromregex', () => {
    it('extracts data from regex with capture groups', () => {
      const text = 'x=1.0, y=2.0\nx=3.0, y=4.0';
      const result = fromregex(text, /x=([\d.]+), y=([\d.]+)/);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
    });

    it('extracts single column data', () => {
      const text = 'val=10\nval=20\nval=30';
      const result = fromregex(text, /val=(\d+)/);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([10, 20, 30]);
    });

    it('returns empty array when no matches', () => {
      const text = 'no matches here';
      const result = fromregex(text, /x=(\d+)/);
      expect(result.shape).toEqual([0]);
    });

    it('works with string regex pattern', () => {
      const text = 'a=1 b=2\na=3 b=4';
      const result = fromregex(text, 'a=(\\d+) b=(\\d+)');
      expect(result.shape).toEqual([2, 2]);
    });

    it('handles non-numeric captures as zero', () => {
      const text = 'name=abc value=42';
      const result = fromregex(text, /name=(\w+) value=(\d+)/);
      // 'abc' can't be parsed as float, becomes 0
      // Single match with 2 groups -> 2D array [[0, 42]]
      const data = result.toArray() as number[][];
      expect(data[0]![0]).toBe(0);
      expect(data[0]![1]).toBe(42);
    });

    it('creates int32 array from regex', () => {
      const text = 'x=1, y=2\nx=3, y=4';
      const result = fromregex(text, /x=(\d+), y=(\d+)/, 'int32');
      expect(result.dtype).toBe('int32');
    });
  });

  describe('diag additional', () => {
    it('extracts diagonal with negative offset from 2D', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diag(arr, -1);
      expect(result.toArray()).toEqual([4, 8]);
    });

    it('extracts diagonal from non-square matrix', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = diag(arr);
      expect(result.toArray()).toEqual([1, 6]);
    });

    it('returns empty for out-of-range diagonal offset', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = diag(arr, 5);
      expect(result.shape).toEqual([0]);
    });

    it('throws for 3D input', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]);
      expect(() => diag(arr)).toThrow('Input must be 1-D or 2-D');
    });
  });

  describe('eye additional', () => {
    it('creates eye with bigint dtype', () => {
      const arr = eye(2, 2, 0, 'int64');
      expect(arr.dtype).toBe('int64');
      expect(arr.shape).toEqual([2, 2]);
    });
  });

  describe('tri additional', () => {
    it('creates tri with negative k', () => {
      const result = tri(3, 3, -1);
      expect(result.toArray()).toEqual([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
      ]);
    });
  });

  describe('tril/triu with 3D arrays', () => {
    it('tril works with 3D array (batch)', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = tril(arr);
      const data = result.toArray() as number[][][];
      expect(data[0]).toEqual([
        [1, 0],
        [3, 4],
      ]);
      expect(data[1]).toEqual([
        [5, 0],
        [7, 8],
      ]);
    });

    it('triu works with 3D array (batch)', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = triu(arr);
      const data = result.toArray() as number[][][];
      expect(data[0]).toEqual([
        [1, 2],
        [0, 4],
      ]);
      expect(data[1]).toEqual([
        [5, 6],
        [0, 8],
      ]);
    });

    it('tril throws for 1D input', () => {
      expect(() => tril(array([1, 2, 3]))).toThrow('Input must be at least 2-D');
    });

    it('triu throws for 1D input', () => {
      expect(() => triu(array([1, 2, 3]))).toThrow('Input must be at least 2-D');
    });
  });

  describe('full with edge cases', () => {
    it('full with bool dtype converts boolean', () => {
      const result = full([3], false, 'bool');
      expect(result.dtype).toBe('bool');
      expect(result.toArray()).toEqual([0, 0, 0]);
    });

    it('full with bigint dtype from number fill', () => {
      const result = full([2], 5, 'int64');
      expect(result.dtype).toBe('int64');
    });
  });

  describe('vander additional', () => {
    it('vander with N=1', () => {
      const x = array([1, 2, 3]);
      const result = vander(x, 1);
      expect(result.shape).toEqual([3, 1]);
      expect(result.toArray()).toEqual([[1], [1], [1]]);
    });
  });
});
