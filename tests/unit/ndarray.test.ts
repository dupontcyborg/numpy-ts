import { describe, it, expect } from 'vitest';
import { zeros, ones, array, Complex } from '../../src';

describe('NDArray Creation', () => {
  describe('zeros', () => {
    it('creates 1D array of zeros', () => {
      const arr = zeros([5]);
      expect(arr.shape).toEqual([5]);
      expect(arr.ndim).toBe(1);
      expect(arr.size).toBe(5);
      expect(arr.dtype).toBe('float64');
      expect(Array.from(arr.data)).toEqual([0, 0, 0, 0, 0]);
    });

    it('creates 2D array of zeros', () => {
      const arr = zeros([2, 3]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.ndim).toBe(2);
      expect(arr.size).toBe(6);
      expect(Array.from(arr.data)).toEqual([0, 0, 0, 0, 0, 0]);
    });

    it('creates 3D array of zeros', () => {
      const arr = zeros([2, 3, 4]);
      expect(arr.shape).toEqual([2, 3, 4]);
      expect(arr.ndim).toBe(3);
      expect(arr.size).toBe(24);
    });
  });

  describe('ones', () => {
    it('creates 1D array of ones', () => {
      const arr = ones([3]);
      expect(arr.shape).toEqual([3]);
      expect(Array.from(arr.data)).toEqual([1, 1, 1]);
    });

    it('creates 2D array of ones', () => {
      const arr = ones([2, 2]);
      expect(arr.shape).toEqual([2, 2]);
      expect(Array.from(arr.data)).toEqual([1, 1, 1, 1]);
    });
  });

  describe('array', () => {
    it('creates 1D array from flat array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      expect(arr.shape).toEqual([5]);
      expect(arr.ndim).toBe(1);
      expect(Array.from(arr.data)).toEqual([1, 2, 3, 4, 5]);
    });

    it('creates 2D array from nested arrays', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.shape).toEqual([2, 3]);
      expect(arr.ndim).toBe(2);
      expect(arr.size).toBe(6);
      expect(Array.from(arr.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('creates 3D array from nested arrays', () => {
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
      expect(arr.shape).toEqual([2, 2, 2]);
      expect(arr.ndim).toBe(3);
      expect(arr.size).toBe(8);
    });
  });
});

describe('NDArray Operations', () => {
  describe('matmul', () => {
    it('multiplies 2x2 matrices', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([
        [5, 6],
        [7, 8],
      ]);
      const C = A.matmul(B);

      expect(C.shape).toEqual([2, 2]);
      expect(C.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('multiplies 2x3 and 3x2 matrices', () => {
      const A = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const B = array([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const C = A.matmul(B);

      expect(C.shape).toEqual([2, 2]);
      // A @ B = [[1*7+2*9+3*11, 1*8+2*10+3*12],
      //          [4*7+5*9+6*11, 4*8+5*10+6*12]]
      //       = [[58, 64], [139, 154]]
      expect(C.toArray()).toEqual([
        [58, 64],
        [139, 154],
      ]);
    });

    it('throws on shape mismatch', () => {
      const A = array([
        [1, 2],
        [3, 4],
      ]);
      const B = array([[1, 2, 3]]);

      expect(() => A.matmul(B)).toThrow('matmul shape mismatch');
    });

    it('throws on non-2D arrays', () => {
      const A = array([1, 2, 3]);
      const B = array([4, 5, 6]);

      expect(() => A.matmul(B)).toThrow('matmul requires 2D arrays');
    });
  });

  describe('toArray', () => {
    it('converts 1D array to JavaScript array', () => {
      const arr = array([1, 2, 3]);
      expect(arr.toArray()).toEqual([1, 2, 3]);
    });

    it('converts 2D array to nested JavaScript array', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      expect(arr.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });
});

describe('NDArray Properties', () => {
  it('has correct strides for 2D C-order array', () => {
    const arr = zeros([3, 4]);
    // C-order: last dimension varies fastest
    // For [3, 4], strides should be [4, 1] (in elements, not bytes)
    // @stdlib uses byte strides, so [4*8, 1*8] = [32, 8] for float64
    expect(arr.strides).toEqual([4, 1]);
  });

  it('toString returns array contents like NumPy', () => {
    const arr = zeros([2, 3]);
    const str = arr.toString();
    expect(str).toContain('[[');
    expect(str).toContain('0.');
    expect(str).not.toContain('shape');
  });
});

describe('NDArray Slicing', () => {
  describe('1D slicing', () => {
    it('slices with start:stop', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('2:7');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([2, 3, 4, 5, 6]);
    });

    it('slices with ":" (full slice)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.slice(':');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('slices with no arguments (returns full array)', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const result = arr.slice();
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('slices with "5:" (start to end)', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('5:');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([5, 6, 7, 8, 9]);
    });

    it('slices with ":5" (beginning to stop)', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice(':5');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('slices with negative start', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('-3:');
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([7, 8, 9]);
    });

    it('slices with negative stop', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice(':-3');
      expect(result.shape).toEqual([7]);
      expect(result.toArray()).toEqual([0, 1, 2, 3, 4, 5, 6]);
    });

    it('slices with negative start and stop', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('-7:-2');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([3, 4, 5, 6, 7]);
    });

    it('slices single index', () => {
      const arr = array([0, 1, 2, 3, 4, 5]);
      const result = arr.slice('2');
      expect(result.shape).toEqual([]);
      expect(result.toArray()).toEqual(2);
    });

    it('slices with negative index', () => {
      const arr = array([0, 1, 2, 3, 4, 5]);
      const result = arr.slice('-1');
      expect(result.shape).toEqual([]);
      expect(result.toArray()).toEqual(5);
    });

    it('slices with step', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('::2');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('slices with start:stop:step', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('1:8:2');
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([1, 3, 5, 7]);
    });

    it('slices with negative step (reverse)', () => {
      const arr = array([0, 1, 2, 3, 4]);
      const result = arr.slice('::-1');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([4, 3, 2, 1, 0]);
    });

    it('slices with negative step and bounds', () => {
      const arr = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const result = arr.slice('7:2:-1');
      expect(result.shape).toEqual([5]);
      expect(result.toArray()).toEqual([7, 6, 5, 4, 3]);
    });
  });

  describe('2D slicing', () => {
    it('slices single row with index', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('0', ':');
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('slices single column with index', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice(':', '1');
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([2, 5, 8]);
    });

    it('slices rows range', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('0:2', ':');
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('slices columns range', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice(':', '1:3');
      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [2, 3],
        [5, 6],
        [8, 9],
      ]);
    });

    it('slices submatrix', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);
      const result = arr.slice('1:3', '1:3');
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [6, 7],
        [10, 11],
      ]);
    });

    it('slices with negative indices', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('-2:', '-2:');
      expect(result.shape).toEqual([2, 2]);
      // -2: means from index 1 to end (indices 1, 2)
      // So we get rows [1, 2] and cols [1, 2]
      expect(result.toArray()).toEqual([
        [5, 6],
        [8, 9],
      ]);
    });

    it('slices bottom-right corner', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = arr.slice('-1', '-1');
      expect(result.shape).toEqual([]);
      expect(result.toArray()).toEqual(9);
    });

    it('slices with step on rows', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const result = arr.slice('::2', ':');
      expect(result.shape).toEqual([2, 3]);
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [7, 8, 9],
      ]);
    });

    it('slices with step on columns', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = arr.slice(':', '::2');
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [5, 7],
      ]);
    });

    it('slices with step on both dimensions', () => {
      const arr = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);
      const result = arr.slice('::2', '::2');
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [9, 11],
      ]);
    });
  });

  describe('convenience methods', () => {
    describe('row()', () => {
      it('gets single row', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = arr.row(1);
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([4, 5, 6]);
      });

      it('gets first row', () => {
        const arr = array([
          [1, 2],
          [3, 4],
        ]);
        const result = arr.row(0);
        expect(result.toArray()).toEqual([1, 2]);
      });

      it('gets last row with negative index', () => {
        const arr = array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const result = arr.row(-1);
        expect(result.toArray()).toEqual([5, 6]);
      });

      it('throws on 1D array', () => {
        const arr = array([1, 2, 3]);
        expect(() => arr.row(0)).toThrow('requires at least 2 dimensions');
      });
    });

    describe('col()', () => {
      it('gets single column', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = arr.col(1);
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([2, 5, 8]);
      });

      it('gets first column', () => {
        const arr = array([
          [1, 2],
          [3, 4],
        ]);
        const result = arr.col(0);
        expect(result.toArray()).toEqual([1, 3]);
      });

      it('gets last column with negative index', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.col(-1);
        expect(result.toArray()).toEqual([3, 6]);
      });

      it('throws on 1D array', () => {
        const arr = array([1, 2, 3]);
        expect(() => arr.col(0)).toThrow('requires at least 2 dimensions');
      });
    });

    describe('rows()', () => {
      it('gets range of rows', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12],
        ]);
        const result = arr.rows(1, 3);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([
          [4, 5, 6],
          [7, 8, 9],
        ]);
      });

      it('gets first two rows', () => {
        const arr = array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const result = arr.rows(0, 2);
        expect(result.toArray()).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('throws on 1D array', () => {
        const arr = array([1, 2, 3]);
        expect(() => arr.rows(0, 2)).toThrow('requires at least 2 dimensions');
      });
    });

    describe('cols()', () => {
      it('gets range of columns', () => {
        const arr = array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ]);
        const result = arr.cols(1, 3);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([
          [2, 3],
          [6, 7],
          [10, 11],
        ]);
      });

      it('gets last two columns', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = arr.cols(1, 3);
        expect(result.toArray()).toEqual([
          [2, 3],
          [5, 6],
        ]);
      });

      it('throws on 1D array', () => {
        const arr = array([1, 2, 3]);
        expect(() => arr.cols(0, 2)).toThrow('requires at least 2 dimensions');
      });
    });
  });

  describe('error cases', () => {
    it('throws on too many indices', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.slice('0', '0')).toThrow(/Too many indices/);
    });

    it('throws on out of bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.slice('10')).toThrow(/out of bounds/);
    });

    it('throws on negative out of bounds index', () => {
      const arr = array([1, 2, 3]);
      expect(() => arr.slice('-10')).toThrow(/out of bounds/);
    });
  });
});

describe('NDArrayCore specific', () => {
  it('fill with Complex', () => {
    const a = array([new Complex(0, 0), new Complex(0, 0)], 'complex128');
    a.fill(new Complex(3, 4) as any);
    expect(a.size).toBe(2);
  });

  it('astype complex to float', () => {
    const a = array([new Complex(1, 0), new Complex(2, 0)], 'complex128');
    const b = a.astype('float64');
    expect(b.dtype).toBe('float64');
  });

  it('iterator 3D', () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const items = [...a];
    expect(items.length).toBe(2);
  });

  it('toArray on 0D', () => {
    const a = array([42]).reshape();
    if (a.ndim === 0) {
      const v = a.toArray();
      expect(v).toBeDefined();
    }
  });

  it('tobytes non-contiguous', () => {
    const a = array([1, 2, 3, 4, 5, 6], 'int32');
    const s = a.slice('::2');
    const b = s.tobytes();
    expect(b).toBeInstanceOf(ArrayBuffer);
  });

  it('get/set with negative indices', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(a.get([-1, -1])).toBe(4);
    a.set([-1, -1], 99);
    expect(a.get([1, 1])).toBe(99);
  });
});
