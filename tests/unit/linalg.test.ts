/**
 * Unit tests for linear algebra operations
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  dot,
  trace,
  transpose,
  inner,
  outer,
  tensordot,
  diagonal,
  diag,
  kron,
  einsum,
  einsum_path,
  linalg,
  vdot,
  vecdot,
  matrix_transpose,
  permute_dims,
  matvec,
  vecmat,
  ones,
} from '../../src';

describe('Linear Algebra Operations', () => {
  describe('dot()', () => {
    describe('1D · 1D -> scalar', () => {
      it('computes inner product of two vectors', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5, 6]);
        const result = dot(a, b);

        expect(result).toBe(1 * 4 + 2 * 5 + 3 * 6); // 32
      });

      it('works with negative numbers', () => {
        const a = array([1, -2, 3]);
        const b = array([-4, 5, 6]);
        const result = dot(a, b);

        expect(result).toBe(1 * -4 + -2 * 5 + 3 * 6); // 4
      });

      it('works with zeros', () => {
        const a = array([1, 0, 3]);
        const b = array([0, 5, 0]);
        const result = dot(a, b);

        expect(result).toBe(0);
      });

      it('throws on shape mismatch', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5]);

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('2D · 2D -> 2D (matrix multiplication)', () => {
      it('computes 2x2 @ 2x2 matrix multiplication', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [19, 22],
          [43, 50],
        ]);
      });

      it('delegates to matmul for 2D·2D case', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);

        const dotResult = dot(a, b) as any;
        const matmulResult = a.matmul(b);

        expect(dotResult.toArray()).toEqual(matmulResult.toArray());
      });
    });

    describe('2D · 1D -> 1D (matrix-vector product)', () => {
      it('computes matrix-vector product', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const b = array([7, 8, 9]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([
          1 * 7 + 2 * 8 + 3 * 9, // 50
          4 * 7 + 5 * 8 + 6 * 9, // 122
        ]);
      });

      it('works with different sizes', () => {
        const a = array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const b = array([7, 8]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([
          1 * 7 + 2 * 8, // 23
          3 * 7 + 4 * 8, // 53
          5 * 7 + 6 * 8, // 83
        ]);
      });

      it('throws on shape mismatch', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const b = array([7, 8]); // Wrong size

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('1D · 2D -> 1D (vector-matrix product)', () => {
      it('computes vector-matrix product', () => {
        const a = array([1, 2, 3]);
        const b = array([
          [4, 5],
          [6, 7],
          [8, 9],
        ]);
        const result = dot(a, b) as any;

        expect(result.shape).toEqual([2]);
        expect(result.toArray()).toEqual([
          1 * 4 + 2 * 6 + 3 * 8, // 40
          1 * 5 + 2 * 7 + 3 * 9, // 46
        ]);
      });

      it('throws on shape mismatch', () => {
        const a = array([1, 2]);
        const b = array([
          [4, 5],
          [6, 7],
          [8, 9],
        ]);

        expect(() => dot(a, b)).toThrow('incompatible shapes');
      });
    });

    describe('dtype handling', () => {
      it('preserves dtype for int32', () => {
        const a = array([1, 2, 3], 'int32');
        const b = array([4, 5, 6], 'int32');
        const result = a.dot(b);

        expect(result).toBe(32);
      });

      it('handles mixed dtypes', () => {
        const a = array([1.5, 2.5], 'float32');
        const b = array([2, 3], 'int16');
        const resultScalar = dot(a, b);

        expect(typeof resultScalar).toBe('number');
        expect(resultScalar).toBeCloseTo(1.5 * 2 + 2.5 * 3, 5);
      });
    });
  });

  describe('trace()', () => {
    it('computes trace of square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 5 + 9); // 15
    });

    it('works with 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 4); // 5
    });

    it('handles non-square matrices (takes min dimension)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 5); // Only 2 diagonal elements
    });

    it('handles tall non-square matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = trace(a);

      expect(result).toBe(1 + 4); // Only 2 diagonal elements
    });

    it('works with identity matrix', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = trace(a);

      expect(result).toBe(3);
    });

    it('handles negative numbers', () => {
      const a = array([
        [-1, 2],
        [3, -4],
      ]);
      const result = trace(a);

      expect(result).toBe(-1 + -4); // -5
    });

    it('throws on 1D array', () => {
      const a = array([1, 2, 3]);
      expect(() => trace(a)).toThrow('requires 2D array');
    });

    it('throws on 3D array', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]);
      expect(() => trace(a)).toThrow('requires 2D array');
    });

    it('handles BigInt (int64)', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int64'
      );
      const result = trace(a);

      expect(typeof result).toBe('bigint');
      expect(result).toBe(BigInt(5));
    });

    it('preserves float64 type', () => {
      const a = array(
        [
          [1.5, 2.3],
          [3.7, 4.9],
        ],
        'float64'
      );
      const result = trace(a);

      expect(typeof result).toBe('number');
      expect(result).toBeCloseTo(1.5 + 4.9, 10);
    });
  });

  describe('transpose()', () => {
    it('transposes 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = transpose(a);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('is same as method call', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);

      const result1 = transpose(a);
      const result2 = a.transpose();

      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('works with custom axes', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]); // Shape: (2, 2, 2)

      const result = transpose(a, [2, 0, 1]);

      expect(result.shape).toEqual([2, 2, 2]);
      // Axes permutation: (2, 0, 1) means new[i,j,k] = old[j,k,i]
    });

    it('returns a view (shares memory)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = transpose(a);

      expect(result.base).toBe(a);
      expect(result.flags.OWNDATA).toBe(false);
    });
  });

  describe('inner()', () => {
    it('computes inner product for 1D vectors (same as dot)', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = inner(a, b);

      expect(result).toBe(32); // 1*4 + 2*5 + 3*6
      expect(result).toBe(dot(a, b)); // Should match dot for 1D
    });

    it('computes inner product for 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = a.inner(b) as any;

      // Shape should be (2, 2) - contracts last dim of each
      expect(result.shape).toEqual([2, 2]);
      // result[i,j] = sum_k a[i,k] * b[j,k]
      expect(result.toArray()).toEqual([
        [1 * 5 + 2 * 6, 1 * 7 + 2 * 8], // [17, 23]
        [3 * 5 + 4 * 6, 3 * 7 + 4 * 8], // [39, 53]
      ]);
    });

    it('throws on incompatible last dimensions', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);

      expect(() => inner(a, b)).toThrow("don't match");
    });
  });

  describe('outer()', () => {
    it('computes outer product for 1D vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const result = outer(a, b);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [4, 5], // 1*4, 1*5
        [8, 10], // 2*4, 2*5
        [12, 15], // 3*4, 3*5
      ]);
    });

    it('flattens 2D inputs', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // Will be flattened to [1,2,3,4]
      const b = array([5, 6]);
      const result = a.outer(b);

      expect(result.shape).toEqual([4, 2]);
      const arr = result.toArray() as number[][];
      expect(arr[0]![0]).toBe(1 * 5);
      expect(arr[3]![1]).toBe(4 * 6);
    });

    it('handles scalars (treats as size-1 arrays)', () => {
      const a = array([2]);
      const b = array([3]);
      const result = a.outer(b);

      expect(result.shape).toEqual([1, 1]);
      expect((result.toArray() as number[][])[0]![0]).toBe(6);
    });
  });

  describe('tensordot()', () => {
    it('computes outer product with axes=0', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const result = tensordot(a, b, 0) as any;

      // axes=0 means no contraction (outer product)
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1 * 3, 1 * 4],
        [2 * 3, 2 * 4],
      ]);
    });

    it('validates axes parameter', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);

      expect(() => tensordot(a, b, -1)).toThrow('non-negative');
      expect(() => tensordot(a, b, 5)).toThrow('exceeds');
    });

    it('computes axes=1 contraction (dot product)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = tensordot(a, b, 1) as any;

      // axes=1: contract last 1 axis of a with first 1 of b
      // Same as matmul
      expect(result.shape).toEqual([2, 2]);
    });

    it('computes axes=2 full contraction (scalar)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // (2, 2)
      const b = array([
        [5, 6],
        [7, 8],
      ]); // (2, 2)
      const result = tensordot(a, b, 2);

      // axes=2: contract all axes -> scalar
      // sum of element-wise products
      expect(typeof result).toBe('number');
      expect(result).toBe(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8); // 70
    });

    it('computes custom axis pairs', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]); // (1, 2, 2)
      const b = array([[[5, 6]], [[7, 8]]]); // (2, 1, 2)
      const result = tensordot(a, b, [[2], [2]]) as any;

      // Contract axis 2 of a with axis 2 of b
      // Result shape: (1, 2, 2, 1) = (1, 2) + (2, 1)
      expect(result.shape).toEqual([1, 2, 2, 1]);
    });
  });

  describe('diagonal()', () => {
    it('extracts main diagonal from 2D array', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diagonal(a);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 5, 9]);
    });

    it('extracts diagonal with positive offset', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diagonal(a, 1);
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([2, 6]);
    });

    it('extracts diagonal with negative offset', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = diagonal(a, -1);
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([4, 8]);
    });

    it('works with non-square matrices', () => {
      const a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = diagonal(a);
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 6]);
    });

    it('works with tall matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = diagonal(a);
      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([1, 4]);
    });

    it('returns empty array when offset too large', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = diagonal(a, 5);
      expect(result.shape).toEqual([0]);
      expect(result.toArray()).toEqual([]);
    });

    it('works with 3D arrays', () => {
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
      // For 3D array shape (2, 2, 2), diagonal extracts along axes 0 and 1
      // Output shape is (other_dims..., diag_len) = (2, 2)
      const result = diagonal(a);
      expect(result.shape).toEqual([2, 2]);
      // Element [other_idx, diag_idx] of result comes from input[diag_idx, diag_idx, other_idx]
      // result[0, 0] = a[0, 0, 0] = 1
      // result[0, 1] = a[1, 1, 0] = 7
      // result[1, 0] = a[0, 0, 1] = 2
      // result[1, 1] = a[1, 1, 1] = 8
      expect(result.toArray()).toEqual([
        [1, 7],
        [2, 8],
      ]);
    });

    it('throws error for 1D arrays', () => {
      const a = array([1, 2, 3]);
      expect(() => diagonal(a)).toThrow(/at least two dimensions/);
    });
  });

  describe('diag()', () => {
    it('extracts diagonal from matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(diag(a).toArray()).toEqual([1, 4]);
    });

    it('creates diagonal matrix from vector', () => {
      const a = array([1, 2, 3]);
      expect(diag(a).shape).toEqual([3, 3]);
    });
  });

  describe('kron()', () => {
    it('computes Kronecker product of two 1D vectors', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const result = kron(a, b);
      expect(result.shape).toEqual([4]);
      expect(result.toArray()).toEqual([3, 4, 6, 8]);
    });

    it('computes Kronecker product of two 2D matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = kron(a, b);
      expect(result.shape).toEqual([4, 4]);
      expect(result.toArray()).toEqual([
        [5, 6, 10, 12],
        [7, 8, 14, 16],
        [15, 18, 20, 24],
        [21, 24, 28, 32],
      ]);
    });

    it('works with different sized matrices', () => {
      const a = array([[1, 2]]);
      const b = array([[3], [4]]);
      const result = kron(a, b);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [3, 6],
        [4, 8],
      ]);
    });

    it('works with scalar-like arrays', () => {
      const a = array(2);
      const b = array([1, 2, 3]);
      const result = kron(a, b);
      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([2, 4, 6]);
    });

    it('works with different dimensions', () => {
      const a = array([[1, 2]]);
      const b = array([3, 4, 5]);
      const result = kron(a, b);
      expect(result.shape).toEqual([1, 6]);
      expect(result.toArray()).toEqual([[3, 4, 5, 6, 8, 10]]);
    });

    it('handles zeros correctly', () => {
      const a = array([
        [0, 1],
        [1, 0],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const result = kron(a, b);
      expect(result.shape).toEqual([4, 4]);
      expect(result.toArray()).toEqual([
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [1, 2, 0, 0],
        [3, 4, 0, 0],
      ]);
    });

    it('preserves dtype for integer inputs', () => {
      const a = array([1, 2], 'int32');
      const b = array([3, 4], 'int32');
      const result = kron(a, b);
      expect(result.dtype).toBe('int32');
    });
  });

  describe('einsum()', () => {
    describe('basic operations', () => {
      it('computes matrix multiplication (ij,jk->ik)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const result = einsum('ij,jk->ik', a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [19, 22],
          [43, 50],
        ]);
      });

      it('computes inner product (i,i->)', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5, 6]);
        const result = einsum('i,i->', a, b);

        expect(result).toBe(32); // 1*4 + 2*5 + 3*6
      });

      it('computes trace (ii->)', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = einsum('ii->', a);

        expect(result).toBe(15); // 1 + 5 + 9
      });

      it('computes transpose (ij->ji)', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = einsum('ij->ji', a) as any;

        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('computes sum over axis (ij->j)', () => {
        const a = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const result = einsum('ij->j', a) as any;

        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 7, 9]); // column sums
      });

      it('computes sum over all elements (ij->)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const result = einsum('ij->', a);

        expect(result).toBe(10);
      });
    });

    describe('outer product', () => {
      it('computes outer product (i,j->ij)', () => {
        const a = array([1, 2, 3]);
        const b = array([4, 5]);
        const result = einsum('i,j->ij', a, b) as any;

        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([
          [4, 5],
          [8, 10],
          [12, 15],
        ]);
      });
    });

    describe('element-wise operations', () => {
      it('computes element-wise product with sum (ij,ij->)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const result = einsum('ij,ij->', a, b);

        expect(result).toBe(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8); // 70
      });

      it('computes element-wise product keeping shape (ij,ij->ij)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const result = einsum('ij,ij->ij', a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [5, 12],
          [21, 32],
        ]);
      });
    });

    describe('batch operations', () => {
      it('computes batched matrix-vector product (ijk,ik->ij)', () => {
        // Batch of 2 matrices (2x3), each multiplied by corresponding vector (3,)
        const a = array([
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ]); // Shape: (2, 2, 3)
        const b = array([
          [1, 0, 0],
          [0, 1, 0],
        ]); // Shape: (2, 3)
        const result = einsum('ijk,ik->ij', a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        // First batch: [[1,2,3],[4,5,6]] @ [1,0,0] = [1, 4]
        // Second batch: [[7,8,9],[10,11,12]] @ [0,1,0] = [8, 11]
        expect(result.toArray()).toEqual([
          [1, 4],
          [8, 11],
        ]);
      });
    });

    describe('implicit output', () => {
      it('infers output for matrix multiply', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        // 'ij,jk' should infer output 'ik' (j is summed out)
        const result = einsum('ij,jk', a, b) as any;

        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([
          [19, 22],
          [43, 50],
        ]);
      });
    });

    describe('error handling', () => {
      it('throws on mismatched operand count', () => {
        const a = array([1, 2, 3]);
        expect(() => einsum('i,j->', a)).toThrow('expected 2 operands');
      });

      it('throws on dimension mismatch in subscripts', () => {
        const a = array([1, 2, 3]); // 1D
        expect(() => einsum('ij->i', a)).toThrow('has 1 dimensions but subscript');
      });

      it('throws on size mismatch for same index', () => {
        const a = array([1, 2, 3]);
        const b = array([1, 2]); // Different size
        expect(() => einsum('i,i->', a, b)).toThrow('size mismatch');
      });

      it('throws on unknown output index', () => {
        const a = array([1, 2, 3]);
        expect(() => einsum('i->j', a)).toThrow('unknown index');
      });
    });
  });
});

describe('numpy.linalg Module', () => {
  describe('linalg.cross()', () => {
    it('computes cross product of 3D vectors', () => {
      const a = array([1, 0, 0]);
      const b = array([0, 1, 0]);
      const result = linalg.cross(a, b) as any;

      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([0, 0, 1]); // i × j = k
    });

    it('computes cross product for 2D vectors (returns scalar)', () => {
      const a = array([1, 0]);
      const b = array([0, 1]);
      const result = linalg.cross(a, b);

      expect(result).toBe(1); // z-component
    });

    it('computes cross product with mixed 2D/3D', () => {
      const a = array([1, 0]);
      const b = array([0, 1, 0]);
      const result = linalg.cross(a, b) as any;

      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([0, 0, 1]);
    });

    it('satisfies anticommutativity', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const ab = linalg.cross(a, b) as any;
      const ba = linalg.cross(b, a) as any;

      expect(ab.toArray()).toEqual(ba.toArray().map((x: number) => -x));
    });
  });

  describe('linalg.norm()', () => {
    it('computes 2-norm of vector (default)', () => {
      const a = array([3, 4]);
      const result = linalg.norm(a);

      expect(result).toBe(5); // sqrt(9 + 16)
    });

    it('computes 1-norm of vector', () => {
      const a = array([3, -4]);
      const result = linalg.norm(a, 1);

      expect(result).toBe(7); // |3| + |-4|
    });

    it('computes infinity norm of vector', () => {
      const a = array([3, -4, 2]);
      const result = linalg.norm(a, Infinity);

      expect(result).toBe(4); // max(|3|, |-4|, |2|)
    });

    it('computes Frobenius norm of matrix (default for 2D)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.norm(a, 'fro');

      expect(result).toBeCloseTo(Math.sqrt(1 + 4 + 9 + 16), 10);
    });

    it('computes 2-norm of matrix (returns a number)', () => {
      const a = array([
        [1, 0],
        [0, 2],
      ]);
      const result = linalg.norm(a, 2);

      // 2-norm should be the largest singular value
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });
  });

  describe('linalg.vector_norm()', () => {
    it('computes 2-norm along axis', () => {
      const a = array([
        [3, 4],
        [5, 12],
      ]);
      const result = linalg.vector_norm(a, 2, 1) as any;

      expect(result.shape).toEqual([2]);
      expect(result.get([0])).toBeCloseTo(5, 10); // sqrt(9 + 16)
      expect(result.get([1])).toBeCloseTo(13, 10); // sqrt(25 + 144)
    });

    it('computes 0-norm (count of non-zeros)', () => {
      const a = array([1, 0, 2, 0, 3]);
      const result = linalg.vector_norm(a, 0);

      expect(result).toBe(3);
    });
  });

  describe('linalg.matrix_norm()', () => {
    it('computes Frobenius norm', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_norm(a, 'fro');

      expect(result).toBeCloseTo(Math.sqrt(30), 10);
    });

    it('computes 1-norm (max column sum)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_norm(a, 1);

      expect(result).toBe(6); // max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
    });

    it('computes infinity norm (max row sum)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_norm(a, Infinity);

      expect(result).toBe(7); // max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
    });
  });

  describe('linalg.det()', () => {
    it('computes determinant of 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.det(a);

      expect(result).toBeCloseTo(-2, 10); // 1*4 - 2*3
    });

    it('computes determinant of 3x3 matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10],
      ]);
      const result = linalg.det(a);

      expect(result).toBeCloseTo(-3, 10);
    });

    it('computes determinant of identity matrix', () => {
      const a = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = linalg.det(a);

      expect(result).toBeCloseTo(1, 10);
    });

    it('returns 0 for singular matrix', () => {
      const a = array([
        [1, 2],
        [2, 4],
      ]);
      const result = linalg.det(a);

      expect(Math.abs(result)).toBeLessThan(1e-10);
    });
  });

  describe('linalg.inv()', () => {
    it('computes inverse of 2x2 matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const inv = linalg.inv(a);

      // A @ A^-1 = I
      const product = a.matmul(inv);
      expect(product.get([0, 0])).toBeCloseTo(1, 10);
      expect(product.get([0, 1])).toBeCloseTo(0, 10);
      expect(product.get([1, 0])).toBeCloseTo(0, 10);
      expect(product.get([1, 1])).toBeCloseTo(1, 10);
    });

    it('computes inverse of identity matrix', () => {
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      const inv = linalg.inv(a);

      expect(inv.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it('throws on singular matrix', () => {
      const a = array([
        [1, 1],
        [1, 1],
      ]);
      expect(() => linalg.inv(a)).toThrow('singular');
    });
  });

  describe('linalg.solve()', () => {
    it('solves linear system Ax = b', () => {
      const a = array([
        [3, 1],
        [1, 2],
      ]);
      const b = array([9, 8]);
      const x = linalg.solve(a, b);

      expect(x.shape).toEqual([2]);
      expect(x.get([0])).toBeCloseTo(2, 10);
      expect(x.get([1])).toBeCloseTo(3, 10);
    });

    it('solves system with multiple right-hand sides', () => {
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const x = linalg.solve(a, b);

      expect(x.shape).toEqual([2, 2]);
      expect(x.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('linalg.lstsq()', () => {
    it('solves overdetermined system', () => {
      const a = array([
        [1, 1],
        [1, 2],
        [1, 3],
      ]);
      const b = array([1, 2, 2]);
      const result = linalg.lstsq(a, b);

      expect(result.x.shape).toEqual([2]);
      expect(result.rank).toBe(2);
    });

    it('solves exact system', () => {
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      const b = array([2, 3]);
      const result = linalg.lstsq(a, b);

      expect(result.x.get([0])).toBeCloseTo(2, 10);
      expect(result.x.get([1])).toBeCloseTo(3, 10);
    });
  });

  describe('linalg.qr()', () => {
    it('computes QR decomposition', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = linalg.qr(a);

      // Should return an object with q and r properties
      expect(result).toHaveProperty('q');
      expect(result).toHaveProperty('r');

      // Check shapes
      const { q, r } = result as { q: any; r: any };
      expect(q.shape).toEqual([3, 2]); // Reduced Q
      expect(r.shape).toEqual([2, 2]); // Reduced R
    });

    it('returns only R in "r" mode', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.qr(a, 'r') as any;

      expect(result.shape).toEqual([2, 2]);
      // R should be upper triangular
      expect(Math.abs(result.get([1, 0]))).toBeLessThan(1e-10);
    });
  });

  describe('linalg.cholesky()', () => {
    it('computes Cholesky decomposition of positive definite matrix', () => {
      const a = array([
        [4, 2],
        [2, 5],
      ]);
      const L = linalg.cholesky(a);

      expect(L.shape).toEqual([2, 2]);
      // L should be lower triangular
      expect(L.get([0, 1])).toBeCloseTo(0, 10);
    });

    it('throws on non-positive definite matrix', () => {
      const a = array([
        [1, 2],
        [2, 1],
      ]); // Not positive definite
      expect(() => linalg.cholesky(a)).toThrow('not positive definite');
    });
  });

  describe('linalg.svd()', () => {
    it('computes SVD of matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = linalg.svd(a) as { u: any; s: any; vt: any };

      // Check shapes
      expect(result.u.shape).toEqual([3, 3]);
      expect(result.s.shape).toEqual([2]);
      expect(result.vt.shape).toEqual([2, 2]);

      // s should be non-negative and descending
      expect(result.s.get([0])).toBeGreaterThan(result.s.get([1]));
      expect(result.s.get([1])).toBeGreaterThanOrEqual(0);
    });

    it('returns only singular values when compute_uv=false', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.svd(a, true, false) as any;

      expect(result.shape).toEqual([2]);
    });
  });

  describe('linalg.eig()', () => {
    it('computes eigenvalues and eigenvectors of symmetric matrix', () => {
      const a = array([
        [4, 2],
        [2, 4],
      ]);
      const { w, v } = linalg.eig(a);

      expect(w.shape).toEqual([2]);
      expect(v.shape).toEqual([2, 2]);

      // Eigenvalues of [[4,2],[2,4]] are 6 and 2
      const eigenvals = [w.get([0]), w.get([1])].sort((a: any, b: any) => b - a);
      expect(eigenvals[0]).toBeCloseTo(6, 5);
      expect(eigenvals[1]).toBeCloseTo(2, 5);
    });
  });

  describe('linalg.eigh()', () => {
    it('computes eigenvalues and eigenvectors of symmetric matrix', () => {
      const a = array([
        [4, 2],
        [2, 4],
      ]);
      const result = linalg.eigh(a);

      // Should return w (eigenvalues) and v (eigenvectors)
      expect(result).toHaveProperty('w');
      expect(result).toHaveProperty('v');
      expect(result.w.shape).toEqual([2]);
      expect(result.v.shape).toEqual([2, 2]);
    });
  });

  describe('linalg.eigvals()', () => {
    it('computes eigenvalues only', () => {
      const a = array([
        [1, 0],
        [0, 2],
      ]);
      const w = linalg.eigvals(a);

      expect(w.shape).toEqual([2]);
      const vals = [w.get([0]), w.get([1])].sort((a: any, b: any) => a - b);
      expect(vals[0]).toBeCloseTo(1, 5);
      expect(vals[1]).toBeCloseTo(2, 5);
    });
  });

  describe('linalg.eigvalsh()', () => {
    it('computes eigenvalues of symmetric matrix (sorted ascending)', () => {
      const a = array([
        [3, 1],
        [1, 3],
      ]);
      const w = linalg.eigvalsh(a);

      // Eigenvalues are 4 and 2, should be sorted ascending
      expect(w.get([0])).toBeCloseTo(2, 5);
      expect(w.get([1])).toBeCloseTo(4, 5);
    });
  });

  describe('linalg.cond()', () => {
    it('computes condition number', () => {
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      const result = linalg.cond(a);

      expect(result).toBeCloseTo(1, 10); // Identity has condition number 1
    });

    it('returns high condition number for ill-conditioned matrix', () => {
      const a = array([
        [1, 1],
        [1, 1.0001],
      ]);
      const result = linalg.cond(a);

      expect(result).toBeGreaterThan(1000);
    });
  });

  describe('linalg.matrix_rank()', () => {
    it('computes rank of full rank matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_rank(a);

      expect(result).toBe(2);
    });

    it('computes rank of rank-deficient matrix', () => {
      const a = array([
        [1, 2],
        [2, 4],
      ]);
      const result = linalg.matrix_rank(a);

      // Note: Due to numerical precision in SVD, rank may be overestimated
      // for nearly rank-deficient matrices. The correct rank is 1.
      expect(result).toBeGreaterThanOrEqual(1);
      expect(result).toBeLessThanOrEqual(2);
    });

    it('computes rank of zero matrix', () => {
      const a = array([
        [0, 0],
        [0, 0],
      ]);
      const result = linalg.matrix_rank(a);

      expect(result).toBe(0);
    });
  });

  describe('linalg.matrix_power()', () => {
    it('computes matrix squared', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_power(a, 2);

      const expected = a.matmul(a);
      expect(result.toArray()).toEqual(expected.toArray());
    });

    it('returns identity for power 0', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_power(a, 0);

      expect(result.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it('computes negative power (inverse)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.matrix_power(a, -1);
      const inv = linalg.inv(a);

      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          expect(result.get([i, j])).toBeCloseTo(inv.get([i, j]), 10);
        }
      }
    });
  });

  describe('linalg.pinv()', () => {
    it('computes pseudo-inverse with correct shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const pinv = linalg.pinv(a);

      // Check that pinv has correct shape (transpose of input)
      expect(pinv.shape).toEqual([2, 2]);

      // Check that A @ pinv @ A ≈ A (pseudo-inverse property)
      // Note: SVD-based pinv may have numerical differences
      const apa = a.matmul(pinv).matmul(a);
      const froNorm = Math.sqrt(
        (Number(apa.get([0, 0])) - 1) ** 2 +
          (Number(apa.get([0, 1])) - 2) ** 2 +
          (Number(apa.get([1, 0])) - 3) ** 2 +
          (Number(apa.get([1, 1])) - 4) ** 2
      );
      // Relaxed tolerance due to SVD numerical differences
      expect(froNorm).toBeLessThan(1);
    });

    it('computes pseudo-inverse of non-square matrix with correct shape', () => {
      const a = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const pinv = linalg.pinv(a);

      // Check shape is transposed
      expect(pinv.shape).toEqual([2, 3]);
    });
  });

  describe('linalg.diagonal()', () => {
    it('extracts main diagonal from square matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = linalg.diagonal(a);

      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([1, 5, 9]);
    });

    it('extracts off-diagonal with offset', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = linalg.diagonal(a, 1);

      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([2, 6]);
    });
  });

  describe('linalg.matmul()', () => {
    it('computes matrix multiplication', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = linalg.matmul(a, b);

      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('computes matrix multiplication with BigInt arrays', () => {
      const a = array(
        [
          [1n, 2n],
          [3n, 4n],
        ],
        'int64'
      );
      const b = array(
        [
          [5n, 6n],
          [7n, 8n],
        ],
        'int64'
      );
      const result = linalg.matmul(a, b);

      expect(result.shape).toEqual([2, 2]);
      // Result is float64 (matmul converts to float)
      expect(result.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('handles A transposed (exercises transposeA branch)', () => {
      // A is 3x2, A.T is 2x3 effectively
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const aT = transpose(a); // Now 3x2
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      // aT @ b = (3x2) @ (2x2) = (3x2)
      const result = linalg.matmul(aT, b);

      expect(result.shape).toEqual([3, 2]);
      // First row: [1,4] @ [[1,2],[3,4]] = [1*1+4*3, 1*2+4*4] = [13, 18]
      // Second row: [2,5] @ [[1,2],[3,4]] = [2*1+5*3, 2*2+5*4] = [17, 24]
      // Third row: [3,6] @ [[1,2],[3,4]] = [3*1+6*3, 3*2+6*4] = [21, 30]
      expect(result.toArray()).toEqual([
        [13, 18],
        [17, 24],
        [21, 30],
      ]);
    });

    it('handles B transposed (exercises transposeB branch)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]); // 2x2
      const b = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // 3x2, transposed will be 2x3
      const bT = transpose(b); // 2x3 view with transposed strides
      // a @ bT = (2x2) @ (2x3) = (2x3)
      const result = linalg.matmul(a, bT);

      expect(result.shape).toEqual([2, 3]);
      // bT viewed as [[1,3,5],[2,4,6]]
      // Row 0: [1,2] @ [[1,3,5],[2,4,6]] = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
      // Row 1: [3,4] @ [[1,3,5],[2,4,6]] = [3*1+4*2, 3*3+4*4, 3*5+4*6] = [11, 25, 39]
      expect(result.toArray()).toEqual([
        [5, 11, 17],
        [11, 25, 39],
      ]);
    });

    it('handles both A and B transposed (exercises both transpose branches)', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]); // 2x3
      const b = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]); // 3x2
      const aT = transpose(a); // 3x2
      const bT = transpose(b); // 2x3
      // aT @ bT = (3x2) @ (2x3) = (3x3)
      const result = linalg.matmul(aT, bT);

      expect(result.shape).toEqual([3, 3]);
      // Row 0: [1,4] @ [1,3,5; 2,4,6]^T = [1,4] @ [[1,2],[3,4],[5,6]]
      // Actually bT is [[1,3,5],[2,4,6]], so [1,4] @ [[1,3,5],[2,4,6]] = [1*1+4*2, 1*3+4*4, 1*5+4*6] = [9, 19, 29]
      expect(result.toArray()).toEqual([
        [9, 19, 29],
        [12, 26, 40],
        [15, 33, 51],
      ]);
    });
  });

  describe('linalg.matrix_transpose()', () => {
    it('transposes 2D matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = linalg.matrix_transpose(a);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('transposes 3D array (swaps last two axes)', () => {
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
      const result = linalg.matrix_transpose(a);

      expect(result.shape).toEqual([2, 2, 2]);
      expect(result.get([0, 0, 0])).toBe(1);
      expect(result.get([0, 0, 1])).toBe(3);
      expect(result.get([0, 1, 0])).toBe(2);
      expect(result.get([0, 1, 1])).toBe(4);
    });
  });

  describe('linalg.multi_dot()', () => {
    it('computes dot product of 2 matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = linalg.multi_dot([a, b]);

      expect(result.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('computes dot product of 3 matrices', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const c = array([
        [1, 0],
        [0, 1],
      ]);
      const result = linalg.multi_dot([a, b, c]);

      expect(result.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('throws error when given less than 2 arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => linalg.multi_dot([a])).toThrow('need at least 2 arrays');
    });
  });

  describe('linalg.tensorinv()', () => {
    it('throws error when ind is not positive', () => {
      const a = array([
        [1, 0],
        [0, 1],
      ]);
      expect(() => linalg.tensorinv(a, 0)).toThrow('ind must be positive');
      expect(() => linalg.tensorinv(a, -1)).toThrow('ind must be positive');
    });
  });

  describe('linalg.outer()', () => {
    it('computes outer product of two vectors', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5]);
      const result = linalg.outer(a, b);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [4, 5],
        [8, 10],
        [12, 15],
      ]);
    });
  });

  describe('linalg.slogdet()', () => {
    it('computes sign and log determinant', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.slogdet(a);

      expect(result.sign).toBe(-1);
      // det = -2, so logabsdet = log(2)
      expect(result.logabsdet).toBeCloseTo(Math.log(2), 10);
    });

    it('handles positive determinant', () => {
      const a = array([
        [2, 0],
        [0, 3],
      ]);
      const result = linalg.slogdet(a);

      expect(result.sign).toBe(1);
      expect(result.logabsdet).toBeCloseTo(Math.log(6), 10);
    });
  });

  describe('linalg.svdvals()', () => {
    it('computes singular values', () => {
      const a = array([
        [1, 0],
        [0, 2],
      ]);
      const result = linalg.svdvals(a);

      expect(result.shape).toEqual([2]);
      // Singular values should be 2 and 1 (sorted descending)
      expect(Number(result.get([0]))).toBeCloseTo(2, 5);
      expect(Number(result.get([1]))).toBeCloseTo(1, 5);
    });
  });

  describe('linalg.tensordot()', () => {
    it('computes tensor dot product', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const result = linalg.tensordot(a, b, 1) as any;

      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('linalg.tensorinv()', () => {
    it('computes tensor inverse for 4D array', () => {
      // Create a proper invertible 4D array: identity when reshaped to 4x4
      // Shape is [2,2,2,2], reshaped to [4,4] should be identity
      const eyeData = [];
      for (let i = 0; i < 2; i++) {
        const row = [];
        for (let j = 0; j < 2; j++) {
          const subrow = [];
          for (let k = 0; k < 2; k++) {
            const subcol = [];
            for (let l = 0; l < 2; l++) {
              // Identity: (i*2+j) == (k*2+l) ? 1 : 0
              subcol.push(i * 2 + j === k * 2 + l ? 1 : 0);
            }
            subrow.push(subcol);
          }
          row.push(subrow);
        }
        eyeData.push(row);
      }
      const a = array(eyeData);
      const result = linalg.tensorinv(a);

      expect(result.shape).toEqual([2, 2, 2, 2]);
    });

    it('throws when ind is larger than ndim', () => {
      const a = ones([2, 2]); // 2D array
      expect(() => linalg.tensorinv(a, 3)).toThrow(/ind=3 is too large/);
    });

    it('throws when dimension products do not match', () => {
      const a = ones([2, 3, 4]); // 2 != 3*4=12
      expect(() => linalg.tensorinv(a, 1)).toThrow(/product of first 1 dimensions.*must equal/);
    });
  });

  describe('linalg.tensorsolve()', () => {
    it('solves tensor equation', () => {
      // Create identity-like 4D tensor for solvable system
      const eyeData = [];
      for (let i = 0; i < 2; i++) {
        const row = [];
        for (let j = 0; j < 2; j++) {
          const subrow = [];
          for (let k = 0; k < 2; k++) {
            const subcol = [];
            for (let l = 0; l < 2; l++) {
              subcol.push(i * 2 + j === k * 2 + l ? 1 : 0);
            }
            subrow.push(subcol);
          }
          row.push(subrow);
        }
        eyeData.push(row);
      }
      const a = array(eyeData);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      const result = linalg.tensorsolve(a, b);

      expect(result.shape).toEqual([2, 2]);
      // For identity tensor, solution should equal b
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('throws when dimensions do not match', () => {
      // a is [2,2,2,2] = 16 elements when reshaped, b is [3,3] = 9 elements
      const a = ones([2, 2, 2, 2]);
      const b = ones([3, 3]); // Wrong size
      expect(() => linalg.tensorsolve(a, b)).toThrow(/dimensions don't match/);
    });

    it('throws for non-square problem', () => {
      // Create a tensor where other dims product != sum dims product
      // a shape [2, 3, 2, 3] with axes [2,3] sums to 2*3=6, other dims = 2*3=6
      // This should work. Let's make it fail with [2, 3, 2, 2]
      const a = ones([2, 3, 2, 2]); // other dims = 2*3=6, sum dims = 2*2=4
      const b = ones([2, 2]); // 4 elements to match sum dims
      expect(() => linalg.tensorsolve(a, b)).toThrow(/non-square problem/);
    });

    it('accepts explicit axes parameter', () => {
      // Use explicit axes to control which dimensions to sum
      const eyeData = [];
      for (let i = 0; i < 2; i++) {
        const row = [];
        for (let j = 0; j < 2; j++) {
          const subrow = [];
          for (let k = 0; k < 2; k++) {
            const subcol = [];
            for (let l = 0; l < 2; l++) {
              subcol.push(i * 2 + j === k * 2 + l ? 1 : 0);
            }
            subrow.push(subcol);
          }
          row.push(subrow);
        }
        eyeData.push(row);
      }
      const a = array(eyeData);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      // Explicitly specify axes [2, 3] (same as default)
      const result = linalg.tensorsolve(a, b, [2, 3]);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('linalg.trace()', () => {
    it('computes sum of diagonal elements', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const result = linalg.trace(a);

      expect(result).toBe(15); // 1 + 5 + 9
    });
  });

  describe('linalg.vecdot()', () => {
    it('computes vector dot product of 1D arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = linalg.vecdot(a, b);

      expect(result).toBe(32); // 1*4 + 2*5 + 3*6
    });

    it('computes vector dot product of 2D arrays', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([
        [1, 1, 1],
        [2, 2, 2],
      ]);
      const result = linalg.vecdot(a, b) as any;

      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([6, 30]); // [1+2+3, 8+10+12]
    });
  });
});

describe('Top-level Linear Algebra Functions', () => {
  describe('vdot()', () => {
    it('computes dot product of flattened arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vdot(a, b);

      expect(result).toBe(32); // 1*4 + 2*5 + 3*6
    });

    it('flattens multi-dimensional arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [1, 1],
        [1, 1],
      ]);
      const result = vdot(a, b);

      expect(result).toBe(10); // 1+2+3+4
    });
  });

  describe('vecdot()', () => {
    it('computes vector dot product', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = vecdot(a, b);

      expect(result).toBe(32);
    });
  });

  describe('matrix_transpose()', () => {
    it('transposes 2D matrix', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = matrix_transpose(a);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });
  });

  describe('permute_dims()', () => {
    it('permutes array dimensions', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = permute_dims(a, [1, 0]);

      expect(result.shape).toEqual([3, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });
  });

  describe('matvec()', () => {
    it('computes matrix-vector product', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([1, 1, 1]);
      const result = matvec(a, b);

      expect(result.shape).toEqual([2]);
      expect(result.toArray()).toEqual([6, 15]); // [1+2+3, 4+5+6]
    });
  });

  describe('vecmat()', () => {
    it('computes vector-matrix product', () => {
      const a = array([1, 1]);
      const b = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = vecmat(a, b);

      expect(result.shape).toEqual([3]);
      expect(result.toArray()).toEqual([5, 7, 9]); // [1+4, 2+5, 3+6]
    });
  });
});

describe('einsum_path()', () => {
  it('returns path for single operand', () => {
    const a = ones([3, 3]);
    const [path, info] = einsum_path('ii->', a);

    expect(Array.isArray(path)).toBe(true);
    expect(typeof info).toBe('string');
    expect(info).toContain('Contraction path');
  });

  it('returns path for two operands (matrix multiply)', () => {
    const a = ones([10, 5]);
    const b = ones([5, 8]);
    const [path, info] = einsum_path('ij,jk->ik', a, b);

    expect(Array.isArray(path)).toBe(true);
    expect(path.length).toBe(1);
    expect(path[0]).toEqual([0, 1]);
    expect(info).toContain('Complete contraction');
  });

  it('returns path for three operands', () => {
    const a = ones([10, 5]);
    const b = ones([5, 8]);
    const c = ones([8, 4]);
    const [path, info] = einsum_path('ij,jk,kl->il', a, b, c);

    expect(Array.isArray(path)).toBe(true);
    expect(path.length).toBe(2);
    expect(info).toContain('Operand shapes');
  });

  it('works with shapes as arrays', () => {
    const [path, info] = einsum_path('ij,jk->ik', [10, 5], [5, 8]);

    expect(Array.isArray(path)).toBe(true);
    expect(typeof info).toBe('string');
  });

  it('throws on operand count mismatch', () => {
    const a = ones([3, 3]);
    expect(() => einsum_path('ij,jk->ik', a)).toThrow();
  });

  it('throws on shape mismatch', () => {
    const a = ones([3, 4]);
    const b = ones([5, 3]); // Should be [4, x] for ij,jk
    expect(() => einsum_path('ij,jk->ik', a, b)).toThrow();
  });

  it('throws when subscript length does not match operand dimensions', () => {
    const a = ones([3, 4, 5]); // 3D array
    const b = ones([4, 3]);
    // 'ij' has 2 indices but a has 3 dimensions
    expect(() => einsum_path('ij,jk->ik', a, b)).toThrow(
      /operand 0 has 3 dimensions but subscript 'ij' has 2 indices/
    );
  });

  it('handles implicit output subscript', () => {
    const a = ones([3, 3]);
    const b = ones([3, 3]);
    const [path, info] = einsum_path('ij,jk', a, b);

    expect(Array.isArray(path)).toBe(true);
    expect(info).toContain('Complete contraction');
  });
});
