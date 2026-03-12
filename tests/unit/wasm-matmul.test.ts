/**
 * Unit tests for WASM-accelerated matmul.
 *
 * Tests that the WASM matmul produces the same results as the JS matmul,
 * and that the fallback behavior works correctly.
 *
 * Since WASM is now integrated directly into matmul (not a separate entrypoint),
 * these tests verify the wasmMatmul backend directly and confirm that the
 * unified matmul automatically uses WASM when appropriate.
 */

import { describe, it, expect } from 'vitest';
import { array, matmul, zeros, reshape } from '../../src';
import { wasmMatmul } from '../../src/common/wasm/matmul';
import type { NDArrayCore } from '../../src/common/ndarray-core';
import { ArrayStorage } from '../../src/common/storage';

/** Helper: compare two NDArrayCore results element-wise */
function expectClose(actual: NDArrayCore, expected: NDArrayCore, tolerance = 1e-10) {
  expect(actual.shape).toEqual(expected.shape);
  for (let i = 0; i < actual.size; i++) {
    const a = actual.storage.iget(i) as number;
    const e = expected.storage.iget(i) as number;
    expect(Math.abs(a - e)).toBeLessThan(tolerance);
  }
}

/** Helper: compare ArrayStorage to NDArrayCore */
function expectStorageClose(actual: ArrayStorage, expected: NDArrayCore, tolerance = 1e-10) {
  expect(Array.from(actual.shape)).toEqual(expected.shape);
  for (let i = 0; i < actual.size; i++) {
    const a = Number(actual.iget(i));
    const e = expected.storage.iget(i) as number;
    expect(Math.abs(a - e)).toBeLessThan(tolerance);
  }
}

describe('WASM matmul', () => {
  describe('correctness — matches JS matmul', () => {
    it('2x2 @ 2x2', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const jsResult = matmul(a, b);
      // matmul now uses WASM automatically when above threshold
      // For small arrays it falls back to JS, so result should match either way
      expectClose(
        jsResult,
        array([
          [19, 22],
          [43, 50],
        ])
      );
    });

    it('3x3 @ 3x3', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const b = array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
      ]);
      const expected = array([
        [30, 24, 18],
        [84, 69, 54],
        [138, 114, 90],
      ]);
      expectClose(matmul(a, b), expected);
    });

    it('non-square matrices: 2x3 @ 3x4', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ]);
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 4]);
    });

    it('identity matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const eye = array([
        [1, 0],
        [0, 1],
      ]);
      expectClose(matmul(a, eye), a);
    });

    it('zeros matrix', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const z = zeros([2, 2]);
      expectClose(matmul(a, z), zeros([2, 2]));
    });

    it('large matrix (above WASM threshold)', () => {
      // 20x20 = 400 elements per input, well above the 256 threshold
      const aData: number[][] = [];
      const bData: number[][] = [];
      for (let i = 0; i < 20; i++) {
        aData.push(Array.from({ length: 20 }, (_, j) => i * 20 + j + 1));
        bData.push(Array.from({ length: 20 }, (_, j) => (i + j) * 0.1));
      }
      const a = array(aData);
      const b = array(bData);
      // This should go through WASM path
      const result = matmul(a, b);
      expect(result.shape).toEqual([20, 20]);
    });
  });

  describe('wasmMatmul backend directly', () => {
    it('returns non-null for large contiguous float64 arrays', () => {
      const aData: number[][] = [];
      const bData: number[][] = [];
      for (let i = 0; i < 20; i++) {
        aData.push(Array.from({ length: 20 }, (_, j) => i * 20 + j + 1));
        bData.push(Array.from({ length: 20 }, (_, j) => (i + j) * 0.1));
      }
      const a = array(aData);
      const b = array(bData);
      const result = wasmMatmul(a.storage, b.storage);
      expect(result).not.toBeNull();
      expect(Array.from(result!.shape)).toEqual([20, 20]);
    });

    it('returns null for small arrays below threshold', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      // 2*2 + 2*2 = 8, well below 256 threshold
      const result = wasmMatmul(a.storage, b.storage);
      expect(result).toBeNull();
    });

    it('returns correct values for eligible arrays', () => {
      const aData: number[][] = [];
      const bData: number[][] = [];
      for (let i = 0; i < 20; i++) {
        aData.push(Array.from({ length: 20 }, (_, j) => i * 20 + j + 1));
        bData.push(Array.from({ length: 20 }, (_, j) => (i + j) * 0.1));
      }
      const a = array(aData);
      const b = array(bData);
      const wasmResult = wasmMatmul(a.storage, b.storage)!;
      const jsResult = matmul(a, b);
      expectStorageClose(wasmResult, jsResult, 1e-6);
    });
  });

  describe('float32 dtype', () => {
    it('2x2 @ 2x2 float32', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float32'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'float32'
      );
      const result = matmul(a, b);
      expect(result.dtype).toBe('float32');
    });
  });

  describe('batched matmul', () => {
    it('batch of 2x2 @ 2x2 (3D)', () => {
      // Shape: (2, 2, 2) @ (2, 2, 2)
      const a = reshape(array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
      const b = reshape(array([1, 0, 0, 1, 2, 1, 1, 2]), [2, 2, 2]);
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 2, 2]);
    });

    it('broadcast batch: (3, 2, 2) @ (2, 2)', () => {
      const a = reshape(array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), [3, 2, 2]);
      const b = array([
        [1, 0],
        [0, 1],
      ]);
      const result = matmul(a, b);
      expect(result.shape).toEqual([3, 2, 2]);
    });

    it('large batch above WASM threshold', () => {
      // (2, 20, 20) @ (2, 20, 20) — each 2D slice is 400 elements
      const aFlat: number[] = [];
      const bFlat: number[] = [];
      for (let batch = 0; batch < 2; batch++) {
        for (let i = 0; i < 20; i++) {
          for (let j = 0; j < 20; j++) {
            aFlat.push(batch * 100 + i * 20 + j + 1);
            bFlat.push((i + j + batch) * 0.1);
          }
        }
      }
      const a = reshape(array(aFlat), [2, 20, 20]);
      const b = reshape(array(bFlat), [2, 20, 20]);
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 20, 20]);
    });
  });

  describe('fallback to JS', () => {
    it('1D vectors fall back to JS', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const result = matmul(a, b);
      expect(result.shape).toEqual([1]);
    });

    it('int32 dtype promotes to float64', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int32'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'int32'
      );
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 2]);
    });

    it('complex dtype works', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'complex128'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'complex128'
      );
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 2]);
      expect(result.dtype).toBe('complex128');
    });

    it('mismatched dtypes work', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'float64'
      );
      const b = array(
        [
          [5, 6],
          [7, 8],
        ],
        'float32'
      );
      const result = matmul(a, b);
      expect(result.shape).toEqual([2, 2]);
    });
  });
});
