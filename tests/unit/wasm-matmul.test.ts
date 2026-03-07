/**
 * Unit tests for WASM-accelerated matmul.
 *
 * Tests that the WASM matmul produces the same results as the JS matmul,
 * and that the fallback behavior works correctly.
 */

import { describe, it, expect } from 'vitest';
import { array, matmul as jsMatmul, zeros, ones } from '../../src';
import { matmul as wasmMatmul } from '../../src/wasm/kernels/matmul';
import type { NDArrayCore } from '../../src/common/ndarray-core';

/** Helper: compare two NDArrayCore results element-wise */
function expectClose(actual: NDArrayCore, expected: NDArrayCore, tolerance = 1e-10) {
  expect(actual.shape).toEqual(expected.shape);
  for (let i = 0; i < actual.size; i++) {
    const a = actual.storage.iget(i) as number;
    const e = expected.storage.iget(i) as number;
    expect(Math.abs(a - e)).toBeLessThan(tolerance);
  }
}

describe('WASM matmul', () => {
  describe('correctness — matches JS matmul', () => {
    it('2x2 @ 2x2', () => {
      const a = array([[1, 2], [3, 4]]);
      const b = array([[5, 6], [7, 8]]);
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult);
    });

    it('3x3 @ 3x3', () => {
      const a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const b = array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult);
    });

    it('non-square matrices: 2x3 @ 3x4', () => {
      const a = array([[1, 2, 3], [4, 5, 6]]);
      const b = array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult);
    });

    it('identity matrix', () => {
      const a = array([[1, 2], [3, 4]]);
      const eye = array([[1, 0], [0, 1]]);
      const wasmResult = wasmMatmul(a, eye);
      expectClose(wasmResult, a);
    });

    it('zeros matrix', () => {
      const a = array([[1, 2], [3, 4]]);
      const z = zeros([2, 2]);
      const wasmResult = wasmMatmul(a, z);
      const expected = zeros([2, 2]);
      expectClose(wasmResult, expected);
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
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult, 1e-6);
    });
  });

  describe('float32 dtype', () => {
    it('2x2 @ 2x2 float32', () => {
      const a = array([[1, 2], [3, 4]], 'float32');
      const b = array([[5, 6], [7, 8]], 'float32');
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expect(wasmResult.dtype).toBe('float32');
      expectClose(wasmResult, jsResult, 1e-4);
    });
  });

  describe('fallback to JS', () => {
    it('1D vectors fall back to JS', () => {
      const a = array([1, 2, 3]);
      const b = array([[1], [2], [3]]);
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult);
    });

    it('int32 dtype falls back to JS', () => {
      const a = array([[1, 2], [3, 4]], 'int32');
      const b = array([[5, 6], [7, 8]], 'int32');
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult);
    });

    it('mismatched dtypes fall back to JS', () => {
      const a = array([[1, 2], [3, 4]], 'float64');
      const b = array([[5, 6], [7, 8]], 'float32');
      const jsResult = jsMatmul(a, b);
      const wasmResult = wasmMatmul(a, b);
      expectClose(wasmResult, jsResult, 1e-4);
    });
  });

  describe('wasm/core entrypoint', () => {
    it('matmul is overridden in wasm/core', async () => {
      // Dynamic import to test the entrypoint
      const wasmCore = await import('../../src/wasm/core');
      const core = await import('../../src/core');

      // Both should export matmul
      expect(typeof wasmCore.matmul).toBe('function');
      expect(typeof core.matmul).toBe('function');

      // They should be different functions (wasm overrides core)
      expect(wasmCore.matmul).not.toBe(core.matmul);

      // Non-overridden functions should be the same
      expect(wasmCore.add).toBe(core.add);
      expect(wasmCore.zeros).toBe(core.zeros);
      expect(wasmCore.reshape).toBe(core.reshape);
    });
  });
});
