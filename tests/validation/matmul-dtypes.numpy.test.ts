/**
 * Cross-validates matmul for all integer and float dtypes (signed + unsigned)
 * against NumPy. Tests small matrices (JS path) and larger matrices
 * (above WASM threshold for float types).
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as np from '../../src/full/index';
import { wasmConfig } from '../../src';
import { checkNumPyAvailable, runNumPy, arraysClose } from './numpy-oracle';
import { supportsRelaxedSimd } from '../../src/common/wasm/detect';

/** NumPy dtype string for a given numpy-ts dtype */
const NP_DTYPE: Record<string, string> = {
  float64: 'np.float64',
  float32: 'np.float32',
  int64: 'np.int64',
  int32: 'np.int32',
  int16: 'np.int16',
  int8: 'np.int8',
  uint64: 'np.uint64',
  uint32: 'np.uint32',
  uint16: 'np.uint16',
  uint8: 'np.uint8',
};

const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1, relaxed: 'auto' as const },
  { name: 'forced WASM (baseline)', multiplier: 0, relaxed: false as const },
  { name: 'forced WASM (relaxed)', multiplier: 0, relaxed: true as const },
] as const;

for (const mode of WASM_MODES) {
  const descFn = mode.relaxed === true && !supportsRelaxedSimd() ? describe.skip : describe;

  descFn(`NumPy Validation: matmul across all dtypes [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      wasmConfig.useRelaxedSimd = mode.relaxed;
      if (!checkNumPyAvailable()) {
        throw new Error('Python NumPy not available');
      }
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      wasmConfig.useRelaxedSimd = mode.relaxed;
    });

    // ============================================================
    // Small 2x2 matmul — exercises the JS path for all types
    // ============================================================
    describe('small 2x2 matmul', () => {
      const DTYPES = [
        'float64',
        'float32',
        'int64',
        'int32',
        'int16',
        'int8',
        'uint64',
        'uint32',
        'uint16',
        'uint8',
      ] as const;

      for (const dtype of DTYPES) {
        it(`${dtype}: 2x2 @ 2x2 matches NumPy`, () => {
          const a = np.array(
            [
              [1, 2],
              [3, 4],
            ],
            dtype
          );
          const b = np.array(
            [
              [5, 6],
              [7, 8],
            ],
            dtype
          );
          const r = np.matmul(a, b);

          const npResult = runNumPy(`
import numpy as np
a = np.array([[1, 2], [3, 4]], dtype=${NP_DTYPE[dtype]})
b = np.array([[5, 6], [7, 8]], dtype=${NP_DTYPE[dtype]})
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
        `);

          expect(r.dtype).toBe(dtype);
          if (dtype === 'int64' || dtype === 'uint64') {
            // BigInt 2D tolist returns bigint[][] — convert to number[][] for comparison
            const got = (r.tolist() as bigint[][]).map((row) => row.map(Number));
            expect(got).toEqual(npResult.value.values);
          } else {
            expect(r.tolist()).toEqual(npResult.value.values);
          }
        });
      }
    });

    // ============================================================
    // Non-square matmul — 2x3 @ 3x2
    // ============================================================
    describe('non-square 2x3 @ 3x2', () => {
      const DTYPES = [
        'float64',
        'float32',
        'int32',
        'int16',
        'int8',
        'uint32',
        'uint16',
        'uint8',
      ] as const;

      for (const dtype of DTYPES) {
        it(`${dtype}: 2x3 @ 3x2 matches NumPy`, () => {
          const a = np.array(
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
            dtype
          );
          const b = np.array(
            [
              [7, 8],
              [9, 10],
              [11, 12],
            ],
            dtype
          );
          const r = np.matmul(a, b);

          const npResult = runNumPy(`
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=${NP_DTYPE[dtype]})
b = np.array([[7, 8], [9, 10], [11, 12]], dtype=${NP_DTYPE[dtype]})
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
        `);

          expect(r.dtype).toBe(dtype);
          expect(r.tolist()).toEqual(npResult.value.values);
        });
      }
    });

    // ============================================================
    // 1D @ 2D and 2D @ 1D
    // ============================================================
    describe('vector-matrix products', () => {
      const DTYPES = ['float64', 'int32', 'int16', 'int8', 'uint8'] as const;

      for (const dtype of DTYPES) {
        it(`${dtype}: 1D @ 2D matches NumPy`, () => {
          const a = np.array([1, 2, 3], dtype);
          const b = np.array(
            [
              [1, 2],
              [3, 4],
              [5, 6],
            ],
            dtype
          );
          const r = np.matmul(a, b);

          const npResult = runNumPy(`
import numpy as np
a = np.array([1, 2, 3], dtype=${NP_DTYPE[dtype]})
b = np.array([[1, 2], [3, 4], [5, 6]], dtype=${NP_DTYPE[dtype]})
result = (a @ b).tolist()
        `);

          expect(r.tolist()).toEqual(npResult.value);
        });

        it(`${dtype}: 2D @ 1D matches NumPy`, () => {
          const a = np.array(
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
            dtype
          );
          const b = np.array([1, 2, 3], dtype);
          const r = np.matmul(a, b);

          const npResult = runNumPy(`
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=${NP_DTYPE[dtype]})
b = np.array([1, 2, 3], dtype=${NP_DTYPE[dtype]})
result = (a @ b).tolist()
        `);

          expect(r.tolist()).toEqual(npResult.value);
        });
      }
    });

    // ============================================================
    // Integer overflow wrapping — values that overflow the dtype
    // ============================================================
    describe('integer overflow wrapping', () => {
      it('int8: accumulation wraps correctly', () => {
        const a = np.array(
          [
            [10, 20],
            [30, 40],
          ],
          'int8'
        );
        const b = np.array(
          [
            [5, 6],
            [7, 8],
          ],
          'int8'
        );
        const r = np.matmul(a, b);

        const npResult = runNumPy(`
import numpy as np
a = np.array([[10, 20], [30, 40]], dtype=np.int8)
b = np.array([[5, 6], [7, 8]], dtype=np.int8)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('int8');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('uint8: accumulation wraps correctly', () => {
        const a = np.array(
          [
            [10, 20],
            [30, 40],
          ],
          'uint8'
        );
        const b = np.array(
          [
            [5, 6],
            [7, 8],
          ],
          'uint8'
        );
        const r = np.matmul(a, b);

        const npResult = runNumPy(`
import numpy as np
a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
b = np.array([[5, 6], [7, 8]], dtype=np.uint8)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('uint8');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('int16: large values wrap correctly', () => {
        const a = np.full([2, 2], 200, 'int16');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 200, dtype=np.int16)
r = a @ a
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('int16');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('uint16: large values wrap correctly', () => {
        const a = np.full([2, 2], 300, 'uint16');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 300, dtype=np.uint16)
r = a @ a
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('uint16');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('int32: large values wrap correctly', () => {
        const a = np.full([2, 2], 50000, 'int32');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 50000, dtype=np.int32)
r = a @ a
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('int32');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('uint32: large values wrap correctly', () => {
        const a = np.full([2, 2], 50000, 'uint32');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 50000, dtype=np.uint32)
r = a @ a
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('uint32');
        expect(r.tolist()).toEqual(npResult.value.values);
      });
    });

    // ============================================================
    // Larger matrices — above WASM threshold for float types,
    // and exercises the JS integer path at moderate scale
    // ============================================================
    describe('larger matrices (10x10)', () => {
      const DTYPES = ['float64', 'float32', 'int32', 'int16', 'int8'] as const;

      for (const dtype of DTYPES) {
        it(`${dtype}: 10x10 @ 10x10 matches NumPy`, () => {
          // Use small values to avoid overflow for narrow int types
          const a = np.remainder(
            np.arange(0, 100, 1, dtype).reshape([10, 10]),
            np.array([5], dtype)
          );
          const b = np.remainder(
            np.arange(0, 100, 1, dtype).reshape([10, 10]),
            np.array([3], dtype)
          );

          const r = np.matmul(a, b);

          const npResult = runNumPy(`
import numpy as np
a = np.arange(100, dtype=${NP_DTYPE[dtype]}).reshape(10, 10) % ${NP_DTYPE[dtype]}(5)
b = np.arange(100, dtype=${NP_DTYPE[dtype]}).reshape(10, 10) % ${NP_DTYPE[dtype]}(3)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
        `);

          expect(r.dtype).toBe(dtype);
          if (dtype === 'float32') {
            expect(arraysClose(r.tolist(), npResult.value.values, 1e-5)).toBe(true);
          } else {
            expect(r.tolist()).toEqual(npResult.value.values);
          }
        });
      }
    });

    // ============================================================
    // Float overflow in matmul
    // ============================================================
    describe('float overflow', () => {
      it('float64: overflows to inf', () => {
        const a = np.full([2, 2], 1e154, 'float64');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 1e154, dtype=np.float64)
result = float((a @ a)[0, 0])
      `);

        const val = (r.tolist() as number[][])[0]![0]!;
        expect(val).toBe(npResult.value);
        expect(val).toBe(Infinity);
      });

      it('float32: overflows to inf', () => {
        const a = np.full([2, 2], 1e20, 'float32');
        const r = np.matmul(a, a);

        const npResult = runNumPy(`
import numpy as np
a = np.full((2, 2), 1e20, dtype=np.float32)
result = float((a @ a)[0, 0])
      `);

        const val = (r.tolist() as number[][])[0]![0]!;
        expect(val).toBe(npResult.value);
      });
    });

    // ============================================================
    // Batched 3D matmul with integer types
    // ============================================================
    describe('batched 3D matmul', () => {
      it('int32: batched 2x2x2 matches NumPy', () => {
        const a = np.array(
          [
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ],
          'int32'
        );
        const b = np.array(
          [
            [
              [1, 0],
              [0, 1],
            ],
            [
              [2, 1],
              [1, 2],
            ],
          ],
          'int32'
        );
        const r = np.matmul(a, b);

        const npResult = runNumPy(`
import numpy as np
a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int32)
b = np.array([[[1,0],[0,1]],[[2,1],[1,2]]], dtype=np.int32)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('int32');
        expect(r.tolist()).toEqual(npResult.value.values);
      });

      it('int8: batched 2x2x2 matches NumPy', () => {
        const a = np.array(
          [
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ],
          'int8'
        );
        const b = np.array(
          [
            [
              [1, 0],
              [0, 1],
            ],
            [
              [2, 1],
              [1, 2],
            ],
          ],
          'int8'
        );
        const r = np.matmul(a, b);

        const npResult = runNumPy(`
import numpy as np
a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int8)
b = np.array([[[1,0],[0,1]],[[2,1],[1,2]]], dtype=np.int8)
r = a @ b
result = {'values': r.tolist(), 'dtype': str(r.dtype)}
      `);

        expect(r.dtype).toBe('int8');
        expect(r.tolist()).toEqual(npResult.value.values);
      });
    });

    // ============================================================
    // dtype preservation — verify output dtype matches input
    // ============================================================
    describe('dtype preservation', () => {
      const ALL_DTYPES = [
        'float64',
        'float32',
        'int64',
        'int32',
        'int16',
        'int8',
        'uint64',
        'uint32',
        'uint16',
        'uint8',
      ] as const;

      for (const dtype of ALL_DTYPES) {
        it(`${dtype}: output dtype matches input`, () => {
          const a = np.array(
            [
              [1, 2],
              [3, 4],
            ],
            dtype
          );
          const b = np.array(
            [
              [1, 0],
              [0, 1],
            ],
            dtype
          );
          const r = np.matmul(a, b);
          expect(r.dtype).toBe(dtype);
        });
      }
    });
  });
} // end WASM_MODES loop
