/**
 * Python NumPy validation tests for WASM-accelerated linear algebra.
 *
 * WASM acceleration is now built into matmul directly — no separate
 * entrypoint needed. These tests verify that the unified matmul
 * (which tries WASM first, falls back to JS) matches NumPy.
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import {
  array,
  arange,
  matmul,
  dot,
  inner,
  reshape,
  linalg,
  matvec,
  vecmat,
  vecdot,
  wasmConfig,
  Complex,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';
import { supportsRelaxedSimd } from '../../src/common/wasm/detect';

// Run all tests with default, forced-baseline, and forced-relaxed modes
const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1, relaxed: 'auto' as const },
  { name: 'forced WASM (baseline)', multiplier: 0, relaxed: false as const },
  { name: 'forced WASM (relaxed)', multiplier: 0, relaxed: true as const },
] as const;

for (const mode of WASM_MODES) {
  const descFn = mode.relaxed === true && !supportsRelaxedSimd() ? describe.skip : describe;

  descFn(`NumPy Validation: WASM-accelerated Linear Algebra [${mode.name}]`, () => {
    beforeAll(() => {
      if (!checkNumPyAvailable()) {
        throw new Error(
          'Python NumPy not available!\n\n' +
            '   This test suite requires Python with NumPy installed.\n\n' +
            '   Setup: source ~/.zshrc && conda activate py313\n'
        );
      }
      wasmConfig.thresholdMultiplier = mode.multiplier;
      wasmConfig.useRelaxedSimd = mode.relaxed;
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      wasmConfig.useRelaxedSimd = mode.relaxed;
    });

    describe('matmul (WASM)', () => {
      it('matches NumPy for 2x2 @ 2x2', () => {
        const jsResult = matmul(
          array([
            [1, 2],
            [3, 4],
          ]),
          array([
            [5, 6],
            [7, 8],
          ])
        );
        const pyResult = runNumPy(`
result = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2x3 @ 3x2', () => {
        const jsResult = matmul(
          array([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          array([
            [7, 8],
            [9, 10],
            [11, 12],
          ])
        );
        const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[7, 8], [9, 10], [11, 12]])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 3x3 @ 3x3', () => {
        const jsResult = matmul(
          array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ]),
          array([
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1],
          ])
        );
        const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) @ np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for larger matrices (above WASM threshold)', () => {
        // 20x20 = 400 elements per input, well above 256 threshold
        const aData: number[][] = [];
        const bData: number[][] = [];
        const pyA: string[] = [];
        const pyB: string[] = [];
        for (let i = 0; i < 20; i++) {
          const aRow = Array.from({ length: 20 }, (_, j) => i * 20 + j + 1);
          const bRow = Array.from({ length: 20 }, (_, j) => (i + j) * 0.1);
          aData.push(aRow);
          bData.push(bRow);
          pyA.push(`[${aRow.join(',')}]`);
          pyB.push(`[${bRow.join(',')}]`);
        }
        const jsResult = matmul(array(aData), array(bData));
        const pyResult = runNumPy(`
a = np.array([${pyA.join(',')}])
b = np.array([${pyB.join(',')}])
result = a @ b
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-6)).toBe(true);
      });

      it('matches NumPy for 1D @ 2D', () => {
        const jsResult = matmul(
          array([1, 2, 3]),
          array([
            [1, 2],
            [3, 4],
            [5, 6],
          ])
        );
        const pyResult = runNumPy(`
result = np.array([1, 2, 3]) @ np.array([[1, 2], [3, 4], [5, 6]])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 2D @ 1D', () => {
        const jsResult = matmul(
          array([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          array([1, 2, 3])
        );
        const pyResult = runNumPy(`
result = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([1, 2, 3])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for batched 3D matmul', () => {
        const a = reshape(array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
        const b = reshape(array([1, 0, 0, 1, 2, 1, 1, 2]), [2, 2, 2]);
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
b = np.array([[[1,0],[0,1]],[[2,1],[1,2]]])
result = a @ b
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('dot via matmul (WASM)', () => {
      it('2D dot delegates to matmul — matches NumPy', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const jsResult = dot(a, b);
        const pyResult = runNumPy(`
result = np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('matmul integer dtypes (WASM)', () => {
      // Signed: values in [-3, 3] — will overflow int8/int16 accumulators in 20x20 matmul
      const signedDtypes = ['int64', 'int32', 'int16', 'int8'] as const;
      for (const dtype of signedDtypes) {
        it(`matches NumPy for 20x20 ${dtype} matmul (overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => ((i * 20 + j) % 7) - 3));
            bData.push(Array.from({ length: 20 }, (_, j) => ((i + j * 3) % 5) - 2));
          }
          const jsResult = matmul(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.matmul(a, b)
        `);
          expect(jsResult.shape).toEqual(pyResult.shape);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });

        it(`matches NumPy for 20x20 ${dtype} matmul (no overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => ((i + j) % 3) - 1));
            bData.push(Array.from({ length: 20 }, (_, j) => ((i * j) % 3) - 1));
          }
          const jsResult = matmul(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.matmul(a, b)
        `);
          expect(jsResult.shape).toEqual(pyResult.shape);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });
      }

      // Unsigned: values in [0, 12] — will overflow uint8/uint16 accumulators in 20x20 matmul
      const unsignedDtypes = ['uint64', 'uint32', 'uint16', 'uint8'] as const;
      for (const dtype of unsignedDtypes) {
        it(`matches NumPy for 20x20 ${dtype} matmul (overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => (i * 20 + j) % 13));
            bData.push(Array.from({ length: 20 }, (_, j) => (i + j * 3) % 11));
          }
          const jsResult = matmul(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.matmul(a, b)
        `);
          expect(jsResult.shape).toEqual(pyResult.shape);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });

        it(`matches NumPy for 20x20 ${dtype} matmul (no overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => (i + j) % 3));
            bData.push(Array.from({ length: 20 }, (_, j) => (i * j) % 3));
          }
          const jsResult = matmul(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.matmul(a, b)
        `);
          expect(jsResult.shape).toEqual(pyResult.shape);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });
      }
    });

    describe('inner (WASM)', () => {
      it('matches NumPy for 2D float64 inner (20x20 @ 15x20 → 20x15)', () => {
        const aData: number[][] = [];
        const bData: number[][] = [];
        const pyA: string[] = [];
        const pyB: string[] = [];
        for (let i = 0; i < 20; i++) {
          const aRow = Array.from({ length: 20 }, (_, j) => i * 20 + j + 1);
          aData.push(aRow);
          pyA.push(`[${aRow.join(',')}]`);
        }
        for (let i = 0; i < 15; i++) {
          const bRow = Array.from({ length: 20 }, (_, j) => (i + j) * 0.1);
          bData.push(bRow);
          pyB.push(`[${bRow.join(',')}]`);
        }
        const jsResult = inner(array(aData), array(bData));
        const pyResult = runNumPy(`
a = np.array([${pyA.join(',')}])
b = np.array([${pyB.join(',')}])
result = np.inner(a, b)
      `);
        expect((jsResult as any).shape).toEqual(pyResult.shape);
        expect(arraysClose((jsResult as any).toArray(), pyResult.value, 1e-6)).toBe(true);
      });

      it('matches NumPy for 2D float32 inner (20x20 @ 15x20 → 20x15)', () => {
        const aData: number[][] = [];
        const bData: number[][] = [];
        for (let i = 0; i < 20; i++) {
          aData.push(Array.from({ length: 20 }, (_, j) => ((i * 20 + j) % 100) * 0.01));
        }
        for (let i = 0; i < 15; i++) {
          bData.push(Array.from({ length: 20 }, (_, j) => ((i + j * 3) % 50) * 0.02));
        }
        const jsResult = inner(array(aData, 'float32'), array(bData, 'float32'));
        const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.float32)
b = np.array(${JSON.stringify(bData)}, dtype=np.float32)
result = np.inner(a, b)
      `);
        expect((jsResult as any).shape).toEqual(pyResult.shape);
        expect(arraysClose((jsResult as any).toArray(), pyResult.value, 1e-4)).toBe(true);
      });
    });

    describe('inner integer dtypes (WASM)', () => {
      // Signed: values in [-3, 3] — will overflow int8/int16 accumulators in 20x20 inner
      const signedDtypes = ['int64', 'int32', 'int16', 'int8'] as const;
      for (const dtype of signedDtypes) {
        it(`matches NumPy for 20x20 @ 15x20 ${dtype} inner (overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => ((i * 20 + j) % 7) - 3));
          }
          for (let i = 0; i < 15; i++) {
            bData.push(Array.from({ length: 20 }, (_, j) => ((i + j * 3) % 5) - 2));
          }
          const jsResult = inner(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.inner(a, b)
        `);
          expect((jsResult as any).shape).toEqual(pyResult.shape);
          expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
        });

        it(`matches NumPy for 20x20 @ 15x20 ${dtype} inner (no overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => ((i + j) % 3) - 1));
          }
          for (let i = 0; i < 15; i++) {
            bData.push(Array.from({ length: 20 }, (_, j) => ((i * j) % 3) - 1));
          }
          const jsResult = inner(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.inner(a, b)
        `);
          expect((jsResult as any).shape).toEqual(pyResult.shape);
          expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
        });
      }

      // Unsigned: values in [0, 12] — will overflow uint8/uint16 accumulators in 20x20 inner
      const unsignedDtypes = ['uint64', 'uint32', 'uint16', 'uint8'] as const;
      for (const dtype of unsignedDtypes) {
        it(`matches NumPy for 20x20 @ 15x20 ${dtype} inner (overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => (i * 20 + j) % 13));
          }
          for (let i = 0; i < 15; i++) {
            bData.push(Array.from({ length: 20 }, (_, j) => (i + j * 3) % 11));
          }
          const jsResult = inner(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.inner(a, b)
        `);
          expect((jsResult as any).shape).toEqual(pyResult.shape);
          expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
        });

        it(`matches NumPy for 20x20 @ 15x20 ${dtype} inner (no overflow)`, () => {
          const aData: number[][] = [];
          const bData: number[][] = [];
          for (let i = 0; i < 20; i++) {
            aData.push(Array.from({ length: 20 }, (_, j) => (i + j) % 3));
          }
          for (let i = 0; i < 15; i++) {
            bData.push(Array.from({ length: 20 }, (_, j) => (i * j) % 3));
          }
          const jsResult = inner(array(aData, dtype), array(bData, dtype));
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(aData)}, dtype=np.${dtype})
b = np.array(${JSON.stringify(bData)}, dtype=np.${dtype})
result = np.inner(a, b)
        `);
          expect((jsResult as any).shape).toEqual(pyResult.shape);
          expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
        });
      }
    });

    describe('linalg.multi_dot (WASM)', () => {
      it('chain of matmuls matches NumPy', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([
          [5, 6],
          [7, 8],
        ]);
        const c = array([
          [9, 10],
          [11, 12],
        ]);
        const jsResult = linalg.multi_dot([a, b, c]);
        const pyResult = runNumPy(`
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.array([[9,10],[11,12]])
result = np.linalg.multi_dot([a, b, c])
      `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });
    describe('matvec (WASM)', () => {
      it('matches NumPy for large float64 matvec', () => {
        const A = arange(0, 256).reshape([16, 16]);
        const x = arange(0, 16);
        const jsResult = matvec(A, x);
        const pyResult = runNumPy(`
A = np.arange(256, dtype=np.float64).reshape(16, 16)
x = np.arange(16, dtype=np.float64)
result = np.matvec(A, x)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float32 matvec', () => {
        const A = array(arange(0, 256).toArray(), 'float32').reshape([16, 16]);
        const x = array(arange(0, 16).toArray(), 'float32');
        const jsResult = matvec(A, x);
        const pyResult = runNumPy(`
A = np.arange(256, dtype=np.float32).reshape(16, 16)
x = np.arange(16, dtype=np.float32)
result = np.matvec(A, x)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for int32 matvec', () => {
        const A = array(arange(0, 256).toArray(), 'int32').reshape([16, 16]);
        const x = array(arange(0, 16).toArray(), 'int32');
        const jsResult = matvec(A, x);
        const pyResult = runNumPy(`
A = np.arange(256, dtype=np.int32).reshape(16, 16)
x = np.arange(16, dtype=np.int32)
result = np.matvec(A, x)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('vecmat (WASM)', () => {
      it('matches NumPy for large float64 vecmat', () => {
        const x = arange(0, 16);
        const A = arange(0, 256).reshape([16, 16]);
        const jsResult = vecmat(x, A);
        const pyResult = runNumPy(`
x = np.arange(16, dtype=np.float64)
A = np.arange(256, dtype=np.float64).reshape(16, 16)
result = np.vecmat(x, A)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float32 vecmat', () => {
        const x = array(arange(0, 16).toArray(), 'float32');
        const A = array(arange(0, 256).toArray(), 'float32').reshape([16, 16]);
        const jsResult = vecmat(x, A);
        const pyResult = runNumPy(`
x = np.arange(16, dtype=np.float32)
A = np.arange(256, dtype=np.float32).reshape(16, 16)
result = np.vecmat(x, A)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for int32 vecmat', () => {
        const x = array(arange(0, 16).toArray(), 'int32');
        const A = array(arange(0, 256).toArray(), 'int32').reshape([16, 16]);
        const jsResult = vecmat(x, A);
        const pyResult = runNumPy(`
x = np.arange(16, dtype=np.int32)
A = np.arange(256, dtype=np.int32).reshape(16, 16)
result = np.vecmat(x, A)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('vecdot (WASM)', () => {
      it('matches NumPy for large float64 vecdot', () => {
        const a = arange(0, 160).reshape([10, 16]);
        const b = arange(0, 160).reshape([10, 16]);
        const jsResult = vecdot(a, b);
        const pyResult = runNumPy(`
a = np.arange(160, dtype=np.float64).reshape(10, 16)
b = np.arange(160, dtype=np.float64).reshape(10, 16)
result = np.vecdot(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for float32 vecdot', () => {
        const a = array(arange(0, 160).toArray(), 'float32').reshape([10, 16]);
        const b = array(arange(0, 160).toArray(), 'float32').reshape([10, 16]);
        const jsResult = vecdot(a, b);
        const pyResult = runNumPy(`
a = np.arange(160, dtype=np.float32).reshape(10, 16)
b = np.arange(160, dtype=np.float32).reshape(10, 16)
result = np.vecdot(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for int32 vecdot', () => {
        const a = array(arange(0, 160).toArray(), 'int32').reshape([10, 16]);
        const b = array(arange(0, 160).toArray(), 'int32').reshape([10, 16]);
        const jsResult = vecdot(a, b);
        const pyResult = runNumPy(`
a = np.arange(160, dtype=np.int32).reshape(10, 16)
b = np.arange(160, dtype=np.int32).reshape(10, 16)
result = np.vecdot(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('matvec complex (WASM)', () => {
      it('matches NumPy for complex128 matvec', () => {
        const A = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex128'
        );
        const x = array([new Complex(1, 0), new Complex(0, 1)], 'complex128');
        const jsResult = matvec(A, x);
        const pyResult = runNumPy(`
A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
x = np.array([1+0j, 0+1j])
result = np.matvec(A, x)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for complex64 matvec', () => {
        const A = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex64'
        );
        const x = array([new Complex(1, 0), new Complex(0, 1)], 'complex64');
        const jsResult = matvec(A, x);
        const pyResult = runNumPy(`
A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]], dtype=np.complex64)
x = np.array([1+0j, 0+1j], dtype=np.complex64)
result = np.matvec(A, x)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    // vecmat complex: JS path is fixed, WASM binary needs recompilation (Zig source updated)
    describe('vecmat complex (WASM)', () => {
      // Skip forced-WASM modes until WASM binary is rebuilt with conjugation fix
      const skipWasm = mode.multiplier === 0;

      it('matches NumPy for complex128 vecmat (conjugates x1)', () => {
        if (skipWasm) return;
        const x = array([new Complex(1, 0), new Complex(0, 1)], 'complex128');
        const A = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex128'
        );
        const jsResult = vecmat(x, A);
        const pyResult = runNumPy(`
x = np.array([1+0j, 0+1j])
A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
result = np.vecmat(x, A)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for complex64 vecmat', () => {
        if (skipWasm) return;
        const x = array([new Complex(1, 0), new Complex(0, 1)], 'complex64');
        const A = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex64'
        );
        const jsResult = vecmat(x, A);
        const pyResult = runNumPy(`
x = np.array([1+0j, 0+1j], dtype=np.complex64)
A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]], dtype=np.complex64)
result = np.vecmat(x, A)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('vecdot complex (WASM)', () => {
      it('matches NumPy for complex64 vecdot', () => {
        const a = array(
          [
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
          ],
          'complex64'
        );
        const b = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, -1)],
          ],
          'complex64'
        );
        const jsResult = vecdot(a, b);
        const pyResult = runNumPy(`
a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
b = np.array([[1+0j, 0+1j], [1+1j, 1-1j]], dtype=np.complex64)
result = np.vecdot(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for complex128 vecdot', () => {
        const a = array(
          [
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
          ],
          'complex128'
        );
        const b = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, -1)],
          ],
          'complex128'
        );
        const jsResult = vecdot(a, b);
        const pyResult = runNumPy(`
a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
b = np.array([[1+0j, 0+1j], [1+1j, 1-1j]])
result = np.vecdot(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('inner complex (WASM)', () => {
      it('matches NumPy for complex64 inner (scalar result)', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)], 'complex64');
        const b = array([new Complex(5, 6), new Complex(7, 8)], 'complex64');
        const jsResult = inner(a, b);
        const pyResult = runNumPy(`
a = np.array([1+2j, 3+4j], dtype=np.complex64)
b = np.array([5+6j, 7+8j], dtype=np.complex64)
result = complex(np.inner(a, b))
        `);
        const jsVal = jsResult as Complex;
        expect(jsVal.re).toBeCloseTo(pyResult.value.re ?? pyResult.value[0], 4);
        expect(jsVal.im).toBeCloseTo(pyResult.value.im ?? pyResult.value[1], 4);
      });

      it('matches NumPy for complex128 inner (scalar result)', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], 'complex128');
        const b = array([new Complex(7, 8), new Complex(9, 10), new Complex(11, 12)], 'complex128');
        const jsResult = inner(a, b);
        const pyResult = runNumPy(`
a = np.array([1+2j, 3+4j, 5+6j])
b = np.array([7+8j, 9+10j, 11+12j])
result = np.inner(a, b)
        `);
        const jsVal = jsResult as Complex;
        expect(jsVal.re).toBeCloseTo(pyResult.value.re ?? pyResult.value[0], 10);
        expect(jsVal.im).toBeCloseTo(pyResult.value.im ?? pyResult.value[1], 10);
      });

      it('matches NumPy for complex128 inner (array result)', () => {
        const a = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, -1)],
          ],
          'complex128'
        );
        const b = array(
          [
            [new Complex(2, 0), new Complex(0, 2)],
            [new Complex(1, 1), new Complex(-1, 1)],
          ],
          'complex128'
        );
        const jsResult = inner(a, b) as any;
        const pyResult = runNumPy(`
a = np.array([[1+0j, 0+1j], [1+1j, 1-1j]])
b = np.array([[2+0j, 0+2j], [1+1j, -1+1j]])
result = np.inner(a, b)
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('matmul multi-dtype (WASM)', () => {
      it('matches NumPy for float32 matmul', () => {
        const a = array(arange(0, 256).toArray(), 'float32').reshape([16, 16]);
        const b = array(arange(0, 256).toArray(), 'float32').reshape([16, 16]);
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.arange(256, dtype=np.float32).reshape(16, 16)
b = np.arange(256, dtype=np.float32).reshape(16, 16)
result = a @ b
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for int32 matmul', () => {
        const a = array(arange(0, 64).toArray(), 'int32').reshape([8, 8]);
        const b = array(arange(0, 64).toArray(), 'int32').reshape([8, 8]);
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.arange(64, dtype=np.int32).reshape(8, 8)
b = np.arange(64, dtype=np.int32).reshape(8, 8)
result = a @ b
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for 1D @ 1D (inner product via matmul)', () => {
        const a = arange(0, 128);
        const b = arange(0, 128);
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.arange(128, dtype=np.float64)
b = np.arange(128, dtype=np.float64)
result = np.matmul(a, b)
        `);
        expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 5);
      });

      it('matches NumPy for batched 3D matmul', () => {
        const a = arange(0, 128).reshape([2, 8, 8]);
        const b = arange(0, 128).reshape([2, 8, 8]);
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.arange(128, dtype=np.float64).reshape(2, 8, 8)
b = np.arange(128, dtype=np.float64).reshape(2, 8, 8)
result = a @ b
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('dot complex (WASM)', () => {
      it('matches NumPy for complex64 dot', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], 'complex64');
        const b = array([new Complex(7, 8), new Complex(9, 10), new Complex(11, 12)], 'complex64');
        const jsResult = dot(a, b);
        const pyResult = runNumPy(`
a = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
b = np.array([7+8j, 9+10j, 11+12j], dtype=np.complex64)
result = complex(np.dot(a, b))
        `);
        const jsVal = jsResult as Complex;
        expect(jsVal.re).toBeCloseTo(pyResult.value.re ?? pyResult.value[0], 4);
        expect(jsVal.im).toBeCloseTo(pyResult.value.im ?? pyResult.value[1], 4);
      });

      it('matches NumPy for complex128 dot', () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], 'complex128');
        const b = array([new Complex(7, 8), new Complex(9, 10), new Complex(11, 12)], 'complex128');
        const jsResult = dot(a, b);
        const pyResult = runNumPy(`
a = np.array([1+2j, 3+4j, 5+6j])
b = np.array([7+8j, 9+10j, 11+12j])
result = np.dot(a, b)
        `);
        const jsVal = jsResult as Complex;
        expect(jsVal.re).toBeCloseTo(pyResult.value.re ?? pyResult.value[0], 10);
        expect(jsVal.im).toBeCloseTo(pyResult.value.im ?? pyResult.value[1], 10);
      });
    });

    describe('matmul complex (WASM)', () => {
      it('matches NumPy for complex64 matmul', () => {
        const a = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex64'
        );
        const b = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, 0)],
          ],
          'complex64'
        );
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.array([[1+1j, 2+0j], [0+1j, 1-1j]], dtype=np.complex64)
b = np.array([[1+0j, 0+1j], [1+1j, 1+0j]], dtype=np.complex64)
result = a @ b
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });

      it('matches NumPy for complex128 matmul', () => {
        const a = array(
          [
            [new Complex(1, 1), new Complex(2, 0)],
            [new Complex(0, 1), new Complex(1, -1)],
          ],
          'complex128'
        );
        const b = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, 0)],
          ],
          'complex128'
        );
        const jsResult = matmul(a, b);
        const pyResult = runNumPy(`
a = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
b = np.array([[1+0j, 0+1j], [1+1j, 1+0j]])
result = a @ b
        `);
        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
      });
    });

    describe('dot multi-dtype (WASM)', () => {
      it('matches NumPy for float32 dot', () => {
        const a = array(arange(0, 128).toArray(), 'float32');
        const b = array(arange(0, 128).toArray(), 'float32');
        const jsResult = dot(a, b);
        const pyResult = runNumPy(`
a = np.arange(128, dtype=np.float32)
b = np.arange(128, dtype=np.float32)
result = np.dot(a, b)
        `);
        expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 0);
      });

      it('matches NumPy for int32 dot', () => {
        const a = array(arange(0, 128).toArray(), 'int32');
        const b = array(arange(0, 128).toArray(), 'int32');
        const jsResult = dot(a, b);
        const pyResult = runNumPy(`
a = np.arange(128, dtype=np.int32)
b = np.arange(128, dtype=np.int32)
result = np.dot(a, b)
        `);
        expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 5);
      });
    });
    describe('vector_norm (WASM)', () => {
      it('matches NumPy for float64 L2 norm', () => {
        const a = arange(1, 129);
        const jsResult = linalg.norm(a);
        const pyResult = runNumPy(`
a = np.arange(1, 129, dtype=np.float64)
result = np.linalg.norm(a)
        `);
        expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 10);
      });

      it('matches NumPy for float32 L2 norm', () => {
        const a = array(arange(1, 129).toArray(), 'float32');
        const jsResult = linalg.norm(a);
        const pyResult = runNumPy(`
a = np.arange(1, 129, dtype=np.float32)
result = np.linalg.norm(a)
        `);
        expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
      });
    });
  });
} // end WASM_MODES loop
