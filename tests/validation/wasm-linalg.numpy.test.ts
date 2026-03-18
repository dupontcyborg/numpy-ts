/**
 * Python NumPy validation tests for WASM-accelerated linear algebra.
 *
 * WASM acceleration is now built into matmul directly — no separate
 * entrypoint needed. These tests verify that the unified matmul
 * (which tries WASM first, falls back to JS) matches NumPy.
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { array, matmul, dot, inner, reshape, linalg, wasmConfig } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

// Run all tests with both default thresholds (multiplier=1) and forced-WASM (multiplier=0)
const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1 },
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
] as const;

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: WASM-accelerated Linear Algebra [${mode.name}]`, () => {
    beforeAll(() => {
      if (!checkNumPyAvailable()) {
        throw new Error(
          'Python NumPy not available!\n\n' +
            '   This test suite requires Python with NumPy installed.\n\n' +
            '   Setup: source ~/.zshrc && conda activate py313\n'
        );
      }
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier; // restore to current mode
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
  });
} // end WASM_MODES loop
