/**
 * Python NumPy validation tests for bitwise operations
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import {
  array,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  bitwise_not,
  invert,
  left_shift,
  right_shift,
  packbits,
  unpackbits,
  bitwise_count,
  bitwise_invert,
  bitwise_left_shift,
  bitwise_right_shift,
  wasmConfig,
} from '../../src/index';
import { runNumPy, checkNumPyAvailable } from './numpy-oracle';

const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1 },
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
] as const;

const INT_DTYPES = ['int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8'] as const;
const NP_DTYPE: Record<string, string> = {
  int64: 'np.int64',
  int32: 'np.int32',
  int16: 'np.int16',
  int8: 'np.int8',
  uint64: 'np.uint64',
  uint32: 'np.uint32',
  uint16: 'np.uint16',
  uint8: 'np.uint8',
};

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: Bitwise Operations [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
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

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    describe('bitwise_and', () => {
      it('matches NumPy for int32 arrays', () => {
        const a = array([0b1100, 0b1010, 0b1111], 'int32');
        const b = array([0b1010, 0b1100, 0b0101], 'int32');
        const jsResult = bitwise_and(a, b);
        const pyResult = runNumPy(`
result = np.bitwise_and(np.array([0b1100, 0b1010, 0b1111], dtype=np.int32), np.array([0b1010, 0b1100, 0b0101], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy with scalar', () => {
        const a = array([0b1111, 0b1010, 0b0101], 'int32');
        const jsResult = bitwise_and(a, 0b1100);
        const pyResult = runNumPy(`
result = np.bitwise_and(np.array([0b1111, 0b1010, 0b0101], dtype=np.int32), 0b1100)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for uint8 arrays', () => {
        const a = array([255, 128, 64], 'uint8');
        const b = array([127, 64, 32], 'uint8');
        const jsResult = bitwise_and(a, b);
        const pyResult = runNumPy(`
result = np.bitwise_and(np.array([255, 128, 64], dtype=np.uint8), np.array([127, 64, 32], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_or', () => {
      it('matches NumPy for int32 arrays', () => {
        const a = array([0b1100, 0b1010, 0b0000], 'int32');
        const b = array([0b1010, 0b0100, 0b1111], 'int32');
        const jsResult = bitwise_or(a, b);
        const pyResult = runNumPy(`
result = np.bitwise_or(np.array([0b1100, 0b1010, 0b0000], dtype=np.int32), np.array([0b1010, 0b0100, 0b1111], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy with scalar', () => {
        const a = array([0b0000, 0b0101], 'int32');
        const jsResult = bitwise_or(a, 0b1010);
        const pyResult = runNumPy(`
result = np.bitwise_or(np.array([0b0000, 0b0101], dtype=np.int32), 0b1010)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_xor', () => {
      it('matches NumPy for int32 arrays', () => {
        const a = array([0b1100, 0b1010, 0b1111], 'int32');
        const b = array([0b1010, 0b1010, 0b1111], 'int32');
        const jsResult = bitwise_xor(a, b);
        const pyResult = runNumPy(`
result = np.bitwise_xor(np.array([0b1100, 0b1010, 0b1111], dtype=np.int32), np.array([0b1010, 0b1010, 0b1111], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy with scalar', () => {
        const a = array([0b1111, 0b0000], 'int32');
        const jsResult = bitwise_xor(a, 0b1010);
        const pyResult = runNumPy(`
result = np.bitwise_xor(np.array([0b1111, 0b0000], dtype=np.int32), 0b1010)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise ops (multi-dtype)', () => {
      for (const dtype of INT_DTYPES) {
        it(`bitwise_and matches NumPy for ${dtype}`, () => {
          const a = array([5, 3, 7], dtype as any);
          const b = array([3, 6, 5], dtype as any);
          const jsResult = bitwise_and(a, b);
          const pyResult = runNumPy(`
result = np.bitwise_and(np.array([5, 3, 7], dtype=${NP_DTYPE[dtype]}), np.array([3, 6, 5], dtype=${NP_DTYPE[dtype]}))
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });

        it(`bitwise_or matches NumPy for ${dtype}`, () => {
          const a = array([5, 3, 7], dtype as any);
          const b = array([3, 6, 5], dtype as any);
          const jsResult = bitwise_or(a, b);
          const pyResult = runNumPy(`
result = np.bitwise_or(np.array([5, 3, 7], dtype=${NP_DTYPE[dtype]}), np.array([3, 6, 5], dtype=${NP_DTYPE[dtype]}))
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });

        it(`bitwise_xor matches NumPy for ${dtype}`, () => {
          const a = array([5, 3, 7], dtype as any);
          const b = array([3, 6, 5], dtype as any);
          const jsResult = bitwise_xor(a, b);
          const pyResult = runNumPy(`
result = np.bitwise_xor(np.array([5, 3, 7], dtype=${NP_DTYPE[dtype]}), np.array([3, 6, 5], dtype=${NP_DTYPE[dtype]}))
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });

        it(`invert matches NumPy for ${dtype}`, () => {
          const a = array([5, 3, 7], dtype as any);
          const jsResult = invert(a);
          const pyResult = runNumPy(`
result = np.invert(np.array([5, 3, 7], dtype=${NP_DTYPE[dtype]}))
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });

        it(`left_shift matches NumPy for ${dtype}`, () => {
          const a = array([1, 2, 3], dtype as any);
          const jsResult = left_shift(a, 1);
          const pyResult = runNumPy(`
result = np.left_shift(np.array([1, 2, 3], dtype=${NP_DTYPE[dtype]}), 1)
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });

        it(`right_shift matches NumPy for ${dtype}`, () => {
          const a = array([4, 8, 16], dtype as any);
          const jsResult = right_shift(a, 1);
          const pyResult = runNumPy(`
result = np.right_shift(np.array([4, 8, 16], dtype=${NP_DTYPE[dtype]}), 1)
`);
          expect(jsResult.toArray()).toEqual(pyResult.value);
        });
      }
    });

    describe('bitwise_not', () => {
      it('matches NumPy for uint8 arrays', () => {
        const a = array([0, 255, 128], 'uint8');
        const jsResult = bitwise_not(a);
        const pyResult = runNumPy(`
result = np.bitwise_not(np.array([0, 255, 128], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for int8 arrays', () => {
        const a = array([0, -1, 127], 'int8');
        const jsResult = bitwise_not(a);
        const pyResult = runNumPy(`
result = np.bitwise_not(np.array([0, -1, 127], dtype=np.int8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('invert', () => {
      it('matches NumPy for uint8 arrays', () => {
        const a = array([0, 255], 'uint8');
        const jsResult = invert(a);
        const pyResult = runNumPy(`
result = np.invert(np.array([0, 255], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for int32 arrays', () => {
        const a = array([0, -1, 12345], 'int32');
        const jsResult = invert(a);
        const pyResult = runNumPy(`
result = np.invert(np.array([0, -1, 12345], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('left_shift', () => {
      it('matches NumPy for int32 arrays', () => {
        const a = array([1, 2, 4], 'int32');
        const jsResult = left_shift(a, 2);
        const pyResult = runNumPy(`
result = np.left_shift(np.array([1, 2, 4], dtype=np.int32), 2)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy with array shift amounts', () => {
        const a = array([1, 1, 1], 'int32');
        const b = array([1, 2, 3], 'int32');
        const jsResult = left_shift(a, b);
        const pyResult = runNumPy(`
result = np.left_shift(np.array([1, 1, 1], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for uint8 arrays', () => {
        const a = array([1, 2, 4], 'uint8');
        const jsResult = left_shift(a, 1);
        const pyResult = runNumPy(`
result = np.left_shift(np.array([1, 2, 4], dtype=np.uint8), 1)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('right_shift', () => {
      it('matches NumPy for int32 arrays', () => {
        const a = array([8, 16, 32], 'int32');
        const jsResult = right_shift(a, 2);
        const pyResult = runNumPy(`
result = np.right_shift(np.array([8, 16, 32], dtype=np.int32), 2)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy with array shift amounts', () => {
        const a = array([8, 16, 32], 'int32');
        const b = array([1, 2, 3], 'int32');
        const jsResult = right_shift(a, b);
        const pyResult = runNumPy(`
result = np.right_shift(np.array([8, 16, 32], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for negative numbers', () => {
        const a = array([-8, -16, -32], 'int32');
        const jsResult = right_shift(a, 1);
        const pyResult = runNumPy(`
result = np.right_shift(np.array([-8, -16, -32], dtype=np.int32), 1)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('packbits', () => {
      it('matches NumPy basic packing', () => {
        const a = array([1, 0, 1, 0, 1, 0, 1, 0], 'uint8');
        const jsResult = packbits(a);
        const pyResult = runNumPy(`
result = np.packbits(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for padding case', () => {
        const a = array([1, 1, 1], 'uint8');
        const jsResult = packbits(a);
        const pyResult = runNumPy(`
result = np.packbits(np.array([1, 1, 1], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for 16 bits', () => {
        const a = array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 'uint8');
        const jsResult = packbits(a);
        const pyResult = runNumPy(`
result = np.packbits(np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('unpackbits', () => {
      it('matches NumPy basic unpacking', () => {
        const a = array([0b10101010], 'uint8');
        const jsResult = unpackbits(a);
        const pyResult = runNumPy(`
result = np.unpackbits(np.array([0b10101010], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for multiple bytes', () => {
        const a = array([0b11110000, 0b00001111], 'uint8');
        const jsResult = unpackbits(a);
        const pyResult = runNumPy(`
result = np.unpackbits(np.array([0b11110000, 0b00001111], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for zero byte', () => {
        const a = array([0], 'uint8');
        const jsResult = unpackbits(a);
        const pyResult = runNumPy(`
result = np.unpackbits(np.array([0], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for 255 byte', () => {
        const a = array([255], 'uint8');
        const jsResult = unpackbits(a);
        const pyResult = runNumPy(`
result = np.unpackbits(np.array([255], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('2D arrays', () => {
      it('bitwise_and matches NumPy on 2D', () => {
        const a = array(
          [
            [0b1111, 0b1010],
            [0b0101, 0b0000],
          ],
          'int32'
        );
        const b = array(
          [
            [0b1010, 0b1111],
            [0b1111, 0b1010],
          ],
          'int32'
        );
        const jsResult = bitwise_and(a, b);
        const pyResult = runNumPy(`
result = np.bitwise_and(np.array([[0b1111, 0b1010], [0b0101, 0b0000]], dtype=np.int32), np.array([[0b1010, 0b1111], [0b1111, 0b1010]], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('left_shift matches NumPy on 2D', () => {
        const a = array(
          [
            [1, 2],
            [4, 8],
          ],
          'int32'
        );
        const jsResult = left_shift(a, 1);
        const pyResult = runNumPy(`
result = np.left_shift(np.array([[1, 2], [4, 8]], dtype=np.int32), 1)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_count', () => {
      it('matches NumPy for uint8 array', () => {
        const a = array([0, 1, 255, 128, 15], 'uint8');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([0, 1, 255, 128, 15], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for int32 array', () => {
        const a = array([0, 1, 7, 16], 'int32');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([0, 1, 7, 16], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for negative int8 values', () => {
        const a = array([-128, -127, -1, -2, -64, 0, 1, 127], 'int8');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([-128, -127, -1, -2, -64, 0, 1, 127], dtype=np.int8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for negative int16 values', () => {
        const a = array([-32768, -1, -256, 0, 1, 32767], 'int16');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([-32768, -1, -256, 0, 1, 32767], dtype=np.int16))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for negative int32 values', () => {
        const a = array([-2147483648, -1, -256, 0, 1, 2147483647], 'int32');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([-2147483648, -1, -256, 0, 1, 2147483647], dtype=np.int32))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });

      it('matches NumPy for negative int64 values', () => {
        const a = array([BigInt(-1), BigInt(-128), BigInt(0), BigInt(1), BigInt(255)], 'int64');
        const jsResult = bitwise_count(a);
        const pyResult = runNumPy(`
result = np.bitwise_count(np.array([-1, -128, 0, 1, 255], dtype=np.int64))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_invert', () => {
      it('matches NumPy (alias for bitwise_not)', () => {
        const a = array([0, 255, 128], 'uint8');
        const jsResult = bitwise_invert(a);
        const pyResult = runNumPy(`
result = np.bitwise_not(np.array([0, 255, 128], dtype=np.uint8))
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_left_shift', () => {
      it('matches NumPy (alias for left_shift)', () => {
        const a = array([1, 2, 4], 'int32');
        const jsResult = bitwise_left_shift(a, 2);
        const pyResult = runNumPy(`
result = np.left_shift(np.array([1, 2, 4], dtype=np.int32), 2)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });

    describe('bitwise_right_shift', () => {
      it('matches NumPy (alias for right_shift)', () => {
        const a = array([8, 16, 32], 'int32');
        const jsResult = bitwise_right_shift(a, 2);
        const pyResult = runNumPy(`
result = np.right_shift(np.array([8, 16, 32], dtype=np.int32), 2)
      `);

        expect(jsResult.shape).toEqual(pyResult.shape);
        expect(jsResult.toArray()).toEqual(pyResult.value);
      });
    });
  });
}
