import { describe, it, expect } from 'vitest';
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
} from '../../src/index';

describe('Bitwise Operations', () => {
  describe('bitwise_and', () => {
    it('performs AND on two int32 arrays', () => {
      const a = array([0b1100, 0b1010, 0b1111], 'int32');
      const b = array([0b1010, 0b1100, 0b0101], 'int32');
      const result = bitwise_and(a, b);
      expect(result.toArray()).toEqual([0b1000, 0b1000, 0b0101]);
      expect(result.dtype).toBe('int32');
    });

    it('performs AND with scalar', () => {
      const a = array([0b1111, 0b1010, 0b0101], 'int32');
      const result = bitwise_and(a, 0b1100);
      expect(result.toArray()).toEqual([0b1100, 0b1000, 0b0100]);
    });

    it('works with uint8 arrays', () => {
      const a = array([255, 128, 64], 'uint8');
      const b = array([127, 64, 32], 'uint8');
      const result = bitwise_and(a, b);
      // 255 & 127 = 127, 128 & 64 = 0, 64 & 32 = 0
      expect(result.toArray()).toEqual([127, 0, 0]);
      expect(result.dtype).toBe('uint8');
    });

    it('works via NDArray method', () => {
      const a = array([0b1100, 0b1010], 'int32');
      const b = array([0b1010, 0b1100], 'int32');
      const result = a.bitwise_and(b);
      expect(result.toArray()).toEqual([0b1000, 0b1000]);
    });

    it('broadcasts 2D with 1D arrays', () => {
      const a = array(
        [
          [0b1111, 0b1010],
          [0b0101, 0b1100],
        ],
        'int32'
      );
      const b = array([0b1100, 0b0011], 'int32');
      const result = bitwise_and(a, b);
      // Row 0: [0b1111 & 0b1100, 0b1010 & 0b0011] = [0b1100, 0b0010]
      // Row 1: [0b0101 & 0b1100, 0b1100 & 0b0011] = [0b0100, 0b0000]
      expect(result.toArray()).toEqual([
        [0b1100, 0b0010],
        [0b0100, 0b0000],
      ]);
    });

    it('works with mixed int32 and int64 arrays', () => {
      const a = array([0b1111, 0b1010], 'int32');
      const b = array([0b1100, 0b0011], 'int64');
      const result = bitwise_and(a, b);
      expect(result.toArray()).toEqual([0b1100n, 0b0010n]);
      expect(result.dtype).toBe('int64');
    });
  });

  describe('bitwise_or', () => {
    it('performs OR on two int32 arrays', () => {
      const a = array([0b1100, 0b1010, 0b0000], 'int32');
      const b = array([0b1010, 0b0100, 0b1111], 'int32');
      const result = bitwise_or(a, b);
      expect(result.toArray()).toEqual([0b1110, 0b1110, 0b1111]);
      expect(result.dtype).toBe('int32');
    });

    it('performs OR with scalar', () => {
      const a = array([0b0000, 0b0101], 'int32');
      const result = bitwise_or(a, 0b1010);
      expect(result.toArray()).toEqual([0b1010, 0b1111]);
    });

    it('works with int16 arrays', () => {
      const a = array([0x00ff, 0x0f0f], 'int16');
      const b = array([0xff00, 0xf0f0], 'int16');
      const result = bitwise_or(a, b);
      expect(result.toArray()).toEqual([-1, -1]); // 0xffff in signed int16 is -1
      expect(result.dtype).toBe('int16');
    });

    it('works via NDArray method', () => {
      const a = array([0b0100, 0b0001], 'int32');
      const b = array([0b1000, 0b0010], 'int32');
      const result = a.bitwise_or(b);
      expect(result.toArray()).toEqual([0b1100, 0b0011]);
    });

    it('broadcasts 2D with 1D arrays', () => {
      const a = array(
        [
          [0b1000, 0b0001],
          [0b0010, 0b0100],
        ],
        'int32'
      );
      const b = array([0b0001, 0b1000], 'int32');
      const result = bitwise_or(a, b);
      expect(result.toArray()).toEqual([
        [0b1001, 0b1001],
        [0b0011, 0b1100],
      ]);
    });

    it('works with mixed int32 and int64 arrays', () => {
      const a = array([0b1000, 0b0001], 'int32');
      const b = array([0b0001, 0b1000], 'int64');
      const result = bitwise_or(a, b);
      expect(result.toArray()).toEqual([0b1001n, 0b1001n]);
      expect(result.dtype).toBe('int64');
    });
  });

  describe('bitwise_xor', () => {
    it('performs XOR on two int32 arrays', () => {
      const a = array([0b1100, 0b1010, 0b1111], 'int32');
      const b = array([0b1010, 0b1010, 0b1111], 'int32');
      const result = bitwise_xor(a, b);
      expect(result.toArray()).toEqual([0b0110, 0b0000, 0b0000]);
      expect(result.dtype).toBe('int32');
    });

    it('performs XOR with scalar', () => {
      const a = array([0b1111, 0b0000], 'int32');
      const result = bitwise_xor(a, 0b1010);
      expect(result.toArray()).toEqual([0b0101, 0b1010]);
    });

    it('works via NDArray method', () => {
      const a = array([0b1111, 0b1010], 'int32');
      const b = array([0b0101, 0b1010], 'int32');
      const result = a.bitwise_xor(b);
      expect(result.toArray()).toEqual([0b1010, 0b0000]);
    });

    it('broadcasts 2D with 1D arrays', () => {
      const a = array(
        [
          [0b1111, 0b0000],
          [0b1010, 0b0101],
        ],
        'int32'
      );
      const b = array([0b1010, 0b0101], 'int32');
      const result = bitwise_xor(a, b);
      expect(result.toArray()).toEqual([
        [0b0101, 0b0101],
        [0b0000, 0b0000],
      ]);
    });

    it('works with mixed int32 and int64 arrays', () => {
      const a = array([0b1111, 0b0000], 'int32');
      const b = array([0b1010, 0b0101], 'int64');
      const result = bitwise_xor(a, b);
      expect(result.toArray()).toEqual([0b0101n, 0b0101n]);
      expect(result.dtype).toBe('int64');
    });
  });

  describe('bitwise_not', () => {
    it('performs NOT on uint8 array', () => {
      const a = array([0, 255, 128], 'uint8');
      const result = bitwise_not(a);
      expect(result.toArray()).toEqual([255, 0, 127]);
      expect(result.dtype).toBe('uint8');
    });

    it('performs NOT on int8 array', () => {
      const a = array([0, -1, 127], 'int8');
      const result = bitwise_not(a);
      expect(result.toArray()).toEqual([-1, 0, -128]);
      expect(result.dtype).toBe('int8');
    });

    it('works via NDArray method', () => {
      const a = array([0b1111, 0b0000], 'uint8');
      const result = a.bitwise_not();
      expect(result.toArray()).toEqual([0b11110000, 0b11111111]);
    });
  });

  describe('invert', () => {
    it('is an alias for bitwise_not', () => {
      const a = array([0, 255], 'uint8');
      const result1 = bitwise_not(a);
      const result2 = invert(a);
      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('works via NDArray method', () => {
      const a = array([0b01010101], 'uint8');
      const result = a.invert();
      expect(result.toArray()).toEqual([0b10101010]);
    });
  });

  describe('left_shift', () => {
    it('shifts bits left', () => {
      const a = array([1, 2, 4], 'int32');
      const result = left_shift(a, 2);
      expect(result.toArray()).toEqual([4, 8, 16]);
      expect(result.dtype).toBe('int32');
    });

    it('shifts by different amounts per element', () => {
      const a = array([1, 1, 1], 'int32');
      const b = array([1, 2, 3], 'int32');
      const result = left_shift(a, b);
      expect(result.toArray()).toEqual([2, 4, 8]);
    });

    it('works with uint8', () => {
      const a = array([1, 2, 4], 'uint8');
      const result = left_shift(a, 1);
      expect(result.toArray()).toEqual([2, 4, 8]);
      expect(result.dtype).toBe('uint8');
    });

    it('works via NDArray method', () => {
      const a = array([1, 2, 4], 'int32');
      const result = a.left_shift(3);
      expect(result.toArray()).toEqual([8, 16, 32]);
    });
  });

  describe('right_shift', () => {
    it('shifts bits right', () => {
      const a = array([8, 16, 32], 'int32');
      const result = right_shift(a, 2);
      expect(result.toArray()).toEqual([2, 4, 8]);
      expect(result.dtype).toBe('int32');
    });

    it('shifts by different amounts per element', () => {
      const a = array([8, 16, 32], 'int32');
      const b = array([1, 2, 3], 'int32');
      const result = right_shift(a, b);
      expect(result.toArray()).toEqual([4, 4, 4]);
    });

    it('preserves sign for signed integers', () => {
      const a = array([-8], 'int32');
      const result = right_shift(a, 1);
      expect(result.toArray()).toEqual([-4]);
    });

    it('works via NDArray method', () => {
      const a = array([64, 128, 256], 'int32');
      const result = a.right_shift(2);
      expect(result.toArray()).toEqual([16, 32, 64]);
    });
  });

  describe('packbits', () => {
    it('packs boolean array into uint8', () => {
      const a = array([1, 0, 1, 0, 1, 0, 1, 0], 'uint8');
      const result = packbits(a);
      expect(result.toArray()).toEqual([0b10101010]);
      expect(result.dtype).toBe('uint8');
    });

    it('handles arrays not divisible by 8', () => {
      const a = array([1, 1, 1], 'uint8');
      const result = packbits(a);
      expect(result.toArray()).toEqual([0b11100000]); // Padded with zeros
    });

    it('packs 16 bits into 2 bytes', () => {
      const a = array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 'uint8');
      const result = packbits(a);
      expect(result.toArray()).toEqual([0b11110000, 0b00001111]);
    });

    it('handles empty array', () => {
      const a = array([], 'uint8');
      const result = packbits(a);
      expect(result.toArray()).toEqual([]);
    });
  });

  describe('unpackbits', () => {
    it('unpacks uint8 to bits', () => {
      const a = array([0b10101010], 'uint8');
      const result = unpackbits(a);
      expect(result.toArray()).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
      expect(result.dtype).toBe('uint8');
    });

    it('unpacks multiple bytes', () => {
      const a = array([0b11110000, 0b00001111], 'uint8');
      const result = unpackbits(a);
      expect(result.toArray()).toEqual([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
    });

    it('handles zero byte', () => {
      const a = array([0], 'uint8');
      const result = unpackbits(a);
      expect(result.toArray()).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    });

    it('handles 255 byte', () => {
      const a = array([255], 'uint8');
      const result = unpackbits(a);
      expect(result.toArray()).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
    });

    it('handles empty array', () => {
      const a = array([], 'uint8');
      const result = unpackbits(a);
      expect(result.toArray()).toEqual([]);
    });

    it('unpacks with count parameter', () => {
      const a = array([0b11110000], 'uint8');
      // Only unpack first 4 bits (axis=-1, count=4, bitorder='big')
      const result = unpackbits(a, -1, 4, 'big');
      expect(result.toArray()).toEqual([1, 1, 1, 1]);
    });

    it('unpacks with bitorder=little', () => {
      const a = array([0b10000001], 'uint8');
      // Little-endian: LSB first (axis=-1, count=-1, bitorder='little')
      const result = unpackbits(a, -1, -1, 'little');
      expect(result.toArray()).toEqual([1, 0, 0, 0, 0, 0, 0, 1]);
    });

    it('unpacks 2D array along axis', () => {
      const a = array([[0b11110000], [0b00001111]], 'uint8');
      // axis=1, count=-1, bitorder='big'
      const result = unpackbits(a, 1, -1, 'big');
      expect(result.shape).toEqual([2, 8]);
      expect(result.toArray()).toEqual([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
      ]);
    });

    it('unpacks 2D array with count and bitorder=little', () => {
      const a = array([[0b10101010]], 'uint8');
      // axis=1, count=4, bitorder='little'
      const result = unpackbits(a, 1, 4, 'little');
      expect(result.shape).toEqual([1, 4]);
      expect(result.toArray()).toEqual([[0, 1, 0, 1]]);
    });
  });

  describe('broadcasting', () => {
    it('broadcasts scalar to array in bitwise_and', () => {
      const a = array(
        [
          [0b1111, 0b1010],
          [0b0101, 0b0000],
        ],
        'int32'
      );
      const result = bitwise_and(a, 0b1100);
      expect(result.toArray()).toEqual([
        [0b1100, 0b1000],
        [0b0100, 0b0000],
      ]);
    });

    it('broadcasts 1D to 2D in bitwise_or', () => {
      const a = array(
        [
          [0b0000, 0b0000],
          [0b0000, 0b0000],
        ],
        'int32'
      );
      const b = array([0b1010, 0b0101], 'int32');
      const result = bitwise_or(a, b);
      expect(result.toArray()).toEqual([
        [0b1010, 0b0101],
        [0b1010, 0b0101],
      ]);
    });
  });

  describe('BigInt support', () => {
    it('supports int64 bitwise_and', () => {
      // Use values that fit in signed int64 range
      const a = array([BigInt(0x7f00ff00ff00ff00n), BigInt(0x00ff00ff00ff00ffn)], 'int64');
      const b = array([BigInt(0x70f0f0f0f0f0f0f0n), BigInt(0x0f0f0f0f0f0f0f0fn)], 'int64');
      const result = bitwise_and(a, b);
      expect(result.dtype).toBe('int64');
      const arr = result.toArray();
      expect(arr[0]).toBe(BigInt(0x7000f000f000f000n));
      expect(arr[1]).toBe(BigInt(0x000f000f000f000fn));
    });

    it('supports uint64 bitwise_or', () => {
      const a = array([BigInt(0xff00000000000000n)], 'uint64');
      const b = array([BigInt(0x00000000000000ffn)], 'uint64');
      const result = bitwise_or(a, b);
      expect(result.dtype).toBe('uint64');
      expect(result.toArray()[0]).toBe(BigInt(0xff000000000000ffn));
    });
  });

  describe('error handling', () => {
    it('throws on float arrays for bitwise_and', () => {
      const a = array([1.5, 2.5], 'float64');
      const b = array([1.0, 2.0], 'float64');
      expect(() => bitwise_and(a, b)).toThrow();
    });

    it('throws on float arrays for bitwise_not', () => {
      const a = array([1.5, 2.5], 'float64');
      expect(() => bitwise_not(a)).toThrow();
    });

    it('packbits works with non-uint8 arrays (treats non-zero as 1)', () => {
      // packbits accepts any dtype and treats non-zero as 1
      const a = array([1, 0, 1], 'int32');
      const result = packbits(a);
      expect(result.toArray()).toEqual([0b10100000]); // Padded with zeros
      expect(result.dtype).toBe('uint8');
    });

    it('throws on non-uint8 arrays for unpackbits', () => {
      const a = array([255], 'int32');
      expect(() => unpackbits(a)).toThrow();
    });
  });

  describe('2D arrays', () => {
    it('bitwise_and on 2D arrays', () => {
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
      const result = bitwise_and(a, b);
      expect(result.toArray()).toEqual([
        [0b1010, 0b1010],
        [0b0101, 0b0000],
      ]);
      expect(result.shape).toEqual([2, 2]);
    });

    it('left_shift on 2D array', () => {
      const a = array(
        [
          [1, 2],
          [4, 8],
        ],
        'int32'
      );
      const result = left_shift(a, 1);
      expect(result.toArray()).toEqual([
        [2, 4],
        [8, 16],
      ]);
    });
  });

  describe('bitwise_count', () => {
    it('counts 1-bits in uint8 array', () => {
      const a = array([0, 1, 255, 128, 15], 'uint8');
      const result = bitwise_count(a);
      // 0b00000000 = 0, 0b00000001 = 1, 0b11111111 = 8, 0b10000000 = 1, 0b00001111 = 4
      expect(result.toArray()).toEqual([0, 1, 8, 1, 4]);
      expect(result.dtype).toBe('uint8');
    });

    it('counts 1-bits in int32 array', () => {
      const a = array([0, 1, -1, 7, 16], 'int32');
      const result = bitwise_count(a);
      // 0 = 0 bits, 1 = 1 bit, -1 = 32 bits (all 1s), 7 = 3 bits, 16 = 1 bit
      expect(result.toArray()).toEqual([0, 1, 32, 3, 1]);
    });

    it('handles 2D arrays', () => {
      const a = array(
        [
          [1, 3],
          [7, 15],
        ],
        'uint8'
      );
      const result = bitwise_count(a);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('throws for float arrays', () => {
      const a = array([1.5, 2.5], 'float64');
      expect(() => bitwise_count(a)).toThrow();
    });

    it('counts 1-bits in int64 BigInt array', () => {
      const a = array([BigInt(0), BigInt(1), BigInt(7), BigInt(255)], 'int64');
      const result = bitwise_count(a);
      // 0 = 0 bits, 1 = 1 bit, 7 = 3 bits, 255 = 8 bits
      expect(result.toArray()).toEqual([0, 1, 3, 8]);
      expect(result.dtype).toBe('uint8');
    });

    it('counts 1-bits in uint64 BigInt array', () => {
      const a = array([BigInt(0), BigInt(1), BigInt(15), BigInt(63)], 'uint64');
      const result = bitwise_count(a);
      // 0 = 0 bits, 1 = 1 bit, 15 = 4 bits, 63 = 6 bits
      expect(result.toArray()).toEqual([0, 1, 4, 6]);
    });

    it('counts 1-bits in negative int64 BigInt', () => {
      // -1 as int64 should have 64 bits set (all 1s in two's complement)
      const a = array([BigInt(-1)], 'int64');
      const result = bitwise_count(a);
      expect(result.toArray()).toEqual([64]);
    });
  });

  describe('bitwise_invert', () => {
    it('is an alias for bitwise_not', () => {
      const a = array([0, 255, 128], 'uint8');
      const result1 = bitwise_not(a);
      const result2 = bitwise_invert(a);
      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('inverts int32 values', () => {
      const a = array([0, 1, -1], 'int32');
      const result = bitwise_invert(a);
      expect(result.toArray()).toEqual([-1, -2, 0]);
    });
  });

  describe('bitwise_left_shift', () => {
    it('is an alias for left_shift', () => {
      const a = array([1, 2, 4], 'int32');
      const result1 = left_shift(a, 2);
      const result2 = bitwise_left_shift(a, 2);
      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('shifts with array', () => {
      const a = array([1, 1, 1], 'int32');
      const b = array([1, 2, 3], 'int32');
      const result = bitwise_left_shift(a, b);
      expect(result.toArray()).toEqual([2, 4, 8]);
    });
  });

  describe('bitwise_right_shift', () => {
    it('is an alias for right_shift', () => {
      const a = array([8, 16, 32], 'int32');
      const result1 = right_shift(a, 2);
      const result2 = bitwise_right_shift(a, 2);
      expect(result1.toArray()).toEqual(result2.toArray());
    });

    it('shifts with array', () => {
      const a = array([8, 16, 32], 'int32');
      const b = array([1, 2, 3], 'int32');
      const result = bitwise_right_shift(a, b);
      expect(result.toArray()).toEqual([4, 4, 4]);
    });
  });
});
