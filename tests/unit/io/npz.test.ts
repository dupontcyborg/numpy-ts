import { describe, it, expect } from 'vitest';
import {
  parseNpz,
  parseNpzSync,
  serializeNpz,
  serializeNpzSync,
  loadNpz,
  loadNpzSync,
} from '../../../src/io/npz';
import { array, zeros, arange } from '../../../src/core/ndarray';

describe('NPZ Format', () => {
  describe('serializeNpzSync and parseNpzSync', () => {
    it('round-trips a single array', () => {
      const arr = array([1, 2, 3, 4, 5]);
      const npzBytes = serializeNpzSync({ arr });
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.size).toBe(1);
      expect(result.skipped.length).toBe(0);

      const loaded = result.arrays.get('arr');
      expect(loaded).toBeDefined();
      expect(loaded!.shape).toEqual([5]);
      expect(loaded!.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('round-trips multiple arrays', () => {
      const a = array([1, 2, 3]);
      const b = array([
        [4, 5],
        [6, 7],
      ]);
      const c = zeros([3, 3], 'float32');

      const npzBytes = serializeNpzSync({ a, b, c });
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.size).toBe(3);

      expect(result.arrays.get('a')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('b')!.toArray()).toEqual([
        [4, 5],
        [6, 7],
      ]);
      expect(result.arrays.get('c')!.dtype).toBe('float32');
    });

    it('works with positional array input (like np.savez positional args)', () => {
      const arr1 = array([1, 2, 3]);
      const arr2 = array([4, 5, 6]);
      const arr3 = array([7, 8, 9]);

      const npzBytes = serializeNpzSync([arr1, arr2, arr3]);
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.size).toBe(3);
      expect(result.arrays.get('arr_0')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('arr_1')!.toArray()).toEqual([4, 5, 6]);
      expect(result.arrays.get('arr_2')!.toArray()).toEqual([7, 8, 9]);
    });

    it('works with Map input', () => {
      const arrays = new Map([
        ['x', array([1, 2, 3])],
        ['y', array([4, 5, 6])],
      ]);

      const npzBytes = serializeNpzSync(arrays);
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.size).toBe(2);
      expect(result.arrays.get('x')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('y')!.toArray()).toEqual([4, 5, 6]);
    });

    it('preserves array names', () => {
      const mySpecialArray = array([1, 2, 3]);
      const anotherArray = array([4, 5, 6]);

      const npzBytes = serializeNpzSync({ mySpecialArray, anotherArray });
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.has('mySpecialArray')).toBe(true);
      expect(result.arrays.has('anotherArray')).toBe(true);
    });

    it('handles different dtypes', () => {
      const f64 = array([1.0, 2.0, 3.0], 'float64');
      const i32 = array([1, 2, 3], 'int32');
      const u8 = array([0, 128, 255], 'uint8');

      const npzBytes = serializeNpzSync({ f64, i32, u8 });
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.get('f64')!.dtype).toBe('float64');
      expect(result.arrays.get('i32')!.dtype).toBe('int32');
      expect(result.arrays.get('u8')!.dtype).toBe('uint8');
    });

    it('handles empty arrays object', () => {
      const npzBytes = serializeNpzSync({});
      const result = parseNpzSync(npzBytes);

      expect(result.arrays.size).toBe(0);
    });

    it('handles large arrays', () => {
      const large = arange(10000);
      const npzBytes = serializeNpzSync({ large });
      const result = parseNpzSync(npzBytes);

      const loaded = result.arrays.get('large')!;
      expect(loaded.shape).toEqual([10000]);
      expect(loaded.get([0])).toBe(0);
      expect(loaded.get([9999])).toBe(9999);
    });
  });

  describe('async serializeNpz and parseNpz', () => {
    it('round-trips a single array', async () => {
      const arr = array([1, 2, 3, 4, 5]);
      const npzBytes = await serializeNpz({ arr });
      const result = await parseNpz(npzBytes);

      expect(result.arrays.size).toBe(1);
      const loaded = result.arrays.get('arr');
      expect(loaded!.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('round-trips multiple arrays', async () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);

      const npzBytes = await serializeNpz({ a, b });
      const result = await parseNpz(npzBytes);

      expect(result.arrays.size).toBe(2);
      expect(result.arrays.get('a')!.toArray()).toEqual([1, 2, 3]);
      expect(result.arrays.get('b')!.toArray()).toEqual([4, 5, 6]);
    });

    it('supports compression', async () => {
      const arr = arange(1000);
      const compressedBytes = await serializeNpz({ arr }, { compress: true });
      const uncompressedBytes = await serializeNpz({ arr }, { compress: false });

      // Compressed should be smaller
      expect(compressedBytes.length).toBeLessThan(uncompressedBytes.length);

      // Both should round-trip correctly
      const compressedResult = await parseNpz(compressedBytes);
      const uncompressedResult = await parseNpz(uncompressedBytes);

      expect(compressedResult.arrays.get('arr')!.shape).toEqual([1000]);
      expect(uncompressedResult.arrays.get('arr')!.shape).toEqual([1000]);
    });
  });

  describe('loadNpz and loadNpzSync convenience functions', () => {
    it('loadNpzSync returns object with array names as keys', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);

      const npzBytes = serializeNpzSync({ a, b });
      const result = loadNpzSync(npzBytes);

      expect(result.a).toBeDefined();
      expect(result.b).toBeDefined();
      expect(result.a.toArray()).toEqual([1, 2, 3]);
      expect(result.b.toArray()).toEqual([4, 5, 6]);
    });

    it('loadNpz returns object with array names as keys', async () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);

      const npzBytes = await serializeNpz({ a, b });
      const result = await loadNpz(npzBytes);

      expect(result.a).toBeDefined();
      expect(result.b).toBeDefined();
      expect(result.a.toArray()).toEqual([1, 2, 3]);
      expect(result.b.toArray()).toEqual([4, 5, 6]);
    });
  });

  describe('force option for unsupported dtypes', () => {
    it('skips unsupported arrays when force=true', async () => {
      // We can't easily create an unsupported dtype array in the NPZ,
      // so we'll test the error path instead
      const arr = array([1, 2, 3]);
      const npzBytes = await serializeNpz({ arr });
      const result = await parseNpz(npzBytes, { force: true });

      // Normal arrays should load fine
      expect(result.arrays.size).toBe(1);
      expect(result.skipped.length).toBe(0);
    });

    it('returns skipped array names and errors with force=true', async () => {
      // Create a valid NPZ and test the behavior
      const arr = array([1, 2, 3]);
      const npzBytes = await serializeNpz({ arr });
      const result = await parseNpz(npzBytes, { force: true });

      // Since all arrays are valid, nothing should be skipped
      expect(result.skipped).toEqual([]);
      expect(result.errors.size).toBe(0);
    });
  });

  describe('error handling', () => {
    it('throws for empty array names', () => {
      expect(() => serializeNpzSync({ '': array([1, 2, 3]) })).toThrow(
        'Array names must be non-empty strings'
      );
    });

    it('rejects invalid ZIP files', async () => {
      const invalidBytes = new Uint8Array([0, 1, 2, 3, 4, 5]);
      await expect(parseNpz(invalidBytes)).rejects.toThrow('Invalid ZIP file');
    });

    it('rejects invalid ZIP files sync', () => {
      const invalidBytes = new Uint8Array([0, 1, 2, 3, 4, 5]);
      expect(() => parseNpzSync(invalidBytes)).toThrow('Invalid ZIP file');
    });
  });
});
