import { describe, it, expect } from 'vitest';
import { parseTxt, genfromtxt, fromregex, serializeTxt } from '../../../src/io/txt';
import { array } from '../../../src';
// Also import from main index to test wrappers
import {
  parseTxt as parseTxtIndex,
  genfromtxt as genfromtxtIndex,
  fromregex as fromregexIndex,
} from '../../../src';

describe('Text I/O', () => {
  describe('parseTxt', () => {
    it('parses whitespace-delimited data', () => {
      const text = '1 2 3\n4 5 6\n7 8 9';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([3, 3]);
      expect(arr.dtype).toBe('float64');
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
    });

    it('parses comma-delimited (CSV) data', () => {
      const text = '1,2,3\n4,5,6';
      const arr = parseTxt(text, { delimiter: ',' });

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('parses tab-delimited (TSV) data', () => {
      const text = '1\t2\t3\n4\t5\t6';
      const arr = parseTxt(text, { delimiter: '\t' });

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('skips comment lines', () => {
      const text = '# This is a comment\n1 2 3\n# Another comment\n4 5 6';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('uses custom comment character', () => {
      const text = '// Comment\n1 2 3\n4 5 6';
      const arr = parseTxt(text, { comments: '//' });

      expect(arr.shape).toEqual([2, 3]);
    });

    it('skips initial rows', () => {
      const text = 'header line\nanother header\n1 2 3\n4 5 6';
      const arr = parseTxt(text, { skiprows: 2 });

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('limits number of rows', () => {
      const text = '1 2 3\n4 5 6\n7 8 9\n10 11 12';
      const arr = parseTxt(text, { max_rows: 2 });

      expect(arr.shape).toEqual([2, 3]);
      expect(arr.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('selects specific columns', () => {
      const text = '1 2 3 4\n5 6 7 8';
      const arr = parseTxt(text, { usecols: [0, 2] });

      expect(arr.shape).toEqual([2, 2]);
      expect(arr.toArray()).toEqual([
        [1, 3],
        [5, 7],
      ]);
    });

    it('handles negative column indices', () => {
      const text = '1 2 3 4\n5 6 7 8';
      const arr = parseTxt(text, { usecols: [-1] });

      expect(arr.shape).toEqual([2]);
      expect(arr.toArray()).toEqual([4, 8]);
    });

    it('uses specified dtype', () => {
      const text = '1 2 3\n4 5 6';
      const arr = parseTxt(text, { dtype: 'int32' });

      expect(arr.dtype).toBe('int32');
    });

    it('parses floating-point values', () => {
      const text = '1.5 2.5 3.5\n-1.0 0.0 1.0';
      const arr = parseTxt(text);

      expect(arr.toArray()).toEqual([
        [1.5, 2.5, 3.5],
        [-1.0, 0.0, 1.0],
      ]);
    });

    it('parses scientific notation', () => {
      const text = '1e-5\n2e10\n3.5e2';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([3]);
      expect(arr.get([0])).toBeCloseTo(1e-5, 10);
      expect(arr.get([1])).toBeCloseTo(2e10, 0);
      expect(arr.get([2])).toBeCloseTo(350, 5);
    });

    it('returns 1D array for single column', () => {
      const text = '1\n2\n3\n4';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([4]);
      expect(arr.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('handles empty lines', () => {
      const text = '1 2 3\n\n4 5 6\n\n';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([2, 3]);
    });

    it('handles Windows line endings', () => {
      const text = '1 2 3\r\n4 5 6\r\n';
      const arr = parseTxt(text);

      expect(arr.shape).toEqual([2, 3]);
    });

    it('throws on inconsistent columns', () => {
      const text = '1 2 3\n4 5';

      expect(() => parseTxt(text)).toThrow('Inconsistent number of columns');
    });

    it('returns empty array for empty input', () => {
      const arr = parseTxt('');
      expect(arr.size).toBe(0);
    });

    it('returns empty array for comments-only input', () => {
      const text = '# comment 1\n# comment 2';
      const arr = parseTxt(text);
      expect(arr.size).toBe(0);
    });
  });

  describe('genfromtxt', () => {
    it('handles missing values', () => {
      const text = '1,2,3\n4,,6\n7,8,9';
      const arr = genfromtxt(text, { delimiter: ',' });

      expect(arr.shape).toEqual([3, 3]);
      expect(arr.get([1, 1])).toBeNaN();
    });

    it('handles various missing value representations', () => {
      const text = '1,2,3\n4,NA,6\n7,nan,9';
      const arr = genfromtxt(text, { delimiter: ',' });

      expect(arr.get([1, 1])).toBeNaN();
      expect(arr.get([2, 1])).toBeNaN();
    });

    it('uses custom filling value', () => {
      const text = '1,2,3\n4,,6';
      const arr = genfromtxt(text, { delimiter: ',', filling_values: -999 });

      expect(arr.get([1, 1])).toBe(-999);
    });

    it('uses custom missing values', () => {
      const text = '1,2,3\n4,MISSING,6';
      const arr = genfromtxt(text, { delimiter: ',', missing_values: 'MISSING' });

      expect(arr.get([1, 1])).toBeNaN();
    });
  });

  describe('fromregex', () => {
    it('extracts values using regex', () => {
      const text = 'x=1.0, y=2.0\nx=3.0, y=4.0';
      const arr = fromregex(text, /x=([\d.]+), y=([\d.]+)/);

      expect(arr.shape).toEqual([2, 2]);
      expect(arr.toArray()).toEqual([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
    });

    it('extracts single column', () => {
      const text = 'value: 10\nvalue: 20\nvalue: 30';
      const arr = fromregex(text, /value: (\d+)/);

      expect(arr.shape).toEqual([3]);
      expect(arr.toArray()).toEqual([10, 20, 30]);
    });

    it('accepts string regex', () => {
      const text = 'a=1 b=2\na=3 b=4';
      const arr = fromregex(text, 'a=(\\d+) b=(\\d+)');

      expect(arr.shape).toEqual([2, 2]);
    });

    it('uses specified dtype', () => {
      const text = 'v=1\nv=2';
      const arr = fromregex(text, /v=(\d+)/, 'int32');

      expect(arr.dtype).toBe('int32');
    });

    it('returns empty array for no matches', () => {
      const text = 'no matches here';
      const arr = fromregex(text, /x=(\d+)/);

      expect(arr.size).toBe(0);
    });

    it('handles multiline text', () => {
      const text = `Point 1: (1.5, 2.5)
Point 2: (3.5, 4.5)
Point 3: (5.5, 6.5)`;
      const arr = fromregex(text, /\(([\d.]+), ([\d.]+)\)/);

      expect(arr.shape).toEqual([3, 2]);
      expect(arr.toArray()).toEqual([
        [1.5, 2.5],
        [3.5, 4.5],
        [5.5, 6.5],
      ]);
    });
  });

  describe('serializeTxt', () => {
    it('serializes 2D array with default format', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const text = serializeTxt(arr);

      // Default format is %.18e (scientific notation)
      expect(text).toContain('e+00');
      const lines = text.trim().split('\n');
      expect(lines.length).toBe(2);
    });

    it('serializes with custom delimiter', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const text = serializeTxt(arr, { delimiter: ',', fmt: '%d' });

      expect(text).toBe('1,2,3\n4,5,6\n');
    });

    it('serializes 1D array', () => {
      const arr = array([1, 2, 3]);
      const text = serializeTxt(arr, { fmt: '%d' });

      expect(text).toBe('1\n2\n3\n');
    });

    it('uses fixed-point format', () => {
      const arr = array([1.5, 2.5, 3.5]);
      const text = serializeTxt(arr, { fmt: '%.2f' });

      expect(text).toBe('1.50\n2.50\n3.50\n');
    });

    it('uses integer format', () => {
      const arr = array([1.9, 2.1, 3.5]);
      const text = serializeTxt(arr, { fmt: '%d' });

      expect(text).toBe('2\n2\n4\n');
    });

    it('adds header', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const text = serializeTxt(arr, { fmt: '%d', header: 'x y' });

      expect(text.startsWith('# x y\n')).toBe(true);
    });

    it('adds footer', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const text = serializeTxt(arr, { fmt: '%d', footer: 'end of file' });

      expect(text.endsWith('# end of file\n')).toBe(true);
    });

    it('uses custom comment prefix', () => {
      const arr = array([[1, 2]]);
      const text = serializeTxt(arr, { fmt: '%d', header: 'data', comments: '// ' });

      expect(text.startsWith('// data\n')).toBe(true);
    });

    it('uses custom newline', () => {
      const arr = array([1, 2, 3]);
      const text = serializeTxt(arr, { fmt: '%d', newline: '\r\n' });

      expect(text).toBe('1\r\n2\r\n3\r\n');
    });

    it('throws for 3D arrays', () => {
      const arr = array([
        [
          [1, 2],
          [3, 4],
        ],
      ]);

      expect(() => serializeTxt(arr)).toThrow('must be 1D or 2D');
    });

    it('formats with scientific notation', () => {
      const arr = array([1e-10, 2e10]);
      const text = serializeTxt(arr, { fmt: '%.2e' });

      expect(text).toContain('1.00e-10');
      expect(text).toContain('2.00e+10');
    });

    it('handles positive format flag', () => {
      const arr = array([1, -2, 3]);
      const text = serializeTxt(arr, { fmt: '%+d' });

      expect(text).toBe('+1\n-2\n+3\n');
    });

    it('handles width padding (right-align)', () => {
      const arr = array([1, 22, 333]);
      const text = serializeTxt(arr, { fmt: '%5d' });

      // Right-aligned: "    1", "   22", "  333"
      expect(text).toBe('    1\n   22\n  333\n');
    });

    it('handles width padding (left-align)', () => {
      const arr = array([1, 22, 333]);
      const text = serializeTxt(arr, { fmt: '%-5d' });

      // Left-aligned: "1    ", "22   ", "333  "
      expect(text).toBe('1    \n22   \n333  \n');
    });

    it('handles string format specifier', () => {
      const arr = array([1.5, 2.75, 3.125]);
      const text = serializeTxt(arr, { fmt: '%s' });

      expect(text).toBe('1.5\n2.75\n3.125\n');
    });

    it('handles header with comment prefix already present', () => {
      const arr = array([1, 2]);
      const text = serializeTxt(arr, { fmt: '%d', header: '# Already commented' });

      expect(text).toBe('# Already commented\n1\n2\n');
    });

    it('handles footer with comment prefix already present', () => {
      const arr = array([1, 2]);
      const text = serializeTxt(arr, { fmt: '%d', footer: '# End of data' });

      expect(text).toBe('1\n2\n# End of data\n');
    });

    it('handles uppercase scientific notation (E)', () => {
      const arr = array([1.5, 2.75]);
      const text = serializeTxt(arr, { fmt: '%.2E' });

      expect(text).toContain('1.50E+00');
      expect(text).toContain('2.75E+00');
    });

    it('handles general format (g)', () => {
      const arr = array([1.5, 12345, 0.000001]);
      const text = serializeTxt(arr, { fmt: '%.4g' });

      // General format uses fixed for small exponents, scientific for large
      const lines = text.trim().split('\n');
      expect(lines.length).toBe(3);
      expect(lines[0]).toContain('1.5'); // Small number: fixed
      expect(lines[1]).toMatch(/1\.23[45]e\+0?4/); // Large: scientific (allow 1.234 or 1.235)
      expect(lines[2]).toMatch(/1(\.\d+)?e-0?6/); // Very small: scientific
    });

    it('handles uppercase general format (G)', () => {
      const arr = array([1e10]);
      const text = serializeTxt(arr, { fmt: '%.2G' });

      expect(text).toContain('E+'); // Uppercase E
    });

    it('handles invalid format string', () => {
      const arr = array([1, 2, 3]);
      const text = serializeTxt(arr, { fmt: 'invalid' });

      // Should fall back to string representation
      expect(text).toBe('1\n2\n3\n');
    });

    it('handles multiline header', () => {
      const arr = array([1, 2]);
      const text = serializeTxt(arr, {
        fmt: '%d',
        header: 'Line 1\nLine 2\nLine 3',
      });

      const lines = text.split('\n');
      expect(lines[0]).toBe('# Line 1');
      expect(lines[1]).toBe('# Line 2');
      expect(lines[2]).toBe('# Line 3');
    });

    it('handles multiline footer', () => {
      const arr = array([1, 2]);
      const text = serializeTxt(arr, {
        fmt: '%d',
        footer: 'Footer line 1\nFooter line 2',
      });

      expect(text).toContain('# Footer line 1\n# Footer line 2\n');
    });
  });

  describe('round-trip', () => {
    it('round-trips 2D array', () => {
      const original = array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5],
      ]);
      const text = serializeTxt(original, { fmt: '%.6f' });
      const loaded = parseTxt(text);

      expect(loaded.shape).toEqual(original.shape);
      for (let i = 0; i < original.shape[0]; i++) {
        for (let j = 0; j < original.shape[1]; j++) {
          expect(loaded.get([i, j])).toBeCloseTo(original.get([i, j]) as number, 5);
        }
      }
    });

    it('round-trips 1D array', () => {
      const original = array([1, 2, 3, 4, 5]);
      const text = serializeTxt(original, { fmt: '%d' });
      const loaded = parseTxt(text, { dtype: 'int32' });

      expect(loaded.shape).toEqual(original.shape);
      expect(loaded.toArray()).toEqual(original.toArray());
    });

    it('round-trips CSV format', () => {
      const original = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const text = serializeTxt(original, { fmt: '%d', delimiter: ',' });
      const loaded = parseTxt(text, { delimiter: ',' });

      expect(loaded.toArray()).toEqual(original.toArray());
    });
  });

  describe('index.ts wrappers upgrade to NDArray', () => {
    it('parseTxtIndex returns NDArray instance', () => {
      const text = '1 2 3\n4 5 6';
      const result = parseTxtIndex(text);

      // Check that result is upgraded to NDArray (has methods)
      expect(result).toBeDefined();
      expect(typeof result.add).toBe('function'); // NDArray has method chaining
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('genfromtxtIndex returns NDArray instance', () => {
      const text = '1 2 3\n4 5 6';
      const result = genfromtxtIndex(text);

      // Check that result is upgraded to NDArray (has methods)
      expect(result).toBeDefined();
      expect(typeof result.add).toBe('function'); // NDArray has method chaining
      expect(result.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('fromregexIndex returns NDArray instance', () => {
      const text = 'value=1.5\nvalue=2.5\nvalue=3.5';
      const result = fromregexIndex(text, /value=([0-9.]+)/);

      // Check that result is upgraded to NDArray (has methods)
      expect(result).toBeDefined();
      expect(typeof result.add).toBe('function'); // NDArray has method chaining
      const vals = result.toArray() as number[];
      expect(vals[0]).toBeCloseTo(1.5);
      expect(vals[1]).toBeCloseTo(2.5);
      expect(vals[2]).toBeCloseTo(3.5);
    });

    it('fromregexIndex with dtype parameter', () => {
      const text = 'value=1\nvalue=2\nvalue=3';
      const result = fromregexIndex(text, /value=([0-9]+)/, 'int32');

      // Check dtype is set correctly
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([1, 2, 3]);
    });
  });
});
