/**
 * Unit tests for formatting functions
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  array,
  arange,
  zeros,
  array2string,
  array_repr,
  array_str,
  base_repr,
  binary_repr,
  format_float_positional,
  format_float_scientific,
  get_printoptions,
  set_printoptions,
  printoptions,
} from '../../src';

describe('Formatting Functions', () => {
  // Store original print options to restore after tests
  let originalOptions: ReturnType<typeof get_printoptions>;

  beforeEach(() => {
    originalOptions = get_printoptions();
  });

  afterEach(() => {
    // Restore original options
    set_printoptions(originalOptions);
  });

  describe('set_printoptions() / get_printoptions()', () => {
    it('returns default options', () => {
      const options = get_printoptions();
      expect(options.precision).toBe(8);
      expect(options.threshold).toBe(1000);
      expect(options.edgeitems).toBe(3);
      expect(options.linewidth).toBe(75);
    });

    it('sets precision', () => {
      set_printoptions({ precision: 4 });
      const options = get_printoptions();
      expect(options.precision).toBe(4);
    });

    it('sets threshold', () => {
      set_printoptions({ threshold: 10 });
      const options = get_printoptions();
      expect(options.threshold).toBe(10);
    });

    it('sets multiple options at once', () => {
      set_printoptions({ precision: 3, threshold: 5, edgeitems: 2 });
      const options = get_printoptions();
      expect(options.precision).toBe(3);
      expect(options.threshold).toBe(5);
      expect(options.edgeitems).toBe(2);
    });

    it('preserves unset options', () => {
      const before = get_printoptions();
      set_printoptions({ precision: 4 });
      const after = get_printoptions();
      expect(after.threshold).toBe(before.threshold);
      expect(after.edgeitems).toBe(before.edgeitems);
    });
  });

  describe('printoptions()', () => {
    it('returns callable object with apply method', () => {
      const ctx = printoptions({ precision: 2 });
      expect(typeof ctx.apply).toBe('function');
    });

    it('applies options during callback execution', () => {
      const result = printoptions({ precision: 2 }).apply(() => {
        return get_printoptions().precision;
      });
      expect(result).toBe(2);
    });

    it('restores options after callback', () => {
      const before = get_printoptions().precision;
      printoptions({ precision: 2 }).apply(() => {
        // Inside: precision is 2
      });
      const after = get_printoptions().precision;
      expect(after).toBe(before);
    });

    it('restores options even if callback throws', () => {
      const before = get_printoptions().precision;
      try {
        printoptions({ precision: 2 }).apply(() => {
          throw new Error('test error');
        });
      } catch {
        // Expected
      }
      const after = get_printoptions().precision;
      expect(after).toBe(before);
    });
  });

  describe('array2string()', () => {
    it('formats 1D array', () => {
      const a = array([1, 2, 3]);
      const s = array2string(a);
      expect(s).toContain('1');
      expect(s).toContain('2');
      expect(s).toContain('3');
    });

    it('formats 2D array', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const s = array2string(a);
      expect(s).toContain('1');
      expect(s).toContain('4');
    });

    it('truncates large arrays', () => {
      set_printoptions({ threshold: 6, edgeitems: 1 });
      const a = arange(10);
      const s = array2string(a);
      expect(s).toContain('...');
    });

    it('uses custom precision', () => {
      const a = array([1.23456789]);
      const s = array2string(a, { precision: 3 });
      expect(s).toContain('1.235');
    });

    it('uses custom separator', () => {
      const a = array([1, 2, 3]);
      const s = array2string(a, { separator: '; ' });
      expect(s).toContain(';');
    });
  });

  describe('array_repr()', () => {
    it('includes array() wrapper', () => {
      const a = array([1, 2, 3]);
      const s = array_repr(a);
      expect(s).toContain('array(');
      expect(s).toContain(')');
    });

    it('includes dtype for non-float64', () => {
      const a = array([1, 2, 3], 'int32');
      const s = array_repr(a);
      expect(s).toContain('dtype');
    });

    it('works with 2D arrays', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const s = array_repr(a);
      expect(s).toContain('array(');
    });
  });

  describe('array_str()', () => {
    it('returns string representation', () => {
      const a = array([1, 2, 3]);
      const s = array_str(a);
      expect(s).toContain('1');
      expect(s).toContain('2');
      expect(s).toContain('3');
    });

    it('formats floats with precision', () => {
      const a = array([1.5, 2.5, 3.5]);
      const s = array_str(a);
      expect(s).toContain('1.5');
    });
  });

  describe('base_repr()', () => {
    it('converts to binary (base 2)', () => {
      expect(base_repr(10, 2)).toBe('1010');
    });

    it('converts to octal (base 8)', () => {
      expect(base_repr(64, 8)).toBe('100');
    });

    it('converts to hexadecimal (base 16)', () => {
      expect(base_repr(255, 16)).toBe('FF');
    });

    it('converts to base 3', () => {
      expect(base_repr(9, 3)).toBe('100');
    });

    it('handles zero', () => {
      expect(base_repr(0, 2)).toBe('0');
    });

    it("handles negative numbers with two's complement", () => {
      // -10 in two's complement (8-bit) is 11110110 = 246
      // But base_repr uses sign + magnitude, so it should return -1010
      const result = base_repr(-10, 2);
      expect(result).toBe('-1010');
    });

    it('pads with zeros', () => {
      // NumPy adds `padding` zeros to the left of the result
      expect(base_repr(5, 2, 8)).toBe('00000000101');
    });

    it('throws for invalid base', () => {
      expect(() => base_repr(10, 1)).toThrow();
      expect(() => base_repr(10, 37)).toThrow();
    });
  });

  describe('binary_repr()', () => {
    it('converts positive numbers', () => {
      expect(binary_repr(5)).toBe('101');
      expect(binary_repr(10)).toBe('1010');
    });

    it('handles zero', () => {
      expect(binary_repr(0)).toBe('0');
    });

    it('uses specified width', () => {
      expect(binary_repr(5, 8)).toBe('00000101');
    });

    it("handles negative numbers with two's complement", () => {
      // -1 in 8-bit two's complement is 11111111
      expect(binary_repr(-1, 8)).toBe('11111111');
      // -5 in 8-bit two's complement
      expect(binary_repr(-5, 8)).toBe('11111011');
    });
  });

  describe('format_float_positional()', () => {
    it('formats with default precision', () => {
      const s = format_float_positional(3.14159265);
      expect(s).toContain('3.14');
    });

    it('formats with custom precision', () => {
      const s = format_float_positional(3.14159265, 2);
      expect(s).toBe('3.14');
    });

    it('handles zero', () => {
      const s = format_float_positional(0.0, 2);
      expect(s).toContain('0');
    });

    it('handles large numbers', () => {
      const s = format_float_positional(1234567.89, 2);
      expect(s).toContain('1234567');
    });

    it('handles small numbers', () => {
      const s = format_float_positional(0.00123, 4);
      expect(s).toContain('0.0012');
    });

    it('respects sign option +', () => {
      // format_float_positional(x, precision, unique, fractional, trim, sign)
      const s = format_float_positional(3.14, 2, false, true, 'k', '+');
      expect(s).toMatch(/^\+/);
    });

    it('respects sign option space', () => {
      // format_float_positional(x, precision, unique, fractional, trim, sign)
      const s = format_float_positional(3.14, 2, false, true, 'k', ' ');
      expect(s).toMatch(/^ /);
    });
  });

  describe('format_float_scientific()', () => {
    it('formats in scientific notation', () => {
      const s = format_float_scientific(12345.0, 2);
      expect(s).toMatch(/e\+/);
    });

    it('formats with custom precision', () => {
      const s = format_float_scientific(3.14159265, 4);
      expect(s).toContain('3.1416');
    });

    it('handles small numbers', () => {
      const s = format_float_scientific(0.00012345, 3);
      expect(s).toMatch(/e-/);
    });

    it('handles negative numbers', () => {
      const s = format_float_scientific(-3.14, 2);
      expect(s).toMatch(/^-/);
    });

    it('respects sign option +', () => {
      const s = format_float_scientific(3.14, 2, true, 'k', '+');
      expect(s).toMatch(/^\+/);
    });

    it('respects exp_digits option', () => {
      const s = format_float_scientific(100.0, 2, true, 'k', '-', null, 3);
      expect(s).toMatch(/e\+0*2/);
    });
  });

  describe('NumPy-compatible formatting', () => {
    // Adaptive precision
    it('formats integer floats with trailing dot', () => {
      const a = array([1.0, 2.0, 3.0]);
      const s = array2string(a);
      expect(s).toBe('[1. 2. 3.]');
    });

    it('formats mixed floats with trimmed precision', () => {
      const a = array([1.0, 1.5, 2.0]);
      const s = array2string(a);
      // 1.0 → "1.", 1.5 → "1.5", 2.0 → "2." — max width 3, right-aligned
      expect(s).toContain('1.5');
      expect(s).toContain('1.');
      expect(s).toContain('2.');
      // Verify right-alignment: " 1." should appear (padded to match 1.5 width)
      expect(s).toMatch(/ 1\./);
    });

    it('uses scientific notation for very large values', () => {
      const a = array([1e17, 2e17]);
      const s = array2string(a);
      expect(s).toMatch(/e\+/);
    });

    it('uses scientific notation for very small values', () => {
      const a = array([1e-5, 2e-5]);
      const s = array2string(a);
      expect(s).toMatch(/e-/);
    });

    // Alignment
    it('right-aligns integer values', () => {
      const a = array([1, 100, 10], 'int32');
      const s = array2string(a);
      expect(s).toBe('[  1 100  10]');
    });

    it('right-aligns float values', () => {
      const a = array([1.0, 100.0, 10.0]);
      const s = array2string(a);
      expect(s).toBe('[  1. 100.  10.]');
    });

    // Structure
    it('no commas between rows in 2D', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const s = array2string(a);
      // Should not have commas followed by newlines
      expect(s).not.toMatch(/,\s*\n/);
    });

    it('blank lines between 3D slices', () => {
      const a = zeros([2, 2, 2]);
      const s = array2string(a);
      // 3D array should have blank line between 2D slices
      expect(s).toContain('\n\n');
    });

    it('wraps long lines at linewidth', () => {
      const a = arange(20);
      const s = array2string(a, { max_line_width: 40 });
      const lines = s.split('\n');
      for (const line of lines) {
        expect(line.length).toBeLessThanOrEqual(41); // allow 1 char slack for closing bracket
      }
    });

    // array_repr uses commas (different from array2string)
    it('array_repr uses commas between elements', () => {
      const a = array([1, 2, 3]);
      const s = array_repr(a);
      expect(s).toContain(', ');
    });

    it('2D array has no commas between rows in toString', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const s = array2string(a);
      const lines = s.split('\n');
      expect(lines.length).toBe(2);
      // First line should end with ] not ],
      expect(lines[0]).toMatch(/\]$/);
    });
  });
});
