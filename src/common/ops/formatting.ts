/**
 * Array printing and formatting operations
 *
 * Functions for converting arrays and numbers to string representations.
 * @module ops/formatting
 */

import { ArrayStorage } from '../storage';
import type { DType } from '../dtype';
import { Complex } from '../complex';

/**
 * Print options configuration
 */
export interface PrintOptions {
  /** Total number of array elements that trigger summarization (default: 1000) */
  threshold: number;
  /** Number of elements to print at edges when summarizing (default: 3) */
  edgeitems: number;
  /** Number of digits of precision for floating point output (default: 8) */
  precision: number;
  /** Characters per line for array output (default: 75) */
  linewidth: number;
  /** If true, always use exponential format; if false, never; if null, auto (default: null) */
  floatmode: 'fixed' | 'unique' | 'maxprec' | 'maxprec_equal';
  /** If true, suppress printing of small floating point values (default: true) */
  suppress: boolean;
  /** String inserted between elements (default: ' ') */
  separator: string;
  /** Prefix for array string */
  prefix: string;
  /** Suffix for array string */
  suffix: string;
  /** String representation of nan (default: 'nan') */
  nanstr: string;
  /** String representation of inf (default: 'inf') */
  infstr: string;
  /** If true, use legacy printing mode (default: false) */
  legacy: string | false;
  /** Whether to print the sign of positive values (default: false) */
  sign: ' ' | '+' | '-';
}

// Default print options
const defaultPrintOptions: PrintOptions = {
  threshold: 1000,
  edgeitems: 3,
  precision: 8,
  linewidth: 75,
  floatmode: 'maxprec',
  suppress: true,
  separator: ' ',
  prefix: '',
  suffix: '',
  nanstr: 'nan',
  infstr: 'inf',
  sign: '-',
  legacy: false,
};

// Current print options (mutable)
let currentPrintOptions: PrintOptions = { ...defaultPrintOptions };

/**
 * Set printing options for array output
 *
 * These options determine how arrays are converted to strings.
 *
 * @param options - Options to set
 *
 * @example
 * ```typescript
 * set_printoptions({ precision: 4, threshold: 100 });
 * ```
 */
export function set_printoptions(options: Partial<PrintOptions>): void {
  currentPrintOptions = { ...currentPrintOptions, ...options };
}

/**
 * Get current print options
 *
 * @returns Current print options
 *
 * @example
 * ```typescript
 * const opts = get_printoptions();
 * console.log(opts.precision); // 8
 * ```
 */
export function get_printoptions(): PrintOptions {
  return { ...currentPrintOptions };
}

/**
 * Context manager for temporarily setting print options
 *
 * In JavaScript, this returns an object with enter/exit methods
 * that can be used with try/finally.
 *
 * @param options - Options to set temporarily
 * @returns Object with enter() and exit() methods
 *
 * @example
 * ```typescript
 * const ctx = printoptions({ precision: 2 });
 * ctx.enter();
 * try {
 *   console.log(array2string(arr));
 * } finally {
 *   ctx.exit();
 * }
 * ```
 */
export function printoptions(options: Partial<PrintOptions>): {
  enter: () => void;
  exit: () => void;
  apply: <T>(fn: () => T) => T;
  _savedOptions: PrintOptions | null;
} {
  let savedOptions: PrintOptions | null = null;

  const ctx = {
    _savedOptions: null as PrintOptions | null,
    enter() {
      savedOptions = { ...currentPrintOptions };
      ctx._savedOptions = savedOptions;
      currentPrintOptions = { ...currentPrintOptions, ...options };
    },
    exit() {
      if (savedOptions) {
        currentPrintOptions = savedOptions;
        savedOptions = null;
        ctx._savedOptions = null;
      }
    },
    apply<T>(fn: () => T): T {
      ctx.enter();
      try {
        return fn();
      } finally {
        ctx.exit();
      }
    },
  };

  return ctx;
}

/**
 * Format a floating-point number in positional notation
 *
 * @param x - Value to format
 * @param precision - Number of digits after decimal point (default: from printoptions)
 * @param unique - If true, use shortest representation (default: true)
 * @param fractional - If true, always include fractional part (default: true)
 * @param trim - Trim trailing zeros: 'k' (keep), '.' (trim after decimal), '0' (trim zeros), '-' (trim point and zeros)
 * @param sign - Sign handling: '-' (only negative), '+' (always), ' ' (space for positive)
 * @param pad_left - Pad left with spaces to this width
 * @param pad_right - Pad right with zeros to this width
 * @param min_digits - Minimum digits after decimal point
 * @returns Formatted string
 *
 * @example
 * ```typescript
 * format_float_positional(3.14159265, 4); // '3.1416'
 * format_float_positional(1000.0, 2);     // '1000.00'
 * ```
 */
export function format_float_positional(
  x: number,
  precision: number | null = null,
  unique: boolean = true,
  fractional: boolean = true,
  trim: 'k' | '.' | '0' | '-' = 'k',
  sign: '-' | '+' | ' ' = '-',
  pad_left: number | null = null,
  pad_right: number | null = null,
  min_digits: number | null = null
): string {
  const prec = precision ?? currentPrintOptions.precision;

  // Handle special values
  if (!Number.isFinite(x)) {
    if (Number.isNaN(x)) return currentPrintOptions.nanstr;
    return (x > 0 ? '' : '-') + currentPrintOptions.infstr;
  }

  // Format the number
  let result: string;
  if (unique && precision === null) {
    // Use shortest representation that round-trips
    result = x.toString();
    // If no decimal point and fractional is required, add .0
    if (fractional && !result.includes('.') && !result.includes('e')) {
      result += '.0';
    }
  } else {
    result = x.toFixed(prec);
  }

  // Apply min_digits
  if (min_digits !== null) {
    const dotIndex = result.indexOf('.');
    if (dotIndex !== -1) {
      const currentDigits = result.length - dotIndex - 1;
      if (currentDigits < min_digits) {
        result += '0'.repeat(min_digits - currentDigits);
      }
    } else if (fractional) {
      result += '.' + '0'.repeat(min_digits);
    }
  }

  // Trim trailing zeros/decimal point
  // Use loop-based trimming instead of regex to avoid ReDoS vulnerability
  // '.' = trim trailing zeros, keep decimal point
  // '0' = trim trailing zeros, ensure at least one digit after decimal
  // '-' = trim trailing zeros and trailing decimal point
  if (trim !== 'k' && result.includes('.')) {
    if (trim === '.' || trim === '0' || trim === '-') {
      let i = result.length;
      while (i > 0 && result[i - 1] === '0') i--;
      result = result.slice(0, i);
    }
    if (trim === '0') {
      if (result.endsWith('.')) {
        result += '0';
      }
    }
    if (trim === '-') {
      if (result.endsWith('.')) {
        result = result.slice(0, -1);
      }
    }
  }

  // Handle sign
  if (x >= 0 && !Object.is(x, -0)) {
    if (sign === '+') {
      result = '+' + result;
    } else if (sign === ' ') {
      result = ' ' + result;
    }
  }

  // Pad left
  if (pad_left !== null && result.length < pad_left) {
    result = ' '.repeat(pad_left - result.length) + result;
  }

  // Pad right
  if (pad_right !== null) {
    const dotIndex = result.indexOf('.');
    if (dotIndex !== -1) {
      const currentDigits = result.length - dotIndex - 1;
      if (currentDigits < pad_right) {
        result += '0'.repeat(pad_right - currentDigits);
      }
    }
  }

  return result;
}

/**
 * Format a floating-point number in scientific notation
 *
 * @param x - Value to format
 * @param precision - Number of digits after decimal point (default: from printoptions)
 * @param unique - If true, use shortest representation (default: true)
 * @param trim - Trim trailing zeros: 'k' (keep), '.' (trim after decimal), '0' (trim zeros), '-' (trim point and zeros)
 * @param sign - Sign handling: '-' (only negative), '+' (always), ' ' (space for positive)
 * @param pad_left - Pad left with spaces to this width
 * @param exp_digits - Minimum digits in exponent
 * @param min_digits - Minimum digits after decimal point
 * @returns Formatted string
 *
 * @example
 * ```typescript
 * format_float_scientific(3.14159265, 4); // '3.1416e+00'
 * format_float_scientific(12345.0, 2);    // '1.23e+04'
 * ```
 */
export function format_float_scientific(
  x: number,
  precision: number | null = null,
  _unique: boolean = true,
  trim: 'k' | '.' | '0' | '-' = 'k',
  sign: '-' | '+' | ' ' = '-',
  pad_left: number | null = null,
  exp_digits: number = 2,
  min_digits: number | null = null
): string {
  const prec = precision ?? currentPrintOptions.precision;

  // Handle special values
  if (!Number.isFinite(x)) {
    if (Number.isNaN(x)) return currentPrintOptions.nanstr;
    return (x > 0 ? '' : '-') + currentPrintOptions.infstr;
  }

  // Format in scientific notation
  let result = x.toExponential(prec);

  // Apply min_digits
  if (min_digits !== null) {
    const eIndex = result.indexOf('e');
    const mantissa = result.slice(0, eIndex);
    const exponent = result.slice(eIndex);
    const dotIndex = mantissa.indexOf('.');
    if (dotIndex !== -1) {
      const currentDigits = mantissa.length - dotIndex - 1;
      if (currentDigits < min_digits) {
        result = mantissa + '0'.repeat(min_digits - currentDigits) + exponent;
      }
    }
  }

  // Trim trailing zeros in mantissa
  if (trim !== 'k') {
    const eIndex = result.indexOf('e');
    let mantissa = result.slice(0, eIndex);
    const exponent = result.slice(eIndex);

    if (mantissa.includes('.')) {
      if (trim === '.' || trim === '0' || trim === '-') {
        mantissa = mantissa.replace(/0+$/, '');
      }
      if (trim === '0') {
        if (mantissa.endsWith('.')) {
          mantissa += '0';
        }
      }
      if (trim === '-') {
        mantissa = mantissa.replace(/\.$/, '');
      }
    }
    result = mantissa + exponent;
  }

  // Fix exponent format (ensure minimum digits)
  const eIndex = result.indexOf('e');
  const mantissa = result.slice(0, eIndex);
  let expPart = result.slice(eIndex + 1);
  const expSign = expPart[0] === '-' ? '-' : '+';
  let expNum = expPart.replace(/^[+-]/, '');

  while (expNum.length < exp_digits) {
    expNum = '0' + expNum;
  }
  result = mantissa + 'e' + expSign + expNum;

  // Handle sign
  if (x >= 0 && !Object.is(x, -0)) {
    if (sign === '+') {
      result = '+' + result;
    } else if (sign === ' ') {
      result = ' ' + result;
    }
  }

  // Pad left
  if (pad_left !== null && result.length < pad_left) {
    result = ' '.repeat(pad_left - result.length) + result;
  }

  return result;
}

/**
 * Return a string representation of a number in the given base
 *
 * For base 2, -36. Negative numbers are represented using two's complement
 * if padding is specified, otherwise prefixed with '-'.
 *
 * @param number - Number to convert
 * @param base - Base for representation (2-36, default: 2)
 * @param padding - Minimum number of digits (pads with zeros)
 * @returns String representation in the given base
 *
 * @example
 * ```typescript
 * base_repr(10, 2);      // '1010'
 * base_repr(10, 16);     // 'A'
 * base_repr(-10, 2, 8);  // '11110110' (two's complement)
 * ```
 */
export function base_repr(number: number, base: number = 2, padding: number = 0): string {
  if (base < 2 || base > 36) {
    throw new Error('base must be between 2 and 36');
  }

  number = Math.trunc(number);

  let result: string;
  if (number < 0) {
    result = '-' + Math.abs(number).toString(base).toUpperCase();
  } else {
    result = number.toString(base).toUpperCase();
  }

  // Apply padding: NumPy adds `padding` zeros to the left of the result
  if (padding > 0) {
    const paddingStr = '0'.repeat(padding);
    if (result.startsWith('-')) {
      result = '-' + paddingStr + result.slice(1);
    } else {
      result = paddingStr + result;
    }
  }

  return result;
}

/**
 * Return the binary representation of the input number as a string
 *
 * For negative numbers, two's complement is used if width is specified,
 * otherwise a minus sign is prefixed.
 *
 * @param num - Integer to convert
 * @param width - Minimum width of the result (pads with zeros, uses two's complement for negatives)
 * @returns Binary string representation
 *
 * @example
 * ```typescript
 * binary_repr(10);      // '1010'
 * binary_repr(-10);     // '-1010'
 * binary_repr(-10, 8);  // '11110110' (two's complement)
 * ```
 */
export function binary_repr(num: number, width: number | null = null): string {
  num = Math.trunc(num);

  if (width !== null && num < 0) {
    // Two's complement representation
    const maxVal = Math.pow(2, width);
    num = maxVal + num;
    if (num < 0) {
      throw new Error('width too small for negative number');
    }
    let result = num.toString(2);
    // Pad to width
    if (result.length < width) {
      result = '0'.repeat(width - result.length) + result;
    }
    return result;
  }

  let result: string;
  if (num < 0) {
    result = '-' + Math.abs(num).toString(2);
  } else {
    result = num.toString(2);
  }

  // Apply width padding for positive numbers
  if (width !== null && result.length < width) {
    result = '0'.repeat(width - result.length) + result;
  }

  return result;
}

/**
 * Format a value for array output
 */
function formatValue(
  value: number | bigint | boolean | Complex,
  dtype: DType,
  opts: PrintOptions
): string {
  if (value instanceof Complex) {
    const re = formatValue(value.re, 'float64', opts);
    const im = formatValue(Math.abs(value.im), 'float64', opts);
    const sign = value.im >= 0 ? '+' : '-';
    return `${re}${sign}${im}j`;
  }

  if (typeof value === 'boolean') {
    return value ? ' True' : 'False';
  }

  if (typeof value === 'bigint') {
    return value.toString();
  }

  // Handle special float values
  if (!Number.isFinite(value)) {
    if (Number.isNaN(value)) return opts.nanstr;
    return (value > 0 ? '' : '-') + opts.infstr;
  }

  // Format based on dtype
  if (dtype === 'float32' || dtype === 'float64') {
    // Check if we should suppress small values
    if (opts.suppress && Math.abs(value) < 1e-10 && value !== 0) {
      return '0.';
    }
    return format_float_positional(value, opts.precision, false, true, 'k', opts.sign);
  }

  // Integer types
  return value.toString();
}

/**
 * Collect visible scalar values from an array, mirroring the summarization logic
 * of formatArrayRecursive. Used for pre-scan to determine uniform formatting.
 */
function collectVisibleValues(
  storage: ArrayStorage,
  opts: PrintOptions
): (number | bigint | boolean | Complex)[] {
  const values: (number | bigint | boolean | Complex)[] = [];
  const shape = storage.shape;
  const ndim = shape.length;
  const strides = storage.strides;

  function walk(indices: number[], depth: number): void {
    if (depth === ndim) {
      let flatIdx = 0;
      for (let i = 0; i < ndim; i++) {
        flatIdx += indices[i]! * strides[i]!;
      }
      values.push(storage.iget(flatIdx));
      return;
    }

    const size = shape[depth]!;
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const shouldSummarize = totalSize > opts.threshold && size > 2 * opts.edgeitems;

    if (shouldSummarize) {
      for (let i = 0; i < opts.edgeitems; i++) {
        indices[depth] = i;
        walk(indices, depth + 1);
      }
      for (let i = size - opts.edgeitems; i < size; i++) {
        indices[depth] = i;
        walk(indices, depth + 1);
      }
    } else {
      for (let i = 0; i < size; i++) {
        indices[depth] = i;
        walk(indices, depth + 1);
      }
    }
  }

  if (ndim > 0) {
    walk(new Array(ndim).fill(0), 0);
  } else {
    values.push(storage.iget(0));
  }

  return values;
}

/**
 * Format an integer-valued float as "N." (e.g., 1., 100., -3.)
 */
function formatIntFloat(v: number, sign: ' ' | '+' | '-'): string {
  if (v < 0 || Object.is(v, -0)) {
    return '-' + Math.abs(v).toString() + '.';
  }
  if (sign === '+') {
    return '+' + v.toString() + '.';
  }
  if (sign === ' ') {
    return ' ' + v.toString() + '.';
  }
  return v.toString() + '.';
}

/**
 * Pad a formatted float string using decimal-point alignment.
 * Left of the decimal is right-justified (padStart), right of the decimal
 * is left-justified (padEnd with spaces). This matches NumPy's alignment.
 */
function padDecimalAlign(s: string, maxLeft: number, maxRight: number): string {
  const dotIdx = s.indexOf('.');
  if (dotIdx === -1) {
    // No decimal point (e.g., special values) — right-justify to total width
    return s.padStart(maxLeft + (maxRight > 0 ? 1 + maxRight : 0));
  }
  const left = s.slice(0, dotIdx);
  const right = s.slice(dotIdx + 1); // after the '.'
  const paddedLeft = left.padStart(maxLeft);
  const paddedRight = maxRight > 0 ? right.padEnd(maxRight) : right;
  return paddedLeft + '.' + paddedRight;
}

/**
 * Compute max chars left of decimal and max chars right of decimal
 * across a set of formatted strings.
 */
function computeDecimalWidths(strings: string[]): { maxLeft: number; maxRight: number } {
  let maxLeft = 0;
  let maxRight = 0;
  for (const s of strings) {
    const dotIdx = s.indexOf('.');
    if (dotIdx === -1) {
      maxLeft = Math.max(maxLeft, s.length);
    } else {
      maxLeft = Math.max(maxLeft, dotIdx);
      maxRight = Math.max(maxRight, s.length - dotIdx - 1);
    }
  }
  return { maxLeft, maxRight };
}

/**
 * Build a formatter closure for float values.
 * Pre-scans values to determine format strategy (scientific/integer/fixed)
 * and computes uniform width using decimal-point alignment.
 */
function buildFloatFormatter(values: number[], opts: PrintOptions): (v: number) => string {
  const finiteValues = values.filter((v) => Number.isFinite(v));

  if (finiteValues.length === 0) {
    // Only special values (nan, inf, -inf) or empty
    const specialStrings = values.map((v) => {
      if (Number.isNaN(v)) return opts.nanstr;
      return (v > 0 ? '' : '-') + opts.infstr;
    });
    const maxWidth =
      specialStrings.length > 0 ? Math.max(...specialStrings.map((s) => s.length)) : 1;
    return (v: number) => {
      if (Number.isNaN(v)) return opts.nanstr.padStart(maxWidth);
      if (!Number.isFinite(v)) return ((v > 0 ? '' : '-') + opts.infstr).padStart(maxWidth);
      return v.toString().padStart(maxWidth);
    };
  }

  // Apply suppress: replace very small values with 0 for formatting decisions
  const processedValues = opts.suppress
    ? finiteValues.map((v) => (Math.abs(v) < 1e-10 && v !== 0 ? 0 : v))
    : finiteValues;

  const absValues = processedValues.map(Math.abs).filter((v) => v > 0);
  const maxAbs = absValues.length > 0 ? Math.max(...absValues) : 0;
  const minAbs = absValues.length > 0 ? Math.min(...absValues) : 0;

  // Determine format strategy (matches NumPy 2.x)
  const useScientific =
    maxAbs >= 1e16 || (minAbs > 0 && minAbs < 1e-4) || (minAbs > 0 && maxAbs / minAbs > 1e3);

  if (useScientific) {
    // First pass: format each trimmed to find max mantissa fraction digits
    const trimmedStrings = processedValues.map((v) =>
      format_float_scientific(v, opts.precision, false, '.', opts.sign)
    );
    let maxMantissaFrac = 0;
    for (const s of trimmedStrings) {
      const eIdx = s.indexOf('e');
      const mantissa = eIdx !== -1 ? s.slice(0, eIdx) : s;
      const dotIdx = mantissa.indexOf('.');
      if (dotIdx !== -1) {
        maxMantissaFrac = Math.max(maxMantissaFrac, mantissa.length - dotIdx - 1);
      }
    }
    // Second pass: re-format all with uniform mantissa precision
    // When uniformPrec is 0, use precision=1 with trim='.' to keep the decimal point
    // (toExponential(0) omits the dot entirely, but NumPy always shows it)
    const uniformPrec = Math.max(maxMantissaFrac, 0);
    const formatPrec = Math.max(uniformPrec, 1);
    const formatTrim: 'k' | '.' = uniformPrec === 0 ? '.' : 'k';
    const strings = processedValues.map((v) =>
      format_float_scientific(v, formatPrec, false, formatTrim, opts.sign)
    );
    for (const v of values) {
      if (!Number.isFinite(v)) {
        strings.push(Number.isNaN(v) ? opts.nanstr : (v > 0 ? '' : '-') + opts.infstr);
      }
    }
    const maxWidth = Math.max(...strings.map((s) => s.length));

    return (v: number) => {
      if (opts.suppress && Math.abs(v) < 1e-10 && v !== 0 && Number.isFinite(v)) {
        v = 0;
      }
      if (Number.isNaN(v)) return opts.nanstr.padStart(maxWidth);
      if (!Number.isFinite(v)) return ((v > 0 ? '' : '-') + opts.infstr).padStart(maxWidth);
      return format_float_scientific(v, formatPrec, false, formatTrim, opts.sign).padStart(
        maxWidth
      );
    };
  }

  // Check if all finite values are integers
  const allIntegers = processedValues.every((v) => Number.isInteger(v));

  if (allIntegers) {
    // Integer float format — all have "N." shape, decimal-align is just padStart
    const strings = processedValues.map((v) => formatIntFloat(v, opts.sign));
    for (const v of values) {
      if (!Number.isFinite(v)) {
        strings.push(Number.isNaN(v) ? opts.nanstr : (v > 0 ? '' : '-') + opts.infstr);
      }
    }
    const { maxLeft, maxRight } = computeDecimalWidths(strings);

    return (v: number) => {
      if (opts.suppress && Math.abs(v) < 1e-10 && v !== 0 && Number.isFinite(v)) {
        v = 0;
      }
      if (Number.isNaN(v)) return padDecimalAlign(opts.nanstr, maxLeft, maxRight);
      if (!Number.isFinite(v))
        return padDecimalAlign((v > 0 ? '' : '-') + opts.infstr, maxLeft, maxRight);
      return padDecimalAlign(formatIntFloat(v, opts.sign), maxLeft, maxRight);
    };
  }

  // Fixed notation with trimmed trailing zeros — decimal-point aligned
  const strings = processedValues.map((v) =>
    format_float_positional(v, opts.precision, false, true, '.', opts.sign)
  );
  for (const v of values) {
    if (!Number.isFinite(v)) {
      strings.push(Number.isNaN(v) ? opts.nanstr : (v > 0 ? '' : '-') + opts.infstr);
    }
  }
  const { maxLeft, maxRight } = computeDecimalWidths(strings);

  return (v: number) => {
    if (opts.suppress && Math.abs(v) < 1e-10 && v !== 0 && Number.isFinite(v)) {
      v = 0;
    }
    if (Number.isNaN(v)) return padDecimalAlign(opts.nanstr, maxLeft, maxRight);
    if (!Number.isFinite(v))
      return padDecimalAlign((v > 0 ? '' : '-') + opts.infstr, maxLeft, maxRight);
    return padDecimalAlign(
      format_float_positional(v, opts.precision, false, true, '.', opts.sign),
      maxLeft,
      maxRight
    );
  };
}

/**
 * Build a formatter for array values based on dtype and visible values.
 * Implements NumPy's two-pass approach: pre-scan to determine format, then apply uniformly.
 */
function buildFormatter(
  storage: ArrayStorage,
  opts: PrintOptions
): (value: number | bigint | boolean | Complex) => string {
  const dtype = storage.dtype as DType;
  const values = collectVisibleValues(storage, opts);

  if (values.length === 0) {
    return (v) => formatValue(v as number | bigint | boolean | Complex, dtype, opts);
  }

  if (dtype === 'bool') {
    return (v) => ((v as boolean) ? ' True' : 'False');
  }

  // BigInt types (int64, uint64 use BigInt64Array/BigUint64Array)
  if (dtype === 'int64' || dtype === 'uint64') {
    const strings = values.map((v) => (v as bigint).toString());
    const maxWidth = Math.max(...strings.map((s) => s.length));
    return (v) => (v as bigint).toString().padStart(maxWidth);
  }

  // Integer types (int8, int16, int32, uint8, uint16, uint32)
  if (dtype.startsWith('int') || dtype.startsWith('uint')) {
    const strings = values.map((v) => (v as number).toString());
    const maxWidth = Math.max(...strings.map((s) => s.length));
    return (v) => (v as number).toString().padStart(maxWidth);
  }

  // Complex types
  if (dtype === 'complex64' || dtype === 'complex128') {
    const reals = values.map((v) => (v as Complex).re);
    const imags = values.map((v) => Math.abs((v as Complex).im));
    const realFmt = buildFloatFormatter(reals, opts);
    const imagFmt = buildFloatFormatter(imags, opts);
    return (v) => {
      const c = v as Complex;
      const re = realFmt(c.re);
      const im = imagFmt(Math.abs(c.im));
      const sign = c.im >= 0 ? '+' : '-';
      return `${re}${sign}${im}j`;
    };
  }

  // Float types (float32, float64)
  const floatFmt = buildFloatFormatter(values as number[], opts);
  return (v) => floatFmt(v as number);
}

/**
 * Recursively format array elements into a nested string
 */
function formatArrayRecursive(
  storage: ArrayStorage,
  indices: number[],
  depth: number,
  opts: PrintOptions,
  formatter: (value: number | bigint | boolean | Complex) => string,
  column: number
): string {
  const shape = storage.shape;
  const ndim = shape.length;

  if (depth === ndim) {
    // Get the value at these indices
    let flatIdx = 0;
    const strides = storage.strides;
    for (let i = 0; i < ndim; i++) {
      flatIdx += indices[i]! * strides[i]!;
    }
    const value = storage.iget(flatIdx);
    return formatter(value as number | bigint | boolean | Complex);
  }

  const size = shape[depth]!;
  const threshold = opts.threshold;
  const edgeitems = opts.edgeitems;

  // Determine if we need to summarize this dimension
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const shouldSummarize = totalSize > threshold && size > 2 * edgeitems;

  const parts: string[] = [];
  const newIndices = [...indices];

  if (shouldSummarize) {
    // Show first edgeitems
    for (let i = 0; i < edgeitems; i++) {
      newIndices[depth] = i;
      parts.push(formatArrayRecursive(storage, newIndices, depth + 1, opts, formatter, column + 1));
    }
    parts.push('...');
    // Show last edgeitems
    for (let i = size - edgeitems; i < size; i++) {
      newIndices[depth] = i;
      parts.push(formatArrayRecursive(storage, newIndices, depth + 1, opts, formatter, column + 1));
    }
  } else {
    for (let i = 0; i < size; i++) {
      newIndices[depth] = i;
      parts.push(formatArrayRecursive(storage, newIndices, depth + 1, opts, formatter, column + 1));
    }
  }

  // Join with appropriate separators
  if (depth === ndim - 1) {
    // Innermost dimension: try single line first
    const singleLine = '[' + parts.join(opts.separator) + ']';
    if (column + singleLine.length < opts.linewidth) {
      return singleLine;
    }
    // Wrap: greedily pack elements onto lines
    const continuationIndent = ' '.repeat(column + 1);
    const maxContentWidth = opts.linewidth - column - 1;
    const lines: string[] = [];
    let currentLine = '';
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]!;
      if (currentLine === '') {
        currentLine = part;
      } else {
        const candidate = currentLine + opts.separator + part;
        if (candidate.length < maxContentWidth) {
          currentLine = candidate;
        } else {
          lines.push(currentLine);
          currentLine = part;
        }
      }
    }
    if (currentLine) {
      lines.push(currentLine);
    }
    return '[' + lines.join('\n' + continuationIndent) + ']';
  } else {
    // Outer dimensions: use newlines with blank lines between blocks
    const indent = ' '.repeat(column + 1);
    const extraNewlines = ndim - depth - 2;
    const innerSep = '\n' + '\n'.repeat(Math.max(0, extraNewlines)) + indent;
    return '[' + parts.join(innerSep) + ']';
  }
}

/**
 * Convert an array to a string representation
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width (default: from printoptions)
 * @param precision - Number of digits of precision (default: from printoptions)
 * @param suppress_small - Suppress small floating point values (default: from printoptions)
 * @param separator - Separator between elements (default: from printoptions)
 * @param prefix - Prefix string (default: '')
 * @param suffix - Suffix string (default: '')
 * @param threshold - Threshold for summarization (default: from printoptions)
 * @param edgeitems - Number of edge items when summarizing (default: from printoptions)
 * @returns String representation of the array
 *
 * @example
 * ```typescript
 * const arr = array([[1, 2, 3], [4, 5, 6]]);
 * console.log(array2string(arr.storage));
 * // [[1 2 3]
 * //  [4 5 6]]
 * ```
 */
export function array2string(
  a: ArrayStorage,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null,
  separator: string = ' ',
  prefix: string = '',
  suffix: string = '',
  threshold: number | null = null,
  edgeitems: number | null = null,
  floatmode: 'fixed' | 'unique' | 'maxprec' | 'maxprec_equal' | null = null,
  sign: ' ' | '+' | '-' | null = null
): string {
  const opts: PrintOptions = {
    ...currentPrintOptions,
    linewidth: max_line_width ?? currentPrintOptions.linewidth,
    precision: precision ?? currentPrintOptions.precision,
    suppress: suppress_small ?? currentPrintOptions.suppress,
    separator,
    prefix,
    suffix,
    threshold: threshold ?? currentPrintOptions.threshold,
    edgeitems: edgeitems ?? currentPrintOptions.edgeitems,
    floatmode: floatmode ?? currentPrintOptions.floatmode,
    sign: sign ?? currentPrintOptions.sign,
  };

  // Handle 0-d array
  if (a.ndim === 0) {
    const value = a.iget(0);
    return formatValue(value as number | bigint | boolean | Complex, a.dtype as DType, opts);
  }

  // Build formatter (pre-scan for uniform formatting)
  const formatter = buildFormatter(a, opts);

  // Format recursively
  const startColumn = opts.prefix.length;
  const result = formatArrayRecursive(
    a,
    new Array(a.ndim).fill(0),
    0,
    opts,
    formatter,
    startColumn
  );

  return opts.prefix + result + opts.suffix;
}

/**
 * Return the string representation of an array
 *
 * Similar to array2string but includes the 'array(' prefix.
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width
 * @param precision - Number of digits of precision
 * @param suppress_small - Suppress small floating point values
 * @returns String representation
 *
 * @example
 * ```typescript
 * const arr = array([1, 2, 3]);
 * console.log(array_repr(arr.storage));
 * // array([1, 2, 3])
 * ```
 */
export function array_repr(
  a: ArrayStorage,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null
): string {
  const dataStr = array2string(a, max_line_width, precision, suppress_small, ', ');

  // Build the repr string
  let result = 'array(' + dataStr;

  // Add dtype if not float64
  if (a.dtype !== 'float64') {
    result += `, dtype='${a.dtype}'`;
  }

  result += ')';

  return result;
}

/**
 * Return a string representation of the data in an array
 *
 * Similar to array2string but returns just the data without array() wrapper.
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width
 * @param precision - Number of digits of precision
 * @param suppress_small - Suppress small floating point values
 * @returns String representation of the data
 *
 * @example
 * ```typescript
 * const arr = array([1, 2, 3]);
 * console.log(array_str(arr.storage));
 * // [1 2 3]
 * ```
 */
export function array_str(
  a: ArrayStorage,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null
): string {
  return array2string(a, max_line_width, precision, suppress_small);
}
