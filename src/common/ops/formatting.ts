/**
 * Array printing and formatting operations
 *
 * Functions for converting arrays and numbers to string representations.
 * @module ops/formatting
 */

import { Complex } from '../complex';
import { type DType, isComplexDType } from '../dtype';
import type { ArrayStorage } from '../storage';

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
  /** If true, print very small floating point values as zero (default: false) */
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
  suppress: false,
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

type FloatKind = 'float16' | 'float32' | 'float64';
type Trim = 'k' | '.' | '0' | '-';
type SignChar = '-' | '+' | ' ';

/** Digits d0 d1 d2... with value d0.d1d2... x 10^e; zero is ds '0', e 0. */
interface Decimal {
  ds: string;
  e: number;
}

const f64view = new DataView(new ArrayBuffer(8));

function normalizeSign(sign: boolean | SignChar | undefined): SignChar {
  if (sign === true) return '+';
  if (sign === false || sign === undefined) return '-';
  return sign;
}

function stripZeros(s: string): string {
  let end = s.length;
  while (end > 0 && s.charCodeAt(end - 1) === 48) end--;
  return s.slice(0, end);
}

/** Exact decimal expansion of a positive finite double. */
function exactDecimal(x: number): Decimal {
  f64view.setFloat64(0, x);
  const hi = f64view.getUint32(0);
  const lo = f64view.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e2: number;
  if (biased === 0) {
    e2 = -1074;
  } else {
    m |= 1n << 52n;
    e2 = biased - 1075;
  }
  let s: string;
  let e10: number;
  if (e2 >= 0) {
    s = (m << BigInt(e2)).toString();
    e10 = s.length - 1;
  } else {
    s = (m * 5n ** BigInt(-e2)).toString();
    e10 = s.length - 1 + e2;
  }
  return { ds: stripZeros(s), e: e10 };
}

function roundHalfEven(t: number): number {
  const f = Math.floor(t);
  const d = t - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Round a finite value to the nearest IEEE half-precision value. */
function roundToFloat16(v: number): number {
  if (!Number.isFinite(v) || v === 0) return v;
  const a = Math.abs(v);
  if (a >= 65520) return v < 0 ? -Infinity : Infinity;
  f64view.setFloat64(0, a);
  const e = Math.max(((f64view.getUint32(0) >>> 20) & 0x7ff) - 1023, -14);
  const q = 2 ** (e - 10);
  const r = roundHalfEven(a / q) * q;
  return v < 0 ? -r : r;
}

/** Round a double to the precision of the given float type. */
function roundToKind(v: number, kind: FloatKind): number {
  if (kind === 'float32') return Math.fround(v);
  if (kind === 'float16') return roundToFloat16(v);
  return v;
}

/**
 * Shortest digit string that reads back as the same value in the given float type. Narrow
 * types round the exact expansion themselves because Number toExponential breaks ties upward
 * where the shortest-digits algorithm breaks them to even.
 */
function uniqueDecimal(a: number, kind: FloatKind): Decimal {
  if (kind === 'float64') {
    const s = a.toExponential();
    const eIdx = s.indexOf('e');
    const ds = stripZeros(s.slice(0, eIdx).replace('.', ''));
    return { ds, e: Number(s.slice(eIdx + 1)) };
  }
  const exact = exactDecimal(a);
  for (let n = 1; n < exact.ds.length; n++) {
    const c = roundDecimal(exact, n);
    const v = Number(`0.${c.ds}e${c.e + 1}`);
    if (roundToKind(v, kind) === a) return { ds: stripZeros(c.ds), e: c.e };
  }
  return exact;
}

/**
 * Round an exact decimal to nd significant digits, ties to even. Digits that were cut
 * keep their trailing zeros, except that a carry shortens the result to the changed digit.
 */
function roundDecimal(d: Decimal, nd: number): Decimal {
  if (d.ds.length <= nd) return d;
  if (nd < 0) return { ds: '0', e: 0 };
  const next = d.ds.charCodeAt(nd) - 48;
  const restNonZero = d.ds.length > nd + 1;
  let up: boolean;
  if (next !== 5) up = next > 5;
  else if (restNonZero) up = true;
  else up = nd > 0 && (d.ds.charCodeAt(nd - 1) - 48) % 2 === 1;
  const kept = d.ds.slice(0, nd);
  if (!up) return nd === 0 ? { ds: '0', e: 0 } : { ds: kept, e: d.e };
  if (nd === 0) return { ds: '1', e: d.e + 1 };
  const digits = kept.split('');
  let i = digits.length - 1;
  while (i >= 0 && digits[i] === '9') i--;
  if (i < 0) return { ds: '1', e: d.e + 1 };
  digits[i] = String(Number(digits[i]) + 1);
  return { ds: digits.slice(0, i + 1).join(''), e: d.e };
}

interface DragonSpec {
  precision: number | null;
  unique: boolean;
  /** Positional only: precision counts digits after the point rather than significant digits. */
  fractional: boolean;
  minDigits: number | null;
  scientific: boolean;
  kind: FloatKind;
}

/** Digits of a positive finite value after applying the precision cutoff and minimum digits. */
function generateDigits(a: number, spec: DragonSpec): Decimal {
  const { precision, minDigits, scientific } = spec;
  const cutoff = (e: number): number | null => {
    if (precision === null) return null;
    return scientific || !spec.fractional ? precision + (scientific ? 1 : 0) : e + 1 + precision;
  };
  const minimum = (e: number, digits: number): number =>
    scientific ? digits + 1 : spec.fractional ? e + 1 + digits : digits;
  if (!spec.unique) {
    const ex = exactDecimal(a);
    const nd = cutoff(ex.e) as number;
    return nd < 0 ? { ds: '0', e: 0 } : roundDecimal(ex, nd);
  }
  // The shortest digits of a narrow float can sit at a different decimal exponent than the
  // exact value, so every rounding position is recomputed against the exact expansion.
  let d = uniqueDecimal(a, spec.kind);
  const nd = cutoff(d.e);
  let exact: Decimal | null = null;
  if (nd !== null && d.ds.length > nd) {
    exact = exactDecimal(a);
    d = roundDecimal(exact, cutoff(exact.e) as number);
  }
  if (minDigits !== null && minDigits > 0) {
    exact ??= exactDecimal(a);
    let min = minimum(exact.e, minDigits);
    const cap = cutoff(exact.e);
    if (cap !== null && min > cap) min = cap;
    if (d.ds.length < min) d = roundDecimal(exact, min);
  }
  return d;
}

/**
 * Apply a trim mode to the digits after the point. In scientific notation the point
 * survives mode '-' whenever digits past the first were generated before trimming.
 */
function trimFraction(
  frac: string,
  trim: Trim,
  keepPoint: boolean,
): { frac: string; point: boolean } {
  if (trim === 'k') return { frac, point: true };
  const f = stripZeros(frac);
  if (trim === '0') return { frac: f === '' ? '0' : f, point: true };
  if (trim === '-') return { frac: f, point: f !== '' || keepPoint };
  return { frac: f, point: true };
}

function checkDigitArgs(
  precision: number | null,
  unique: boolean,
  minDigits: number | null,
  fractional = true,
): void {
  if (!fractional && precision === 0) {
    throw new Error('precision must be greater than 0 when fractional is false');
  }
  if (!unique && precision === null) {
    throw new Error('in non-unique mode `precision` must be supplied');
  }
  if (precision !== null && precision > 0 && minDigits !== null && minDigits > precision) {
    throw new Error('min_digits must be less than or equal to precision');
  }
}

function nonFinite(x: number, sign: SignChar): string {
  if (Number.isNaN(x)) return 'nan';
  if (x < 0) return '-inf';
  return sign === '+' ? '+inf' : 'inf';
}

function signPrefix(x: number, sign: SignChar): string {
  if (x < 0 || Object.is(x, -0)) return '-';
  return sign === '-' ? '' : sign;
}

function dragonPositional(
  x: number,
  kind: FloatKind,
  precision: number | null,
  unique: boolean,
  fractional: boolean,
  trim: Trim,
  sign: SignChar,
  padLeft: number | null,
  padRight: number | null,
  minDigits: number | null,
): string {
  if (!Number.isFinite(x)) return nonFinite(x, sign);
  checkDigitArgs(precision, unique, minDigits, fractional);
  const a = Math.abs(x);
  let intPart = '0';
  let frac = '';
  if (a !== 0) {
    const d = generateDigits(a, {
      precision,
      unique,
      fractional,
      minDigits,
      scientific: false,
      kind,
    });
    if (d.ds === '0') {
      if (precision !== null && fractional) frac = '0'.repeat(precision);
    } else if (d.e >= 0) {
      intPart = d.ds.slice(0, d.e + 1).padEnd(d.e + 1, '0');
      frac = d.ds.slice(d.e + 1);
    } else {
      frac = '0'.repeat(-d.e - 1) + d.ds;
    }
  }
  if (!unique) {
    // Significant-digit mode pads until the integer digits plus fraction digits reach precision.
    const want = fractional ? (precision as number) : (precision as number) - intPart.length;
    frac = frac.padEnd(want, '0');
  } else if (minDigits !== null && minDigits > 0) {
    if (fractional) {
      frac = frac.padEnd(minDigits, '0');
    } else {
      // Counting digits from the first nonzero integer digit; the zero before the point of a
      // fractional value does not count, but leading zeros after the point do.
      const have = (intPart === '0' && a !== 0 ? 0 : intPart.length) + frac.length;
      frac = frac.padEnd(frac.length + Math.max(0, minDigits - have), '0');
    }
  }
  const t = trimFraction(frac, trim, false);
  let head = signPrefix(x, sign) + intPart;
  if (padLeft !== null && head.length < padLeft) head = head.padStart(padLeft);
  if (!t.point) return padRight === null ? head : `${head} ${' '.repeat(padRight)}`;
  const tail = padRight !== null && t.frac.length < padRight ? t.frac.padEnd(padRight) : t.frac;
  return `${head}.${tail}`;
}

function dragonScientific(
  x: number,
  kind: FloatKind,
  precision: number | null,
  unique: boolean,
  trim: Trim,
  sign: SignChar,
  padLeft: number | null,
  expDigits: number | null,
  minDigits: number | null,
): string {
  if (!Number.isFinite(x)) return nonFinite(x, sign);
  checkDigitArgs(precision, unique, minDigits);
  const a = Math.abs(x);
  let ds: string;
  let e: number;
  if (a === 0) {
    ds = '0';
    e = 0;
  } else {
    const d = generateDigits(a, {
      precision,
      unique,
      fractional: true,
      minDigits,
      scientific: true,
      kind,
    });
    ds = d.ds;
    e = d.e;
  }
  let frac = ds.slice(1);
  if (!unique) frac = frac.padEnd(precision as number, '0');
  else if (minDigits !== null && minDigits > 0) frac = frac.padEnd(minDigits, '0');
  const extended =
    unique &&
    a !== 0 &&
    minDigits !== null &&
    minDigits > 0 &&
    (precision === null || precision > 0);
  const t = trimFraction(frac, trim, ds.length > 1 || extended);
  let head = signPrefix(x, sign) + ds[0];
  if (padLeft !== null && head.length < padLeft) head = head.padStart(padLeft);
  const expStr =
    e === 0 && expDigits === 0 ? '' : String(Math.abs(e)).padStart(expDigits ?? 2, '0');
  return `${head}${t.point ? `.${t.frac}` : ''}e${e < 0 ? '-' : '+'}${expStr}`;
}

/**
 * Format a floating-point number in positional notation. With unique true the shortest
 * digits that identify the value are printed and precision only caps them, so precision
 * never pads zeros; pass unique false to print exactly precision digits.
 *
 * @param x - Value to format
 * @param precision - Maximum digits after the decimal point (all unique digits when null)
 * @param unique - If true, use the shortest representation that round-trips (default: true)
 * @param fractional - If true, precision counts digits after the point, otherwise significant digits
 * @param trim - 'k' keep trailing zeros, '.' trim zeros and keep the point, '0' trim zeros but keep one digit after the point, '-' trim zeros and the point
 * @param sign - true or '+' always shows a sign, ' ' leaves a space for positive values, false or '-' only shows negative signs
 * @param pad_left - Pad the part left of the decimal point with spaces to this width
 * @param pad_right - Pad the part right of the decimal point with spaces to this width
 * @param min_digits - Minimum digits after the decimal point when unique is true
 * @returns Formatted string
 *
 * @example
 * ```typescript
 * format_float_positional(3.14159265, 4);        // '3.1416'
 * format_float_positional(3.14, 4);              // '3.14'
 * format_float_positional(3.14, 4, false);       // '3.1400'
 * ```
 */
export function format_float_positional(
  x: number,
  precision: number | null = null,
  unique: boolean = true,
  fractional: boolean = true,
  trim: Trim = 'k',
  sign: boolean | SignChar = '-',
  pad_left: number | null = null,
  pad_right: number | null = null,
  min_digits: number | null = null,
): string {
  return dragonPositional(
    x,
    'float64',
    precision,
    unique,
    fractional,
    trim,
    normalizeSign(sign),
    pad_left,
    pad_right,
    min_digits,
  );
}

/**
 * Format a floating-point number in scientific notation. With unique true the shortest
 * digits that identify the value are printed and precision only caps them; pass unique
 * false to print exactly precision digits after the point.
 *
 * @param x - Value to format
 * @param precision - Maximum digits after the decimal point of the mantissa (all unique digits when null)
 * @param unique - If true, use the shortest representation that round-trips (default: true)
 * @param trim - 'k' keep trailing zeros, '.' trim zeros and keep the point, '0' trim zeros but keep one digit after the point, '-' trim zeros and the point
 * @param sign - true or '+' always shows a sign, ' ' leaves a space for positive values, false or '-' only shows negative signs
 * @param pad_left - Pad the part left of the decimal point with spaces to this width
 * @param exp_digits - Minimum digits in the exponent (default: 2)
 * @param min_digits - Minimum digits after the decimal point when unique is true
 * @returns Formatted string
 *
 * @example
 * ```typescript
 * format_float_scientific(3.14159265, 4);        // '3.1416e+00'
 * format_float_scientific(1500.0);               // '1.5e+03'
 * format_float_scientific(1500.0, 3, false);     // '1.500e+03'
 * ```
 */
export function format_float_scientific(
  x: number,
  precision: number | null = null,
  unique: boolean = true,
  trim: Trim = 'k',
  sign: boolean | SignChar = '-',
  pad_left: number | null = null,
  exp_digits: number | null = null,
  min_digits: number | null = null,
): string {
  return dragonScientific(
    x,
    'float64',
    precision,
    unique,
    trim,
    normalizeSign(sign),
    pad_left,
    exp_digits,
    min_digits,
  );
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
    const maxVal = 2 ** width;
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

type ElementValue = number | bigint | Complex;
type ElementFormatter = (value: ElementValue) => string;

function floatKindOf(dtype: DType): FloatKind {
  if (dtype === 'float16') return 'float16';
  if (dtype === 'float32' || dtype === 'complex64') return 'float32';
  return 'float64';
}

/**
 * Read the element at a multi-index. Strides are applied here, once; iget takes
 * a logical row-major index and applies strides itself, so it cannot be fed a
 * stride-based offset.
 */
function readAt(storage: ArrayStorage, indices: readonly number[]): ElementValue {
  const strides = storage.strides;
  let pos = storage.offset;
  for (let i = 0; i < indices.length; i++) {
    pos += indices[i]! * strides[i]!;
  }
  if (isComplexDType(storage.dtype)) {
    return new Complex(storage.data[pos * 2] as number, storage.data[pos * 2 + 1] as number);
  }
  const v = storage.data[pos]!;
  // Without native Float16Array the buffer holds float32 values; print what a half would hold.
  return storage.dtype === 'float16' ? roundToFloat16(v as number) : v;
}

/** Positions shown along an axis of length n: all of them, or the edge items when summarizing. */
function shownPositions(n: number, summarize: boolean, edgeitems: number): number[] {
  const out: number[] = [];
  if (summarize && n > 2 * edgeitems) {
    for (let i = 0; i < edgeitems; i++) out.push(i);
    for (let i = n - edgeitems; i < n; i++) out.push(i);
  } else {
    for (let i = 0; i < n; i++) out.push(i);
  }
  return out;
}

/** The values that will actually be printed, used to size the columns uniformly. */
function collectVisibleValues(
  storage: ArrayStorage,
  summarize: boolean,
  edgeitems: number,
): ElementValue[] {
  const values: ElementValue[] = [];
  const shape = storage.shape;
  const ndim = shape.length;
  const indices = new Array<number>(ndim).fill(0);

  function walk(depth: number): void {
    if (depth === ndim) {
      values.push(readAt(storage, indices));
      return;
    }
    for (const i of shownPositions(shape[depth]!, summarize, edgeitems)) {
      indices[depth] = i;
      walk(depth + 1);
    }
  }

  walk(0);
  return values;
}

// Largest magnitude printed positionally; wider float types reach 1e8, narrower ones less.
const EXP_CUTOFF_MAX: Record<FloatKind, number> = { float16: 1e3, float32: 1e6, float64: 1e8 };

/**
 * Float element formatter that sizes every column from the visible values: scientific
 * notation when the magnitudes span too many orders, a shared number of digits after
 * the point in fixed modes, and alignment on the decimal point.
 */
function buildFloatFormatter(
  data: number[],
  kind: FloatKind,
  opts: PrintOptions,
  sign: SignChar,
): (x: number) => string {
  const floatmode = opts.floatmode;
  const plusSign: SignChar = sign === '+' ? '+' : '-';
  let precision: number | null = floatmode === 'unique' ? null : opts.precision;
  const finite = data.filter((v) => Number.isFinite(v));

  let expFormat = false;
  let padLeft: number;
  let padRight: number;
  let trim: Trim;
  let expSize = -1;
  let unique: boolean;
  let minDigits: number | null;

  let maxVal = 0;
  let minVal = Infinity;
  for (const v of finite) {
    const a = Math.abs(v);
    if (a === 0) continue;
    if (a > maxVal) maxVal = a;
    if (a < minVal) minVal = a;
  }
  if (maxVal > 0) {
    if (
      maxVal >= EXP_CUTOFF_MAX[kind] ||
      (!opts.suppress &&
        (minVal < roundToKind(0.0001, kind) || roundToKind(maxVal / minVal, kind) > 1000))
    ) {
      expFormat = true;
    }
  }

  if (finite.length === 0) {
    padLeft = 0;
    padRight = 0;
    trim = '.';
    unique = true;
    minDigits = null;
  } else if (expFormat) {
    unique = floatmode !== 'fixed';
    const first: Trim = unique ? '.' : 'k';
    const strs = finite.map((v) =>
      dragonScientific(v, kind, precision, unique, first, plusSign, null, null, null),
    );
    let maxFrac = 0;
    let maxInt = 0;
    let maxExp = 0;
    for (const s of strs) {
      const eIdx = s.indexOf('e');
      const mant = s.slice(0, eIdx);
      const dot = mant.indexOf('.');
      maxInt = Math.max(maxInt, dot);
      maxFrac = Math.max(maxFrac, mant.length - dot - 1);
      maxExp = Math.max(maxExp, s.length - eIdx - 2);
    }
    expSize = maxExp;
    trim = 'k';
    precision = maxFrac;
    minDigits = maxFrac;
    padLeft = maxInt;
    padRight = expSize + 2 + precision;
  } else {
    unique = floatmode !== 'fixed';
    const first: Trim = unique ? '.' : 'k';
    const strs = finite.map((v) =>
      dragonPositional(v, kind, precision, unique, true, first, plusSign, null, null, null),
    );
    let maxInt = 0;
    let maxFrac = 0;
    for (const s of strs) {
      const dot = s.indexOf('.');
      maxInt = Math.max(maxInt, dot);
      maxFrac = Math.max(maxFrac, s.length - dot - 1);
    }
    padLeft = maxInt;
    padRight = maxFrac;
    if (floatmode === 'fixed' || floatmode === 'maxprec_equal') {
      precision = maxFrac;
      minDigits = maxFrac;
      trim = 'k';
    } else {
      trim = '.';
      minDigits = 0;
    }
  }

  if (sign === ' ' && !finite.some((v) => v < 0 || Object.is(v, -0))) padLeft += 1;

  if (data.length !== finite.length) {
    const neginf = sign !== '-' || data.some((v) => v === -Infinity);
    const offset = padRight + 1;
    padLeft = Math.max(
      padLeft,
      opts.nanstr.length - offset,
      opts.infstr.length + (neginf ? 1 : 0) - offset,
    );
  }

  return (x: number) => {
    if (!Number.isFinite(x)) {
      let ret: string;
      if (Number.isNaN(x)) ret = (sign === '+' ? '+' : '') + opts.nanstr;
      else ret = (x < 0 ? '-' : sign === '+' ? '+' : '') + opts.infstr;
      return ' '.repeat(Math.max(0, padLeft + padRight + 1 - ret.length)) + ret;
    }
    if (expFormat) {
      return dragonScientific(
        x,
        kind,
        precision,
        unique,
        trim,
        plusSign,
        padLeft,
        expSize,
        minDigits,
      );
    }
    return dragonPositional(
      x,
      kind,
      precision,
      unique,
      true,
      trim,
      plusSign,
      padLeft,
      padRight,
      minDigits,
    );
  };
}

/** Builds the element formatter for a dtype from the values that will be shown. */
function buildFormatter(
  dtype: DType,
  values: ElementValue[],
  opts: PrintOptions,
  scalar: boolean,
): ElementFormatter {
  if (dtype === 'bool') {
    const t = scalar ? 'True' : ' True';
    return (v) => (v ? t : 'False');
  }

  if (isComplexDType(dtype)) {
    const kind = floatKindOf(dtype);
    const cs = values as Complex[];
    const realFmt = buildFloatFormatter(
      cs.map((c) => c.re),
      kind,
      opts,
      opts.sign,
    );
    const imagFmt = buildFloatFormatter(
      cs.map((c) => c.im),
      kind,
      opts,
      '+',
    );
    return (v) => {
      const c = v as Complex;
      const r = realFmt(c.re);
      let i = imagFmt(c.im);
      const end = i.trimEnd().length;
      i = `${i.slice(0, end)}j${i.slice(end)}`;
      return r + i;
    };
  }

  if (dtype === 'float16' || dtype === 'float32' || dtype === 'float64') {
    const fmt = buildFloatFormatter(values as number[], floatKindOf(dtype), opts, opts.sign);
    return (v) => fmt(v as number);
  }

  // Integers, including the bigint-backed int64 and uint64. Callers pass at least one value.
  let max = values[0] as number | bigint;
  let min = max;
  for (const v of values as (number | bigint)[]) {
    if (v > max) max = v;
    if (v < min) min = v;
  }
  let sign = opts.sign;
  if (sign === ' ' && min < 0) sign = '-';
  let maxLen = String(max).length;
  if (max >= 0 && (sign === '+' || sign === ' ')) maxLen += 1;
  const width = Math.max(maxLen, String(min).length);
  const plus = sign === '+' || sign === ' ';
  const lead = sign === '+' ? '+' : ' ';
  return (v) => {
    const n = v as number | bigint;
    const s = String(n);
    return (plus && n >= 0 ? lead + s : s).padStart(width);
  };
}

function extendLine(
  parts: string[],
  line: string,
  word: string,
  lineWidth: number,
  nextLinePrefix: string,
): string {
  let needsWrap = line.length + word.length > lineWidth;
  if (line.length <= nextLinePrefix.length) needsWrap = false;
  if (needsWrap) {
    parts.push(`${line.trimEnd()}\n`);
    line = nextLinePrefix;
  }
  return line + word;
}

/**
 * Lay the elements out recursively. The last axis wraps greedily at the line width and
 * continuation lines align under the first element; outer axes put each sub-array on its
 * own line, separated by blank lines for every extra dimension.
 */
function formatArrayRecursive(
  storage: ArrayStorage,
  formatter: ElementFormatter,
  lineWidth: number,
  nextLinePrefix: string,
  separator: string,
  edgeitems: number,
  summarize: boolean,
): string {
  const shape = storage.shape;
  const ndim = shape.length;
  const indices = new Array<number>(ndim).fill(0);
  const sepTrim = separator.trimEnd();

  function recurse(depth: number, hangingIndent: string, currWidth: number): string {
    const axesLeft = ndim - depth;
    if (axesLeft === 0) return formatter(readAt(storage, indices));

    const nextHangingIndent = `${hangingIndent} `;
    const nextWidth = currWidth - 1;
    const n = shape[depth]!;
    const showSummary = summarize && 2 * edgeitems < n;
    const positions = shownPositions(n, showSummary, edgeitems);
    if (showSummary) positions.splice(edgeitems, 0, -1);

    const out: string[] = [];
    if (axesLeft === 1) {
      const elemWidth = currWidth - Math.max(sepTrim.length, 1);
      let line = hangingIndent;
      for (let k = 0; k < positions.length; k++) {
        const pos = positions[k]!;
        const last = k === positions.length - 1;
        let word: string;
        if (pos === -1) {
          word = '...';
        } else {
          indices[depth] = pos;
          word = recurse(depth + 1, nextHangingIndent, nextWidth);
        }
        line = extendLine(out, line, word, elemWidth, hangingIndent);
        if (!last) line += separator;
      }
      out.push(line);
    } else {
      const lineSep = sepTrim + '\n'.repeat(axesLeft - 1);
      for (let k = 0; k < positions.length; k++) {
        const pos = positions[k]!;
        let nested: string;
        if (pos === -1) {
          nested = '...';
        } else {
          indices[depth] = pos;
          nested = recurse(depth + 1, nextHangingIndent, nextWidth);
        }
        out.push(hangingIndent + nested);
        if (k < positions.length - 1) out.push(lineSep);
      }
    }
    return `[${out.join('').slice(hangingIndent.length)}]`;
  }

  return recurse(0, nextLinePrefix, lineWidth);
}

const SCALAR_SCI_UPPER: Record<FloatKind, number> = {
  float16: 1e3,
  float32: 1e6,
  float64: 1e16,
};

/**
 * Shortest round-trip text of a float scalar the way NumPy prints a scalar: positional
 * with a trailing .0 for whole values, scientific below 1e-4 or from an upper bound that
 * depends on the float width.
 */
function floatScalarString(x: number, kind: FloatKind, trim: Trim): string {
  if (!Number.isFinite(x)) return nonFinite(x, '-');
  const a = Math.abs(x);
  if (a !== 0 && (a >= SCALAR_SCI_UPPER[kind] || a < 1e-4)) {
    return dragonScientific(x, kind, null, true, trim === '0' ? '-' : trim, '-', null, null, null);
  }
  return dragonPositional(x, kind, null, true, true, trim, '-', null, null, null);
}

/** The str() of a single element, as used for 0-d arrays. */
function scalarString(value: ElementValue, dtype: DType): string {
  if (dtype === 'bool') return value ? 'True' : 'False';
  if (isComplexDType(dtype)) {
    const c = value as Complex;
    const kind = floatKindOf(dtype);
    const re = floatScalarString(c.re, kind, '-');
    const imNeg = c.im < 0 || Object.is(c.im, -0);
    const im = `${floatScalarString(Math.abs(c.im), kind, '-')}j`;
    if (c.re === 0 && !Object.is(c.re, -0)) return (imNeg ? '-' : '') + im;
    return `(${re}${imNeg ? '-' : '+'}${im})`;
  }
  if (dtype === 'float16' || dtype === 'float32' || dtype === 'float64') {
    return floatScalarString(value as number, floatKindOf(dtype), '0');
  }
  return String(value);
}

/**
 * Convert an array to a string representation, matching numpy.array2string. The prefix
 * and suffix are not part of the result; they only reserve width and set the indent of
 * continuation lines.
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width (default: from printoptions)
 * @param precision - Number of digits of precision (default: from printoptions)
 * @param suppress_small - Print very small values as zero (default: from printoptions)
 * @param separator - Separator between elements (default: ' ')
 * @param prefix - Text that will precede the result, used for alignment
 * @param suffix - Text that will follow the result, used for line width
 * @param threshold - Element count above which the output is summarized (default: from printoptions)
 * @param edgeitems - Number of items at each edge when summarizing (default: from printoptions)
 * @param floatmode - Float format mode (default: from printoptions)
 * @param sign - Sign handling for numbers (default: from printoptions)
 * @returns String representation of the array
 *
 * @example
 * ```typescript
 * const arr = array([[1, 2, 3], [4, 5, 6]]);
 * console.log(array2string(arr.storage));
 * // [[1. 2. 3.]
 * //  [4. 5. 6.]]
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
  sign: ' ' | '+' | '-' | null = null,
): string {
  const opts: PrintOptions = {
    ...currentPrintOptions,
    linewidth: (max_line_width ?? currentPrintOptions.linewidth) - suffix.length,
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

  if (a.size === 0) return '[]';

  const dtype = a.dtype as DType;
  const summarize = a.size > opts.threshold;
  const values = collectVisibleValues(a, summarize, opts.edgeitems);
  const formatter = buildFormatter(dtype, values, opts, a.ndim === 0);
  return formatArrayRecursive(
    a,
    formatter,
    opts.linewidth,
    ` ${' '.repeat(prefix.length)}`,
    separator,
    opts.edgeitems,
    summarize,
  );
}

const IMPLIED_DTYPES: ReadonlySet<string> = new Set(['int64', 'float64', 'complex128', 'bool']);

/**
 * Return the string representation of an array, matching numpy.array_repr: rows are
 * comma separated and aligned under the opening bracket, and the dtype is appended unless
 * it is implied by the values (int64, float64, complex128, bool). Empty arrays and arrays
 * larger than the print threshold also show their shape.
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width
 * @param precision - Number of digits of precision
 * @param suppress_small - Print very small values as zero
 * @returns String representation
 *
 * @example
 * ```typescript
 * const arr = array([1, 2, 3], 'int8');
 * console.log(array_repr(arr.storage));
 * // array([1, 2, 3], dtype=int8)
 * ```
 */
export function array_repr(
  a: ArrayStorage,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null,
): string {
  const width = max_line_width ?? currentPrintOptions.linewidth;
  const prefix = 'array(';
  const lst = array2string(a, width, precision, suppress_small, ', ', prefix, ')');

  const extras: string[] = [];
  const shape = a.shape;
  const isEmpty1d = shape.length === 1 && shape[0] === 0;
  if ((a.size === 0 && !isEmpty1d) || a.size > currentPrintOptions.threshold) {
    extras.push(`shape=(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`);
  }
  if (!IMPLIED_DTYPES.has(a.dtype) || a.size === 0) {
    extras.push(`dtype=${a.dtype}`);
  }
  if (extras.length === 0) return `${prefix}${lst})`;

  const head = `${prefix}${lst},`;
  const extra = `${extras.join(', ')})`;
  const lastLineLen = head.length - (head.lastIndexOf('\n') + 1);
  const spacer = lastLineLen + extra.length + 1 > width ? `\n${' '.repeat(prefix.length)}` : ' ';
  return head + spacer + extra;
}

/**
 * Return a string representation of the data in an array, matching numpy.array_str.
 * A 0-d array prints as its scalar value.
 *
 * @param a - Input array storage
 * @param max_line_width - Maximum line width
 * @param precision - Number of digits of precision
 * @param suppress_small - Print very small values as zero
 * @returns String representation of the data
 *
 * @example
 * ```typescript
 * const arr = array([1, 2, 3]);
 * console.log(array_str(arr.storage));
 * // [1. 2. 3.]
 * ```
 */
export function array_str(
  a: ArrayStorage,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null,
): string {
  if (a.ndim === 0) return scalarString(readAt(a, []), a.dtype as DType);
  return array2string(a, max_line_width, precision, suppress_small, ' ', '');
}
