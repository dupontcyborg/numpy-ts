/**
 * Text file serialization for numpy-ts
 *
 * Provides NumPy-compatible savetxt functionality.
 * These functions work with strings and are environment-agnostic.
 */

import { NDArray } from '../../full/ndarray';

/**
 * Options for serializing array to text
 */
export interface SerializeTxtOptions {
  /**
   * Format string for a single value.
   * Uses a simplified printf-style format:
   * - '%.6f' - 6 decimal places (default for floats)
   * - '%.2e' - scientific notation with 2 decimal places
   * - '%d' - integer
   * - '%s' - string representation
   *
   * Default: '%.18e' (NumPy default for full precision)
   */
  fmt?: string;

  /**
   * String or character separating columns.
   * Default: ' ' (single space)
   */
  delimiter?: string;

  /**
   * String that will be written at the end of each row.
   * Default: '\n'
   */
  newline?: string;

  /**
   * String that will be written at the beginning of the file.
   * Will be prepended with the comment character if it doesn't start with one.
   */
  header?: string;

  /**
   * String that will be written at the end of the file.
   */
  footer?: string;

  /**
   * String that will be prepended to the header and footer strings,
   * to mark them as comments.
   * Default: '# '
   */
  comments?: string;
}

/**
 * Format a number according to a printf-style format string.
 */
function formatValue(value: number | bigint, fmt: string): string {
  const num = typeof value === 'bigint' ? Number(value) : value;

  // Parse format string: %[flags][width][.precision]specifier
  const match = fmt.match(/^%([+-]?)(\d*)(?:\.(\d+))?([dfeEgGs])$/);

  if (!match) {
    // Default to string representation
    return String(num);
  }

  const [, flags, width, precisionStr, specifier] = match;
  const precision = precisionStr !== undefined ? parseInt(precisionStr, 10) : undefined;

  let result: string;

  switch (specifier) {
    case 'd':
      // Integer
      result = Math.round(num).toString();
      break;

    case 'f':
      // Fixed-point
      result = num.toFixed(precision ?? 6);
      break;

    case 'e':
      // Scientific notation (lowercase)
      // Ensure exponent has at least 2 digits (e.g., e+00, not e+0) like NumPy
      result = num.toExponential(precision ?? 6).replace(/e([+-])(\d)$/, 'e$10$2');
      break;

    case 'E':
      // Scientific notation (uppercase)
      // Ensure exponent has at least 2 digits like NumPy
      result = num
        .toExponential(precision ?? 6)
        .toUpperCase()
        .replace(/E([+-])(\d)$/, 'E$10$2');
      break;

    case 'g':
    case 'G': {
      // General format - use fixed or scientific based on magnitude
      const p = precision ?? 6;
      const exp = Math.floor(Math.log10(Math.abs(num)));
      if (exp >= -4 && exp < p) {
        result = num.toPrecision(p);
        // Remove trailing zeros after decimal point
        if (result.includes('.')) {
          result = result.replace(/\.?0+$/, '');
        }
      } else {
        result = num.toExponential(p - 1);
      }
      if (specifier === 'G') {
        result = result.toUpperCase();
      }
      break;
    }

    case 's':
      result = String(num);
      break;

    default:
      result = String(num);
  }

  // Apply width padding
  if (width) {
    const w = parseInt(width, 10);
    if (result.length < w) {
      const padding = ' '.repeat(w - result.length);
      if (flags === '-') {
        result = result + padding; // Left-align
      } else {
        result = padding + result; // Right-align (default)
      }
    }
  }

  // Apply + flag for positive numbers
  if (flags === '+' && num >= 0 && !result.startsWith('-')) {
    result = '+' + result;
  }

  return result;
}

/**
 * Serialize an NDArray to text format.
 *
 * This is the browser-compatible core function. For file operations in Node.js,
 * use `savetxt` from 'numpy-ts/node'.
 *
 * @param arr - The array to serialize (must be 1D or 2D)
 * @param options - Serialization options
 * @returns String representation of the array
 *
 * @example
 * ```typescript
 * const arr = np.array([[1, 2, 3], [4, 5, 6]]);
 * const text = serializeTxt(arr, { delimiter: ',' });
 * // "1.000000000000000000e+00,2.000000000000000000e+00,3.000000000000000000e+00\n4.000000000000000000e+00,5.000000000000000000e+00,6.000000000000000000e+00\n"
 * ```
 */
export function serializeTxt(arr: NDArray, options: SerializeTxtOptions = {}): string {
  const {
    fmt = '%.18e',
    delimiter = ' ',
    newline = '\n',
    header,
    footer,
    comments = '# ',
  } = options;

  if (arr.ndim > 2) {
    throw new Error('savetxt: array must be 1D or 2D');
  }

  const lines: string[] = [];

  // Add header
  if (header !== undefined) {
    const headerLines = header.split(/\r?\n/);
    for (const line of headerLines) {
      if (line.startsWith(comments.trimEnd())) {
        lines.push(line);
      } else {
        lines.push(comments + line);
      }
    }
  }

  // Get array as nested array for easier iteration
  const data = arr.toArray();

  if (arr.ndim === 1) {
    // 1D array: each element on its own line
    for (const value of data as (number | bigint)[]) {
      lines.push(formatValue(value, fmt));
    }
  } else {
    // 2D array: each row on its own line
    for (const row of data as (number | bigint)[][]) {
      const formattedValues = row.map((v) => formatValue(v, fmt));
      lines.push(formattedValues.join(delimiter));
    }
  }

  // Add footer
  if (footer !== undefined) {
    const footerLines = footer.split(/\r?\n/);
    for (const line of footerLines) {
      if (line.startsWith(comments.trimEnd())) {
        lines.push(line);
      } else {
        lines.push(comments + line);
      }
    }
  }

  return lines.join(newline) + newline;
}
