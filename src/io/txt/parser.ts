/**
 * Text file parsing for numpy-ts
 *
 * Provides NumPy-compatible loadtxt/genfromtxt functionality.
 * These functions work with strings and are environment-agnostic.
 */

import { array } from '../../core/creation';
import { NDArrayCore } from '../../common/ndarray-core';
import type { DType } from '../../common/dtype';

/**
 * Options for parsing text data
 */
export interface ParseTxtOptions {
  /**
   * The string used to separate values.
   * By default, any consecutive whitespace acts as delimiter.
   * If specified, the exact delimiter is used.
   */
  delimiter?: string;

  /**
   * The character used to indicate the start of a comment.
   * Lines beginning with this character (after stripping whitespace) will be skipped.
   * Default: '#'
   */
  comments?: string;

  /**
   * Skip the first `skiprows` lines.
   * Default: 0
   */
  skiprows?: number;

  /**
   * Which columns to read, with 0 being the first.
   * If not specified, all columns are read.
   */
  usecols?: number | number[];

  /**
   * Read only the first `max_rows` lines of content after `skiprows`.
   * Default: read all rows
   */
  max_rows?: number;

  /**
   * Data type of the resulting array.
   * Default: 'float64'
   */
  dtype?: DType;

  /**
   * What encoding to use when reading the file.
   * Only relevant for Node.js file operations.
   * Default: 'utf-8'
   */
  encoding?: string;

  /**
   * The string representation of a missing value.
   * Used by genfromtxt. When encountered, the value is replaced with `filling_values`.
   */
  missing_values?: string | string[];

  /**
   * The value to use for missing values.
   * Default: NaN for floating point, 0 for integers
   */
  filling_values?: number;
}

/**
 * Parse text data into an NDArray.
 *
 * This is the browser-compatible core function. For file operations in Node.js,
 * use `loadtxt` from 'numpy-ts/node'.
 *
 * @param text - The text content to parse
 * @param options - Parsing options
 * @returns NDArray with the parsed data
 *
 * @example
 * ```typescript
 * const text = "1 2 3\n4 5 6\n7 8 9";
 * const arr = parseTxt(text);
 * // arr.shape = [3, 3]
 * ```
 */
export function parseTxt(text: string, options: ParseTxtOptions = {}): NDArrayCore {
  const {
    delimiter,
    comments = '#',
    skiprows = 0,
    usecols,
    max_rows,
    dtype = 'float64',
    missing_values,
    filling_values,
  } = options;

  // Split into lines
  let lines = text.split(/\r?\n/);

  // Skip initial rows
  if (skiprows > 0) {
    lines = lines.slice(skiprows);
  }

  // Filter out comment lines and empty lines
  lines = lines.filter((line) => {
    const trimmed = line.trim();
    if (trimmed === '') return false;
    if (comments && trimmed.startsWith(comments)) return false;
    return true;
  });

  // Limit rows if max_rows specified
  if (max_rows !== undefined && max_rows > 0) {
    lines = lines.slice(0, max_rows);
  }

  if (lines.length === 0) {
    // Return empty array
    return array([], dtype);
  }

  // Parse each line
  const data: number[][] = [];
  const missingSet = new Set(
    missing_values ? (Array.isArray(missing_values) ? missing_values : [missing_values]) : []
  );

  // Determine fill value
  const fillValue =
    filling_values !== undefined
      ? filling_values
      : dtype.includes('int') || dtype === 'bool'
        ? 0
        : NaN;

  for (const line of lines) {
    let values: string[];

    if (delimiter === undefined) {
      // Split on whitespace (any consecutive whitespace)
      values = line.trim().split(/\s+/);
    } else {
      values = line.split(delimiter);
    }

    // Select specific columns if usecols is specified
    if (usecols !== undefined) {
      const cols = Array.isArray(usecols) ? usecols : [usecols];
      values = cols.map((col) => {
        if (col < 0) col = values.length + col;
        return values[col] ?? '';
      });
    }

    // Parse values to numbers
    const row = values.map((v) => {
      const trimmed = v.trim();
      if (missingSet.has(trimmed) || trimmed === '') {
        return fillValue;
      }
      const num = parseFloat(trimmed);
      return isNaN(num) ? fillValue : num;
    });

    data.push(row);
  }

  // Check if all rows have the same length
  const ncols = data[0]?.length ?? 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i]!.length !== ncols) {
      throw new Error(
        `Inconsistent number of columns: row 0 has ${ncols} columns, row ${i} has ${data[i]!.length} columns`
      );
    }
  }

  // If only one column, return 1D array
  if (ncols === 1) {
    return array(
      data.map((row) => row[0]),
      dtype
    );
  }

  return array(data, dtype);
}

/**
 * Parse text data into an NDArray with more flexible handling.
 *
 * Similar to parseTxt but handles missing values more gracefully.
 * This is the browser-compatible version.
 *
 * @param text - The text content to parse
 * @param options - Parsing options
 * @returns NDArray with the parsed data
 */
export function genfromtxt(text: string, options: ParseTxtOptions = {}): NDArrayCore {
  // genfromtxt is essentially parseTxt with different defaults for missing value handling
  const opts: ParseTxtOptions = {
    ...options,
    // Default missing values for genfromtxt
    missing_values: options.missing_values ?? ['', 'nan', 'NaN', 'NA', 'N/A', '-'],
    filling_values: options.filling_values ?? NaN,
  };

  return parseTxt(text, opts);
}

/**
 * Parse text data using a regular expression.
 *
 * Extract data from each line using regex groups.
 *
 * @param text - The text content to parse
 * @param regexp - Regular expression with groups to extract values
 * @param dtype - Data type of the resulting array (default: 'float64')
 * @returns NDArray with the parsed data
 *
 * @example
 * ```typescript
 * const text = "x=1.0, y=2.0\nx=3.0, y=4.0";
 * const arr = fromregex(text, /x=([\d.]+), y=([\d.]+)/);
 * // arr = [[1.0, 2.0], [3.0, 4.0]]
 * ```
 */
export function fromregex(
  text: string,
  regexp: RegExp | string,
  dtype: DType = 'float64'
): NDArrayCore {
  const re =
    typeof regexp === 'string' ? new RegExp(regexp, 'gm') : new RegExp(regexp.source, 'gm');

  const data: number[][] = [];

  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    // Skip the full match (index 0), use only capture groups
    if (match.length > 1) {
      const row = match.slice(1).map((v) => {
        const num = parseFloat(v);
        return isNaN(num) ? 0 : num;
      });
      data.push(row);
    }
  }

  if (data.length === 0) {
    return array([], dtype);
  }

  // If only one column, return 1D array
  if (data[0]!.length === 1) {
    return array(
      data.map((row) => row[0]),
      dtype
    );
  }

  return array(data, dtype);
}
