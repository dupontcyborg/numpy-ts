/**
 * Slicing utilities for NumPy-compatible array indexing
 *
 * Supports Python-style slice syntax via strings: "0:5", ":", "::2", "-1"
 */

/**
 * Represents a parsed slice specification
 */
export interface SliceSpec {
  start: number | null;
  stop: number | null;
  step: number;
  isIndex: boolean; // true if this is a single index, not a slice
}

/**
 * Parse a slice string into a SliceSpec
 *
 * Supports:
 * - Single index: "5", "-1"
 * - Full slice: "0:5", "2:8"
 * - With step: "0:10:2", "::2"
 * - Partial: "5:", ":10", ":"
 * - Negative: "-5:", ":-2", "::-1"
 *
 * @param sliceStr - String representation of slice (e.g., "0:5", ":", "::2")
 * @returns Parsed slice specification
 *
 * @example
 * ```typescript
 * parseSlice("0:5")    // {start: 0, stop: 5, step: 1, isIndex: false}
 * parseSlice(":")      // {start: null, stop: null, step: 1, isIndex: false}
 * parseSlice("::2")    // {start: null, stop: null, step: 2, isIndex: false}
 * parseSlice("-1")     // {start: -1, stop: null, step: 1, isIndex: true}
 * parseSlice("5")      // {start: 5, stop: null, step: 1, isIndex: true}
 * ```
 */
export function parseSlice(sliceStr: string): SliceSpec {
  // Check if it's a single index (no colons)
  if (!sliceStr.includes(':')) {
    // Reject decimal points - indices must be integers
    if (sliceStr.includes('.')) {
      throw new Error(`Invalid slice index: "${sliceStr}" (must be integer)`);
    }
    const index = parseInt(sliceStr, 10);
    if (isNaN(index)) {
      throw new Error(`Invalid slice index: "${sliceStr}"`);
    }
    return {
      start: index,
      stop: null,
      step: 1,
      isIndex: true,
    };
  }

  // Parse slice notation: start:stop:step
  const parts = sliceStr.split(':');

  if (parts.length > 3) {
    throw new Error(`Invalid slice notation: "${sliceStr}" (too many colons)`);
  }

  const start = parts[0] === '' ? null : parseInt(parts[0]!, 10);
  const stop = parts[1] === '' || parts[1] === undefined ? null : parseInt(parts[1], 10);
  const step = parts[2] === '' || parts[2] === undefined ? 1 : parseInt(parts[2], 10);

  // Validate parsed values
  if (start !== null && isNaN(start)) {
    throw new Error(`Invalid start index in slice: "${sliceStr}"`);
  }
  if (stop !== null && isNaN(stop)) {
    throw new Error(`Invalid stop index in slice: "${sliceStr}"`);
  }
  if (isNaN(step)) {
    throw new Error(`Invalid step in slice: "${sliceStr}"`);
  }
  if (step === 0) {
    throw new Error(`Slice step cannot be zero`);
  }

  return {
    start,
    stop,
    step,
    isIndex: false,
  };
}

/**
 * Normalize a slice specification to absolute indices
 *
 * Handles negative indices and defaults:
 * - Negative indices count from the end
 * - null start becomes 0 (or size-1 for negative step)
 * - null stop becomes size (or -1 for negative step)
 *
 * @param spec - Parsed slice specification
 * @param size - Size of the dimension being sliced
 * @returns Normalized slice with absolute start, stop, step
 *
 * @example
 * ```typescript
 * normalizeSlice({start: -1, stop: null, step: 1, isIndex: true}, 10)
 * // {start: 9, stop: 10, step: 1, isIndex: true}
 *
 * normalizeSlice({start: null, stop: -2, step: 1, isIndex: false}, 10)
 * // {start: 0, stop: 8, step: 1, isIndex: false}
 * ```
 */
export function normalizeSlice(
  spec: SliceSpec,
  size: number
): { start: number; stop: number; step: number; isIndex: boolean } {
  let { start, stop } = spec;
  const { step, isIndex } = spec;

  // For single index, normalize and return
  if (isIndex) {
    if (start === null) {
      throw new Error('Index cannot be null');
    }
    const normalizedStart = start < 0 ? size + start : start;
    if (normalizedStart < 0 || normalizedStart >= size) {
      throw new Error(`Index ${start} is out of bounds for size ${size}`);
    }
    return {
      start: normalizedStart,
      stop: normalizedStart + 1,
      step: 1,
      isIndex: true,
    };
  }

  // Handle slice defaults based on step direction
  if (step > 0) {
    // Forward slice
    if (start === null) start = 0;
    if (stop === null) stop = size;
  } else {
    // Backward slice
    if (start === null) start = size - 1;
    if (stop === null) stop = -size - 1; // Will be normalized to before start
  }

  // Normalize negative indices
  if (start < 0) start = size + start;
  if (stop < 0) stop = size + stop;

  // Clamp to valid range
  start = Math.max(0, Math.min(start, size));
  stop = Math.max(-1, Math.min(stop, size)); // -1 allowed for backward slices

  return {
    start,
    stop,
    step,
    isIndex: false,
  };
}

/**
 * Compute the length of a slice result
 *
 * @param start - Normalized start index
 * @param stop - Normalized stop index
 * @param step - Step value
 * @returns Number of elements in the slice
 */
export function computeSliceLength(start: number, stop: number, step: number): number {
  if (step > 0) {
    if (start >= stop) return 0;
    return Math.ceil((stop - start) / step);
  } else {
    if (start <= stop) return 0;
    return Math.ceil((start - stop) / -step);
  }
}

/**
 * Parse multiple slice specifications for multi-dimensional indexing
 *
 * @param sliceStrs - Array of slice strings, one per dimension
 * @returns Array of parsed slice specifications
 *
 * @example
 * ```typescript
 * parseSlices(["0:5", ":", "::2"])
 * // [
 * //   {start: 0, stop: 5, step: 1, isIndex: false},
 * //   {start: null, stop: null, step: 1, isIndex: false},
 * //   {start: null, stop: null, step: 2, isIndex: false}
 * // ]
 * ```
 */
export function parseSlices(sliceStrs: string[]): SliceSpec[] {
  return sliceStrs.map((s) => parseSlice(s));
}
