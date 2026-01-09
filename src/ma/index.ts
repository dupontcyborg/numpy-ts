/**
 * Masked Array module (np.ma)
 *
 * Provides NumPy-compatible masked array functionality for handling
 * arrays with invalid or missing data.
 */

import { MaskedArray, masked, nomask, default_fill_value, MaskInput } from './MaskedArray';
import { NDArray } from '../core/ndarray';
import { DType } from '../core/dtype';
import * as creation from '../core/ndarray';
import * as logicOps from '../ops/logic';
import * as comparisonOps from '../ops/comparison';
import * as trigOps from '../ops/trig';
import * as hyperbolicOps from '../ops/hyperbolic';
import * as exponentialOps from '../ops/exponential';
import * as roundingOps from '../ops/rounding';
import * as bitwiseOps from '../ops/bitwise';
import * as arithmeticOps from '../ops/arithmetic';
import * as linalgOps from '../ops/linalg';
import { ArrayStorage } from '../core/storage';

// Re-export core types and constants
export { masked, nomask, default_fill_value, MaskedArray };
export type { MaskInput };

// ============================================================================
// Array Creation Functions
// ============================================================================

/**
 * Create a MaskedArray
 *
 * @param data - Input data
 * @param options - mask, dtype, fill_value, copy, hard_mask
 */
export function array(
  data: NDArray | number[] | number[][] | number[][][],
  options?: {
    mask?: MaskInput;
    dtype?: DType;
    fill_value?: number | bigint | boolean;
    copy?: boolean;
    hard_mask?: boolean;
  }
): MaskedArray {
  return new MaskedArray(data, options);
}

/**
 * Alias for array() - create a MaskedArray
 */
export const masked_array = array;

/**
 * Alias for MaskedArray constructor
 */
export const MaskedArray_ = MaskedArray;

/**
 * Convert input to MaskedArray
 */
export function asarray(a: MaskedArray | NDArray | number[] | number[][]): MaskedArray {
  if (a instanceof MaskedArray) return a;
  return new MaskedArray(a instanceof NDArray ? a : creation.array(a));
}

/**
 * Convert input to MaskedArray (allows subclasses)
 */
export function asanyarray(a: MaskedArray | NDArray | number[] | number[][]): MaskedArray {
  return asarray(a);
}

/**
 * Create MaskedArray of zeros
 */
export function zeros(shape: number | number[], _dtype?: DType, fill_value?: number): MaskedArray {
  const shapeArr = typeof shape === 'number' ? [shape] : shape;
  return new MaskedArray(creation.zeros(shapeArr) as NDArray, { fill_value });
}

/**
 * Create MaskedArray of ones
 */
export function ones(shape: number | number[], dtype?: DType, fill_value?: number): MaskedArray {
  const shapeArr = typeof shape === 'number' ? [shape] : shape;
  return new MaskedArray(creation.ones(shapeArr, dtype) as NDArray, { fill_value });
}

/**
 * Create empty MaskedArray
 */
export function empty(shape: number | number[], dtype?: DType, fill_value?: number): MaskedArray {
  const shapeArr = typeof shape === 'number' ? [shape] : shape;
  return new MaskedArray(creation.empty(shapeArr, dtype) as NDArray, { fill_value });
}

/**
 * Create MaskedArray like another array but with zeros
 */
export function zeros_like(a: MaskedArray | NDArray, _dtype?: DType): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a;
  const result = creation.zeros_like(arr) as NDArray;
  return new MaskedArray(result, {
    mask: a instanceof MaskedArray ? a.mask : false,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Create MaskedArray like another array but with ones
 */
export function ones_like(a: MaskedArray | NDArray, _dtype?: DType): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a;
  const result = creation.ones_like(arr) as NDArray;
  return new MaskedArray(result, {
    mask: a instanceof MaskedArray ? a.mask : false,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Create empty MaskedArray like another array
 */
export function empty_like(a: MaskedArray | NDArray, _dtype?: DType): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a;
  const result = creation.empty_like(arr) as NDArray;
  return new MaskedArray(result, {
    mask: a instanceof MaskedArray ? a.mask : false,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Create a MaskedArray with all elements masked
 */
export function masked_all(shape: number | number[], _dtype?: DType): MaskedArray {
  const shapeArr = typeof shape === 'number' ? [shape] : shape;
  return new MaskedArray(creation.zeros(shapeArr) as NDArray, { mask: true });
}

/**
 * Create a MaskedArray like another with all elements masked
 */
export function masked_all_like(a: MaskedArray | NDArray): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a;
  return new MaskedArray(creation.zeros_like(arr) as NDArray, { mask: true });
}

/**
 * Create MaskedArray with values in range
 */
export function arange(start: number, stop?: number, step?: number, dtype?: DType): MaskedArray {
  return new MaskedArray(creation.arange(start, stop, step, dtype));
}

/**
 * Create MaskedArray from indices
 */
export function indices(dimensions: number[], _dtype?: 'int32' | 'int64' | 'float64'): MaskedArray {
  const result = creation.indices(dimensions);
  return new MaskedArray(result);
}

/**
 * Create identity MaskedArray
 */
export function identity(n: number, _dtype?: DType): MaskedArray {
  return new MaskedArray(creation.identity(n));
}

// ============================================================================
// Masking Functions
// ============================================================================

/**
 * Mask where condition is true
 */
export function masked_where(
  condition: NDArray | boolean[] | number[],
  a: MaskedArray | NDArray | number[]
): MaskedArray {
  const data = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const condArray = condition instanceof NDArray ? condition : creation.array(condition, 'bool');

  // Combine with existing mask if present
  let finalMask: NDArray;
  if (a instanceof MaskedArray && a.mask !== false) {
    finalMask = NDArray._fromStorage(logicOps.logical_or(a.mask.storage, condArray.storage));
  } else {
    finalMask = condArray.astype('bool');
  }

  return new MaskedArray(data as NDArray, {
    mask: finalMask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values equal given value
 */
export function masked_equal(a: NDArray | MaskedArray | number[], value: number): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.equal(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values do not equal given value
 */
export function masked_not_equal(a: NDArray | MaskedArray | number[], value: number): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.notEqual(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are greater than given value
 */
export function masked_greater(a: NDArray | MaskedArray | number[], value: number): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.greater(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are greater than or equal to given value
 */
export function masked_greater_equal(
  a: NDArray | MaskedArray | number[],
  value: number
): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.greaterEqual(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are less than given value
 */
export function masked_less(a: NDArray | MaskedArray | number[], value: number): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.less(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are less than or equal to given value
 */
export function masked_less_equal(a: NDArray | MaskedArray | number[], value: number): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const mask = NDArray._fromStorage(comparisonOps.lessEqual(arr.storage, value));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are inside interval [v1, v2]
 */
export function masked_inside(
  a: NDArray | MaskedArray | number[],
  v1: number,
  v2: number
): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const ge = NDArray._fromStorage(comparisonOps.greaterEqual(arr.storage, Math.min(v1, v2)));
  const le = NDArray._fromStorage(comparisonOps.lessEqual(arr.storage, Math.max(v1, v2)));
  const mask = NDArray._fromStorage(logicOps.logical_and(ge.storage, le.storage));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are outside interval [v1, v2]
 */
export function masked_outside(
  a: NDArray | MaskedArray | number[],
  v1: number,
  v2: number
): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const lt = NDArray._fromStorage(comparisonOps.less(arr.storage, Math.min(v1, v2)));
  const gt = NDArray._fromStorage(comparisonOps.greater(arr.storage, Math.max(v1, v2)));
  const mask = NDArray._fromStorage(logicOps.logical_or(lt.storage, gt.storage));
  return new MaskedArray(arr, {
    mask,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask where values are approximately equal to given value
 */
export function masked_values(
  a: NDArray | MaskedArray | number[],
  value: number,
  rtol: number = 1e-5,
  atol: number = 1e-8
): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  // |a - value| <= atol + rtol * |value|
  const tol = atol + rtol * Math.abs(value);
  const result = masked_inside(arr, value - tol, value + tol);
  return result;
}

/**
 * Mask invalid values (NaN, Inf)
 */
export function masked_invalid(a: NDArray | MaskedArray | number[]): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const data = arr.data;

  // Create mask array for invalid values
  const maskArr = creation.zeros([...arr.shape], 'bool') as NDArray;
  const maskArrData = maskArr.data;

  for (let i = 0; i < arr.size; i++) {
    const v = Number(data[i]);
    if (!Number.isFinite(v)) {
      (maskArrData as Uint8Array)[i] = 1;
    }
  }

  return new MaskedArray(arr, {
    mask: maskArr,
    fill_value: a instanceof MaskedArray ? a.fill_value : undefined,
  });
}

/**
 * Mask object-type elements (not applicable in numpy-ts, but provided for compatibility)
 */
export function masked_object(a: NDArray | MaskedArray | number[], value: unknown): MaskedArray {
  // In numpy-ts we don't have object arrays, so this just does equality comparison
  return masked_equal(a, value as number);
}

// ============================================================================
// Mask Utilities
// ============================================================================

/**
 * Get the mask of an array (returns False for regular arrays)
 */
export function getmask(a: MaskedArray | NDArray): NDArray | false {
  if (a instanceof MaskedArray) {
    return a.mask;
  }
  return false;
}

/**
 * Get the mask as an array (returns array of False for regular arrays)
 */
export function getmaskarray(a: MaskedArray | NDArray): NDArray {
  if (a instanceof MaskedArray) {
    if (a.mask === false) {
      return creation.zeros(a.shape as number[], 'bool') as NDArray;
    }
    return a.mask;
  }
  return creation.zeros(a.shape as number[], 'bool') as NDArray;
}

/**
 * Get the data from a MaskedArray (returns the array itself for regular arrays)
 */
export function getdata(a: MaskedArray | NDArray): NDArray {
  if (a instanceof MaskedArray) {
    return a.data;
  }
  return a;
}

/**
 * Test whether input is a masked array
 */
export function isMaskedArray(a: unknown): a is MaskedArray {
  return a instanceof MaskedArray;
}

/**
 * Alias for isMaskedArray
 */
export const isMA = isMaskedArray;

/**
 * Alias for isMaskedArray
 */
export function isarray(a: unknown): a is MaskedArray {
  return a instanceof MaskedArray;
}

/**
 * Check if any element is masked
 */
export function is_masked(a: MaskedArray | NDArray): boolean {
  if (!(a instanceof MaskedArray)) return false;
  if (a.mask === false) return false;

  const maskData = a.mask.data;
  for (let i = 0; i < maskData.length; i++) {
    if (maskData[i]) return true;
  }
  return false;
}

/**
 * Test whether input is a valid mask
 */
export function is_mask(m: unknown): boolean {
  if (m === false || m === nomask) return true;
  if (m instanceof NDArray && m.dtype === 'bool') return true;
  if (Array.isArray(m) && m.every((v) => typeof v === 'boolean' || v === 0 || v === 1)) {
    return true;
  }
  return false;
}

/**
 * Create a boolean mask from an array
 */
export function make_mask(
  m: MaskInput,
  copy: boolean = false,
  shrink: boolean = true,
  _dtype?: DType
): NDArray | false {
  // Check for falsy boolean values
  if (typeof m === 'boolean' && m === false) return false;

  let mask: NDArray;
  if (typeof m === 'boolean' && m === true) {
    // Can't create mask without shape info
    throw new Error('Cannot create mask from True without shape information');
  } else if (m instanceof NDArray) {
    mask = copy ? m.copy() : m;
    if (mask.dtype !== 'bool') {
      mask = mask.astype('bool');
    }
  } else {
    mask = creation.array(m as number[], 'bool');
  }

  if (shrink) {
    // Check if all false
    const data = mask.data;
    let anyTrue = false;
    for (let i = 0; i < data.length; i++) {
      if (data[i]) {
        anyTrue = true;
        break;
      }
    }
    if (!anyTrue) return false;
  }

  return mask;
}

/**
 * Create a mask with all False values
 */
export function make_mask_none(shape: number | number[]): NDArray {
  const shapeArr = typeof shape === 'number' ? [shape] : shape;
  return creation.zeros(shapeArr, 'bool') as NDArray;
}

/**
 * Construct a dtype description for a masked array
 */
export function make_mask_descr(_dtype: DType): DType {
  // In NumPy this returns a structured dtype for the mask
  // We just return bool since we don't support structured dtypes
  return 'bool';
}

/**
 * Combine two masks with OR
 */
export function mask_or(m1: MaskInput, m2: MaskInput, copy: boolean = false): NDArray | false {
  if (m1 === false && m2 === false) return false;
  if (m1 === false) return make_mask(m2, copy) as NDArray;
  if (m2 === false) return make_mask(m1, copy) as NDArray;

  const mask1 = m1 instanceof NDArray ? m1 : creation.array(m1 as number[], 'bool');
  const mask2 = m2 instanceof NDArray ? m2 : creation.array(m2 as number[], 'bool');

  return NDArray._fromStorage(logicOps.logical_or(mask1.storage, mask2.storage));
}

/**
 * Flatten a mask
 */
export function flatten_mask(m: MaskInput): NDArray | false {
  if (typeof m === 'boolean' && m === false) return false;
  if (typeof m === 'boolean' && m === true) {
    throw new Error('Cannot flatten mask from True without shape information');
  }
  const mask = m instanceof NDArray ? m : creation.array(m as number[], 'bool');
  return mask.ravel();
}

// ============================================================================
// Fill Value Functions
// ============================================================================

/**
 * Return data with masked values filled
 */
export function filled(a: MaskedArray, fill_value?: number | bigint | boolean): NDArray {
  return a.filled(fill_value);
}

/**
 * Fix invalid values by masking and optionally filling
 */
export function fix_invalid(
  a: NDArray | MaskedArray | number[],
  mask?: MaskInput,
  copy: boolean = true,
  fill_value?: number
): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const result = masked_invalid(arr);

  if (mask !== undefined && mask !== false) {
    // Combine with additional mask
    const additionalMask =
      mask instanceof NDArray ? mask : creation.array(mask as number[], 'bool');
    if (result.mask === false) {
      result.mask = additionalMask;
    } else {
      result.mask = NDArray._fromStorage(
        logicOps.logical_or(result.mask.storage, additionalMask.storage)
      );
    }
  }

  if (fill_value !== undefined) {
    result.fill_value = fill_value;
  }

  return copy ? result.copy() : result;
}

/**
 * Return the minimum fill value for a dtype
 */
export function minimum_fill_value(a: MaskedArray | NDArray | DType): number | bigint {
  const dtype = typeof a === 'string' ? a : a.dtype;
  if (dtype.startsWith('float')) return -Infinity;
  if (dtype === 'int8') return -128;
  if (dtype === 'int16') return -32768;
  if (dtype === 'int32') return -2147483648;
  if (dtype === 'int64') return BigInt('-9223372036854775808');
  if (dtype.startsWith('uint')) return 0;
  return -Infinity;
}

/**
 * Return the maximum fill value for a dtype
 */
export function maximum_fill_value(a: MaskedArray | NDArray | DType): number | bigint {
  const dtype = typeof a === 'string' ? a : a.dtype;
  if (dtype.startsWith('float')) return Infinity;
  if (dtype === 'int8') return 127;
  if (dtype === 'int16') return 32767;
  if (dtype === 'int32') return 2147483647;
  if (dtype === 'int64') return BigInt('9223372036854775807');
  if (dtype === 'uint8') return 255;
  if (dtype === 'uint16') return 65535;
  if (dtype === 'uint32') return 4294967295;
  if (dtype === 'uint64') return BigInt('18446744073709551615');
  return Infinity;
}

/**
 * Get common fill value of two masked arrays
 */
export function common_fill_value(
  a: MaskedArray,
  b: MaskedArray
): number | bigint | boolean | null {
  if (a.fill_value === b.fill_value) {
    return a.fill_value;
  }
  return null;
}

/**
 * Set the fill value of a masked array
 */
export function set_fill_value(a: MaskedArray, fill_value: number | bigint | boolean): void {
  a.fill_value = fill_value;
}

// ============================================================================
// Compressed/Clumped Data Functions
// ============================================================================

/**
 * Return non-masked data as 1-D array
 */
export function compressed(a: MaskedArray): NDArray {
  return a.compressed();
}

/**
 * Return slices of masked/unmasked regions
 */
export function clump_masked(a: MaskedArray): Array<[number, number]> {
  if (a.mask === false) return [];

  const mask = a.mask.ravel().data;
  const clumps: Array<[number, number]> = [];
  let start: number | null = null;

  for (let i = 0; i < mask.length; i++) {
    if (mask[i] && start === null) {
      start = i;
    } else if (!mask[i] && start !== null) {
      clumps.push([start, i]);
      start = null;
    }
  }
  if (start !== null) {
    clumps.push([start, mask.length]);
  }

  return clumps;
}

/**
 * Return slices of unmasked regions
 */
export function clump_unmasked(a: MaskedArray): Array<[number, number]> {
  if (a.mask === false) {
    return [[0, a.size]];
  }

  const mask = a.mask.ravel().data;
  const clumps: Array<[number, number]> = [];
  let start: number | null = null;

  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] && start === null) {
      start = i;
    } else if (mask[i] && start !== null) {
      clumps.push([start, i]);
      start = null;
    }
  }
  if (start !== null) {
    clumps.push([start, mask.length]);
  }

  return clumps;
}

/**
 * Find contiguous unmasked data in flattened array
 */
export function flatnotmasked_contiguous(a: MaskedArray): Array<[number, number]> | null {
  const clumps = clump_unmasked(a);
  return clumps.length > 0 ? clumps : null;
}

/**
 * Find edges of unmasked data in flattened array
 */
export function flatnotmasked_edges(a: MaskedArray): [number, number] | null {
  if (a.mask === false) {
    return [0, a.size - 1];
  }

  const mask = a.mask.ravel().data;
  let first: number | null = null;
  let last: number | null = null;

  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) {
      if (first === null) first = i;
      last = i;
    }
  }

  if (first === null) return null;
  return [first, last!];
}

/**
 * Find contiguous unmasked data
 */
export function notmasked_contiguous(
  a: MaskedArray,
  axis?: number
): Array<[number, number]> | null {
  if (axis !== undefined) {
    throw new Error('axis parameter not yet implemented');
  }
  return flatnotmasked_contiguous(a);
}

/**
 * Find edges of unmasked data
 */
export function notmasked_edges(a: MaskedArray, axis?: number): [number, number] | null {
  if (axis !== undefined) {
    throw new Error('axis parameter not yet implemented');
  }
  return flatnotmasked_edges(a);
}

// ============================================================================
// Row/Column Masking
// ============================================================================

/**
 * Mask rows of a 2D array that contain masked values
 */
export function mask_rows(a: MaskedArray, _axis?: number): MaskedArray {
  if (a.ndim !== 2) {
    throw new Error('mask_rows requires a 2D array');
  }
  if (a.mask === false) return a.copy();

  const nrows = a.shape[0] as number;
  const ncols = a.shape[1] as number;
  const mask = a.mask.data;
  const newMask = creation.zeros([...a.shape], 'bool') as NDArray;
  const newMaskData = newMask.data;

  for (let i = 0; i < nrows; i++) {
    let rowHasMasked = false;
    for (let j = 0; j < ncols; j++) {
      if (mask[i * ncols + j]) {
        rowHasMasked = true;
        break;
      }
    }
    if (rowHasMasked) {
      for (let j = 0; j < ncols; j++) {
        (newMaskData as Uint8Array)[i * ncols + j] = 1;
      }
    }
  }

  return new MaskedArray(a.data, { mask: newMask, fill_value: a.fill_value });
}

/**
 * Mask columns of a 2D array that contain masked values
 */
export function mask_cols(a: MaskedArray, _axis?: number): MaskedArray {
  if (a.ndim !== 2) {
    throw new Error('mask_cols requires a 2D array');
  }
  if (a.mask === false) return a.copy();

  const nrows = a.shape[0] as number;
  const ncols = a.shape[1] as number;
  const mask = a.mask.data;
  const newMask = creation.zeros([...a.shape], 'bool') as NDArray;
  const newMaskData = newMask.data;

  for (let j = 0; j < ncols; j++) {
    let colHasMasked = false;
    for (let i = 0; i < nrows; i++) {
      if (mask[i * ncols + j]) {
        colHasMasked = true;
        break;
      }
    }
    if (colHasMasked) {
      for (let i = 0; i < nrows; i++) {
        (newMaskData as Uint8Array)[i * ncols + j] = 1;
      }
    }
  }

  return new MaskedArray(a.data, { mask: newMask, fill_value: a.fill_value });
}

/**
 * Mask rows and columns that contain masked values
 */
export function mask_rowcols(a: MaskedArray, _axis?: number): MaskedArray {
  if (a.ndim !== 2) {
    throw new Error('mask_rowcols requires a 2D array');
  }

  // First mask rows, then mask cols of the result
  const rowMasked = mask_rows(a);
  return mask_cols(rowMasked);
}

/**
 * Compress rows with masked values
 */
export function compress_rows(a: MaskedArray): NDArray {
  if (a.ndim !== 2) {
    throw new Error('compress_rows requires a 2D array');
  }
  if (a.mask === false) return a.data.copy();

  const nrows = a.shape[0] as number;
  const ncols = a.shape[1] as number;
  const mask = a.mask.data;
  const validRows: number[] = [];

  for (let i = 0; i < nrows; i++) {
    let rowValid = true;
    for (let j = 0; j < ncols; j++) {
      if (mask[i * ncols + j]) {
        rowValid = false;
        break;
      }
    }
    if (rowValid) validRows.push(i);
  }

  if (validRows.length === 0) {
    return creation.empty([0, ncols], a.dtype as DType) as NDArray;
  }

  const result = creation.zeros([validRows.length, ncols], a.dtype as DType) as NDArray;
  const srcData = a.data.data;
  const dstData = result.data;

  for (let i = 0; i < validRows.length; i++) {
    const srcRow = validRows[i]!;
    for (let j = 0; j < ncols; j++) {
      (dstData as Float64Array)[i * ncols + j] = (srcData as Float64Array)[
        srcRow * ncols + j
      ] as number;
    }
  }

  return result;
}

/**
 * Compress columns with masked values
 */
export function compress_cols(a: MaskedArray): NDArray {
  if (a.ndim !== 2) {
    throw new Error('compress_cols requires a 2D array');
  }
  if (a.mask === false) return a.data.copy();

  const nrows = a.shape[0] as number;
  const ncols = a.shape[1] as number;
  const mask = a.mask.data;
  const validCols: number[] = [];

  for (let j = 0; j < ncols; j++) {
    let colValid = true;
    for (let i = 0; i < nrows; i++) {
      if (mask[i * ncols + j]) {
        colValid = false;
        break;
      }
    }
    if (colValid) validCols.push(j);
  }

  if (validCols.length === 0) {
    return creation.empty([nrows, 0], a.dtype as DType) as NDArray;
  }

  const result = creation.zeros([nrows, validCols.length], a.dtype as DType) as NDArray;
  const srcData = a.data.data;
  const dstData = result.data;

  for (let i = 0; i < nrows; i++) {
    for (let jNew = 0; jNew < validCols.length; jNew++) {
      const jOld = validCols[jNew]!;
      (dstData as Float64Array)[i * validCols.length + jNew] = (srcData as Float64Array)[
        i * ncols + jOld
      ] as number;
    }
  }

  return result;
}

/**
 * Compress rows and columns with masked values
 */
export function compress_rowcols(a: MaskedArray, axis?: number): NDArray {
  if (axis === 0) return compress_rows(a);
  if (axis === 1) return compress_cols(a);
  // Default: compress both
  return compress_cols(new MaskedArray(compress_rows(a)));
}

/**
 * Compress array along axis
 */
export function compress_nd(a: MaskedArray, _axis?: number | number[]): NDArray {
  // Simplified implementation - just returns compressed data
  return a.compressed();
}

// ============================================================================
// Counting Functions
// ============================================================================

/**
 * Count non-masked elements
 */
export function count(a: MaskedArray, axis?: number, _keepdims?: boolean): number | NDArray {
  return a.count(axis);
}

/**
 * Count masked elements
 */
export function count_masked(a: MaskedArray, axis?: number): number | NDArray {
  if (a.mask === false) {
    if (axis === undefined) return 0;
    const newShape = [...a.shape];
    newShape.splice(axis, 1);
    return creation.zeros(newShape.length === 0 ? [1] : newShape, 'int64') as NDArray;
  }

  if (axis === undefined) {
    const maskData = a.mask.data;
    let count = 0;
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i]) count++;
    }
    return count;
  }

  // Sum mask along axis
  return a.mask.sum(axis) as NDArray;
}

// ============================================================================
// Reduction Operations (function form)
// ============================================================================

export function sum(
  a: MaskedArray,
  axis?: number,
  _dtype?: DType,
  keepdims?: boolean
): number | MaskedArray {
  return a.sum(axis, keepdims);
}

export function prod(
  a: MaskedArray,
  axis?: number,
  _dtype?: DType,
  _keepdims?: boolean
): number | MaskedArray {
  return a.prod(axis);
}

export function mean(
  a: MaskedArray,
  axis?: number,
  _dtype?: DType,
  _keepdims?: boolean
): number | MaskedArray {
  return a.mean(axis);
}

export function var_(
  a: MaskedArray,
  axis?: number,
  _dtype?: DType,
  ddof?: number,
  keepdims?: boolean
): number | MaskedArray {
  return a.var(axis, ddof, keepdims);
}

// Alias
export { var_ as variance };

export function std(
  a: MaskedArray,
  axis?: number,
  _dtype?: DType,
  ddof?: number,
  _keepdims?: boolean
): number | MaskedArray {
  return a.std(axis, ddof);
}

export function min(a: MaskedArray, axis?: number, _keepdims?: boolean): number | MaskedArray {
  return a.min(axis);
}

// Alias
export { min as amin };

export function max(a: MaskedArray, axis?: number, _keepdims?: boolean): number | MaskedArray {
  return a.max(axis);
}

// Alias
export { max as amax };

export function ptp(a: MaskedArray, axis?: number, _keepdims?: boolean): number | MaskedArray {
  return a.ptp(axis);
}

export function argmin(a: MaskedArray, axis?: number): number | NDArray {
  return a.argmin(axis);
}

export function argmax(a: MaskedArray, axis?: number): number | NDArray {
  return a.argmax(axis);
}

export function all(a: MaskedArray, axis?: number, _keepdims?: boolean): boolean | MaskedArray {
  return a.all(axis);
}

// Alias
export { all as alltrue };

export function any(a: MaskedArray, axis?: number, _keepdims?: boolean): boolean | MaskedArray {
  return a.any(axis);
}

// Alias
export { any as sometrue };

export function cumsum(a: MaskedArray, axis?: number, _dtype?: DType): MaskedArray {
  return a.cumsum(axis);
}

export function cumprod(a: MaskedArray, axis?: number, _dtype?: DType): MaskedArray {
  return a.cumprod(axis);
}

// ============================================================================
// Comparison Functions
// ============================================================================

/**
 * Check if two arrays are equal element-wise (ignoring masked values)
 */
export function allequal(a: MaskedArray, b: MaskedArray, fill_value?: boolean): boolean {
  const fv = fill_value ?? true;

  // Get unmasked values from both
  const aData = a.data.data;
  const bData = b.data.data;
  const aMask = a.mask === false ? null : a.mask.data;
  const bMask = b.mask === false ? null : b.mask.data;

  if (a.size !== b.size) return false;

  for (let i = 0; i < a.size; i++) {
    const aIsMasked = aMask ? aMask[i] : false;
    const bIsMasked = bMask ? bMask[i] : false;

    if (aIsMasked && bIsMasked) continue; // Both masked, skip
    if (aIsMasked || bIsMasked) {
      if (!fv) return false; // One masked, one not
      continue;
    }
    if (aData[i] !== bData[i]) return false;
  }

  return true;
}

/**
 * Check if two arrays are close (ignoring masked values)
 */
export function allclose(
  a: MaskedArray,
  b: MaskedArray,
  rtol?: number,
  atol?: number,
  masked_equal?: boolean
): boolean {
  const rt = rtol ?? 1e-5;
  const at = atol ?? 1e-8;
  const me = masked_equal ?? true;

  const aData = a.data.data;
  const bData = b.data.data;
  const aMask = a.mask === false ? null : a.mask.data;
  const bMask = b.mask === false ? null : b.mask.data;

  if (a.size !== b.size) return false;

  for (let i = 0; i < a.size; i++) {
    const aIsMasked = aMask ? aMask[i] : false;
    const bIsMasked = bMask ? bMask[i] : false;

    if (aIsMasked && bIsMasked) continue;
    if (aIsMasked || bIsMasked) {
      if (!me) return false;
      continue;
    }

    const av = Number(aData[i]);
    const bv = Number(bData[i]);
    if (Math.abs(av - bv) > at + rt * Math.abs(bv)) {
      return false;
    }
  }

  return true;
}

// ============================================================================
// Arithmetic Operations (function form)
// ============================================================================

export function add(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.add(b);
}

export function subtract(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.subtract(b);
}

export function multiply(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.multiply(b);
}

export function divide(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.divide(b);
}

export function true_divide(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.divide(b);
}

export function floor_divide(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.floor_divide(b);
}

export function mod(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.mod(b);
}

export function remainder(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.mod(b);
}

export function power(a: MaskedArray, b: MaskedArray | NDArray | number): MaskedArray {
  return a.power(b);
}

export function negative(a: MaskedArray): MaskedArray {
  return a.negative();
}

export function absolute(a: MaskedArray): MaskedArray {
  return a.abs();
}

// Alias
export { absolute as abs };

export function sqrt(a: MaskedArray): MaskedArray {
  return a.sqrt();
}

// ============================================================================
// Shape Operations (function form)
// ============================================================================

export function reshape(a: MaskedArray, newshape: number[]): MaskedArray {
  return a.reshape(newshape);
}

export function ravel(a: MaskedArray): MaskedArray {
  return a.ravel();
}

export function transpose(a: MaskedArray, axes?: number[]): MaskedArray {
  return a.transpose(axes);
}

export function swapaxes(a: MaskedArray, axis1: number, axis2: number): MaskedArray {
  return a.swapaxes(axis1, axis2);
}

export function squeeze(a: MaskedArray, axis?: number): MaskedArray {
  return a.squeeze(axis);
}

export function expand_dims(a: MaskedArray, axis: number): MaskedArray {
  const data = a.data;
  // Insert dimension at axis
  const newShape = [...data.shape];
  newShape.splice(axis < 0 ? data.ndim + 1 + axis : axis, 0, 1);
  return a.reshape(newShape);
}

// ============================================================================
// Concatenation Functions
// ============================================================================

/**
 * Concatenate masked arrays along an axis
 */
export function concatenate(arrays: MaskedArray[], axis: number = 0): MaskedArray {
  const datas = arrays.map((a) => a.data);
  const resultData = creation.concatenate(datas, axis);

  // Concatenate masks
  const masks = arrays.map((a) =>
    a.mask === false ? (creation.zeros(a.shape as number[], 'bool') as NDArray) : a.mask
  );
  const resultMask = creation.concatenate(masks, axis);

  // Check if all masks are false
  let anyMasked = false;
  const maskData = resultMask.data;
  for (let i = 0; i < maskData.length; i++) {
    if (maskData[i]) {
      anyMasked = true;
      break;
    }
  }

  return new MaskedArray(resultData, {
    mask: anyMasked ? resultMask : false,
    fill_value: arrays[0]?.fill_value,
  });
}

export function vstack(arrays: MaskedArray[]): MaskedArray {
  return concatenate(
    arrays.map((a) => (a.ndim === 1 ? a.reshape([1, ...a.shape]) : a)),
    0
  );
}

export function hstack(arrays: MaskedArray[]): MaskedArray {
  if (arrays.length === 0) {
    throw new Error('hstack requires at least one array');
  }
  if (arrays[0]!.ndim === 1) {
    return concatenate(arrays, 0);
  }
  return concatenate(arrays, 1);
}

export function dstack(arrays: MaskedArray[]): MaskedArray {
  // Stack along third axis
  const expanded = arrays.map((a) => {
    if (a.ndim === 1) return a.reshape([1, a.size, 1]);
    if (a.ndim === 2) return a.reshape([...a.shape, 1]);
    return a;
  });
  return concatenate(expanded, 2);
}

export function stack(arrays: MaskedArray[], axis: number = 0): MaskedArray {
  // Add new axis to each array then concatenate
  const expanded = arrays.map((a) => expand_dims(a, axis));
  return concatenate(expanded, axis);
}

export function column_stack(arrays: MaskedArray[]): MaskedArray {
  // Stack 1D arrays as columns
  const cols = arrays.map((a) => (a.ndim === 1 ? a.reshape([a.size, 1]) : a));
  return concatenate(cols, 1);
}

export function row_stack(arrays: MaskedArray[]): MaskedArray {
  return vstack(arrays);
}

// ============================================================================
// Statistics
// ============================================================================

/**
 * Anomalies (deviations from mean)
 */
export function anom(a: MaskedArray, axis?: number): MaskedArray {
  const meanVal = a.mean(axis, true) as MaskedArray;
  return a.subtract(meanVal);
}

// Alias
export { anom as anomalies };

/**
 * Covariance matrix
 */
export function cov(
  a: MaskedArray,
  y?: MaskedArray,
  rowvar?: boolean,
  bias?: boolean,
  ddof?: number
): MaskedArray {
  // Simplified implementation - compute on filled data
  const filled = a.filled(a.mean() as number);
  const result = creation.cov(
    filled,
    y ? y.filled(y.mean() as number) : undefined,
    rowvar,
    bias,
    ddof
  );
  return new MaskedArray(result);
}

/**
 * Correlation coefficients
 */
export function corrcoef(a: MaskedArray, y?: MaskedArray, rowvar?: boolean): MaskedArray {
  const filled = a.filled(a.mean() as number);
  const result = creation.corrcoef(filled, y ? y.filled(y.mean() as number) : undefined, rowvar);
  return new MaskedArray(result);
}

// ============================================================================
// Other Operations
// ============================================================================

/**
 * Median (with mask support)
 */
export function median(a: MaskedArray, axis?: number, _keepdims?: boolean): number | MaskedArray {
  if (axis !== undefined) {
    throw new Error('axis parameter not yet implemented for median');
  }

  const values = a.compressed().data;
  if (values.length === 0) return NaN;

  // Convert typed array to number array for sorting
  const numValues: number[] = [];
  for (let i = 0; i < values.length; i++) {
    numValues.push(Number(values[i]));
  }
  const sorted = numValues.sort((x, y) => x - y);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
  }
  return sorted[mid] as number;
}

/**
 * Sort masked array
 */
export function sort(a: MaskedArray, axis?: number, _kind?: string): MaskedArray {
  // Fill masked values with max value, sort, then restore mask
  const fv = maximum_fill_value(a) as number;
  const filled = a.filled(fv);
  const sorted = filled.sort(axis) as NDArray;

  // Recreate mask (masked values will be at the end after sorting)
  if (a.mask === false) {
    return new MaskedArray(sorted, { fill_value: a.fill_value });
  }

  // Count masked values
  const maskCount = count_masked(a) as number;
  if (maskCount === 0) {
    return new MaskedArray(sorted, { fill_value: a.fill_value });
  }

  // Create new mask with masked values at end
  const newMask = creation.zeros(a.shape as number[], 'bool') as NDArray;
  const newMaskData = newMask.data;
  for (let i = a.size - maskCount; i < a.size; i++) {
    (newMaskData as Uint8Array)[i] = 1;
  }

  return new MaskedArray(sorted, { mask: newMask, fill_value: a.fill_value });
}

/**
 * Argsort with mask support
 */
export function argsort(a: MaskedArray, axis?: number, _kind?: string): NDArray {
  const fv = maximum_fill_value(a) as number;
  const filled = a.filled(fv);
  return filled.argsort(axis) as NDArray;
}

/**
 * Where with mask support
 */
export function where(
  condition: NDArray | MaskedArray,
  x?: MaskedArray | NDArray | number,
  y?: MaskedArray | NDArray | number
): MaskedArray | NDArray[] {
  const cond = condition instanceof MaskedArray ? condition.data : condition;

  if (x === undefined && y === undefined) {
    // Return indices of non-zero elements
    return creation.where(cond) as NDArray[];
  }

  // Convert x and y to NDArray if they are numbers
  const xArr = x instanceof MaskedArray ? x.data : x instanceof NDArray ? x : undefined;
  const yArr = y instanceof MaskedArray ? y.data : y instanceof NDArray ? y : undefined;
  const result = creation.where(cond, xArr, yArr);

  // Combine masks if any
  let resultMask: NDArray | false = false;
  if (x instanceof MaskedArray || y instanceof MaskedArray || condition instanceof MaskedArray) {
    const masks: NDArray[] = [];
    if (condition instanceof MaskedArray && condition.mask !== false) {
      masks.push(condition.mask);
    }
    if (x instanceof MaskedArray && x.mask !== false) {
      masks.push(x.mask);
    }
    if (y instanceof MaskedArray && y.mask !== false) {
      masks.push(y.mask);
    }

    if (masks.length > 0) {
      resultMask = masks[0]!;
      for (let i = 1; i < masks.length; i++) {
        resultMask = NDArray._fromStorage(
          logicOps.logical_or(resultMask.storage, masks[i]!.storage)
        );
      }
    }
  }

  return new MaskedArray(result as NDArray, { mask: resultMask });
}

/**
 * Return indices of non-masked, non-zero elements
 */
export function nonzero(a: MaskedArray): NDArray[] {
  const filled = a.filled(0);
  return creation.nonzero(filled);
}

/**
 * Harden mask of array
 */
export function harden_mask(a: MaskedArray): MaskedArray {
  return a.harden_mask();
}

/**
 * Soften mask of array
 */
export function soften_mask(a: MaskedArray): MaskedArray {
  return a.soften_mask();
}

/**
 * Copy a masked array
 */
export function copy(a: MaskedArray): MaskedArray {
  return a.copy();
}

/**
 * Return array dimensions
 */
export function shape(a: MaskedArray): readonly number[] {
  return a.shape;
}

/**
 * Return number of dimensions
 */
export function ndim(a: MaskedArray): number {
  return a.ndim;
}

/**
 * Return number of elements
 */
export function size(a: MaskedArray): number {
  return a.size;
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

/** Apply unary operation with mask propagation */
function _unaryMaskedOp(
  a: MaskedArray | NDArray | number[],
  op: (storage: ArrayStorage) => ArrayStorage
): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const result = NDArray._fromStorage(op(arr.data.storage));
  return new MaskedArray(result, {
    mask: arr instanceof MaskedArray ? arr.mask : false,
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

/** Apply binary operation with mask propagation */
function _binaryMaskedOp(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number,
  op: (x: ArrayStorage, y: ArrayStorage | number) => ArrayStorage
): MaskedArray {
  const arrA = a instanceof MaskedArray ? a : asarray(a);
  const isScalarB = typeof b === 'number';
  const arrB = isScalarB ? null : b instanceof MaskedArray ? b : asarray(b);

  const result = isScalarB
    ? NDArray._fromStorage(op(arrA.data.storage, b))
    : NDArray._fromStorage(op(arrA.data.storage, arrB!.data.storage));

  // Combine masks
  let resultMask: NDArray | false = false;
  const maskA = arrA instanceof MaskedArray ? arrA.mask : false;
  const maskB = arrB instanceof MaskedArray ? arrB.mask : false;

  if (maskA !== false && maskB !== false) {
    resultMask = NDArray._fromStorage(logicOps.logical_or(maskA.storage, maskB.storage));
  } else if (maskA !== false) {
    resultMask = maskA;
  } else if (maskB !== false) {
    resultMask = maskB;
  }

  return new MaskedArray(result, {
    mask: resultMask,
    fill_value: arrA instanceof MaskedArray ? arrA.fill_value : undefined,
  });
}

export function sin(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.sin);
}

export function cos(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.cos);
}

export function tan(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.tan);
}

export function arcsin(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.arcsin);
}

export function arccos(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.arccos);
}

export function arctan(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, trigOps.arctan);
}

export function arctan2(
  y: MaskedArray | NDArray | number[],
  x: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(y, x, trigOps.arctan2);
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

export function sinh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.sinh);
}

export function cosh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.cosh);
}

export function tanh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.tanh);
}

export function arcsinh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.arcsinh);
}

export function arccosh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.arccosh);
}

export function arctanh(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, hyperbolicOps.arctanh);
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

export function exp(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.exp);
}

export function log(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.log);
}

export function log10(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.log10);
}

export function log2(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.log2);
}

export function exp2(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.exp2);
}

export function expm1(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.expm1);
}

export function log1p(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, exponentialOps.log1p);
}

// ============================================================================
// Rounding Functions
// ============================================================================

export function around(a: MaskedArray | NDArray | number[], decimals: number = 0): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const result = NDArray._fromStorage(roundingOps.around(arr.data.storage, decimals));
  return new MaskedArray(result, {
    mask: arr instanceof MaskedArray ? arr.mask : false,
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

// Alias
export { around as round_ };

export function ceil(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, roundingOps.ceil);
}

export function floor(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, roundingOps.floor);
}

export function trunc(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, roundingOps.trunc);
}

export function rint(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, roundingOps.rint);
}

export function fix(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, roundingOps.fix);
}

// ============================================================================
// Logical Functions
// ============================================================================

export function logical_and(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, logicOps.logical_and);
}

export function logical_or(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, logicOps.logical_or);
}

export function logical_xor(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, logicOps.logical_xor);
}

export function logical_not(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, logicOps.logical_not);
}

// ============================================================================
// Bitwise Functions
// ============================================================================

export function bitwise_and(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, bitwiseOps.bitwise_and);
}

export function bitwise_or(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, bitwiseOps.bitwise_or);
}

export function bitwise_xor(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, bitwiseOps.bitwise_xor);
}

export function left_shift(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, bitwiseOps.left_shift);
}

export function right_shift(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, bitwiseOps.right_shift);
}

export function invert(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, bitwiseOps.bitwise_not);
}

// ============================================================================
// Comparison Functions (element-wise)
// ============================================================================

export function equal(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.equal);
}

export function not_equal(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.notEqual);
}

export function greater(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.greater);
}

export function greater_equal(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.greaterEqual);
}

export function less(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.less);
}

export function less_equal(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, comparisonOps.lessEqual);
}

// ============================================================================
// Additional Math Functions
// ============================================================================

export function fabs(a: MaskedArray | NDArray | number[]): MaskedArray {
  return _unaryMaskedOp(a, arithmeticOps.absolute);
}

export function hypot(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, trigOps.hypot);
}

export function angle(a: MaskedArray | NDArray | number[]): MaskedArray {
  // For real arrays, angle is 0 for positive, pi for negative
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const data = arr.data.data;
  const resultData = creation.zeros([...arr.shape], 'float64') as NDArray;
  const result = resultData.data as Float64Array;
  for (let i = 0; i < data.length; i++) {
    result[i] = Number(data[i]) < 0 ? Math.PI : 0;
  }
  return new MaskedArray(resultData, {
    mask: arr instanceof MaskedArray ? arr.mask : false,
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

export function fmod(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[] | number
): MaskedArray {
  return _binaryMaskedOp(a, b, arithmeticOps.mod);
}

export function maximum(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, arithmeticOps.maximum);
}

export function minimum(
  a: MaskedArray | NDArray | number[],
  b: MaskedArray | NDArray | number[]
): MaskedArray {
  return _binaryMaskedOp(a, b, arithmeticOps.minimum);
}

export function clip(
  a: MaskedArray | NDArray | number[],
  a_min: number | null,
  a_max: number | null
): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const result = creation.clip(arr.data, a_min, a_max);
  return new MaskedArray(result, {
    mask: arr instanceof MaskedArray ? arr.mask : false,
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

export function conjugate(a: MaskedArray | NDArray | number[]): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  // For real arrays, conjugate is identity
  return new MaskedArray(arr.data.copy(), {
    mask: arr instanceof MaskedArray ? arr.mask : false,
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

// Alias
export { conjugate as conj };

// ============================================================================
// Array Manipulation Functions
// ============================================================================

export function append(
  arr: MaskedArray,
  values: MaskedArray | NDArray | number[],
  axis?: number
): MaskedArray {
  const valuesArr = values instanceof MaskedArray ? values : asarray(values);
  if (axis === undefined) {
    // Flatten both and concatenate
    return concatenate([arr.ravel(), valuesArr.ravel()], 0);
  }
  return concatenate([arr, valuesArr], axis);
}

export function diag(a: MaskedArray | NDArray | number[], k: number = 0): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const result = creation.diag(arr.data, k);
  // For diag, mask handling is complex - simplified version
  return new MaskedArray(result, {
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

export function diagflat(a: MaskedArray | NDArray | number[], k: number = 0): MaskedArray {
  const arr = a instanceof MaskedArray ? a : asarray(a);
  const flat = arr.data.ravel();
  const result = creation.diag(flat, k);
  return new MaskedArray(result, {
    fill_value: arr instanceof MaskedArray ? arr.fill_value : undefined,
  });
}

export function diagonal(
  a: MaskedArray,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): MaskedArray {
  const result = creation.diagonal(a.data, offset, axis1, axis2);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function trace(a: MaskedArray, offset: number = 0): number {
  const diagonalArr = diagonal(a, offset);
  return diagonalArr.sum() as number;
}

export function diff(a: MaskedArray, n: number = 1, axis: number = -1): MaskedArray {
  let result = a;
  for (let i = 0; i < n; i++) {
    const data = creation.diff(result.data, 1, axis);

    // Compute mask for diff: diff[i] is masked if either a[i] or a[i+1] is masked
    let resultMask: NDArray | false = false;
    if (result.mask !== false) {
      // For 1D or along axis: mask[i] || mask[i+1]
      const mask = result.mask;
      const size = data.size;
      const maskData = new Uint8Array(size);

      // Handle axis normalization
      const ndim = result.ndim;
      const actualAxis = axis < 0 ? ndim + axis : axis;

      if (ndim === 1) {
        // Simple 1D case
        const srcMask = mask.ravel().data;
        for (let j = 0; j < size; j++) {
          maskData[j] = srcMask[j] || srcMask[j + 1] ? 1 : 0;
        }
      } else {
        // For multi-dimensional, iterate through in C-order
        const srcMask = mask.ravel().data;
        const srcShape = result.shape;
        const axisLen = srcShape[actualAxis] ?? 1;
        const outerSize = srcShape.slice(0, actualAxis).reduce((a, b) => a * b, 1);
        const innerSize = srcShape.slice(actualAxis + 1).reduce((a, b) => a * b, 1);

        let outIdx = 0;
        for (let outer = 0; outer < outerSize; outer++) {
          for (let alongAxis = 0; alongAxis < axisLen - 1; alongAxis++) {
            for (let inner = 0; inner < innerSize; inner++) {
              const srcIdx1 = outer * axisLen * innerSize + alongAxis * innerSize + inner;
              const srcIdx2 = outer * axisLen * innerSize + (alongAxis + 1) * innerSize + inner;
              maskData[outIdx++] = srcMask[srcIdx1] || srcMask[srcIdx2] ? 1 : 0;
            }
          }
        }
      }

      // Create mask NDArray using ArrayStorage
      const maskStorage = ArrayStorage.fromData(maskData, [...data.shape] as number[], 'bool');
      resultMask = NDArray._fromStorage(maskStorage);
    }

    result = new MaskedArray(data, { mask: resultMask, fill_value: a.fill_value });
  }
  return result;
}

export function ediff1d(a: MaskedArray): MaskedArray {
  return diff(a.ravel(), 1, 0);
}

export function take(a: MaskedArray, indices: number[] | NDArray, axis?: number): MaskedArray {
  const idx = indices instanceof NDArray ? (indices.tolist() as number[]) : indices;
  const result = creation.take(a.data, idx, axis);
  // Take corresponding mask values too
  let resultMask: NDArray | false = false;
  if (a.mask !== false) {
    resultMask = creation.take(a.mask, idx, axis);
  }
  return new MaskedArray(result, { mask: resultMask, fill_value: a.fill_value });
}

export function put(a: MaskedArray, indices: number[], values: number[]): void {
  creation.put(a.data, indices, creation.array(values));
}

export function putmask(a: MaskedArray, mask: NDArray | boolean[], values: number[]): void {
  const maskArr = mask instanceof NDArray ? mask : creation.array(mask, 'bool');
  const maskData = maskArr.data;
  const aData = a.data.data;
  let vi = 0;
  for (let i = 0; i < maskData.length; i++) {
    if (maskData[i]) {
      (aData as Float64Array)[i] = values[vi % values.length]!;
      vi++;
    }
  }
}

export function repeat(a: MaskedArray, repeats: number, axis?: number): MaskedArray {
  const result = creation.repeat(a.data, repeats, axis);
  let resultMask: NDArray | false = false;
  if (a.mask !== false) {
    resultMask = creation.repeat(a.mask, repeats, axis);
  }
  return new MaskedArray(result, { mask: resultMask, fill_value: a.fill_value });
}

export function resize(a: MaskedArray, new_shape: number[]): MaskedArray {
  const result = creation.resize(a.data, new_shape);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function atleast_1d(a: MaskedArray): MaskedArray {
  if (a.ndim >= 1) return a;
  return a.reshape([1]);
}

export function atleast_2d(a: MaskedArray): MaskedArray {
  if (a.ndim >= 2) return a;
  if (a.ndim === 1) return a.reshape([1, ...a.shape]);
  return a.reshape([1, 1]);
}

export function atleast_3d(a: MaskedArray): MaskedArray {
  if (a.ndim >= 3) return a;
  if (a.ndim === 2) return a.reshape([...a.shape, 1]);
  if (a.ndim === 1) return a.reshape([1, a.shape[0] as number, 1]);
  return a.reshape([1, 1, 1]);
}

export function hsplit(a: MaskedArray, indices_or_sections: number | number[]): MaskedArray[] {
  const results = creation.hsplit(a.data, indices_or_sections);
  return results.map((r) => new MaskedArray(r, { fill_value: a.fill_value }));
}

export function choose(a: MaskedArray | NDArray | number[], choices: MaskedArray[]): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a instanceof NDArray ? a : creation.array(a);
  const choiceArrays = choices.map((c) => c.data);
  const result = creation.choose(arr, choiceArrays);
  return new MaskedArray(result, { fill_value: choices[0]?.fill_value });
}

// ============================================================================
// Linear Algebra Functions
// ============================================================================

export function dot(a: MaskedArray, b: MaskedArray): MaskedArray | number {
  const result = linalgOps.dot(a.data.storage, b.data.storage);
  // Mask handling for dot product is complex - if any element in computation is masked,
  // result could be affected. Simplified: combine input masks
  if (typeof result === 'number' || typeof result === 'bigint') {
    return Number(result);
  }
  const resultMask: NDArray | false = false;
  return new MaskedArray(NDArray._fromStorage(result as ArrayStorage), {
    mask: resultMask,
    fill_value: a.fill_value,
  });
}

export function inner(a: MaskedArray, b: MaskedArray): MaskedArray | number {
  const result = linalgOps.inner(a.data.storage, b.data.storage);
  if (typeof result === 'number' || typeof result === 'bigint') {
    return Number(result);
  }
  return new MaskedArray(NDArray._fromStorage(result as ArrayStorage), {
    fill_value: a.fill_value,
  });
}

export { inner as innerproduct };

export function outer(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = linalgOps.outer(a.data.storage, b.data.storage);
  return new MaskedArray(NDArray._fromStorage(result), { fill_value: a.fill_value });
}

export { outer as outerproduct };

// ============================================================================
// Set Operations
// ============================================================================

export function unique(a: MaskedArray): MaskedArray {
  // Get non-masked unique values
  const compressed = a.compressed();
  const result = creation.unique(compressed) as NDArray;

  // If there are masked values in the input, add one masked entry at the end
  // This matches NumPy's behavior where unique includes masked values as a single entry
  const hasMasked =
    a.mask !== false && (a.mask.ravel().data as Uint8Array).some((v: number) => v !== 0);

  if (hasMasked) {
    // Append a masked entry (with fill_value as the data)
    const dataArr = result.tolist() as number[];
    dataArr.push(a.fill_value as number);
    const newData = creation.array(dataArr);
    // Mask only the last element
    const mask = new Array(dataArr.length).fill(false);
    mask[mask.length - 1] = true;
    return new MaskedArray(newData, { mask, fill_value: a.fill_value });
  }

  return new MaskedArray(result);
}

export function intersect1d(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = creation.intersect1d(a.compressed(), b.compressed());
  return new MaskedArray(result);
}

export function union1d(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = creation.union1d(a.compressed(), b.compressed());
  return new MaskedArray(result);
}

export function setdiff1d(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = creation.setdiff1d(a.compressed(), b.compressed());
  return new MaskedArray(result);
}

export function setxor1d(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = creation.setxor1d(a.compressed(), b.compressed());
  return new MaskedArray(result);
}

export function in1d(a: MaskedArray, b: MaskedArray): MaskedArray {
  const result = creation.in1d(a.data, b.compressed());
  return new MaskedArray(result, { mask: a.mask, fill_value: a.fill_value });
}

export function isin(a: MaskedArray, test_elements: MaskedArray | NDArray | number[]): MaskedArray {
  const testArr =
    test_elements instanceof MaskedArray
      ? test_elements.compressed()
      : test_elements instanceof NDArray
        ? test_elements
        : creation.array(test_elements);
  const result = creation.isin(a.data, testArr);
  return new MaskedArray(result, { mask: a.mask, fill_value: a.fill_value });
}

// ============================================================================
// Statistics - Additional
// ============================================================================

export function average(
  a: MaskedArray,
  axis?: number,
  weights?: MaskedArray | NDArray | number[]
): number | MaskedArray {
  if (weights === undefined) {
    return a.mean(axis);
  }

  // Weighted average
  const w =
    weights instanceof MaskedArray
      ? weights.data
      : weights instanceof NDArray
        ? weights
        : creation.array(weights);

  if (axis === undefined) {
    // Global weighted average
    const values = a.compressed();
    // Simple case - assume weights align with compressed values
    let sum = 0;
    let wsum = 0;
    const vdata = values.data;
    const wdata = w.data;
    const len = Math.min(vdata.length, wdata.length);
    for (let i = 0; i < len; i++) {
      sum += Number(vdata[i]) * Number(wdata[i]);
      wsum += Number(wdata[i]);
    }
    return sum / wsum;
  }

  // Axis-aware weighted average - simplified
  throw new Error('Weighted average with axis not yet fully implemented');
}

// ============================================================================
// Miscellaneous
// ============================================================================

export function apply_along_axis(
  func1d: (arr: NDArray) => NDArray | number,
  axis: number,
  a: MaskedArray
): MaskedArray {
  const result = creation.apply_along_axis(func1d, axis, a.data);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function apply_over_axes(
  func: (arr: NDArray, axis: number) => NDArray,
  a: MaskedArray,
  axes: number[]
): MaskedArray {
  let result = a.data;
  for (const axis of axes) {
    result = func(result, axis);
  }
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function vander(a: MaskedArray, N?: number, increasing: boolean = false): MaskedArray {
  const result = creation.vander(a.compressed(), N, increasing);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function convolve(
  a: MaskedArray,
  v: MaskedArray,
  mode: 'full' | 'same' | 'valid' = 'full'
): MaskedArray {
  const result = creation.convolve(a.filled(0), v.filled(0), mode);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

export function correlate(
  a: MaskedArray,
  v: MaskedArray,
  mode: 'full' | 'same' | 'valid' = 'valid'
): MaskedArray {
  const result = creation.correlate(a.filled(0), v.filled(0), mode);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

// Alias for product
export { prod as product };

// Alias for round (around is the primary)
export { around as round };

// Alias for var (var_ is the primary due to reserved word)
export { var_ as var };

// ============================================================================
// Additional Creation Functions
// ============================================================================

/**
 * Create MaskedArray from a buffer
 */
export function frombuffer(
  buffer: ArrayBuffer | ArrayBufferView,
  dtype: DType = 'float64',
  count: number = -1,
  offset: number = 0
): MaskedArray {
  const result = creation.frombuffer(buffer, dtype, count, offset);
  return new MaskedArray(result);
}

/**
 * Create MaskedArray from a function
 */
export function fromfunction(
  func: (...indices: number[]) => number,
  shape: number[],
  dtype: DType = 'float64'
): MaskedArray {
  const result = creation.fromfunction(func, shape, dtype);
  return new MaskedArray(result);
}

/**
 * Create MaskedArray from flexible-type array (simplified - treats as regular array)
 */
export function fromflex(fxarray: NDArray | number[]): MaskedArray {
  const arr = fxarray instanceof NDArray ? fxarray : creation.array(fxarray);
  return new MaskedArray(arr);
}

/**
 * Compress array along axis, removing masked elements
 */
export function compress(
  condition: NDArray | boolean[],
  a: MaskedArray,
  axis?: number
): MaskedArray {
  const condArr = condition instanceof NDArray ? condition : creation.array(condition, 'bool');
  const result = creation.compress(condArr, a.data, axis);
  return new MaskedArray(result, { fill_value: a.fill_value });
}

/**
 * Return the memory IDs of data and mask arrays
 */
export function ids(a: MaskedArray): [number, number] {
  // In JavaScript we don't have memory addresses, so return unique identifiers
  // This is a simplified implementation
  const dataId = a.data.data.byteOffset;
  const maskId = a.mask !== false ? a.mask.data.byteOffset : 0;
  return [dataId, maskId];
}

/**
 * Multidimensional index iterator for MaskedArray
 */
export function* ndenumerate(
  a: MaskedArray
): Generator<[number[], number | bigint | typeof masked]> {
  const shape = a.shape;
  const ndim = shape.length;
  const size = a.size;

  for (let flatIdx = 0; flatIdx < size; flatIdx++) {
    // Convert flat index to multi-dimensional index
    const idx: number[] = [];
    let remaining = flatIdx;
    for (let d = ndim - 1; d >= 0; d--) {
      const dimSize = shape[d] as number;
      idx.unshift(remaining % dimSize);
      remaining = Math.floor(remaining / dimSize);
    }

    // Get value (or masked)
    const value = a.item(...idx);
    yield [idx, value];
  }
}

/**
 * Polynomial fit with masked values ignored
 */
export function polyfit(
  x: MaskedArray,
  y: MaskedArray,
  deg: number,
  _rcond?: number,
  _full?: boolean,
  _w?: MaskedArray,
  _cov?: boolean
): NDArray {
  // Use compressed (non-masked) values for fitting
  const xComp = x.compressed();
  const yComp = y.compressed();

  // Simple polynomial fit using least squares
  // This is a simplified implementation
  const result = creation.polyfit(xComp, yComp, deg);
  return result;
}

/**
 * Flatten a structured array (simplified - just returns flattened data)
 */
export function flatten_structured_array(a: MaskedArray): MaskedArray {
  return a.ravel();
}

// ============================================================================
// Convenience exports for compatibility
// ============================================================================

// Additional aliases that NumPy has
export const core = {
  MaskedArray,
  masked,
  nomask,
};
