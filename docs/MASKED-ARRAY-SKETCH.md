# MaskedArray Implementation Sketch

A design sketch for implementing NumPy's masked array (`np.ma`) functionality.

## Core Class: MaskedArray

```typescript
// src/core/masked-array.ts

import { NDArray, array, zeros, ones, logical_or, logical_and, logical_not } from './ndarray';
import type { DType } from './dtype';

/**
 * MaskedArray - Array with masked (invalid) elements
 *
 * Similar to NumPy's np.ma.MaskedArray. Wraps an NDArray with a boolean mask
 * indicating which elements should be ignored in computations.
 */
export class MaskedArray {
  private _data: NDArray;      // Underlying data
  private _mask: NDArray;      // Boolean mask (true = masked/invalid)
  private _fill_value: number; // Default fill value for masked positions
  private _hardmask: boolean;  // If true, mask cannot be unset

  constructor(
    data: NDArray | number[] | number[][],
    options?: {
      mask?: NDArray | boolean[] | boolean;
      fill_value?: number;
      dtype?: DType;
      copy?: boolean;
      hardmask?: boolean;
    }
  ) {
    const opts = options ?? {};

    // Convert data to NDArray if needed
    this._data = data instanceof NDArray
      ? (opts.copy ? data.copy() : data)
      : array(data, opts.dtype);

    // Initialize mask
    if (opts.mask === undefined || opts.mask === false) {
      // No mask - all elements valid (mask is all false)
      this._mask = zeros(this._data.shape, 'bool') as NDArray;
    } else if (opts.mask === true) {
      // All masked
      this._mask = ones(this._data.shape, 'bool') as NDArray;
    } else if (opts.mask instanceof NDArray) {
      this._mask = opts.copy ? opts.mask.copy() : opts.mask;
    } else {
      this._mask = array(opts.mask, 'bool');
    }

    // Set fill value (NumPy uses dtype-specific defaults)
    this._fill_value = opts.fill_value ?? this._getDefaultFillValue();
    this._hardmask = opts.hardmask ?? false;
  }

  // ============ Properties ============

  /** The underlying data array (includes masked values) */
  get data(): NDArray {
    return this._data;
  }

  /** Boolean mask array (true = masked/invalid) */
  get mask(): NDArray {
    return this._mask;
  }

  /** Fill value for masked positions */
  get fill_value(): number {
    return this._fill_value;
  }

  set fill_value(value: number) {
    this._fill_value = value;
  }

  /** Whether mask is hard (cannot unset masked values) */
  get hardmask(): boolean {
    return this._hardmask;
  }

  // Forward NDArray properties
  get shape(): readonly number[] { return this._data.shape; }
  get ndim(): number { return this._data.ndim; }
  get size(): number { return this._data.size; }
  get dtype(): string { return this._data.dtype; }
  get T(): MaskedArray { return this.transpose(); }

  // ============ Core Methods ============

  /**
   * Return data with masked values replaced by fill_value
   */
  filled(fill_value?: number): NDArray {
    const fv = fill_value ?? this._fill_value;
    const result = this._data.copy();
    // Where mask is true, replace with fill_value
    // Implementation: iterate and replace masked positions
    const maskData = this._mask.data;
    const resultData = result.data;
    for (let i = 0; i < this.size; i++) {
      if (maskData[i]) {
        resultData[i] = fv;
      }
    }
    return result;
  }

  /**
   * Return 1-D array of non-masked data
   */
  compressed(): NDArray {
    // Count non-masked elements
    const maskData = this._mask.data;
    let count = 0;
    for (let i = 0; i < this.size; i++) {
      if (!maskData[i]) count++;
    }

    // Extract non-masked values
    const result = zeros([count], this.dtype) as NDArray;
    const srcData = this._data.data;
    const dstData = result.data;
    let j = 0;
    for (let i = 0; i < this.size; i++) {
      if (!maskData[i]) {
        dstData[j++] = srcData[i];
      }
    }
    return result;
  }

  /**
   * Count non-masked elements
   */
  count(axis?: number): number | NDArray {
    if (axis === undefined) {
      const maskData = this._mask.data;
      let count = 0;
      for (let i = 0; i < this.size; i++) {
        if (!maskData[i]) count++;
      }
      return count;
    }
    // Axis-aware counting would use reduction ops
    // return sum(logical_not(this._mask), axis);
    throw new Error('axis parameter not yet implemented');
  }

  // ============ Arithmetic Operations ============
  // These wrap existing NDArray ops and propagate masks

  add(other: MaskedArray | NDArray | number): MaskedArray {
    return ma_binary_op(this, other, (a, b) => a.add(b));
  }

  subtract(other: MaskedArray | NDArray | number): MaskedArray {
    return ma_binary_op(this, other, (a, b) => a.subtract(b));
  }

  multiply(other: MaskedArray | NDArray | number): MaskedArray {
    return ma_binary_op(this, other, (a, b) => a.multiply(b));
  }

  divide(other: MaskedArray | NDArray | number): MaskedArray {
    return ma_binary_op(this, other, (a, b) => a.divide(b));
  }

  // ... similar for all arithmetic ops

  // ============ Reductions ============
  // These skip masked values (similar to nan* functions)

  sum(axis?: number, keepdims?: boolean): number | MaskedArray {
    // Use filled(0) so masked values contribute 0 to sum
    // Or implement mask-aware reduction
    return ma_reduction(this, 'sum', axis, keepdims);
  }

  mean(axis?: number, keepdims?: boolean): number | MaskedArray {
    // sum of non-masked / count of non-masked
    return ma_reduction(this, 'mean', axis, keepdims);
  }

  std(axis?: number, keepdims?: boolean, ddof?: number): number | MaskedArray {
    return ma_reduction(this, 'std', axis, keepdims, { ddof });
  }

  var(axis?: number, keepdims?: boolean, ddof?: number): number | MaskedArray {
    return ma_reduction(this, 'var', axis, keepdims, { ddof });
  }

  min(axis?: number, keepdims?: boolean): number | MaskedArray {
    return ma_reduction(this, 'min', axis, keepdims);
  }

  max(axis?: number, keepdims?: boolean): number | MaskedArray {
    return ma_reduction(this, 'max', axis, keepdims);
  }

  // ============ Shape Operations ============

  reshape(newShape: number[]): MaskedArray {
    return new MaskedArray(this._data.reshape(newShape), {
      mask: this._mask.reshape(newShape),
      fill_value: this._fill_value,
    });
  }

  transpose(axes?: number[]): MaskedArray {
    return new MaskedArray(this._data.transpose(axes), {
      mask: this._mask.transpose(axes),
      fill_value: this._fill_value,
    });
  }

  ravel(): MaskedArray {
    return new MaskedArray(this._data.ravel(), {
      mask: this._mask.ravel(),
      fill_value: this._fill_value,
    });
  }

  // ============ Masking Operations ============

  harden_mask(): void {
    this._hardmask = true;
  }

  soften_mask(): void {
    this._hardmask = false;
  }

  // ============ Private Helpers ============

  private _getDefaultFillValue(): number {
    // NumPy uses dtype-specific defaults
    const dtype = this._data.dtype;
    if (dtype.startsWith('float')) return 1e20;
    if (dtype.startsWith('int')) return 999999;
    if (dtype === 'bool') return 1;
    return 1e20;
  }

  // ============ String Representation ============

  toString(): string {
    // Format like NumPy: [1 -- 3 4 --]
    const data = this._data.tolist();
    const mask = this._mask.tolist();
    const formatted = this._formatMasked(data, mask);
    return `MaskedArray(${formatted})`;
  }

  private _formatMasked(data: any, mask: any): string {
    if (Array.isArray(data)) {
      const items = data.map((d, i) => this._formatMasked(d, mask[i]));
      return `[${items.join(', ')}]`;
    }
    return mask ? '--' : String(data);
  }
}

// ============ Helper Functions ============

/**
 * Binary operation with mask propagation
 */
function ma_binary_op(
  a: MaskedArray,
  b: MaskedArray | NDArray | number,
  op: (x: NDArray, y: NDArray | number) => NDArray
): MaskedArray {
  if (b instanceof MaskedArray) {
    // Combine masks: result is masked if either input is masked
    const resultData = op(a.data, b.data);
    const resultMask = logical_or(a.mask, b.mask);
    return new MaskedArray(resultData, { mask: resultMask });
  } else {
    // Scalar or NDArray - only a's mask matters
    const resultData = op(a.data, b instanceof NDArray ? b : b);
    return new MaskedArray(resultData, { mask: a.mask });
  }
}

/**
 * Reduction with mask awareness
 */
function ma_reduction(
  arr: MaskedArray,
  op: 'sum' | 'mean' | 'std' | 'var' | 'min' | 'max',
  axis?: number,
  keepdims?: boolean,
  options?: { ddof?: number }
): number | MaskedArray {
  // For full reduction (no axis), return scalar
  if (axis === undefined) {
    const data = arr.data.data;
    const mask = arr.mask.data;

    // Collect non-masked values
    const values: number[] = [];
    for (let i = 0; i < arr.size; i++) {
      if (!mask[i]) values.push(data[i] as number);
    }

    if (values.length === 0) {
      return NaN; // All masked
    }

    switch (op) {
      case 'sum':
        return values.reduce((a, b) => a + b, 0);
      case 'mean':
        return values.reduce((a, b) => a + b, 0) / values.length;
      case 'min':
        return Math.min(...values);
      case 'max':
        return Math.max(...values);
      case 'var': {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const ddof = options?.ddof ?? 0;
        const sumSq = values.reduce((a, b) => a + (b - mean) ** 2, 0);
        return sumSq / (values.length - ddof);
      }
      case 'std': {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const ddof = options?.ddof ?? 0;
        const sumSq = values.reduce((a, b) => a + (b - mean) ** 2, 0);
        return Math.sqrt(sumSq / (values.length - ddof));
      }
    }
  }

  // Axis-aware reduction would follow similar pattern to existing ops
  throw new Error('Axis-aware masked reduction not yet implemented');
}
```

## Module Functions (np.ma.*)

```typescript
// src/ma/index.ts - Top-level ma.* functions

import { MaskedArray } from './masked-array';
import { NDArray, array as np_array, isnan } from '../core/ndarray';

// ============ Array Creation ============

/** Create a MaskedArray */
export function array(
  data: number[] | number[][] | NDArray,
  options?: { mask?: boolean[] | NDArray; dtype?: DType; fill_value?: number }
): MaskedArray {
  return new MaskedArray(data instanceof NDArray ? data : np_array(data), options);
}

/** Create MaskedArray with all elements masked */
export function masked_all(shape: number[], dtype?: DType): MaskedArray {
  return new MaskedArray(zeros(shape, dtype), { mask: true });
}

/** Create MaskedArray like another array with all masked */
export function masked_all_like(a: NDArray | MaskedArray): MaskedArray {
  const arr = a instanceof MaskedArray ? a.data : a;
  return new MaskedArray(zeros(arr.shape, arr.dtype as DType), { mask: true });
}

// ============ Masking Functions ============

/** Mask where condition is true */
export function masked_where(
  condition: NDArray | boolean[],
  a: NDArray | MaskedArray
): MaskedArray {
  const data = a instanceof MaskedArray ? a.data : a;
  const existingMask = a instanceof MaskedArray ? a.mask : null;

  const condArray = condition instanceof NDArray ? condition : np_array(condition, 'bool');
  const newMask = existingMask
    ? logical_or(existingMask, condArray)
    : condArray;

  return new MaskedArray(data, { mask: newMask });
}

/** Mask where values equal given value */
export function masked_equal(a: NDArray, value: number): MaskedArray {
  const mask = a.equal(value);
  return new MaskedArray(a, { mask });
}

/** Mask where values are greater than given value */
export function masked_greater(a: NDArray, value: number): MaskedArray {
  const mask = a.greater(value);
  return new MaskedArray(a, { mask });
}

/** Mask where values are less than given value */
export function masked_less(a: NDArray, value: number): MaskedArray {
  const mask = a.less(value);
  return new MaskedArray(a, { mask });
}

/** Mask where values are inside interval [v1, v2] */
export function masked_inside(a: NDArray, v1: number, v2: number): MaskedArray {
  const mask = logical_and(a.greater_equal(v1), a.less_equal(v2));
  return new MaskedArray(a, { mask });
}

/** Mask where values are outside interval [v1, v2] */
export function masked_outside(a: NDArray, v1: number, v2: number): MaskedArray {
  const mask = logical_or(a.less(v1), a.greater(v2));
  return new MaskedArray(a, { mask });
}

/** Mask invalid values (NaN, Inf) */
export function masked_invalid(a: NDArray): MaskedArray {
  const mask = logical_or(isnan(a), isinf(a));
  return new MaskedArray(a, { mask });
}

// ============ Mask Utilities ============

/** Get the mask of a masked array */
export function getmask(a: MaskedArray | NDArray): NDArray | boolean {
  if (a instanceof MaskedArray) {
    return a.mask;
  }
  return false; // Regular arrays have no mask
}

/** Get the data of a masked array */
export function getdata(a: MaskedArray | NDArray): NDArray {
  return a instanceof MaskedArray ? a.data : a;
}

/** Test whether input is a masked array */
export function isMaskedArray(a: any): a is MaskedArray {
  return a instanceof MaskedArray;
}

/** Check if any element is masked */
export function is_masked(a: MaskedArray | NDArray): boolean {
  if (!(a instanceof MaskedArray)) return false;
  const mask = a.mask.data;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) return true;
  }
  return false;
}

/** Create a boolean mask from an array */
export function make_mask(
  m: boolean[] | NDArray,
  copy?: boolean,
  shrink?: boolean
): NDArray {
  const mask = m instanceof NDArray ? m : np_array(m, 'bool');
  if (shrink) {
    // If all false, return false (nomask)
    const data = mask.data;
    let anyTrue = false;
    for (let i = 0; i < data.length; i++) {
      if (data[i]) { anyTrue = true; break; }
    }
    if (!anyTrue) return np_array([false], 'bool'); // nomask sentinel
  }
  return copy ? mask.copy() : mask;
}

// ============ Fill Value Functions ============

/** Return data with masked values filled */
export function filled(a: MaskedArray, fill_value?: number): NDArray {
  return a.filled(fill_value);
}

/** Fix invalid (NaN/Inf) values by replacing with fill_value */
export function fix_invalid(
  a: NDArray,
  fill_value?: number
): MaskedArray {
  const fv = fill_value ?? 0;
  const result = a.copy();
  const data = result.data;
  const mask = zeros(a.shape, 'bool') as NDArray;
  const maskData = mask.data;

  for (let i = 0; i < a.size; i++) {
    const v = data[i] as number;
    if (!Number.isFinite(v)) {
      data[i] = fv;
      maskData[i] = 1;
    }
  }

  return new MaskedArray(result, { mask });
}

// ============ Operations (wrappers) ============

// These just call the existing functions and return MaskedArray

export function sum(a: MaskedArray, axis?: number, keepdims?: boolean) {
  return a.sum(axis, keepdims);
}

export function mean(a: MaskedArray, axis?: number, keepdims?: boolean) {
  return a.mean(axis, keepdims);
}

export function std(a: MaskedArray, axis?: number, keepdims?: boolean, ddof?: number) {
  return a.std(axis, keepdims, ddof);
}

// ... 150+ more wrapper functions that just delegate to MaskedArray methods
// or call existing NDArray functions with mask handling
```

## Usage Example

```typescript
import * as np from 'numpy-ts';

// Create a masked array
const data = np.array([1, 2, -999, 4, 5]);
const ma = np.ma.masked_equal(data, -999);

console.log(ma.toString());  // MaskedArray([1, 2, --, 4, 5])
console.log(ma.sum());       // 12 (skips masked)
console.log(ma.mean());      // 3.0
console.log(ma.compressed()); // [1, 2, 4, 5]

// Mask invalid values
const withNaN = np.array([1, 2, NaN, 4, Infinity]);
const clean = np.ma.masked_invalid(withNaN);
console.log(clean.mean());   // 2.333...

// Operations propagate masks
const a = np.ma.array([1, 2, 3], { mask: [false, true, false] });
const b = np.ma.array([4, 5, 6], { mask: [false, false, true] });
const c = a.add(b);
console.log(c.toString());   // MaskedArray([5, --, --])
```

## Implementation Effort Summary

| Component | LOC Estimate | Reuses Existing |
|-----------|--------------|-----------------|
| MaskedArray class | ~400 | NDArray, ArrayStorage |
| Mask utilities (~25 funcs) | ~300 | logical_*, comparison ops |
| Reduction wrappers | ~200 | nan* pattern |
| Operation wrappers (~150) | ~500 | All arithmetic/math ops |
| Tests | ~1000 | Test patterns |
| **Total** | **~2400** | **80%+ reuse** |

## Key Design Decisions

1. **Composition over inheritance**: MaskedArray wraps NDArray rather than extending it
2. **Mask is always NDArray**: Simplifies operations (no special "nomask" sentinel)
3. **Reuse existing ops**: All math operations delegate to NDArray methods
4. **Pattern follows nan* functions**: Reductions skip masked values like nansum skips NaN
