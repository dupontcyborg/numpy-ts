/**
 * MaskedArray - NumPy-compatible masked array implementation
 *
 * Provides arrays with masks for invalid/missing data, matching NumPy's np.ma module.
 */

import { NDArray } from '../core/ndarray';
import { ArrayStorage } from '../core/storage';
import { DType, isBigIntDType } from '../core/dtype';
import * as creation from '../core/ndarray';
import * as arithmeticOps from '../ops/arithmetic';
import * as exponentialOps from '../ops/exponential';
import * as comparisonOps from '../ops/comparison';
import * as logicOps from '../ops/logic';

/**
 * Sentinel value for masked elements (like NumPy's np.ma.masked)
 */
export const masked = Symbol('masked');

/**
 * Sentinel for "no mask" (like NumPy's np.ma.nomask which is False)
 */
export const nomask = false;

/**
 * Type for mask input - can be boolean, array of booleans, or NDArray
 */
export type MaskInput = boolean | boolean[] | number[] | NDArray;

/**
 * Get the default fill value for a dtype (matches NumPy)
 */
export function default_fill_value(dtype: DType | string): number | bigint | boolean {
  const dt = dtype as DType;
  if (dt.startsWith('float') || dt.startsWith('complex')) {
    return 1e20;
  }
  if (dt.startsWith('int') || dt.startsWith('uint')) {
    if (isBigIntDType(dt)) {
      return BigInt(999999);
    }
    return 999999;
  }
  if (dt === 'bool') {
    return true;
  }
  return 1e20;
}

/**
 * MaskedArray - Array class with masked (invalid) elements
 *
 * A MaskedArray wraps an NDArray with a boolean mask indicating which
 * elements should be ignored in computations.
 */
export class MaskedArray {
  private _data: NDArray;
  private _mask: NDArray | false; // false means nomask (no elements masked)
  private _fill_value: number | bigint | boolean;
  private _hardmask: boolean;
  private _baseclass: typeof NDArray = NDArray;

  /**
   * Create a MaskedArray
   *
   * @param data - Input data (array-like or NDArray)
   * @param options - Configuration options
   * @param options.mask - Boolean mask (true = masked/invalid)
   * @param options.fill_value - Value to use for masked positions when filling
   * @param options.dtype - Data type for the array
   * @param options.copy - Whether to copy the data
   * @param options.hard_mask - If true, masked values cannot be unmasked
   */
  constructor(
    data: NDArray | number[] | number[][] | number[][][] | (number | typeof masked)[],
    options?: {
      mask?: MaskInput;
      fill_value?: number | bigint | boolean;
      dtype?: DType;
      copy?: boolean;
      hard_mask?: boolean;
    }
  ) {
    const opts = options ?? {};

    // Handle masked constant in data
    let processedData: NDArray | number[] | number[][] | number[][][] = data as
      | NDArray
      | number[]
      | number[][]
      | number[][][];
    let inferredMask: boolean[] | undefined;

    if (Array.isArray(data)) {
      const flat = this._flattenArray(data);
      if (flat.some((v) => v === masked)) {
        // Replace masked with 0 (or appropriate fill) and create mask
        inferredMask = flat.map((v) => v === masked);
        processedData = this._replaceWithShape(data, masked, 0) as
          | number[]
          | number[][]
          | number[][][];
      }
    }

    // Convert data to NDArray
    if (processedData instanceof NDArray) {
      this._data = opts.copy ? processedData.copy() : processedData;
    } else {
      this._data = creation.array(processedData as number[], opts.dtype);
    }

    // Set fill value (before processing mask, in case we need it)
    this._fill_value = opts.fill_value ?? default_fill_value(this._data.dtype);

    // Initialize mask
    const maskInput = opts.mask ?? inferredMask;
    if (maskInput === undefined || (typeof maskInput === 'boolean' && !maskInput)) {
      // No mask
      this._mask = false;
    } else if (typeof maskInput === 'boolean' && maskInput === true) {
      // All masked
      this._mask = creation.ones([...this._data.shape], 'bool') as NDArray;
    } else if (maskInput instanceof NDArray) {
      // Ensure it's boolean dtype and correct shape
      const maskArr = maskInput.dtype === 'bool' ? maskInput : maskInput.astype('bool');
      this._mask = opts.copy ? maskArr.copy() : maskArr;
    } else {
      // Array of booleans/numbers
      this._mask = creation.array(maskInput as number[], 'bool');
    }

    // Verify mask shape matches data shape
    if (this._mask !== false) {
      const maskShape = this._mask.shape;
      const dataShape = this._data.shape;
      if (maskShape.length !== dataShape.length || !maskShape.every((s, i) => s === dataShape[i])) {
        throw new Error(
          `Mask shape ${JSON.stringify(maskShape)} does not match data shape ${JSON.stringify(dataShape)}`
        );
      }
    }

    this._hardmask = opts.hard_mask ?? false;
  }

  // ============ Private Helpers ============

  private _flattenArray(arr: unknown[]): unknown[] {
    const result: unknown[] = [];
    const flatten = (a: unknown) => {
      if (Array.isArray(a)) {
        a.forEach(flatten);
      } else {
        result.push(a);
      }
    };
    flatten(arr);
    return result;
  }

  private _replaceWithShape(arr: unknown, target: unknown, replacement: unknown): unknown {
    if (Array.isArray(arr)) {
      return arr.map((item) => this._replaceWithShape(item, target, replacement));
    }
    return arr === target ? replacement : arr;
  }

  // ============ Properties ============

  /**
   * The underlying data array (includes masked values)
   */
  get data(): NDArray {
    return this._data;
  }

  /**
   * Boolean mask array (true = masked/invalid), or False if no mask
   */
  get mask(): NDArray | false {
    return this._mask;
  }

  /**
   * Set the mask
   */
  set mask(value: MaskInput) {
    if (this._hardmask && this._mask !== false) {
      // Hard mask: can only set more masked values, not unmask
      if (typeof value === 'boolean' && !value) {
        throw new Error('Cannot unmask elements with hard_mask=True');
      }
    }

    if (typeof value === 'boolean' && !value) {
      this._mask = false;
    } else if (typeof value === 'boolean' && value === true) {
      this._mask = creation.ones([...this._data.shape], 'bool') as NDArray;
    } else if (value instanceof NDArray) {
      this._mask = value.dtype === 'bool' ? value : value.astype('bool');
    } else {
      this._mask = creation.array(value as number[], 'bool');
    }
  }

  /**
   * Fill value for masked positions
   */
  get fill_value(): number | bigint | boolean {
    return this._fill_value;
  }

  set fill_value(value: number | bigint | boolean) {
    this._fill_value = value;
  }

  /**
   * Whether mask is hard (cannot unset masked values)
   */
  get hardmask(): boolean {
    return this._hardmask;
  }

  /**
   * Shape of the array
   */
  get shape(): readonly number[] {
    return this._data.shape;
  }

  /**
   * Number of dimensions
   */
  get ndim(): number {
    return this._data.ndim;
  }

  /**
   * Total number of elements
   */
  get size(): number {
    return this._data.size;
  }

  /**
   * Data type
   */
  get dtype(): string {
    return this._data.dtype;
  }

  /**
   * Transposed array
   */
  get T(): MaskedArray {
    return this.transpose();
  }

  /**
   * Size of one element in bytes
   */
  get itemsize(): number {
    return this._data.itemsize;
  }

  /**
   * Total bytes consumed by the array elements
   */
  get nbytes(): number {
    return this._data.nbytes;
  }

  /**
   * Base array if this is a view
   */
  get base(): NDArray | null {
    return this._data.base;
  }

  /**
   * The class used for the underlying data (NDArray)
   */
  get baseclass(): typeof NDArray {
    return this._baseclass;
  }

  // ============ Core Methods ============

  /**
   * Return data with masked values replaced by fill_value
   */
  filled(fill_value?: number | bigint | boolean): NDArray {
    const fv = fill_value ?? this._fill_value;
    const result = this._data.copy(); // This creates a contiguous C-order copy

    if (this._mask === false) {
      return result; // No mask, return copy of data
    }

    // Make the mask contiguous to match the result's layout
    // This is important for strided arrays (e.g., after transpose)
    const contiguousMask = this._mask.ravel();
    const maskData = contiguousMask.data;
    const resultData = result.data;
    const isBigInt = isBigIntDType(this._data.dtype as DType);

    for (let i = 0; i < this.size; i++) {
      if (maskData[i]) {
        if (isBigInt) {
          (resultData as BigInt64Array | BigUint64Array)[i] =
            typeof fv === 'bigint' ? fv : BigInt(Number(fv));
        } else {
          (resultData as Float64Array)[i] = Number(fv);
        }
      }
    }
    return result;
  }

  /**
   * Return 1-D array of non-masked data
   */
  compressed(): NDArray {
    if (this._mask === false) {
      return this._data.ravel();
    }

    const maskData = this._mask.data;
    const srcData = this._data.data;

    // Count non-masked elements
    let count = 0;
    for (let i = 0; i < this.size; i++) {
      if (!maskData[i]) count++;
    }

    // Extract non-masked values
    const result = creation.zeros([count], this.dtype as DType) as NDArray;
    const dstData = result.data;
    let j = 0;

    const isBigInt = isBigIntDType(this._data.dtype as DType);
    for (let i = 0; i < this.size; i++) {
      if (!maskData[i]) {
        if (isBigInt) {
          (dstData as BigInt64Array)[j++] = (srcData as BigInt64Array)[i]!;
        } else {
          (dstData as Float64Array)[j++] = (srcData as Float64Array)[i]! as number;
        }
      }
    }
    return result;
  }

  /**
   * Count non-masked elements along axis
   */
  count(axis?: number): number | NDArray {
    if (this._mask === false) {
      // No mask - all elements are valid
      if (axis === undefined) {
        return this.size;
      }
      // Return shape along other axes
      const newShape = [...this.shape];
      newShape.splice(axis, 1);
      const axisSize = this.shape[axis] as number;
      if (newShape.length === 0) {
        return axisSize;
      }
      return creation.full(newShape, axisSize, 'int64') as NDArray;
    }

    if (axis === undefined) {
      // Count all non-masked
      const maskData = this._mask.data;
      let count = 0;
      for (let i = 0; i < this.size; i++) {
        if (!maskData[i]) count++;
      }
      return count;
    }

    // Axis-aware count: sum of ~mask along axis
    const notMask = logicOps.logical_not(this._mask.storage);
    const notMaskArr = NDArray._fromStorage(notMask);
    return notMaskArr.sum(axis) as NDArray;
  }

  /**
   * Copy the array
   */
  copy(): MaskedArray {
    return new MaskedArray(this._data.copy(), {
      mask: this._mask === false ? false : this._mask.copy(),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Convert to regular NDArray (masked values become fill_value)
   */
  toNDArray(): NDArray {
    return this.filled();
  }

  /**
   * Convert to nested JavaScript arrays
   */
  tolist(): unknown {
    return this.filled().tolist();
  }

  /**
   * Get item at index
   */
  item(...indices: number[]): number | bigint | typeof masked {
    const flatIndex =
      indices.length === 1
        ? (indices[0] as number)
        : indices.reduce((acc, idx, i) => {
            const stride = this.shape.slice(i + 1).reduce((a, b) => a * b, 1);
            return acc + idx * stride;
          }, 0);

    if (this._mask !== false && this._mask.data[flatIndex]) {
      return masked;
    }

    const result = this._data.item(...indices);
    // item() can return Complex for complex dtypes, but we don't support that in MaskedArray
    return result as number | bigint;
  }

  // ============ Arithmetic Operations ============

  private _binaryOp(
    other: MaskedArray | NDArray | number | bigint,
    op: (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage
  ): MaskedArray {
    let resultData: NDArray;
    let resultMask: NDArray | false;

    if (other instanceof MaskedArray) {
      resultData = NDArray._fromStorage(op(this._data.storage, other._data.storage));
      // Combine masks (OR)
      if (this._mask === false && other._mask === false) {
        resultMask = false;
      } else if (this._mask === false) {
        resultMask = other._mask;
      } else if (other._mask === false) {
        resultMask = this._mask;
      } else {
        resultMask = NDArray._fromStorage(
          logicOps.logical_or(this._mask.storage, other._mask.storage)
        );
      }
    } else if (other instanceof NDArray) {
      resultData = NDArray._fromStorage(op(this._data.storage, other.storage));
      resultMask = this._mask;
    } else {
      // Convert bigint to number if needed
      const scalar = typeof other === 'bigint' ? Number(other) : other;
      resultData = NDArray._fromStorage(op(this._data.storage, scalar));
      resultMask = this._mask;
    }

    return new MaskedArray(resultData, {
      mask: resultMask,
      fill_value: this._fill_value,
    });
  }

  add(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.add);
  }

  subtract(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.subtract);
  }

  multiply(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.multiply);
  }

  divide(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.divide);
  }

  floor_divide(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.floorDivide);
  }

  mod(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, arithmeticOps.mod);
  }

  power(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._binaryOp(other, exponentialOps.power);
  }

  // Unary operations
  private _unaryOp(op: (a: ArrayStorage) => ArrayStorage): MaskedArray {
    const resultData = NDArray._fromStorage(op(this._data.storage));
    return new MaskedArray(resultData, {
      mask: this._mask,
      fill_value: this._fill_value,
    });
  }

  negative(): MaskedArray {
    return this._unaryOp(arithmeticOps.negative);
  }

  abs(): MaskedArray {
    return this._unaryOp(arithmeticOps.absolute);
  }

  sqrt(): MaskedArray {
    return this._unaryOp(exponentialOps.sqrt);
  }

  square(): MaskedArray {
    return this._unaryOp(arithmeticOps.square);
  }

  // ============ Comparison Operations ============

  private _comparisonOp(
    other: MaskedArray | NDArray | number | bigint,
    op: (a: ArrayStorage, b: ArrayStorage | number) => ArrayStorage
  ): MaskedArray {
    let resultData: NDArray;
    let resultMask: NDArray | false;

    if (other instanceof MaskedArray) {
      resultData = NDArray._fromStorage(op(this._data.storage, other._data.storage));
      if (this._mask === false && other._mask === false) {
        resultMask = false;
      } else if (this._mask === false) {
        resultMask = other._mask;
      } else if (other._mask === false) {
        resultMask = this._mask;
      } else {
        resultMask = NDArray._fromStorage(
          logicOps.logical_or(this._mask.storage, other._mask.storage)
        );
      }
    } else if (other instanceof NDArray) {
      resultData = NDArray._fromStorage(op(this._data.storage, other.storage));
      resultMask = this._mask;
    } else {
      // Convert bigint to number if needed
      const scalar = typeof other === 'bigint' ? Number(other) : other;
      resultData = NDArray._fromStorage(op(this._data.storage, scalar));
      resultMask = this._mask;
    }

    return new MaskedArray(resultData, {
      mask: resultMask,
      fill_value: this._fill_value,
    });
  }

  equal(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.equal);
  }

  not_equal(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.notEqual);
  }

  less(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.less);
  }

  less_equal(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.lessEqual);
  }

  greater(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.greater);
  }

  greater_equal(other: MaskedArray | NDArray | number | bigint): MaskedArray {
    return this._comparisonOp(other, comparisonOps.greaterEqual);
  }

  // ============ Reduction Operations ============

  private _getMaskedValues(): number[] {
    if (this._mask === false) {
      // No mask - all values
      const data = this._data.data;
      const values: number[] = [];
      for (let i = 0; i < this.size; i++) {
        values.push(Number(data[i]));
      }
      return values;
    }

    const data = this._data.data;
    const mask = this._mask.data;
    const values: number[] = [];
    for (let i = 0; i < this.size; i++) {
      if (!mask[i]) {
        values.push(Number(data[i]));
      }
    }
    return values;
  }

  /**
   * Sum of non-masked elements
   */
  sum(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return 0;
      return values.reduce((a, b) => a + b, 0);
    }
    return this._axisReduction('sum', axis, keepdims);
  }

  /**
   * Product of non-masked elements
   */
  prod(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return 1;
      return values.reduce((a, b) => a * b, 1);
    }
    return this._axisReduction('prod', axis, keepdims);
  }

  /**
   * Mean of non-masked elements
   */
  mean(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return NaN;
      return values.reduce((a, b) => a + b, 0) / values.length;
    }
    return this._axisReduction('mean', axis, keepdims);
  }

  /**
   * Variance of non-masked elements
   */
  var(axis?: number, ddof: number = 0, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length <= ddof) return NaN;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const sumSq = values.reduce((a, b) => a + (b - mean) ** 2, 0);
      return sumSq / (values.length - ddof);
    }
    return this._axisReduction('var', axis, keepdims, { ddof });
  }

  /**
   * Standard deviation of non-masked elements
   */
  std(axis?: number, ddof: number = 0, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const variance = this.var(undefined, ddof) as number;
      return Math.sqrt(variance);
    }
    return this._axisReduction('std', axis, keepdims, { ddof });
  }

  /**
   * Minimum of non-masked elements
   */
  min(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return NaN;
      return Math.min(...values);
    }
    return this._axisReduction('min', axis, keepdims);
  }

  /**
   * Maximum of non-masked elements
   */
  max(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return NaN;
      return Math.max(...values);
    }
    return this._axisReduction('max', axis, keepdims);
  }

  /**
   * Peak-to-peak (max - min) of non-masked elements
   */
  ptp(axis?: number, keepdims: boolean = false): number | MaskedArray {
    if (axis === undefined) {
      const values = this._getMaskedValues();
      if (values.length === 0) return NaN;
      return Math.max(...values) - Math.min(...values);
    }
    return this._axisReduction('ptp', axis, keepdims);
  }

  /**
   * Index of minimum along axis
   */
  argmin(axis?: number): number | NDArray {
    if (axis === undefined) {
      if (this._mask === false) {
        return this._data.argmin() as number;
      }
      const data = this._data.data;
      const mask = this._mask.data;
      let minVal = Infinity;
      let minIdx = -1;
      for (let i = 0; i < this.size; i++) {
        if (!mask[i] && Number(data[i]) < minVal) {
          minVal = Number(data[i]);
          minIdx = i;
        }
      }
      return minIdx;
    }
    // For axis-aware argmin, we need more complex logic
    throw new Error('axis parameter not yet implemented for argmin');
  }

  /**
   * Index of maximum along axis
   */
  argmax(axis?: number): number | NDArray {
    if (axis === undefined) {
      if (this._mask === false) {
        return this._data.argmax() as number;
      }
      const data = this._data.data;
      const mask = this._mask.data;
      let maxVal = -Infinity;
      let maxIdx = -1;
      for (let i = 0; i < this.size; i++) {
        if (!mask[i] && Number(data[i]) > maxVal) {
          maxVal = Number(data[i]);
          maxIdx = i;
        }
      }
      return maxIdx;
    }
    throw new Error('axis parameter not yet implemented for argmax');
  }

  /**
   * Cumulative sum
   */
  cumsum(axis?: number): MaskedArray {
    // Fill masked with 0, then cumsum
    const filled = this.filled(0);
    const result = filled.cumsum(axis);
    return new MaskedArray(result, { mask: this._mask, fill_value: this._fill_value });
  }

  /**
   * Cumulative product
   */
  cumprod(axis?: number): MaskedArray {
    // Fill masked with 1, then cumprod
    const filled = this.filled(1);
    const result = filled.cumprod(axis);
    return new MaskedArray(result, { mask: this._mask, fill_value: this._fill_value });
  }

  /**
   * All elements are true (non-masked)
   */
  all(axis?: number): boolean | MaskedArray {
    if (axis === undefined) {
      if (this._mask === false) {
        return this._data.all() as boolean;
      }
      const data = this._data.data;
      const mask = this._mask.data;
      for (let i = 0; i < this.size; i++) {
        if (!mask[i] && !data[i]) return false;
      }
      return true;
    }
    return this._axisReduction('all', axis, false) as MaskedArray;
  }

  /**
   * Any element is true (non-masked)
   */
  any(axis?: number): boolean | MaskedArray {
    if (axis === undefined) {
      if (this._mask === false) {
        return this._data.any() as boolean;
      }
      const data = this._data.data;
      const mask = this._mask.data;
      for (let i = 0; i < this.size; i++) {
        if (!mask[i] && data[i]) return true;
      }
      return false;
    }
    return this._axisReduction('any', axis, false) as MaskedArray;
  }

  private _axisReduction(
    op: string,
    axis: number,
    keepdims: boolean,
    opts?: { ddof?: number }
  ): MaskedArray {
    // For axis-aware reductions, we use filled data with appropriate fill values
    // and then adjust for count
    const shape = this.shape;
    const ndim = this.ndim;

    // Normalize negative axis
    if (axis < 0) axis = ndim + axis;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`Axis ${axis} is out of bounds for array with ${ndim} dimensions`);
    }

    // Calculate output shape (used for validation, may be needed in future)
    const _outShape = keepdims
      ? shape.map((s, i) => (i === axis ? 1 : s))
      : [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    void _outShape; // Prevent unused warning

    // For simple cases, we can use the data's reduction and adjust
    // This is a simplified implementation
    let result: NDArray;
    let resultMask: NDArray | false = false;

    switch (op) {
      case 'sum': {
        const filled = this.filled(0);
        result = filled.sum(axis, keepdims) as NDArray;
        break;
      }
      case 'prod': {
        const filled = this.filled(1);
        result = filled.prod(axis, keepdims) as NDArray;
        break;
      }
      case 'mean': {
        const filled = this.filled(0);
        const sumResult = filled.sum(axis, keepdims) as NDArray;
        const countResult = this.count(axis) as NDArray;
        // Divide sum by count
        result = NDArray._fromStorage(arithmeticOps.divide(sumResult.storage, countResult.storage));
        break;
      }
      case 'min': {
        const filled = this.filled(Infinity);
        result = filled.min(axis, keepdims) as NDArray;
        break;
      }
      case 'max': {
        const filled = this.filled(-Infinity);
        result = filled.max(axis, keepdims) as NDArray;
        break;
      }
      case 'var':
      case 'std': {
        // Compute mean first
        const meanArr = this._axisReduction('mean', axis, true) as MaskedArray;
        // Compute (x - mean)^2
        const diff = this.subtract(meanArr);
        const sq = diff.multiply(diff);
        // Sum of squares
        const sqSum = sq._axisReduction('sum', axis, keepdims) as MaskedArray;
        // Divide by count - ddof
        const countResult = this.count(axis) as NDArray;
        const ddof = opts?.ddof ?? 0;
        const divisor = NDArray._fromStorage(arithmeticOps.subtract(countResult.storage, ddof));
        result = NDArray._fromStorage(arithmeticOps.divide(sqSum._data.storage, divisor.storage));
        if (op === 'std') {
          result = NDArray._fromStorage(exponentialOps.sqrt(result.storage));
        }
        break;
      }
      case 'ptp': {
        const maxArr = this._axisReduction('max', axis, keepdims) as MaskedArray;
        const minArr = this._axisReduction('min', axis, keepdims) as MaskedArray;
        result = NDArray._fromStorage(
          arithmeticOps.subtract(maxArr._data.storage, minArr._data.storage)
        );
        break;
      }
      case 'all': {
        const filled = this.filled(true);
        result = filled.all(axis) as NDArray;
        break;
      }
      case 'any': {
        const filled = this.filled(false);
        result = filled.any(axis) as NDArray;
        break;
      }
      default:
        throw new Error(`Unknown reduction operation: ${op}`);
    }

    return new MaskedArray(result, { mask: resultMask, fill_value: this._fill_value });
  }

  // ============ Shape Operations ============

  /**
   * Reshape the array
   */
  reshape(...newShape: number[] | [number[]]): MaskedArray {
    const shape = Array.isArray(newShape[0]) ? newShape[0] : (newShape as number[]);
    return new MaskedArray(this._data.reshape(...shape), {
      mask: this._mask === false ? false : this._mask.reshape(...shape),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Transpose the array
   */
  transpose(axes?: number[]): MaskedArray {
    return new MaskedArray(this._data.transpose(axes), {
      mask: this._mask === false ? false : this._mask.transpose(axes),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Flatten to 1D
   */
  ravel(): MaskedArray {
    return new MaskedArray(this._data.ravel(), {
      mask: this._mask === false ? false : this._mask.ravel(),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Flatten to 1D (copy)
   */
  flatten(): MaskedArray {
    return new MaskedArray(this._data.flatten(), {
      mask: this._mask === false ? false : this._mask.flatten(),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Squeeze out dimensions of size 1
   */
  squeeze(axis?: number): MaskedArray {
    return new MaskedArray(this._data.squeeze(axis), {
      mask: this._mask === false ? false : this._mask.squeeze(axis),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Swap two axes
   */
  swapaxes(axis1: number, axis2: number): MaskedArray {
    return new MaskedArray(this._data.swapaxes(axis1, axis2), {
      mask: this._mask === false ? false : this._mask.swapaxes(axis1, axis2),
      fill_value: this._fill_value,
      hard_mask: this._hardmask,
    });
  }

  /**
   * Convert to specified dtype
   */
  astype(dtype: DType): MaskedArray {
    return new MaskedArray(this._data.astype(dtype), {
      mask: this._mask,
      fill_value: default_fill_value(dtype),
      hard_mask: this._hardmask,
    });
  }

  // ============ Mask Operations ============

  /**
   * Harden the mask (prevent unmasking)
   */
  harden_mask(): MaskedArray {
    this._hardmask = true;
    return this;
  }

  /**
   * Soften the mask (allow unmasking)
   */
  soften_mask(): MaskedArray {
    this._hardmask = false;
    return this;
  }

  /**
   * Shrink mask to nomask if no values are masked
   */
  shrink_mask(): MaskedArray {
    if (this._mask === false) return this;

    const maskData = this._mask.data;
    let anyMasked = false;
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i]) {
        anyMasked = true;
        break;
      }
    }

    if (!anyMasked) {
      this._mask = false;
    }
    return this;
  }

  /**
   * Unmask all elements (if not hardmask)
   */
  unshare_mask(): MaskedArray {
    if (this._mask !== false) {
      this._mask = this._mask.copy();
    }
    return this;
  }

  // ============ String Representation ============

  /**
   * String representation
   */
  toString(): string {
    return this._formatMaskedArray();
  }

  private _formatMaskedArray(): string {
    if (this.ndim === 0) {
      if (this._mask !== false && this._mask.data[0]) {
        return '--';
      }
      return String(this._data.data[0]);
    }

    const data = this._data.tolist();
    const mask = this._mask === false ? null : this._mask.tolist();
    const formatted = this._formatRecursive(data, mask);
    return formatted;
  }

  private _formatRecursive(data: unknown, mask: unknown): string {
    if (!Array.isArray(data)) {
      if (mask) return '--';
      return String(data);
    }

    const items = data.map((d, i) => {
      const m = mask ? (mask as unknown[])[i] : false;
      return this._formatRecursive(d, m);
    });

    return `[${items.join(' ')}]`;
  }

  /**
   * Detailed string representation
   */
  repr(): string {
    return `MaskedArray(${this.toString()}, mask=${this._mask === false ? 'False' : this._mask.toString()}, fill_value=${this._fill_value})`;
  }
}

// Re-export for convenience
export { NDArray };
