/**
 * ArrayStorage - Internal storage abstraction
 *
 * Stores array data directly using TypedArrays without external dependencies.
 *
 * @internal - This is not part of the public API
 */

import {
  type DType,
  type TypedArray,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  isBigIntDType,
  isComplexDType,
} from './dtype';
import { Complex } from './complex';

/**
 * Internal storage for NDArray data
 * Manages the underlying TypedArray and metadata
 */
export class ArrayStorage {
  // Underlying TypedArray data buffer
  private _data: TypedArray;
  // Array shape
  private _shape: readonly number[];
  // Strides for each dimension
  private _strides: readonly number[];
  // Offset into the data buffer
  private _offset: number;
  // Data type
  private _dtype: DType;

  constructor(
    data: TypedArray,
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    dtype: DType
  ) {
    this._data = data;
    this._shape = shape;
    this._strides = strides;
    this._offset = offset;
    this._dtype = dtype;
  }

  /**
   * Shape of the array
   */
  get shape(): readonly number[] {
    return this._shape;
  }

  /**
   * Number of dimensions
   */
  get ndim(): number {
    return this._shape.length;
  }

  /**
   * Total number of elements
   */
  get size(): number {
    return this._shape.reduce((a, b) => a * b, 1);
  }

  /**
   * Data type
   */
  get dtype(): DType {
    return this._dtype;
  }

  /**
   * Underlying data buffer
   */
  get data(): TypedArray {
    return this._data;
  }

  /**
   * Strides (steps in each dimension)
   */
  get strides(): readonly number[] {
    return this._strides;
  }

  /**
   * Offset into the data buffer
   */
  get offset(): number {
    return this._offset;
  }

  /**
   * Check if array is C-contiguous (row-major, no gaps)
   */
  get isCContiguous(): boolean {
    const shape = this._shape;
    const strides = this._strides;
    const ndim = shape.length;

    if (ndim === 0) return true;
    if (ndim === 1) return strides[0] === 1;

    // Check if strides match row-major order
    let expectedStride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      if (strides[i] !== expectedStride) return false;
      expectedStride *= shape[i]!;
    }
    return true;
  }

  /**
   * Check if array is F-contiguous (column-major, no gaps)
   */
  get isFContiguous(): boolean {
    const shape = this._shape;
    const strides = this._strides;
    const ndim = shape.length;

    if (ndim === 0) return true;
    if (ndim === 1) return strides[0] === 1;

    // Check if strides match column-major order
    let expectedStride = 1;
    for (let i = 0; i < ndim; i++) {
      if (strides[i] !== expectedStride) return false;
      expectedStride *= shape[i]!;
    }
    return true;
  }

  /**
   * Get element at linear index (respects strides and offset)
   * For complex dtypes, returns a Complex object.
   */
  iget(linearIndex: number): number | bigint | Complex {
    // Convert linear index to multi-index, then to actual buffer position
    const shape = this._shape;
    const strides = this._strides;
    const ndim = shape.length;
    const isComplex = isComplexDType(this._dtype);

    let bufferIndex: number;

    if (ndim === 0) {
      bufferIndex = this._offset;
    } else {
      // Convert linear index to multi-index in row-major order
      let remaining = linearIndex;
      bufferIndex = this._offset;

      for (let i = 0; i < ndim; i++) {
        // Compute size of remaining dimensions
        let dimSize = 1;
        for (let j = i + 1; j < ndim; j++) {
          dimSize *= shape[j]!;
        }
        const idx = Math.floor(remaining / dimSize);
        remaining = remaining % dimSize;
        bufferIndex += idx * strides[i]!;
      }
    }

    if (isComplex) {
      // Complex: read two consecutive values (re, im)
      // Buffer index is in complex element units, multiply by 2 for physical index
      const physicalIndex = bufferIndex * 2;
      const re = this._data[physicalIndex] as number;
      const im = this._data[physicalIndex + 1] as number;
      return new Complex(re, im);
    }

    return this._data[bufferIndex]!;
  }

  /**
   * Set element at linear index (respects strides and offset)
   * For complex dtypes, value can be a Complex object, {re, im} object, or number.
   */
  iset(linearIndex: number, value: number | bigint | Complex | { re: number; im: number }): void {
    const shape = this._shape;
    const strides = this._strides;
    const ndim = shape.length;
    const isComplex = isComplexDType(this._dtype);

    let bufferIndex: number;

    if (ndim === 0) {
      bufferIndex = this._offset;
    } else {
      let remaining = linearIndex;
      bufferIndex = this._offset;

      for (let i = 0; i < ndim; i++) {
        let dimSize = 1;
        for (let j = i + 1; j < ndim; j++) {
          dimSize *= shape[j]!;
        }
        const idx = Math.floor(remaining / dimSize);
        remaining = remaining % dimSize;
        bufferIndex += idx * strides[i]!;
      }
    }

    if (isComplex) {
      // Complex: write two consecutive values (re, im)
      const physicalIndex = bufferIndex * 2;
      let re: number, im: number;

      if (value instanceof Complex) {
        re = value.re;
        im = value.im;
      } else if (typeof value === 'object' && value !== null && 're' in value) {
        re = value.re;
        im = value.im ?? 0;
      } else {
        // Scalar number - treat as real with 0 imaginary
        re = Number(value);
        im = 0;
      }

      (this._data as Float64Array | Float32Array)[physicalIndex] = re;
      (this._data as Float64Array | Float32Array)[physicalIndex + 1] = im;
    } else {
      (this._data as unknown as (number | bigint)[])[bufferIndex] = value as number | bigint;
    }
  }

  /**
   * Get element at multi-index position
   * For complex dtypes, returns a Complex object.
   */
  get(...indices: number[]): number | bigint | Complex {
    const strides = this._strides;
    let bufferIndex = this._offset;

    for (let i = 0; i < indices.length; i++) {
      bufferIndex += indices[i]! * strides[i]!;
    }

    if (isComplexDType(this._dtype)) {
      const physicalIndex = bufferIndex * 2;
      const re = this._data[physicalIndex] as number;
      const im = this._data[physicalIndex + 1] as number;
      return new Complex(re, im);
    }

    return this._data[bufferIndex]!;
  }

  /**
   * Set element at multi-index position
   * For complex dtypes, value can be a Complex object, {re, im} object, or number.
   */
  set(indices: number[], value: number | bigint | Complex | { re: number; im: number }): void {
    const strides = this._strides;
    let bufferIndex = this._offset;

    for (let i = 0; i < indices.length; i++) {
      bufferIndex += indices[i]! * strides[i]!;
    }

    if (isComplexDType(this._dtype)) {
      const physicalIndex = bufferIndex * 2;
      let re: number, im: number;

      if (value instanceof Complex) {
        re = value.re;
        im = value.im;
      } else if (typeof value === 'object' && value !== null && 're' in value) {
        re = value.re;
        im = value.im ?? 0;
      } else {
        re = Number(value);
        im = 0;
      }

      (this._data as Float64Array | Float32Array)[physicalIndex] = re;
      (this._data as Float64Array | Float32Array)[physicalIndex + 1] = im;
    } else {
      (this._data as unknown as (number | bigint)[])[bufferIndex] = value as number | bigint;
    }
  }

  /**
   * Create a deep copy of this storage
   */
  copy(): ArrayStorage {
    const shape = Array.from(this._shape);
    const dtype = this._dtype;
    const size = this.size;
    const isComplex = isComplexDType(dtype);

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot copy array with dtype ${dtype}`);
    }

    // For complex types, physical size is 2x logical size
    const physicalSize = isComplex ? size * 2 : size;
    const newData = new Constructor(physicalSize);

    if (this.isCContiguous && this._offset === 0) {
      // Fast path: direct copy of physical data
      if (isBigIntDType(dtype)) {
        const src = this._data as BigInt64Array | BigUint64Array;
        const dst = newData as BigInt64Array | BigUint64Array;
        for (let i = 0; i < physicalSize; i++) {
          dst[i] = src[i]!;
        }
      } else {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).set(
          (this._data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).subarray(
            0,
            physicalSize
          )
        );
      }
    } else {
      // Slow path: respect strides
      if (isBigIntDType(dtype)) {
        const dst = newData as BigInt64Array | BigUint64Array;
        for (let i = 0; i < size; i++) {
          dst[i] = this.iget(i) as bigint;
        }
      } else if (isComplex) {
        // For complex, copy element by element
        const dst = newData as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          const val = this.iget(i) as Complex;
          dst[i * 2] = val.re;
          dst[i * 2 + 1] = val.im;
        }
      } else {
        for (let i = 0; i < size; i++) {
          newData[i] = this.iget(i) as number;
        }
      }
    }

    return new ArrayStorage(newData, shape, ArrayStorage._computeStrides(shape), 0, dtype);
  }

  /**
   * Create storage from TypedArray data
   */
  static fromData(
    data: TypedArray,
    shape: number[],
    dtype: DType,
    strides?: number[],
    offset?: number
  ): ArrayStorage {
    const finalStrides = strides ?? ArrayStorage._computeStrides(shape);
    const finalOffset = offset ?? 0;
    return new ArrayStorage(data, shape, finalStrides, finalOffset, dtype);
  }

  /**
   * Create storage with zeros
   */
  static zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): ArrayStorage {
    const size = shape.reduce((a, b) => a * b, 1);
    const isComplex = isComplexDType(dtype);

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create array with dtype ${dtype}`);
    }

    // For complex types, physical size is 2x logical size
    const physicalSize = isComplex ? size * 2 : size;
    const data = new Constructor(physicalSize);
    // TypedArrays are initialized to 0 by default, so no need to fill

    return new ArrayStorage(data, shape, ArrayStorage._computeStrides(shape), 0, dtype);
  }

  /**
   * Create storage with ones
   */
  static ones(shape: number[], dtype: DType = DEFAULT_DTYPE): ArrayStorage {
    const size = shape.reduce((a, b) => a * b, 1);
    const isComplex = isComplexDType(dtype);

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create array with dtype ${dtype}`);
    }

    // For complex types, physical size is 2x logical size
    const physicalSize = isComplex ? size * 2 : size;
    const data = new Constructor(physicalSize);

    // Fill with ones
    if (isBigIntDType(dtype)) {
      (data as BigInt64Array | BigUint64Array).fill(BigInt(1));
    } else if (isComplex) {
      // For complex, ones means 1+0j, so fill with [1, 0, 1, 0, ...]
      const floatData = data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        floatData[i * 2] = 1; // real part
        floatData[i * 2 + 1] = 0; // imaginary part
      }
    } else {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(1);
    }

    return new ArrayStorage(data, shape, ArrayStorage._computeStrides(shape), 0, dtype);
  }

  /**
   * Compute strides for row-major (C-order) layout
   * @private
   */
  private static _computeStrides(shape: readonly number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i]!;
    }
    return strides;
  }
}

/**
 * Compute strides for a given shape (row-major order)
 * @internal
 */
export function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}
