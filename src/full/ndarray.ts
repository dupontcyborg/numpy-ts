// AUTO-GENERATED - DO NOT EDIT
// Run `npm run generate` to regenerate this file from scripts/ndarray-methods.ts

import { parseSlice, normalizeSlice } from '../common/slicing';
import {
  type DType,
  type TypedArray,
  getTypedArrayConstructor,
  getDTypeSize,
  isBigIntDType,
  isComplexDType,
} from '../common/dtype';
import { Complex } from '../common/complex';
import { ArrayStorage } from '../common/storage';
import { NDArrayCore } from '../common/ndarray-core';
import * as core from '../core';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

export class NDArray extends NDArrayCore {
  // ========================================
  // Manual methods
  // ========================================

  /**
   * Override _base with NDArray type
   */
  protected override _base?: NDArray;

  constructor(storage: ArrayStorage, base?: NDArray) {
    super(storage, base);
    this._base = base;
  }

  /**
   * Create NDArray from storage (for ops modules)
   * @internal
   */
  static override fromStorage(storage: ArrayStorage, base?: NDArray): NDArray {
    return new NDArray(storage, base);
  }

  /**
   * Base array if this is a view, null if this array owns its data
   * Similar to NumPy's base attribute
   */
  override get base(): NDArray | null {
    return this._base ?? null;
  }

  /**
   * Transpose of the array (shorthand for transpose())
   * Returns a view with axes reversed
   */
  get T(): NDArray {
    return this.transpose();
  }

  /**
   * Size of one array element in bytes
   */
  override get itemsize(): number {
    return getDTypeSize(this._storage.dtype);
  }

  /**
   * Total bytes consumed by the elements of the array
   */
  override get nbytes(): number {
    return this.size * this.itemsize;
  }

  /**
   * Fill the array with a scalar value (in-place)
   * @param value - Value to fill with
   */
  override fill(value: number | bigint): void {
    const dtype = this._storage.dtype;
    const size = this.size;

    if (isBigIntDType(dtype)) {
      const bigintValue = typeof value === 'bigint' ? value : BigInt(Math.round(Number(value)));
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, bigintValue);
      }
    } else if (dtype === 'bool') {
      const boolValue = value ? 1 : 0;
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, boolValue);
      }
    } else {
      const numValue = Number(value);
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, numValue);
      }
    }
  }

  /**
   * Iterator protocol - iterate over the first axis
   * For 1D arrays, yields elements; for ND arrays, yields (N-1)D subarrays
   */
  override *[Symbol.iterator](): Iterator<NDArray | number | bigint | Complex> {
    if (this.ndim === 0) {
      yield this._storage.iget(0);
    } else if (this.ndim === 1) {
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this._storage.iget(i);
      }
    } else {
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this.slice(String(i));
      }
    }
  }

  /**
   * Get a single element from the array
   * @param indices - Array of indices, one per dimension
   * @returns The element value
   */
  override get(indices: number[]): number | bigint | Complex {
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    return this._storage.get(...normalizedIndices);
  }

  /**
   * Set a single element in the array
   * @param indices - Array of indices, one per dimension
   * @param value - Value to set
   */
  override set(
    indices: number[],
    value: number | bigint | Complex | { re: number; im: number }
  ): void {
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    const currentDtype = this.dtype as DType;

    if (isComplexDType(currentDtype)) {
      this._storage.set(normalizedIndices, value);
    } else if (isBigIntDType(currentDtype)) {
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = typeof value === 'bigint' ? value : BigInt(Math.round(numValue));
      this._storage.set(normalizedIndices, convertedValue);
    } else if (currentDtype === 'bool') {
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = numValue ? 1 : 0;
      this._storage.set(normalizedIndices, convertedValue);
    } else {
      const convertedValue = value instanceof Complex ? value.re : Number(value);
      this._storage.set(normalizedIndices, convertedValue);
    }
  }

  /**
   * Return a deep copy of the array
   */
  override copy(): NDArray {
    return new NDArray(this._storage.copy());
  }

  /**
   * Cast array to a different dtype
   * @param dtype - Target dtype
   * @param copy - If false and dtype matches, return self
   * @returns Array with specified dtype
   */
  override astype(dtype: DType, copy: boolean = true): NDArray {
    const currentDtype = this.dtype as DType;

    if (currentDtype === dtype && !copy) {
      return this;
    }

    if (currentDtype === dtype && copy) {
      return this.copy();
    }

    const shape = Array.from(this.shape);
    const size = this.size;

    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot convert to dtype ${dtype}`);
    }
    const newData = new Constructor(size);
    const oldData = this.data;

    if (isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      if (dtype === 'bool') {
        for (let i = 0; i < size; i++) {
          (newData as Uint8Array)[i] = typedOldData[i] !== BigInt(0) ? 1 : 0;
        }
      } else {
        for (let i = 0; i < size; i++) {
          (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(
            typedOldData[i]
          );
        }
      }
    } else if (!isBigIntDType(currentDtype) && isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = BigInt(
          Math.round(Number(typedOldData[i]))
        );
      }
    } else if (dtype === 'bool') {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Uint8Array)[i] = typedOldData[i] !== 0 ? 1 : 0;
      }
    } else if (currentDtype === 'bool' && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Uint8Array;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    } else if (!isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    } else {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = typedOldData[i]!;
      }
    }

    const storage = ArrayStorage.fromData(newData, shape, dtype);
    return new NDArray(storage);
  }

  /**
   * Slice the array using NumPy-style string syntax
   * @param sliceStrs - Slice specifications, one per dimension
   * @returns Sliced view of the array
   */
  override slice(...sliceStrs: string[]): NDArray {
    if (sliceStrs.length === 0) {
      return this;
    }

    if (sliceStrs.length > this.ndim) {
      throw new Error(
        `Too many indices for array: array is ${this.ndim}-dimensional, but ${sliceStrs.length} were indexed`
      );
    }

    const sliceSpecs = sliceStrs.map((str, i) => {
      const spec = parseSlice(str);
      const normalized = normalizeSlice(spec, this.shape[i]!);
      return normalized;
    });

    while (sliceSpecs.length < this.ndim) {
      sliceSpecs.push({
        start: 0,
        stop: this.shape[sliceSpecs.length]!,
        step: 1,
        isIndex: false,
      });
    }

    const newShape: number[] = [];
    const newStrides: number[] = [];
    let newOffset = this._storage.offset;

    for (let i = 0; i < sliceSpecs.length; i++) {
      const spec = sliceSpecs[i]!;
      const stride = this._storage.strides[i]!;

      newOffset += spec.start * stride;

      if (!spec.isIndex) {
        let dimSize: number;
        if (spec.step > 0) {
          dimSize = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
        } else {
          dimSize = Math.max(0, Math.ceil((spec.start - spec.stop) / Math.abs(spec.step)));
        }
        newShape.push(dimSize);
        newStrides.push(stride * spec.step);
      }
    }

    const slicedStorage = ArrayStorage.fromData(
      this._storage.data,
      newShape,
      this._storage.dtype,
      newStrides,
      newOffset
    );

    const base = this._base ?? this;
    return new NDArray(slicedStorage, base);
  }

  /**
   * Get a single row (convenience method)
   * @param i - Row index
   * @returns Row as 1D or (n-1)D array
   */
  row(i: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('row() requires at least 2 dimensions');
    }
    return this.slice(String(i), ':');
  }

  /**
   * Get a single column (convenience method)
   * @param j - Column index
   * @returns Column as 1D or (n-1)D array
   */
  col(j: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('col() requires at least 2 dimensions');
    }
    return this.slice(':', String(j));
  }

  /**
   * Get a range of rows (convenience method)
   * @param start - Start row index
   * @param stop - Stop row index (exclusive)
   * @returns Rows as array
   */
  rows(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('rows() requires at least 2 dimensions');
    }
    return this.slice(`${start}:${stop}`, ':');
  }

  /**
   * Get a range of columns (convenience method)
   * @param start - Start column index
   * @param stop - Stop column index (exclusive)
   * @returns Columns as array
   */
  cols(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('cols() requires at least 2 dimensions');
    }
    return this.slice(':', `${start}:${stop}`);
  }

  /**
   * Reshape array to a new shape
   * Returns a new array with the specified shape
   * @param shape - New shape (must be compatible with current size)
   * @returns Reshaped array
   */
  reshape(...shape: number[]): NDArray {
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = core.reshape(this, newShape).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base);
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * @returns 1D array containing all elements
   */
  ravel(): NDArray {
    const resultStorage = core.ravel(this).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base);
  }

  /**
   * Put values at specified indices (modifies array in-place)
   * @param indices - Indices at which to place values
   * @param values - Values to put
   */
  put(indices: number[], values: NDArray | number | bigint): void {
    const valuesStorage = values instanceof NDArray ? values._storage : values;
    core.put(this, indices, valuesStorage as never);
  }

  /**
   * Return selected slices of this array along given axis
   * @param condition - Boolean array that selects which entries to return
   * @param axis - Axis along which to take slices
   * @returns Array with selected entries
   */
  compress(condition: NDArray | boolean[], axis?: number): NDArray {
    const condStorage =
      condition instanceof NDArray
        ? condition
        : NDArray.fromStorage(
            ArrayStorage.fromData(
              new Uint8Array(condition.map((b) => (b ? 1 : 0))),
              [condition.length],
              'bool'
            )
          );
    return up(core.compress(condStorage, this, axis));
  }

  /**
   * Construct an array from an index array and a list of arrays to choose from
   * @param choices - Array of NDArrays to choose from
   * @returns New array with selected elements
   */
  choose(choices: NDArray[]): NDArray {
    return up(core.choose(this, choices));
  }

  /**
   * Clip (limit) the values in an array
   * @param a_min - Minimum value (null for no minimum)
   * @param a_max - Maximum value (null for no maximum)
   * @returns Array with values clipped to [a_min, a_max]
   */
  clip(a_min: number | NDArray | null, a_max: number | NDArray | null): NDArray {
    return up(core.clip(this, a_min, a_max));
  }

  /**
   * Integer array indexing (fancy indexing)
   *
   * Select elements using an array of indices.
   * @param indices - Array of integer indices
   * @param axis - Axis along which to index (default: 0)
   * @returns New array with selected elements
   */
  iindex(indices: NDArray | number[] | number[][], axis: number = 0): NDArray {
    let indexArray: number[];
    if (indices instanceof NDArray) {
      indexArray = [];
      for (let i = 0; i < indices.size; i++) {
        const val = indices.storage.iget(i);
        const numVal =
          typeof val === 'bigint' ? Number(val) : val instanceof Complex ? val.re : val;
        indexArray.push(numVal);
      }
    } else if (Array.isArray(indices) && indices.length > 0 && Array.isArray(indices[0])) {
      indexArray = (indices as number[][]).flat();
    } else {
      indexArray = indices as number[];
    }

    return this.take(indexArray, axis);
  }

  /**
   * Boolean array indexing (fancy indexing with mask)
   *
   * Select elements where a boolean mask is true.
   * @param mask - Boolean NDArray mask
   * @param axis - Axis along which to apply the mask
   * @returns New 1D array with selected elements
   */
  bindex(mask: NDArray, axis?: number): NDArray {
    return up(core.compress(mask, this, axis));
  }

  /**
   * String representation of the array
   * @returns String describing the array shape and dtype
   */
  override toString(): string {
    return `NDArray(shape=${JSON.stringify(this.shape)}, dtype=${this.dtype})`;
  }

  /**
   * Convert to nested JavaScript array
   * @returns Nested JavaScript array representation
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override toArray(): any {
    if (this.ndim === 0) {
      return this._storage.iget(0);
    }

    const shape = this.shape;
    const ndim = shape.length;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildNestedArray = (indices: number[], dim: number): any => {
      if (dim === ndim) {
        return this._storage.get(...indices);
      }

      const arr = [];
      for (let i = 0; i < shape[dim]!; i++) {
        indices[dim] = i;
        arr.push(buildNestedArray(indices, dim + 1));
      }
      return arr;
    };

    return buildNestedArray(new Array(ndim), 0);
  }

  /**
   * Return the array as a nested list (same as toArray)
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override tolist(): any {
    return this.toArray();
  }

  /**
   * Return the raw bytes of the array data
   */
  override tobytes(): ArrayBuffer {
    if (this._storage.isCContiguous) {
      const data = this._storage.data;
      const bytesPerElement = data.BYTES_PER_ELEMENT;
      const offset = this._storage.offset * bytesPerElement;
      const length = this.size * bytesPerElement;
      return data.buffer.slice(offset, offset + length) as ArrayBuffer;
    }
    const copy = this.copy();
    const data = copy._storage.data;
    return data.buffer.slice(0, this.size * data.BYTES_PER_ELEMENT) as ArrayBuffer;
  }

  /**
   * Copy an element of an array to a standard scalar and return it
   */
  override item(...args: number[]): number | bigint | Complex {
    if (args.length === 0) {
      if (this.size !== 1) {
        throw new Error('can only convert an array of size 1 to a Python scalar');
      }
      return this._storage.iget(0);
    }
    if (args.length === 1) {
      const flatIdx = args[0]!;
      if (flatIdx < 0 || flatIdx >= this.size) {
        throw new Error(`index ${flatIdx} is out of bounds for size ${this.size}`);
      }
      return this._storage.iget(flatIdx);
    }
    return this.get(args);
  }

  /**
   * Swap the bytes of the array elements
   */
  byteswap(inplace: boolean = false): NDArray {
    const target = inplace ? this : this.copy();
    const data = target._storage.data;
    const bytesPerElement = data.BYTES_PER_ELEMENT;
    if (bytesPerElement === 1) return target;

    const buffer = data.buffer;
    const view = new DataView(buffer);

    for (let i = 0; i < data.length; i++) {
      const byteOffset = i * bytesPerElement;
      if (bytesPerElement === 2) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        view.setUint8(byteOffset, b1);
        view.setUint8(byteOffset + 1, b0);
      } else if (bytesPerElement === 4) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        const b2 = view.getUint8(byteOffset + 2);
        const b3 = view.getUint8(byteOffset + 3);
        view.setUint8(byteOffset, b3);
        view.setUint8(byteOffset + 1, b2);
        view.setUint8(byteOffset + 2, b1);
        view.setUint8(byteOffset + 3, b0);
      } else if (bytesPerElement === 8) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        const b2 = view.getUint8(byteOffset + 2);
        const b3 = view.getUint8(byteOffset + 3);
        const b4 = view.getUint8(byteOffset + 4);
        const b5 = view.getUint8(byteOffset + 5);
        const b6 = view.getUint8(byteOffset + 6);
        const b7 = view.getUint8(byteOffset + 7);
        view.setUint8(byteOffset, b7);
        view.setUint8(byteOffset + 1, b6);
        view.setUint8(byteOffset + 2, b5);
        view.setUint8(byteOffset + 3, b4);
        view.setUint8(byteOffset + 4, b3);
        view.setUint8(byteOffset + 5, b2);
        view.setUint8(byteOffset + 6, b1);
        view.setUint8(byteOffset + 7, b0);
      }
    }
    return target;
  }

  /**
   * Return a view of the array with a different dtype
   */
  view(dtype?: DType): NDArray {
    if (!dtype || dtype === this.dtype) {
      return NDArray.fromStorage(this._storage, this._base ?? this);
    }
    const oldSize = getDTypeSize(this.dtype as DType);
    const newSize = getDTypeSize(dtype);
    if (oldSize !== newSize) {
      throw new Error(
        'When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.'
      );
    }
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) throw new Error(`Unsupported dtype: ${dtype}`);
    const data = this._storage.data;
    const byteOffset = data.byteOffset + this._storage.offset * oldSize;
    const newData = new Constructor(data.buffer as ArrayBuffer, byteOffset, this.size);
    const storage = ArrayStorage.fromData(
      newData as TypedArray,
      [...this.shape],
      dtype,
      [...this._storage.strides],
      0
    );
    return NDArray.fromStorage(storage, this._base ?? this);
  }

  /**
   * Write array to a file (stub)
   */
  tofile(_file: string, _sep: string = '', _format: string = ''): void {
    throw new Error(
      'tofile() requires file system access. Use the node module: import { save } from "numpy-ts/node"'
    );
  }

  /**
   * Round an array to the given number of decimals (alias for around)
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  round(decimals: number = 0): NDArray {
    return this.around(decimals);
  }

  /**
   * Return the complex conjugate, element-wise (alias for conj)
   * @returns Complex conjugate of the array
   */
  conjugate(): NDArray {
    return this.conj();
  }

  /**
   * Round an array to the given number of decimals
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  around(decimals: number = 0): NDArray {
    return up(core.around(this, decimals));
  }

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns boolean
   */
  allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    return core.allclose(this, other, rtol, atol);
  }

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns Boolean array
   */
  isclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
    return up(core.isclose(this, other, rtol, atol));
  }

  /**
   * Compute the weighted average along the specified axis
   * @param weights - Array of weights (optional)
   * @param axis - Axis along which to compute average
   * @returns Weighted average of array elements
   */
  average(weights?: NDArray, axis?: number): NDArray | number | Complex {
    const r = core.average(this, axis, weights);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Dot product (matching NumPy behavior)
   * @param other - Array to dot with
   * @returns Result of dot product
   */
  dot(other: NDArray): NDArray | number | bigint | Complex {
    const r = core.dot(this, other);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Sum of diagonal elements (trace)
   * @returns Sum of diagonal elements
   */
  trace(): number | bigint | Complex {
    return core.trace(this);
  }

  /**
   * Inner product (contracts over last axes of both arrays)
   * @param other - Array to compute inner product with
   * @returns Inner product result
   */
  inner(other: NDArray): NDArray | number | bigint | Complex {
    const r = core.inner(this, other);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Tensor dot product along specified axes
   * @param other - Array to contract with
   * @param axes - Axes to contract
   * @returns Tensor dot product result
   */
  tensordot(
    other: NDArray,
    axes: number | [number[], number[]] = 2
  ): NDArray | number | bigint | Complex {
    const r = core.tensordot(this, other, axes);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Returns both quotient and remainder (floor divide and modulo)
   * @param divisor - Array or scalar divisor
   * @returns Tuple of [quotient, remainder] arrays
   */
  divmod(divisor: NDArray | number): [NDArray, NDArray] {
    const r = core.divmod(this, divisor);
    return [up(r[0]), up(r[1])] as [NDArray, NDArray];
  }

  /**
   * Find indices where elements should be inserted to maintain order
   * @param v - Values to insert
   * @param side - "left" or "right" side to insert
   * @returns Indices where values should be inserted
   */
  searchsorted(v: NDArray, side: 'left' | 'right' = 'left'): NDArray {
    return up(core.searchsorted(this, v, side));
  }

  // ========================================
  // Unary operations
  // ========================================

  /**
   * Square root of each element
   * Promotes integer types to float64
   */
  sqrt(): NDArray {
    return up(core.sqrt(this));
  }

  /**
   * Natural exponential (e^x) of each element
   * Promotes integer types to float64
   */
  exp(): NDArray {
    return up(core.exp(this));
  }

  /**
   * Base-2 exponential (2^x) of each element
   * Promotes integer types to float64
   */
  exp2(): NDArray {
    return up(core.exp2(this));
  }

  /**
   * Exponential minus one (e^x - 1) of each element
   * More accurate than exp(x) - 1 for small x
   */
  expm1(): NDArray {
    return up(core.expm1(this));
  }

  /**
   * Natural logarithm (ln) of each element
   * Promotes integer types to float64
   */
  log(): NDArray {
    return up(core.log(this));
  }

  /**
   * Base-2 logarithm of each element
   * Promotes integer types to float64
   */
  log2(): NDArray {
    return up(core.log2(this));
  }

  /**
   * Base-10 logarithm of each element
   * Promotes integer types to float64
   */
  log10(): NDArray {
    return up(core.log10(this));
  }

  /**
   * Natural logarithm of (1 + x) of each element
   * More accurate than log(1 + x) for small x
   */
  log1p(): NDArray {
    return up(core.log1p(this));
  }

  /**
   * Absolute value of each element
   */
  absolute(): NDArray {
    return up(core.absolute(this));
  }

  /**
   * Numerical negative (element-wise negation)
   */
  negative(): NDArray {
    return up(core.negative(this));
  }

  /**
   * Sign of each element (-1, 0, or 1)
   */
  sign(): NDArray {
    return up(core.sign(this));
  }

  /**
   * Numerical positive (element-wise +x)
   * @returns Copy of the array
   */
  positive(): NDArray {
    return up(core.positive(this));
  }

  /**
   * Element-wise reciprocal (1/x)
   */
  reciprocal(): NDArray {
    return up(core.reciprocal(this));
  }

  /**
   * Return the ceiling of the input, element-wise
   */
  ceil(): NDArray {
    return up(core.ceil(this));
  }

  /**
   * Round to nearest integer towards zero
   */
  fix(): NDArray {
    return up(core.fix(this));
  }

  /**
   * Return the floor of the input, element-wise
   */
  floor(): NDArray {
    return up(core.floor(this));
  }

  /**
   * Round elements to the nearest integer
   */
  rint(): NDArray {
    return up(core.rint(this));
  }

  /**
   * Return the truncated value of the input, element-wise
   */
  trunc(): NDArray {
    return up(core.trunc(this));
  }

  /**
   * Sine of each element (in radians)
   * Promotes integer types to float64
   */
  sin(): NDArray {
    return up(core.sin(this));
  }

  /**
   * Cosine of each element (in radians)
   * Promotes integer types to float64
   */
  cos(): NDArray {
    return up(core.cos(this));
  }

  /**
   * Tangent of each element (in radians)
   * Promotes integer types to float64
   */
  tan(): NDArray {
    return up(core.tan(this));
  }

  /**
   * Inverse sine of each element
   * Promotes integer types to float64
   */
  arcsin(): NDArray {
    return up(core.arcsin(this));
  }

  /**
   * Inverse cosine of each element
   * Promotes integer types to float64
   */
  arccos(): NDArray {
    return up(core.arccos(this));
  }

  /**
   * Inverse tangent of each element
   * Promotes integer types to float64
   */
  arctan(): NDArray {
    return up(core.arctan(this));
  }

  /**
   * Convert angles from radians to degrees
   */
  degrees(): NDArray {
    return up(core.degrees(this));
  }

  /**
   * Convert angles from degrees to radians
   */
  radians(): NDArray {
    return up(core.radians(this));
  }

  /**
   * Hyperbolic sine of each element
   * Promotes integer types to float64
   */
  sinh(): NDArray {
    return up(core.sinh(this));
  }

  /**
   * Hyperbolic cosine of each element
   * Promotes integer types to float64
   */
  cosh(): NDArray {
    return up(core.cosh(this));
  }

  /**
   * Hyperbolic tangent of each element
   * Promotes integer types to float64
   */
  tanh(): NDArray {
    return up(core.tanh(this));
  }

  /**
   * Inverse hyperbolic sine of each element
   * Promotes integer types to float64
   */
  arcsinh(): NDArray {
    return up(core.arcsinh(this));
  }

  /**
   * Inverse hyperbolic cosine of each element
   * Promotes integer types to float64
   */
  arccosh(): NDArray {
    return up(core.arccosh(this));
  }

  /**
   * Inverse hyperbolic tangent of each element
   * Promotes integer types to float64
   */
  arctanh(): NDArray {
    return up(core.arctanh(this));
  }

  /**
   * Bitwise NOT (inversion) element-wise
   */
  bitwise_not(): NDArray {
    return up(core.bitwise_not(this));
  }

  /**
   * Invert (bitwise NOT) element-wise - alias for bitwise_not
   */
  invert(): NDArray {
    return up(core.invert(this));
  }

  /**
   * Logical NOT element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_not(): NDArray {
    return up(core.logical_not(this));
  }

  /**
   * Test element-wise for finiteness (not infinity and not NaN)
   */
  isfinite(): NDArray {
    return up(core.isfinite(this));
  }

  /**
   * Test element-wise for positive or negative infinity
   */
  isinf(): NDArray {
    return up(core.isinf(this));
  }

  /**
   * Test element-wise for NaN (Not a Number)
   */
  isnan(): NDArray {
    return up(core.isnan(this));
  }

  /**
   * Test element-wise for NaT (Not a Time)
   * @returns Boolean array (always false without datetime support)
   */
  isnat(): NDArray {
    return up(core.isnat(this));
  }

  /**
   * Returns element-wise True where signbit is set (less than zero)
   */
  signbit(): NDArray {
    return up(core.signbit(this));
  }

  /**
   * Return the distance between x and the nearest adjacent number
   */
  spacing(): NDArray {
    return up(core.spacing(this));
  }

  /**
   * Element-wise cube root
   * Promotes integer types to float64
   */
  cbrt(): NDArray {
    return up(core.cbrt(this));
  }

  /**
   * Element-wise absolute value (always returns float)
   */
  fabs(): NDArray {
    return up(core.fabs(this));
  }

  /**
   * Element-wise square (x**2)
   */
  square(): NDArray {
    return up(core.square(this));
  }

  /**
   * Return the complex conjugate, element-wise
   */
  conj(): NDArray {
    return up(core.conj(this));
  }

  /**
   * Return a flattened copy of the array
   * @returns 1D array containing all elements
   */
  flatten(): NDArray {
    return up(core.flatten(this));
  }

  /**
   * Find the indices of array elements that are non-zero, grouped by element
   * @returns 2D array of shape (N, ndim)
   */
  argwhere(): NDArray {
    return up(core.argwhere(this));
  }

  // ========================================
  // Binary operations
  // ========================================

  /**
   * Element-wise addition
   * @param other - Array or scalar to add
   */
  add(other: NDArray | number): NDArray {
    return up(core.add(this, other));
  }

  /**
   * Element-wise subtraction
   * @param other - Array or scalar to subtract
   */
  subtract(other: NDArray | number): NDArray {
    return up(core.subtract(this, other));
  }

  /**
   * Element-wise multiplication
   * @param other - Array or scalar to multiply
   */
  multiply(other: NDArray | number): NDArray {
    return up(core.multiply(this, other));
  }

  /**
   * Element-wise division
   * @param other - Array or scalar to divide by
   */
  divide(other: NDArray | number): NDArray {
    return up(core.divide(this, other));
  }

  /**
   * Element-wise modulo operation
   * @param other - Array or scalar divisor
   */
  mod(other: NDArray | number): NDArray {
    return up(core.mod(this, other));
  }

  /**
   * Element-wise floor division
   * @param other - Array or scalar to divide by
   */
  floor_divide(other: NDArray | number): NDArray {
    return up(core.floor_divide(this, other));
  }

  /**
   * Raise elements to power
   * @param exponent - Power to raise to (array or scalar)
   */
  power(exponent: NDArray | number): NDArray {
    return up(core.power(this, exponent));
  }

  /**
   * Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))
   * @param x2 - Second operand
   */
  logaddexp(x2: NDArray | number): NDArray {
    return up(core.logaddexp(this, x2));
  }

  /**
   * Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)
   * @param x2 - Second operand
   */
  logaddexp2(x2: NDArray | number): NDArray {
    return up(core.logaddexp2(this, x2));
  }

  /**
   * Element-wise arc tangent of this/other choosing the quadrant correctly
   * @param other - x-coordinates
   */
  arctan2(other: NDArray | number): NDArray {
    return up(core.arctan2(this, other));
  }

  /**
   * Given the "legs" of a right triangle, return its hypotenuse
   * @param other - Second leg
   */
  hypot(other: NDArray | number): NDArray {
    return up(core.hypot(this, other));
  }

  /**
   * Element-wise greater than comparison
   * @returns Boolean array
   */
  greater(other: NDArray | number): NDArray {
    return up(core.greater(this, other));
  }

  /**
   * Element-wise greater than or equal comparison
   * @returns Boolean array
   */
  greater_equal(other: NDArray | number): NDArray {
    return up(core.greater_equal(this, other));
  }

  /**
   * Element-wise less than comparison
   * @returns Boolean array
   */
  less(other: NDArray | number): NDArray {
    return up(core.less(this, other));
  }

  /**
   * Element-wise less than or equal comparison
   * @returns Boolean array
   */
  less_equal(other: NDArray | number): NDArray {
    return up(core.less_equal(this, other));
  }

  /**
   * Element-wise equality comparison
   * @returns Boolean array
   */
  equal(other: NDArray | number): NDArray {
    return up(core.equal(this, other));
  }

  /**
   * Element-wise not equal comparison
   * @returns Boolean array
   */
  not_equal(other: NDArray | number): NDArray {
    return up(core.not_equal(this, other));
  }

  /**
   * Bitwise AND element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_and(other: NDArray | number): NDArray {
    return up(core.bitwise_and(this, other));
  }

  /**
   * Bitwise OR element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_or(other: NDArray | number): NDArray {
    return up(core.bitwise_or(this, other));
  }

  /**
   * Bitwise XOR element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_xor(other: NDArray | number): NDArray {
    return up(core.bitwise_xor(this, other));
  }

  /**
   * Left shift elements by positions
   * @param shift - Shift amount
   */
  left_shift(shift: NDArray | number): NDArray {
    return up(core.left_shift(this, shift));
  }

  /**
   * Right shift elements by positions
   * @param shift - Shift amount
   */
  right_shift(shift: NDArray | number): NDArray {
    return up(core.right_shift(this, shift));
  }

  /**
   * Logical AND element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_and(other: NDArray | number): NDArray {
    return up(core.logical_and(this, other));
  }

  /**
   * Logical OR element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_or(other: NDArray | number): NDArray {
    return up(core.logical_or(this, other));
  }

  /**
   * Logical XOR element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_xor(other: NDArray | number): NDArray {
    return up(core.logical_xor(this, other));
  }

  /**
   * Change the sign of x1 to that of x2, element-wise
   * @param x2 - Values whose sign is used
   */
  copysign(x2: NDArray | number): NDArray {
    return up(core.copysign(this, x2));
  }

  /**
   * Return the next floating-point value after x1 towards x2, element-wise
   * @param x2 - Direction to look
   */
  nextafter(x2: NDArray | number): NDArray {
    return up(core.nextafter(this, x2));
  }

  /**
   * Element-wise remainder (same as mod)
   * @param divisor - Array or scalar divisor
   */
  remainder(divisor: NDArray | number): NDArray {
    return up(core.remainder(this, divisor));
  }

  /**
   * Heaviside step function
   * @param x2 - Value to use when this array element is 0
   */
  heaviside(x2: NDArray | number): NDArray {
    return up(core.heaviside(this, x2));
  }

  /**
   * Matrix multiplication
   * @param other - Array to multiply with
   */
  matmul(other: NDArray): NDArray {
    return up(core.matmul(this, other));
  }

  /**
   * Outer product (flattens inputs then computes a[i]*b[j])
   * @param other - Array to compute outer product with
   */
  outer(other: NDArray): NDArray {
    return up(core.outer(this, other));
  }

  // ========================================
  // Reduction operations
  // ========================================

  /**
   * Sum array elements over a given axis
   */
  sum(axis?: number, keepdims: boolean = false): NDArray | number | bigint | Complex {
    const r = core.sum(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the arithmetic mean along the specified axis
   */
  mean(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.mean(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Product of array elements over a given axis
   */
  prod(axis?: number, keepdims: boolean = false): NDArray | number | bigint | Complex {
    const r = core.prod(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the maximum along a given axis
   */
  max(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.max(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the minimum along a given axis
   */
  min(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.min(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Peak to peak (maximum - minimum) value along a given axis
   */
  ptp(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.ptp(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the sum of array elements, treating NaNs as zero
   */
  nansum(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.nansum(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the product of array elements, treating NaNs as ones
   */
  nanprod(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.nanprod(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the arithmetic mean, ignoring NaNs
   */
  nanmean(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.nanmean(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return minimum of an array, ignoring NaNs
   */
  nanmin(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.nanmin(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return maximum of an array, ignoring NaNs
   */
  nanmax(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const r = core.nanmax(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Indices of the minimum values along an axis
   */
  argmin(axis?: number): NDArray | number {
    const r = core.argmin(this, axis);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Indices of the maximum values along an axis
   */
  argmax(axis?: number): NDArray | number {
    const r = core.argmax(this, axis);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the indices of the minimum values, ignoring NaNs
   */
  nanargmin(axis?: number): NDArray | number {
    const r = core.nanargmin(this, axis);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Return the indices of the maximum values, ignoring NaNs
   */
  nanargmax(axis?: number): NDArray | number {
    const r = core.nanargmax(this, axis);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute variance along the specified axis
   * @param axis - Axis along which to compute variance
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  var(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const r = core.variance(this, axis, ddof, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute standard deviation along the specified axis
   * @param axis - Axis along which to compute std
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  std(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const r = core.std(this, axis, ddof, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the variance, ignoring NaNs
   * @param axis - Axis along which to compute variance
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  nanvar(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const r = core.nanvar(this, axis, ddof, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the standard deviation, ignoring NaNs
   * @param axis - Axis along which to compute std
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  nanstd(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const r = core.nanstd(this, axis, ddof, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Test whether all array elements along a given axis evaluate to True
   */
  all(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const r = core.all(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Test whether any array elements along a given axis evaluate to True
   */
  any(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const r = core.any(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the median along the specified axis
   */
  median(axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.median(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the median, ignoring NaNs
   */
  nanmedian(axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.nanmedian(this, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the q-th percentile of the data along the specified axis
   * @param q - Percentile to compute (0-100)
   */
  percentile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.percentile(this, q, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the q-th quantile of the data along the specified axis
   * @param q - Quantile to compute (0-1)
   */
  quantile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.quantile(this, q, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the q-th quantile, ignoring NaNs
   * @param q - Quantile to compute (0-1)
   */
  nanquantile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.nanquantile(this, q, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  /**
   * Compute the q-th percentile, ignoring NaNs
   * @param q - Percentile to compute (0-100)
   */
  nanpercentile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const r = core.nanpercentile(this, q, axis, keepdims);
    return r instanceof NDArrayCore ? up(r) : r;
  }

  // ========================================
  // Passthrough operations
  // ========================================

  /**
   * Return the cumulative sum of elements along a given axis
   */
  cumsum(axis?: number): NDArray {
    return up(core.cumsum(this, axis));
  }

  /**
   * Return the cumulative product of elements along a given axis
   */
  cumprod(axis?: number): NDArray {
    return up(core.cumprod(this, axis));
  }

  /**
   * Return the cumulative sum of elements, treating NaNs as zero
   */
  nancumsum(axis?: number): NDArray {
    return up(core.nancumsum(this, axis));
  }

  /**
   * Return the cumulative product of elements, treating NaNs as one
   */
  nancumprod(axis?: number): NDArray {
    return up(core.nancumprod(this, axis));
  }

  /**
   * Return a sorted copy of the array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  sort(axis: number = -1): NDArray {
    return up(core.sort(this, axis));
  }

  /**
   * Returns the indices that would sort this array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  argsort(axis: number = -1): NDArray {
    return up(core.argsort(this, axis));
  }

  /**
   * Partially sort the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  partition(kth: number, axis: number = -1): NDArray {
    return up(core.partition(this, kth, axis));
  }

  /**
   * Returns indices that would partition the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  argpartition(kth: number, axis: number = -1): NDArray {
    return up(core.argpartition(this, kth, axis));
  }

  /**
   * Return specified diagonals
   * @param offset - Offset of the diagonal from the main diagonal
   * @param axis1 - First axis of the 2-D sub-arrays
   * @param axis2 - Second axis of the 2-D sub-arrays
   */
  diagonal(offset: number = 0, axis1: number = 0, axis2: number = 1): NDArray {
    return up(core.diagonal(this, offset, axis1, axis2));
  }

  /**
   * Return a new array with the specified shape
   * If larger, filled with repeated copies of the original data
   * @param newShape - Shape of the resized array
   */
  resize(newShape: number[]): NDArray {
    return up(core.resize(this, newShape));
  }

  /**
   * Calculate the n-th discrete difference along the given axis
   * @param n - Number of times values are differenced (default: 1)
   * @param axis - Axis along which to compute difference (default: -1)
   */
  diff(n: number = 1, axis: number = -1): NDArray {
    return up(core.diff(this, n, axis));
  }

  /**
   * Take elements from array along an axis
   * @param indices - Indices of elements to take
   * @param axis - Axis along which to take
   */
  take(indices: number[], axis?: number): NDArray {
    return up(core.take(this, indices, axis));
  }

  /**
   * Repeat elements of an array
   * @param repeats - Number of repetitions for each element
   * @param axis - Axis along which to repeat
   */
  repeat(repeats: number | number[], axis?: number): NDArray {
    return up(core.repeat(this, repeats, axis));
  }

  /**
   * Transpose array (permute dimensions)
   * @param axes - Permutation of axes. If undefined, reverse the dimensions
   * @returns Transposed array (always a view)
   */
  transpose(axes?: number[]): NDArray {
    return up(core.transpose(this, axes));
  }

  /**
   * Remove axes of length 1
   * @param axis - Axis to squeeze
   * @returns Array with specified dimensions removed (always a view)
   */
  squeeze(axis?: number): NDArray {
    return up(core.squeeze(this, axis));
  }

  /**
   * Expand the shape by inserting a new axis of length 1
   * @param axis - Position where new axis is placed
   * @returns Array with additional dimension (always a view)
   */
  expand_dims(axis: number): NDArray {
    return up(core.expand_dims(this, axis));
  }

  /**
   * Swap two axes of an array
   * @param axis1 - First axis
   * @param axis2 - Second axis
   * @returns Array with swapped axes (always a view)
   */
  swapaxes(axis1: number, axis2: number): NDArray {
    return up(core.swapaxes(this, axis1, axis2));
  }

  /**
   * Move axes to new positions
   * @param source - Original positions of axes to move
   * @param destination - New positions for axes
   * @returns Array with moved axes (always a view)
   */
  moveaxis(source: number | number[], destination: number | number[]): NDArray {
    return up(core.moveaxis(this, source, destination));
  }

  // ========================================
  // Array return operations
  // ========================================

  /**
   * Return the indices of non-zero elements
   * @returns Tuple of arrays, one for each dimension
   */
  nonzero(): NDArray[] {
    return core.nonzero(this).map(up);
  }
}

/**
 * Return coordinate matrices from coordinate vectors
 * @param arrays - 1D coordinate arrays
 * @param indexing - 'xy' (Cartesian, default) or 'ij' (matrix indexing)
 * @returns Array of coordinate grids
 */
export function meshgrid(...args: (NDArray | { indexing?: 'xy' | 'ij' })[]): NDArray[] {
  let arrays: NDArray[] = [];
  let indexing: 'xy' | 'ij' = 'xy';

  for (const arg of args) {
    if (arg instanceof NDArray) {
      arrays.push(arg);
    } else if (typeof arg === 'object' && 'indexing' in arg) {
      indexing = arg.indexing || 'xy';
    }
  }

  if (arrays.length === 0) {
    return [];
  }

  if (arrays.length === 1) {
    return [arrays[0]!.copy()];
  }

  const sizes = arrays.map((a) => a.size);

  if (indexing === 'xy' && arrays.length >= 2) {
    arrays = [arrays[1]!, arrays[0]!, ...arrays.slice(2)];
    [sizes[0], sizes[1]] = [sizes[1]!, sizes[0]!];
  }

  const outputShape = sizes;
  const ndim = outputShape.length;

  const results: NDArray[] = [];

  for (let i = 0; i < arrays.length; i++) {
    const inputArr = arrays[i]!;
    const inputSize = inputArr.size;

    const broadcastShape: number[] = new Array(ndim).fill(1);
    broadcastShape[i] = inputSize;

    const reshaped = inputArr.reshape(...broadcastShape);
    const resultStorage = core.broadcast_to(reshaped, outputShape);
    const result = NDArray.fromStorage(resultStorage.storage.copy());
    results.push(result);
  }

  if (indexing === 'xy' && results.length >= 2) {
    [results[0], results[1]] = [results[1]!, results[0]!];
  }

  return results;
}
