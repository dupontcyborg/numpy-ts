// AUTO-GENERATED - DO NOT EDIT
// Run `pnpm run generate` to regenerate this file from scripts/ndarray-methods.ts

import { Complex } from '../common/complex';
import {
  type DType,
  getDTypeSize,
  getTypedArrayConstructor,
  isBigIntDType,
  isComplexDType,
  type Scalar,
  type TypedArray,
} from '../common/dtype';
import type {
  Abs,
  BoolArith,
  Divide,
  MathBinary,
  MathResult,
  Power,
  Promote,
  ReductionAccum,
  StdVar,
  TrueDivide,
} from '../common/dtype-promotion';
import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import * as core from '../core';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

export class NDArray<D extends DType = DType> extends NDArrayCore<D> {
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
   * Iterator protocol - iterate over the first axis
   * For 1D arrays, yields elements; for ND arrays, yields (N-1)D subarrays
   */
  override *[Symbol.iterator](): Iterator<NDArray<D> | Scalar<D>> {
    if (this.ndim === 0) {
      yield this._storage.iget(0) as Scalar<D>;
    } else if (this.ndim === 1) {
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this._storage.iget(i) as Scalar<D>;
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
  override get(indices: number[]): Scalar<D> {
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`,
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`,
        );
      }
      return normalized;
    });

    return this._storage.get(...normalizedIndices) as Scalar<D>;
  }

  /**
   * Set a single element in the array
   * @param indices - Array of indices, one per dimension
   * @param value - Value to set
   */
  override set(
    indices: number[],
    value: number | bigint | Complex | { re: number; im: number },
  ): void {
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`,
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`,
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
   * Cast array to a different dtype
   * @param dtype - Target dtype (tracked in the result type)
   * @param copy - Whether to copy when the dtype is unchanged (default true)
   * @returns A new NDArray with the target dtype
   */
  override astype<E extends DType>(dtype: E, copy: boolean = true): NDArray<E> {
    return super.astype(dtype, copy) as NDArray<E>;
  }

  /**
   * Get a single row (convenience method)
   * @param i - Row index
   * @returns Row as 1D or (n-1)D array
   */
  row(i: number): NDArray<D> {
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
  col(j: number): NDArray<D> {
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
  rows(start: number, stop: number): NDArray<D> {
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
  cols(start: number, stop: number): NDArray<D> {
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
  reshape(...shape: number[]): NDArray<D> {
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = core.reshape(this, newShape).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base) as NDArray<D>;
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * @returns 1D array containing all elements
   */
  ravel(): NDArray<D> {
    const resultStorage = core.ravel(this).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base) as NDArray<D>;
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
  compress(condition: NDArray | boolean[], axis?: number): NDArray<D> {
    const condStorage =
      condition instanceof NDArray
        ? condition
        : NDArray.fromStorage(
            ArrayStorage.fromData(
              new Uint8Array(condition.map((b) => (b ? 1 : 0))),
              [condition.length],
              'bool',
            ),
          );
    return up(core.compress(condStorage, this, axis)) as NDArray<D>;
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
  clip(a_min: number | NDArray | null, a_max: number | NDArray | null): NDArray<D> {
    return up(core.clip(this, a_min, a_max)) as NDArray<D>;
  }

  /**
   * Integer array indexing (fancy indexing)
   *
   * Select elements using an array of indices.
   * @param indices - Array of integer indices
   * @param axis - Axis along which to index (default: 0)
   * @returns New array with selected elements
   */
  iindex(indices: NDArray | number[] | number[][], axis: number = 0): NDArray<D> {
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
  bindex(mask: NDArray, axis?: number): NDArray<D> {
    return up(core.compress(mask, this, axis)) as NDArray<D>;
  }

  /**
   * String representation of the array
   * @returns String describing the array shape and dtype
   */
  override toString(): string {
    return core.array_str(this);
  }

  /**
   * Return the raw bytes of the array data
   */
  override tobytes(): ArrayBuffer {
    if (this._storage.isCContiguous) {
      const data = this._storage.data;
      const bytesPerElement = data.BYTES_PER_ELEMENT;
      const offset = data.byteOffset + this._storage.offset * bytesPerElement;
      const length = this.size * bytesPerElement;
      return data.buffer.slice(offset, offset + length) as ArrayBuffer;
    }
    const copy = this.copy();
    const data = copy._storage.data;
    return data.buffer.slice(
      data.byteOffset,
      data.byteOffset + this.size * data.BYTES_PER_ELEMENT,
    ) as ArrayBuffer;
  }

  /**
   * Copy an element of an array to a standard scalar and return it
   */
  override item(...args: number[]): Scalar<D> {
    if (args.length === 0) {
      if (this.size !== 1) {
        throw new Error('can only convert an array of size 1 to a Python scalar');
      }
      return this._storage.iget(0) as Scalar<D>;
    }
    if (args.length === 1) {
      const flatIdx = args[0]!;
      if (flatIdx < 0 || flatIdx >= this.size) {
        throw new Error(`index ${flatIdx} is out of bounds for size ${this.size}`);
      }
      return this._storage.iget(flatIdx) as Scalar<D>;
    }
    return this.get(args);
  }

  /**
   * Swap the bytes of the array elements
   */
  byteswap(inplace: boolean = false): NDArray<D> {
    const target = inplace ? this : this.copy();
    const data = target._storage.data;
    const bytesPerElement = data.BYTES_PER_ELEMENT;
    if (bytesPerElement === 1) return target;

    const buffer = data.buffer;
    const dataByteOff = data.byteOffset;
    const view = new DataView(buffer);

    for (let i = 0; i < data.length; i++) {
      const byteOffset = dataByteOff + i * bytesPerElement;
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
  view<E extends DType = D>(dtype?: E): NDArray<E> {
    if (!dtype || dtype === (this.dtype as DType)) {
      return NDArray.fromStorage(this._storage, this._base ?? this) as NDArray<E>;
    }
    const oldSize = getDTypeSize(this.dtype as DType);
    const newSize = getDTypeSize(dtype);
    if (oldSize !== newSize) {
      throw new Error(
        'When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.',
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
      0,
    );
    return NDArray.fromStorage(storage, this._base ?? this) as NDArray<E>;
  }

  /**
   * Write array to a file (stub)
   */
  tofile(_file: string, _sep: string = '', _format: string = ''): void {
    throw new Error(
      'tofile() requires file system access. Use the node module: import { save } from "numpy-ts/node"',
    );
  }

  /**
   * Round an array to the given number of decimals (alias for around)
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  round(decimals: number = 0): NDArray<D> {
    return this.around(decimals);
  }

  /**
   * Return the complex conjugate, element-wise (alias for conj)
   * @returns Complex conjugate of the array
   */
  conjugate(): NDArray<D> {
    return this.conj();
  }

  /**
   * Round an array to the given number of decimals
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  around(decimals: number = 0): NDArray<D> {
    return up(core.around(this, decimals)) as NDArray<D>;
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
  isclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): NDArray<'bool'> {
    return up(core.isclose(this, other, rtol, atol)) as NDArray<'bool'>;
  }

  /**
   * Compute the weighted average along the specified axis
   * @param weights - Array of weights (optional)
   * @param axis - Axis along which to compute average
   * @returns Weighted average of array elements
   */
  average(weights?: NDArray, axis?: number): NDArray | number | Complex {
    const r = core.average(this, axis, weights, false, false) as NDArrayCore | number | Complex;
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
  trace(): NDArray | number | bigint | Complex {
    const r = core.trace(this);
    return r instanceof NDArrayCore ? up(r) : r;
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
    axes: number | [number[], number[]] = 2,
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
  searchsorted(v: NDArray, side: 'left' | 'right' = 'left'): NDArray<'int64'> {
    return up(core.searchsorted(this, v, side)) as NDArray<'int64'>;
  }

  // ========================================
  // Unary operations
  // ========================================

  /**
   * Square root of each element
   * Promotes integer types to float64
   */
  sqrt(): NDArray<MathResult<D>> {
    return up(core.sqrt(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Natural exponential (e^x) of each element
   * Promotes integer types to float64
   */
  exp(): NDArray<MathResult<D>> {
    return up(core.exp(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Base-2 exponential (2^x) of each element
   * Promotes integer types to float64
   */
  exp2(): NDArray<MathResult<D>> {
    return up(core.exp2(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Exponential minus one (e^x - 1) of each element
   * More accurate than exp(x) - 1 for small x
   */
  expm1(): NDArray<MathResult<D>> {
    return up(core.expm1(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Natural logarithm (ln) of each element
   * Promotes integer types to float64
   */
  log(): NDArray<MathResult<D>> {
    return up(core.log(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Base-2 logarithm of each element
   * Promotes integer types to float64
   */
  log2(): NDArray<MathResult<D>> {
    return up(core.log2(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Base-10 logarithm of each element
   * Promotes integer types to float64
   */
  log10(): NDArray<MathResult<D>> {
    return up(core.log10(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Natural logarithm of (1 + x) of each element
   * More accurate than log(1 + x) for small x
   */
  log1p(): NDArray<MathResult<D>> {
    return up(core.log1p(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Absolute value of each element
   */
  absolute(): NDArray<Abs<D>> {
    return up(core.absolute(this)) as NDArray<Abs<D>>;
  }

  /**
   * Numerical negative (element-wise negation)
   */
  negative(): NDArray<D> {
    return up(core.negative(this)) as NDArray<D>;
  }

  /**
   * Sign of each element (-1, 0, or 1)
   */
  sign(): NDArray<D> {
    return up(core.sign(this)) as NDArray<D>;
  }

  /**
   * Numerical positive (element-wise +x)
   * @returns Copy of the array
   */
  positive(): NDArray<D> {
    return up(core.positive(this)) as NDArray<D>;
  }

  /**
   * Element-wise reciprocal (1/x)
   */
  reciprocal(): NDArray<D> {
    return up(core.reciprocal(this)) as NDArray<D>;
  }

  /**
   * Return the ceiling of the input, element-wise
   */
  ceil(): NDArray<MathResult<D>> {
    return up(core.ceil(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Round to nearest integer towards zero
   */
  fix(): NDArray<MathResult<D>> {
    return up(core.fix(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Return the floor of the input, element-wise
   */
  floor(): NDArray<MathResult<D>> {
    return up(core.floor(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Round elements to the nearest integer
   */
  rint(): NDArray<MathResult<D>> {
    return up(core.rint(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Return the truncated value of the input, element-wise
   */
  trunc(): NDArray<MathResult<D>> {
    return up(core.trunc(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Sine of each element (in radians)
   * Promotes integer types to float64
   */
  sin(): NDArray<MathResult<D>> {
    return up(core.sin(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Cosine of each element (in radians)
   * Promotes integer types to float64
   */
  cos(): NDArray<MathResult<D>> {
    return up(core.cos(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Tangent of each element (in radians)
   * Promotes integer types to float64
   */
  tan(): NDArray<MathResult<D>> {
    return up(core.tan(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse sine of each element
   * Promotes integer types to float64
   */
  arcsin(): NDArray<MathResult<D>> {
    return up(core.arcsin(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse cosine of each element
   * Promotes integer types to float64
   */
  arccos(): NDArray<MathResult<D>> {
    return up(core.arccos(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse tangent of each element
   * Promotes integer types to float64
   */
  arctan(): NDArray<MathResult<D>> {
    return up(core.arctan(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Convert angles from radians to degrees
   */
  degrees(): NDArray<MathResult<D>> {
    return up(core.degrees(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Convert angles from degrees to radians
   */
  radians(): NDArray<MathResult<D>> {
    return up(core.radians(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Hyperbolic sine of each element
   * Promotes integer types to float64
   */
  sinh(): NDArray<MathResult<D>> {
    return up(core.sinh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Hyperbolic cosine of each element
   * Promotes integer types to float64
   */
  cosh(): NDArray<MathResult<D>> {
    return up(core.cosh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Hyperbolic tangent of each element
   * Promotes integer types to float64
   */
  tanh(): NDArray<MathResult<D>> {
    return up(core.tanh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse hyperbolic sine of each element
   * Promotes integer types to float64
   */
  arcsinh(): NDArray<MathResult<D>> {
    return up(core.arcsinh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse hyperbolic cosine of each element
   * Promotes integer types to float64
   */
  arccosh(): NDArray<MathResult<D>> {
    return up(core.arccosh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Inverse hyperbolic tangent of each element
   * Promotes integer types to float64
   */
  arctanh(): NDArray<MathResult<D>> {
    return up(core.arctanh(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Bitwise NOT (inversion) element-wise
   */
  bitwise_not(): NDArray<D> {
    return up(core.bitwise_not(this)) as NDArray<D>;
  }

  /**
   * Invert (bitwise NOT) element-wise - alias for bitwise_not
   */
  invert(): NDArray<D> {
    return up(core.invert(this)) as NDArray<D>;
  }

  /**
   * Logical NOT element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_not(): NDArray<'bool'> {
    return up(core.logical_not(this)) as NDArray<'bool'>;
  }

  /**
   * Test element-wise for finiteness (not infinity and not NaN)
   */
  isfinite(): NDArray<'bool'> {
    return up(core.isfinite(this)) as NDArray<'bool'>;
  }

  /**
   * Test element-wise for positive or negative infinity
   */
  isinf(): NDArray<'bool'> {
    return up(core.isinf(this)) as NDArray<'bool'>;
  }

  /**
   * Test element-wise for NaN (Not a Number)
   */
  isnan(): NDArray<'bool'> {
    return up(core.isnan(this)) as NDArray<'bool'>;
  }

  /**
   * Test element-wise for NaT (Not a Time)
   * @returns Boolean array (always false without datetime support)
   */
  isnat(): NDArray<'bool'> {
    return up(core.isnat(this)) as NDArray<'bool'>;
  }

  /**
   * Returns element-wise True where signbit is set (less than zero)
   */
  signbit(): NDArray<'bool'> {
    return up(core.signbit(this)) as NDArray<'bool'>;
  }

  /**
   * Return the distance between x and the nearest adjacent number
   */
  spacing(): NDArray<MathResult<D>> {
    return up(core.spacing(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Element-wise cube root
   * Promotes integer types to float64
   */
  cbrt(): NDArray<MathResult<D>> {
    return up(core.cbrt(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Element-wise absolute value (always returns float)
   */
  fabs(): NDArray<MathResult<D>> {
    return up(core.fabs(this)) as NDArray<MathResult<D>>;
  }

  /**
   * Element-wise square (x**2)
   */
  square(): NDArray<BoolArith<D>> {
    return up(core.square(this)) as NDArray<BoolArith<D>>;
  }

  /**
   * Return the complex conjugate, element-wise
   */
  conj(): NDArray<D> {
    return up(core.conj(this)) as NDArray<D>;
  }

  /**
   * Return a flattened copy of the array
   * @returns 1D array containing all elements
   */
  flatten(): NDArray<D> {
    return up(core.flatten(this)) as NDArray<D>;
  }

  /**
   * Find the indices of array elements that are non-zero, grouped by element
   * @returns 2D array of shape (N, ndim)
   */
  argwhere(): NDArray<'int64'> {
    return up(core.argwhere(this)) as NDArray<'int64'>;
  }

  // ========================================
  // Binary operations
  // ========================================

  /**
   * Element-wise addition
   * @param other - Array or scalar to add
   */
  add<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  add(other: number): NDArray<D>;
  add(other: NDArray | number): NDArray {
    return up(core.add(this, other));
  }

  /**
   * Element-wise subtraction
   * @param other - Array or scalar to subtract
   */
  subtract<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  subtract(other: number): NDArray<D>;
  subtract(other: NDArray | number): NDArray {
    return up(core.subtract(this, other));
  }

  /**
   * Element-wise multiplication
   * @param other - Array or scalar to multiply
   */
  multiply<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  multiply(other: number): NDArray<D>;
  multiply(other: NDArray | number): NDArray {
    return up(core.multiply(this, other));
  }

  /**
   * Element-wise division
   * @param other - Array or scalar to divide by
   */
  divide<B extends DType>(other: NDArray<B>): NDArray<Divide<D, B>>;
  divide(other: number): NDArray<TrueDivide<D>>;
  divide(other: NDArray | number): NDArray {
    return up(core.divide(this, other));
  }

  /**
   * Element-wise modulo operation
   * @param other - Array or scalar divisor
   */
  mod<B extends DType>(other: NDArray<B>): NDArray<Power<D, B>>;
  mod(other: number): NDArray<D>;
  mod(other: NDArray | number): NDArray {
    return up(core.mod(this, other));
  }

  /**
   * Element-wise floor division
   * @param other - Array or scalar to divide by
   */
  floor_divide<B extends DType>(other: NDArray<B>): NDArray<Power<D, B>>;
  floor_divide(other: number): NDArray<D>;
  floor_divide(other: NDArray | number): NDArray {
    return up(core.floor_divide(this, other));
  }

  /**
   * Raise elements to power
   * @param exponent - Power to raise to (array or scalar)
   */
  power<B extends DType>(exponent: NDArray<B>): NDArray<Power<D, B>>;
  power(exponent: number): NDArray<D>;
  power(exponent: NDArray | number): NDArray {
    return up(core.power(this, exponent));
  }

  /**
   * Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))
   * @param x2 - Second operand
   */
  logaddexp<B extends DType>(x2: NDArray<B>): NDArray<MathBinary<D, B>>;
  logaddexp(x2: number): NDArray<MathResult<D>>;
  logaddexp(x2: NDArray | number): NDArray {
    return up(core.logaddexp(this, x2));
  }

  /**
   * Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)
   * @param x2 - Second operand
   */
  logaddexp2<B extends DType>(x2: NDArray<B>): NDArray<MathBinary<D, B>>;
  logaddexp2(x2: number): NDArray<MathResult<D>>;
  logaddexp2(x2: NDArray | number): NDArray {
    return up(core.logaddexp2(this, x2));
  }

  /**
   * Element-wise arc tangent of this/other choosing the quadrant correctly
   * @param other - x-coordinates
   */
  arctan2<B extends DType>(other: NDArray<B>): NDArray<MathBinary<D, B>>;
  arctan2(other: number): NDArray<MathResult<D>>;
  arctan2(other: NDArray | number): NDArray {
    return up(core.arctan2(this, other));
  }

  /**
   * Given the "legs" of a right triangle, return its hypotenuse
   * @param other - Second leg
   */
  hypot<B extends DType>(other: NDArray<B>): NDArray<MathBinary<D, B>>;
  hypot(other: number): NDArray<MathResult<D>>;
  hypot(other: NDArray | number): NDArray {
    return up(core.hypot(this, other));
  }

  /**
   * Element-wise greater than comparison
   * @returns Boolean array
   */
  greater(other: NDArray | number): NDArray<'bool'> {
    return up(core.greater(this, other)) as NDArray<'bool'>;
  }

  /**
   * Element-wise greater than or equal comparison
   * @returns Boolean array
   */
  greater_equal(other: NDArray | number): NDArray<'bool'> {
    return up(core.greater_equal(this, other)) as NDArray<'bool'>;
  }

  /**
   * Element-wise less than comparison
   * @returns Boolean array
   */
  less(other: NDArray | number): NDArray<'bool'> {
    return up(core.less(this, other)) as NDArray<'bool'>;
  }

  /**
   * Element-wise less than or equal comparison
   * @returns Boolean array
   */
  less_equal(other: NDArray | number): NDArray<'bool'> {
    return up(core.less_equal(this, other)) as NDArray<'bool'>;
  }

  /**
   * Element-wise equality comparison
   * @returns Boolean array
   */
  equal(other: NDArray | number): NDArray<'bool'> {
    return up(core.equal(this, other)) as NDArray<'bool'>;
  }

  /**
   * Element-wise not equal comparison
   * @returns Boolean array
   */
  not_equal(other: NDArray | number): NDArray<'bool'> {
    return up(core.not_equal(this, other)) as NDArray<'bool'>;
  }

  /**
   * Bitwise AND element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_and<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  bitwise_and(other: number): NDArray<D>;
  bitwise_and(other: NDArray | number): NDArray {
    return up(core.bitwise_and(this, other));
  }

  /**
   * Bitwise OR element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_or<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  bitwise_or(other: number): NDArray<D>;
  bitwise_or(other: NDArray | number): NDArray {
    return up(core.bitwise_or(this, other));
  }

  /**
   * Bitwise XOR element-wise
   * @param other - Array or scalar (must be integer type)
   */
  bitwise_xor<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  bitwise_xor(other: number): NDArray<D>;
  bitwise_xor(other: NDArray | number): NDArray {
    return up(core.bitwise_xor(this, other));
  }

  /**
   * Left shift elements by positions
   * @param shift - Shift amount
   */
  left_shift<B extends DType>(shift: NDArray<B>): NDArray<Promote<D, B>>;
  left_shift(shift: number): NDArray<D>;
  left_shift(shift: NDArray | number): NDArray {
    return up(core.left_shift(this, shift));
  }

  /**
   * Right shift elements by positions
   * @param shift - Shift amount
   */
  right_shift<B extends DType>(shift: NDArray<B>): NDArray<Promote<D, B>>;
  right_shift(shift: number): NDArray<D>;
  right_shift(shift: NDArray | number): NDArray {
    return up(core.right_shift(this, shift));
  }

  /**
   * Logical AND element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_and(other: NDArray | number): NDArray<'bool'> {
    return up(core.logical_and(this, other)) as NDArray<'bool'>;
  }

  /**
   * Logical OR element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_or(other: NDArray | number): NDArray<'bool'> {
    return up(core.logical_or(this, other)) as NDArray<'bool'>;
  }

  /**
   * Logical XOR element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_xor(other: NDArray | number): NDArray<'bool'> {
    return up(core.logical_xor(this, other)) as NDArray<'bool'>;
  }

  /**
   * Change the sign of x1 to that of x2, element-wise
   * @param x2 - Values whose sign is used
   */
  copysign<B extends DType>(x2: NDArray<B>): NDArray<MathBinary<D, B>>;
  copysign(x2: number): NDArray<MathResult<D>>;
  copysign(x2: NDArray | number): NDArray {
    return up(core.copysign(this, x2));
  }

  /**
   * Return the next floating-point value after x1 towards x2, element-wise
   * @param x2 - Direction to look
   */
  nextafter<B extends DType>(x2: NDArray<B>): NDArray<MathBinary<D, B>>;
  nextafter(x2: number): NDArray<MathResult<D>>;
  nextafter(x2: NDArray | number): NDArray {
    return up(core.nextafter(this, x2));
  }

  /**
   * Element-wise remainder (same as mod)
   * @param divisor - Array or scalar divisor
   */
  remainder<B extends DType>(divisor: NDArray<B>): NDArray<Power<D, B>>;
  remainder(divisor: number): NDArray<D>;
  remainder(divisor: NDArray | number): NDArray {
    return up(core.remainder(this, divisor));
  }

  /**
   * Heaviside step function
   * @param x2 - Value to use when this array element is 0
   */
  heaviside<B extends DType>(x2: NDArray<B>): NDArray<MathBinary<D, B>>;
  heaviside(x2: number): NDArray<MathResult<D>>;
  heaviside(x2: NDArray | number): NDArray {
    return up(core.heaviside(this, x2));
  }

  /**
   * Matrix multiplication
   * @param other - Array to multiply with
   */
  matmul<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  matmul(other: NDArray): NDArray {
    return up(core.matmul(this, other));
  }

  /**
   * Outer product (flattens inputs then computes a[i]*b[j])
   * @param other - Array to compute outer product with
   */
  outer<B extends DType>(other: NDArray<B>): NDArray<Promote<D, B>>;
  outer(other: NDArray): NDArray {
    return up(core.outer(this, other));
  }

  // ========================================
  // Reduction operations
  // ========================================

  /**
   * Sum array elements over a given axis
   */
  sum(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
    const r = core.sum(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<ReductionAccum<D>>
      | Scalar<ReductionAccum<D>>;
  }

  /**
   * Compute the arithmetic mean along the specified axis
   */
  mean(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.mean(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Product of array elements over a given axis
   */
  prod(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
    const r = core.prod(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<ReductionAccum<D>>
      | Scalar<ReductionAccum<D>>;
  }

  /**
   * Return the maximum along a given axis
   */
  max(axis?: number | number[], keepdims: boolean = false): NDArray<D> | Scalar<D> {
    const r = core.max(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
  }

  /**
   * Return the minimum along a given axis
   */
  min(axis?: number | number[], keepdims: boolean = false): NDArray<D> | Scalar<D> {
    const r = core.min(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
  }

  /**
   * Peak to peak (maximum - minimum) value along a given axis
   */
  ptp(axis?: number, keepdims: boolean = false): NDArray<D> | Scalar<D> {
    const r = core.ptp(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
  }

  /**
   * Return the sum of array elements, treating NaNs as zero
   */
  nansum(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
    const r = core.nansum(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<ReductionAccum<D>>
      | Scalar<ReductionAccum<D>>;
  }

  /**
   * Return the product of array elements, treating NaNs as ones
   */
  nanprod(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>> {
    const r = core.nanprod(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<ReductionAccum<D>>
      | Scalar<ReductionAccum<D>>;
  }

  /**
   * Compute the arithmetic mean, ignoring NaNs
   */
  nanmean(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.nanmean(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Return minimum of an array, ignoring NaNs
   */
  nanmin(axis?: number | number[], keepdims: boolean = false): NDArray<D> | Scalar<D> {
    const r = core.nanmin(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
  }

  /**
   * Return maximum of an array, ignoring NaNs
   */
  nanmax(axis?: number | number[], keepdims: boolean = false): NDArray<D> | Scalar<D> {
    const r = core.nanmax(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<D> | Scalar<D>;
  }

  /**
   * Indices of the minimum values along an axis
   */
  argmin(axis?: number): NDArray<'int64'> | number {
    const r = core.argmin(this, axis);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int64'> | number;
  }

  /**
   * Indices of the maximum values along an axis
   */
  argmax(axis?: number): NDArray<'int64'> | number {
    const r = core.argmax(this, axis);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int64'> | number;
  }

  /**
   * Return the indices of the minimum values, ignoring NaNs
   */
  nanargmin(axis?: number): NDArray<'int64'> | number {
    const r = core.nanargmin(this, axis);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int64'> | number;
  }

  /**
   * Return the indices of the maximum values, ignoring NaNs
   */
  nanargmax(axis?: number): NDArray<'int64'> | number {
    const r = core.nanargmax(this, axis);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'int64'> | number;
  }

  /**
   * Compute variance along the specified axis
   * @param axis - Axis along which to compute variance
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  var(
    axis?: number,
    ddof: number = 0,
    keepdims: boolean = false,
  ): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
    const r = core.variance(this, axis, ddof, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<StdVar<D>>
      | Scalar<StdVar<D>>;
  }

  /**
   * Compute standard deviation along the specified axis
   * @param axis - Axis along which to compute std
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  std(
    axis?: number,
    ddof: number = 0,
    keepdims: boolean = false,
  ): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
    const r = core.std(this, axis, ddof, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<StdVar<D>>
      | Scalar<StdVar<D>>;
  }

  /**
   * Compute the variance, ignoring NaNs
   * @param axis - Axis along which to compute variance
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  nanvar(
    axis?: number | number[],
    ddof: number = 0,
    keepdims: boolean = false,
  ): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
    const r = core.nanvar(this, axis, ddof, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<StdVar<D>>
      | Scalar<StdVar<D>>;
  }

  /**
   * Compute the standard deviation, ignoring NaNs
   * @param axis - Axis along which to compute std
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   */
  nanstd(
    axis?: number | number[],
    ddof: number = 0,
    keepdims: boolean = false,
  ): NDArray<StdVar<D>> | Scalar<StdVar<D>> {
    const r = core.nanstd(this, axis, ddof, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<StdVar<D>>
      | Scalar<StdVar<D>>;
  }

  /**
   * Test whether all array elements along a given axis evaluate to True
   */
  all(axis?: number | number[], keepdims: boolean = false): NDArray<'bool'> | boolean {
    const r = core.all(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'bool'> | boolean;
  }

  /**
   * Test whether any array elements along a given axis evaluate to True
   */
  any(axis?: number | number[], keepdims: boolean = false): NDArray<'bool'> | boolean {
    const r = core.any(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as NDArray<'bool'> | boolean;
  }

  /**
   * Compute the median along the specified axis
   */
  median(
    axis?: number | number[],
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.median(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Compute the median, ignoring NaNs
   */
  nanmedian(
    axis?: number,
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.nanmedian(this, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Compute the q-th percentile of the data along the specified axis
   * @param q - Percentile to compute (0-100)
   */
  percentile(
    q: number,
    axis?: number,
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.percentile(this, q, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Compute the q-th quantile of the data along the specified axis
   * @param q - Quantile to compute (0-1)
   */
  quantile(
    q: number,
    axis?: number,
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.quantile(this, q, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Compute the q-th quantile, ignoring NaNs
   * @param q - Quantile to compute (0-1)
   */
  nanquantile(
    q: number,
    axis?: number,
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.nanquantile(this, q, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  /**
   * Compute the q-th percentile, ignoring NaNs
   * @param q - Percentile to compute (0-100)
   */
  nanpercentile(
    q: number,
    axis?: number,
    keepdims: boolean = false,
  ): NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> {
    const r = core.nanpercentile(this, q, axis, keepdims);
    return (r instanceof NDArrayCore ? up(r) : r) as unknown as
      | NDArray<TrueDivide<D>>
      | Scalar<TrueDivide<D>>;
  }

  // ========================================
  // Passthrough operations
  // ========================================

  /**
   * Return the cumulative sum of elements along a given axis
   */
  cumsum(axis?: number): NDArray<ReductionAccum<D>> {
    return up(core.cumsum(this, axis)) as NDArray<ReductionAccum<D>>;
  }

  /**
   * Return the cumulative product of elements along a given axis
   */
  cumprod(axis?: number): NDArray<ReductionAccum<D>> {
    return up(core.cumprod(this, axis)) as NDArray<ReductionAccum<D>>;
  }

  /**
   * Return the cumulative sum of elements, treating NaNs as zero
   */
  nancumsum(axis?: number): NDArray<ReductionAccum<D>> {
    return up(core.nancumsum(this, axis)) as NDArray<ReductionAccum<D>>;
  }

  /**
   * Return the cumulative product of elements, treating NaNs as one
   */
  nancumprod(axis?: number): NDArray<ReductionAccum<D>> {
    return up(core.nancumprod(this, axis)) as NDArray<ReductionAccum<D>>;
  }

  /**
   * Return a sorted copy of the array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  sort(axis: number = -1): NDArray<D> {
    return up(core.sort(this, axis)) as NDArray<D>;
  }

  /**
   * Returns the indices that would sort this array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  argsort(axis: number = -1): NDArray<'int64'> {
    return up(core.argsort(this, axis)) as NDArray<'int64'>;
  }

  /**
   * Partially sort the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  partition(kth: number, axis: number = -1): NDArray<D> {
    return up(core.partition(this, kth, axis)) as NDArray<D>;
  }

  /**
   * Returns indices that would partition the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   */
  argpartition(kth: number, axis: number = -1): NDArray<'int64'> {
    return up(core.argpartition(this, kth, axis)) as NDArray<'int64'>;
  }

  /**
   * Return specified diagonals
   * @param offset - Offset of the diagonal from the main diagonal
   * @param axis1 - First axis of the 2-D sub-arrays
   * @param axis2 - Second axis of the 2-D sub-arrays
   */
  diagonal(offset: number = 0, axis1: number = 0, axis2: number = 1): NDArray<D> {
    return up(core.diagonal(this, offset, axis1, axis2)) as NDArray<D>;
  }

  /**
   * Return a new array with the specified shape
   * If larger, filled with repeated copies of the original data
   * @param newShape - Shape of the resized array
   */
  resize(newShape: number[]): NDArray<D> {
    return up(core.resize(this, newShape)) as NDArray<D>;
  }

  /**
   * Calculate the n-th discrete difference along the given axis
   * @param n - Number of times values are differenced (default: 1)
   * @param axis - Axis along which to compute difference (default: -1)
   */
  diff(n: number = 1, axis: number = -1): NDArray<D> {
    return up(core.diff(this, n, axis)) as NDArray<D>;
  }

  /**
   * Take elements from array along an axis
   * @param indices - Indices of elements to take
   * @param axis - Axis along which to take
   */
  take(indices: number[], axis?: number): NDArray<D> {
    return up(core.take(this, indices, axis)) as NDArray<D>;
  }

  /**
   * Repeat elements of an array
   * @param repeats - Number of repetitions for each element
   * @param axis - Axis along which to repeat
   */
  repeat(repeats: number | number[], axis?: number): NDArray<D> {
    return up(core.repeat(this, repeats, axis)) as NDArray<D>;
  }

  /**
   * Transpose array (permute dimensions)
   * @param axes - Permutation of axes. If undefined, reverse the dimensions
   * @returns Transposed array (always a view)
   */
  transpose(axes?: number[]): NDArray<D> {
    return up(core.transpose(this, axes)) as NDArray<D>;
  }

  /**
   * Remove axes of length 1
   * @param axis - Axis to squeeze
   * @returns Array with specified dimensions removed (always a view)
   */
  squeeze(axis?: number): NDArray<D> {
    return up(core.squeeze(this, axis)) as NDArray<D>;
  }

  /**
   * Expand the shape by inserting a new axis of length 1
   * @param axis - Position where new axis is placed
   * @returns Array with additional dimension (always a view)
   */
  expand_dims(axis: number): NDArray<D> {
    return up(core.expand_dims(this, axis)) as NDArray<D>;
  }

  /**
   * Swap two axes of an array
   * @param axis1 - First axis
   * @param axis2 - Second axis
   * @returns Array with swapped axes (always a view)
   */
  swapaxes(axis1: number, axis2: number): NDArray<D> {
    return up(core.swapaxes(this, axis1, axis2)) as NDArray<D>;
  }

  /**
   * Move axes to new positions
   * @param source - Original positions of axes to move
   * @param destination - New positions for axes
   * @returns Array with moved axes (always a view)
   */
  moveaxis(source: number | number[], destination: number | number[]): NDArray<D> {
    return up(core.moveaxis(this, source, destination)) as NDArray<D>;
  }

  // ========================================
  // Array return operations
  // ========================================

  /**
   * Return the indices of non-zero elements
   * @returns Tuple of arrays, one for each dimension
   */
  nonzero(): NDArray<'int64'>[] {
    return core.nonzero(this).map(up) as NDArray<'int64'>[];
  }
}

/**
 * Return coordinate matrices from coordinate vectors
 * @param arrays - 1D coordinate arrays
 * @param indexing - 'xy' (Cartesian, default) or 'ij' (matrix indexing)
 * @returns Array of coordinate grids
 */
export interface MeshgridOptions {
  /** 'xy' (default, NumPy-compatible) or 'ij' Cartesian/matrix indexing */
  indexing?: 'xy' | 'ij';
  /** If true, return open (non-broadcasted) grids — outputs have shape 1 except along their own axis */
  sparse?: boolean;
  /** If true (default), each output owns its data. If false, outputs are views (broadcast). */
  copy?: boolean;
}

export function meshgrid(...args: (NDArray | MeshgridOptions)[]): NDArray[] {
  let arrays: NDArray[] = [];
  let indexing: 'xy' | 'ij' = 'xy';
  let sparse = false;
  let copy = true;

  for (const arg of args) {
    if (arg instanceof NDArray) {
      arrays.push(arg);
    } else if (arg && typeof arg === 'object') {
      if ('indexing' in arg && arg.indexing) indexing = arg.indexing;
      if ('sparse' in arg && arg.sparse !== undefined) sparse = arg.sparse;
      if ('copy' in arg && arg.copy !== undefined) copy = arg.copy;
    }
  }

  if (arrays.length === 0) return [];
  if (arrays.length === 1) {
    const a = arrays[0]!;
    return [copy ? a.copy() : a];
  }

  if (indexing === 'xy' && arrays.length >= 2) {
    arrays = [arrays[1]!, arrays[0]!, ...arrays.slice(2)];
  }

  const sizes = arrays.map((a) => a.size);
  const ndim = sizes.length;

  const results: NDArray[] = [];
  for (let i = 0; i < arrays.length; i++) {
    const broadcastShape: number[] = new Array(ndim).fill(1);
    broadcastShape[i] = sizes[i]!;

    const reshaped = arrays[i]!.reshape(...broadcastShape);
    if (sparse) {
      results.push(copy ? reshaped.copy() : reshaped);
      continue;
    }
    const broadcasted = core.broadcast_to(reshaped, sizes);
    const out = NDArray.fromStorage(copy ? broadcasted.storage.copy() : broadcasted.storage);
    results.push(out);
  }

  if (indexing === 'xy' && results.length >= 2) {
    [results[0], results[1]] = [results[1]!, results[0]!];
  }

  return results;
}
