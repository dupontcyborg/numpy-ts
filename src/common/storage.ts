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
import {
  type WasmRegion,
  wasmMalloc,
  getSharedMemory,
  registerForCleanup,
  unregisterCleanup,
} from './wasm/runtime';

/**
 * Maximum number of dimensions an array can have (matches NumPy's limit).
 */
export const MAX_NDIM = 64;

/**
 * Internal storage for NDArray data
 * Manages the underlying TypedArray and metadata
 */
export class ArrayStorage {
  // Symbol.dispose for `using` keyword support (conditionally defined below class)
  [Symbol.dispose]?: () => void;

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
  // Cached contiguity flag (-1 = not computed, 0 = false, 1 = true)
  private _isCContiguous: number = -1;
  // WASM memory region (null for JS-fallback arrays)
  private _wasmRegion: WasmRegion | null;

  constructor(
    data: TypedArray,
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    dtype: DType,
    wasmRegion: WasmRegion | null = null
  ) {
    this._data = data;
    this._shape = shape;
    this._strides = strides;
    this._offset = offset;
    this._dtype = dtype;
    this._wasmRegion = wasmRegion;
    if (wasmRegion) {
      registerForCleanup(this, wasmRegion);
    }
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
   * Whether this storage is backed by WASM linear memory (zero-copy eligible).
   */
  get isWasmBacked(): boolean {
    return this._wasmRegion !== null;
  }

  /**
   * Eagerly free the WASM memory backing this storage.
   *
   * In normal usage, WASM memory is freed automatically via FinalizationRegistry
   * when the storage is garbage collected. Call dispose() to free immediately
   * when you know this storage will not be used again — useful in tight loops,
   * benchmarks, or resource-sensitive contexts.
   *
   * Also available as `[Symbol.dispose]` on runtimes that support it (Node 18+,
   * Chrome 134+, Firefox 132+), enabling the `using` keyword for automatic
   * scope-based cleanup. Safari does not yet support `Symbol.dispose`, so use
   * this method directly for cross-browser compatibility.
   */
  dispose(): void {
    if (this._wasmRegion) {
      unregisterCleanup(this);
      this._wasmRegion.release();
      this._wasmRegion = null;
    }
  }

  /**
   * Byte offset of the start of the allocation in WASM memory.
   * Returns 0 if not WASM-backed (safe for use with resolveInputPtr).
   * @internal
   */
  get wasmPtr(): number {
    return this._wasmRegion ? this._wasmRegion.ptr : 0;
  }

  /**
   * The underlying WasmRegion for view sharing.
   * @internal
   */
  get wasmRegion(): WasmRegion | null {
    return this._wasmRegion;
  }

  /**
   * Check if array is C-contiguous (row-major, no gaps)
   */
  get isCContiguous(): boolean {
    if (this._isCContiguous !== -1) return this._isCContiguous === 1;

    const shape = this._shape;
    const strides = this._strides;
    const ndim = shape.length;

    let result: boolean;
    if (ndim === 0) {
      result = true;
    } else if (ndim === 1) {
      result = strides[0] === 1;
    } else {
      // Check if strides match row-major order
      result = true;
      let expectedStride = 1;
      for (let i = ndim - 1; i >= 0; i--) {
        if (strides[i] !== expectedStride) {
          result = false;
          break;
        }
        expectedStride *= shape[i]!;
      }
    }

    this._isCContiguous = result ? 1 : 0;
    return result;
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
      // Fast path: direct copy via TypedArray.set() (works for all types including BigInt)
      (newData as unknown as { set(src: ArrayLike<number>): void }).set(
        (
          this._data as unknown as { subarray(begin: number, end: number): ArrayLike<number> }
        ).subarray(0, physicalSize)
      );
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

    const region = wasmMalloc(newData.byteLength);
    if (region) {
      const mem = getSharedMemory();
      const wasmData = new Constructor(mem.buffer, region.ptr, physicalSize) as TypedArray;
      (wasmData as unknown as { set(src: ArrayLike<number>): void }).set(
        newData as unknown as ArrayLike<number>
      );
      return new ArrayStorage(
        wasmData,
        shape,
        ArrayStorage._computeStrides(shape),
        0,
        dtype,
        region
      );
    }
    return new ArrayStorage(newData, shape, ArrayStorage._computeStrides(shape), 0, dtype);
  }

  /**
   * Create storage from TypedArray data.
   * If the data is not already in WASM memory, copies it there (one-time cost).
   */
  static fromData(
    data: TypedArray,
    shape: number[],
    dtype: DType,
    strides?: number[],
    offset?: number
  ): ArrayStorage {
    if (shape.length > MAX_NDIM) {
      throw new Error(
        `maximum supported dimension for an ndarray is currently ${MAX_NDIM}, found ${shape.length}`
      );
    }
    const finalStrides = strides ?? ArrayStorage._computeStrides(shape);
    const finalOffset = offset ?? 0;

    // If data is already a view into WASM memory, use directly (no region — caller passes it via constructor)
    const mem = getSharedMemory();
    if (data.buffer === mem.buffer) {
      return new ArrayStorage(data, shape, finalStrides, finalOffset, dtype);
    }

    // External data: try to copy into WASM memory
    const region = wasmMalloc(data.byteLength);
    if (region) {
      const Ctor = data.constructor as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray;
      const wasmData = new Ctor(mem.buffer, region.ptr, data.length);
      (wasmData as unknown as { set(src: ArrayLike<number>): void }).set(
        data as unknown as ArrayLike<number>
      );
      return new ArrayStorage(wasmData, shape, finalStrides, finalOffset, dtype, region);
    }

    // JS fallback
    return new ArrayStorage(data, shape, finalStrides, finalOffset, dtype);
  }

  /**
   * Create a view sharing the same backing WASM region.
   * Increments the WasmRegion refcount so the memory is not freed
   * until all views are garbage collected.
   * @internal
   */
  static fromDataShared(
    data: TypedArray,
    shape: number[],
    dtype: DType,
    strides: number[],
    offset: number,
    wasmRegion: WasmRegion | null
  ): ArrayStorage {
    if (wasmRegion) wasmRegion.retain();
    return new ArrayStorage(data, shape, strides, offset, dtype, wasmRegion);
  }

  /**
   * Create storage with a WASM-backed output region.
   * Used by kernel wrappers to construct output directly in WASM memory.
   * @internal
   */
  static fromWasmRegion(
    shape: number[],
    dtype: DType,
    region: WasmRegion,
    elementCount: number,
    Ctor: new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  ): ArrayStorage {
    const mem = getSharedMemory();
    const data = new Ctor(mem.buffer, region.ptr, elementCount);
    return new ArrayStorage(data, shape, ArrayStorage._computeStrides(shape), 0, dtype, region);
  }

  /**
   * Create storage with zeros
   */
  static zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): ArrayStorage {
    if (shape.length > MAX_NDIM) {
      throw new Error(
        `maximum supported dimension for an ndarray is currently ${MAX_NDIM}, found ${shape.length}`
      );
    }
    const size = shape.reduce((a, b) => a * b, 1);
    const isComplex = isComplexDType(dtype);

    // Get TypedArray constructor
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot create array with dtype ${dtype}`);
    }

    // For complex types, physical size is 2x logical size
    const physicalSize = isComplex ? size * 2 : size;
    const byteLength =
      physicalSize * (Constructor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    // Try WASM allocation first
    const region = wasmMalloc(byteLength);
    if (region) {
      const mem = getSharedMemory();
      // Zero-fill (freed WASM regions may have stale data)
      new Uint8Array(mem.buffer, region.ptr, byteLength).fill(0);
      const data = new Constructor(mem.buffer, region.ptr, physicalSize);
      return new ArrayStorage(data, shape, ArrayStorage._computeStrides(shape), 0, dtype, region);
    }

    // JS fallback
    const data = new Constructor(physicalSize);
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
    const byteLength =
      physicalSize * (Constructor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    // Try WASM allocation first
    const region = wasmMalloc(byteLength);
    let data: TypedArray;
    if (region) {
      const mem = getSharedMemory();
      data = new Constructor(mem.buffer, region.ptr, physicalSize);
    } else {
      data = new Constructor(physicalSize);
    }

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

    return new ArrayStorage(
      data,
      shape,
      ArrayStorage._computeStrides(shape),
      0,
      dtype,
      region ?? null
    );
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

// Symbol.dispose support for the `using` keyword (automatic scope-based cleanup).
// Safari does not yet support Symbol.dispose, so we define it conditionally.
// Users on Safari can call .dispose() directly instead.
if (typeof Symbol.dispose !== 'undefined') {
  ArrayStorage.prototype[Symbol.dispose] = ArrayStorage.prototype.dispose;
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
