/**
 * NDArray Core - Minimal NDArray class for tree-shaking
 *
 * This module contains the minimal NDArray class without operation methods.
 * It only depends on core modules (storage, dtype, slicing, complex) and
 * does NOT import any ops modules.
 *
 * For the full NDArray with all methods, use ndarray-full.ts
 */

import { parseSlice, normalizeSlice } from './slicing';
import {
  type DType,
  type TypedArray,
  getTypedArrayConstructor,
  getDTypeSize,
  isBigIntDType,
  isComplexDType,
  isFloatDType,
} from './dtype';
import { Complex } from './complex';
import { ArrayStorage } from './storage';
import { array_str } from './ops/formatting';

/**
 * Minimal NDArray class - core functionality without operation methods
 *
 * This class provides:
 * - Array properties (shape, dtype, data, etc.)
 * - Element access (get, set, iget, iset)
 * - Basic methods (copy, astype, fill, slice)
 * - Conversion methods (toArray, tolist, tobytes)
 *
 * Operation methods (add, sin, reshape, etc.) are NOT included.
 * Use NDArray from ndarray-full.ts for the complete API.
 */
/**
 * NumPy-compatible float-to-integer conversion.
 *
 * NumPy converts float→int by:
 * - NaN → 0
 * - For 32/64-bit target types: clamp to target range (saturation)
 * - For 8/16-bit target types: clamp to int64 range, then bit-truncate
 *
 * JS TypedArrays differ: they return 0 for inf/NaN and wrap (modular) for
 * out-of-range values on 32-bit types. This function pre-processes the value
 * so that the subsequent TypedArray assignment produces the NumPy result.
 */
const TWO_63 = 2 ** 63; // 9223372036854775808

// Precomputed: INT64_MAX/MIN bit-truncated to each narrow type
const INT64_MAX_AS: Record<string, number> = {
  int8: -1,
  int16: -1,
  uint8: 255,
  uint16: 65535,
};
const INT64_MIN_AS: Record<string, number> = {
  int8: 0,
  int16: 0,
  uint8: 0,
  uint16: 0,
};

const INT_RANGE: Record<string, [number, number]> = {
  int32: [-2147483648, 2147483647],
  uint32: [0, 4294967295],
};

function floatToInt(value: number, targetDtype: DType): number {
  if (isNaN(value)) return 0;

  // Narrow types (8/16-bit): NumPy converts via int64 then bit-truncates
  if (targetDtype in INT64_MAX_AS) {
    if (value >= TWO_63 || value === Infinity) return INT64_MAX_AS[targetDtype]!;
    if (value <= -TWO_63 || value === -Infinity) return INT64_MIN_AS[targetDtype]!;
    return Math.trunc(value);
  }

  // Wide types (32-bit): NumPy saturates at target bounds
  if (targetDtype in INT_RANGE) {
    const [min, max] = INT_RANGE[targetDtype]!;
    if (value >= max || value === Infinity) return max;
    if (value <= min || value === -Infinity) return min;
    return Math.trunc(value);
  }

  return Math.trunc(value);
}

export class NDArrayCore {
  // Internal storage
  protected _storage: ArrayStorage;
  // Track if this array is a view of another array
  protected _base?: NDArrayCore;

  // Allows bracket access: arr[0], arr[-1], arr[0][1], etc.
  // Implemented via Proxy in the constructor.
  [key: number]: NDArrayCore | number | bigint | Complex;

  // Shared proxy handler — one object for all instances
  private static readonly _proxyHandler: ProxyHandler<NDArrayCore> = {
    get(target, prop, receiver) {
      if (typeof prop === 'string') {
        const idx = parseInt(prop, 10);
        if (!isNaN(idx) && String(idx) === prop) {
          const len = target._storage.ndim > 0 ? target._storage.shape[0]! : 1;
          const normalized = idx < 0 ? len + idx : idx;
          if (target._storage.ndim <= 1) {
            return target._storage.iget(normalized);
          }
          return target.slice(prop);
        }
      }
      // Pass receiver so getters (e.g. .T, .base) execute with this = Proxy
      return Reflect.get(target, prop, receiver);
    },
    set(target, prop, value, receiver) {
      if (typeof prop === 'string') {
        const idx = parseInt(prop, 10);
        if (!isNaN(idx) && String(idx) === prop) {
          const len = target._storage.shape[0]!;
          const normalized = idx < 0 ? len + idx : idx;
          if (target._storage.ndim === 1) {
            target.set([normalized], value as number | bigint | Complex);
            return true;
          }
          // ND: get the slice view and fill/copy into it
          const sliceView = target.slice(String(idx));
          const n = sliceView.size;
          if (typeof value === 'number' || typeof value === 'bigint') {
            sliceView.fill(value);
          } else if (value instanceof Complex) {
            for (let i = 0; i < n; i++) sliceView._storage.iset(i, value);
          } else if (value instanceof NDArrayCore) {
            if (value.size !== n) {
              throw new Error(`Cannot assign array of size ${value.size} into slice of size ${n}`);
            }
            for (let i = 0; i < n; i++) sliceView._storage.iset(i, value._storage.iget(i));
          } else if (Array.isArray(value)) {
            const flat = (value as unknown[]).flat(Infinity) as number[];
            if (flat.length !== n) {
              throw new Error(
                `Cannot assign array of length ${flat.length} into slice of size ${n}`
              );
            }
            for (let i = 0; i < n; i++) sliceView._storage.iset(i, flat[i]!);
          } else {
            throw new Error(`Cannot assign value of type ${typeof value} via bracket operator`);
          }
          return true;
        }
      }
      return Reflect.set(target, prop, value, receiver);
    },
  };

  constructor(storage: ArrayStorage, base?: NDArrayCore) {
    this._storage = storage;
    this._base = base;
    return new Proxy(this, NDArrayCore._proxyHandler);
  }

  /**
   * Get internal storage (for ops modules)
   * @internal
   */
  get storage(): ArrayStorage {
    return this._storage;
  }

  /**
   * Create NDArray from storage (for ops modules)
   * @internal
   */
  static fromStorage(storage: ArrayStorage, base?: NDArrayCore): NDArrayCore {
    return new NDArrayCore(storage, base);
  }

  // NumPy properties
  get shape(): readonly number[] {
    return this._storage.shape;
  }

  get ndim(): number {
    return this._storage.ndim;
  }

  get size(): number {
    return this._storage.size;
  }

  get dtype(): string {
    return this._storage.dtype;
  }

  get data(): TypedArray {
    return this._storage.data;
  }

  get strides(): readonly number[] {
    return this._storage.strides;
  }

  /**
   * Array flags (similar to NumPy's flags)
   */
  get flags(): {
    C_CONTIGUOUS: boolean;
    F_CONTIGUOUS: boolean;
    OWNDATA: boolean;
  } {
    return {
      C_CONTIGUOUS: this._storage.isCContiguous,
      F_CONTIGUOUS: this._storage.isFContiguous,
      OWNDATA: this._base === undefined,
    };
  }

  /**
   * Base array if this is a view, null if this array owns its data
   */
  get base(): NDArrayCore | null {
    return this._base ?? null;
  }

  /**
   * Size of one array element in bytes
   */
  get itemsize(): number {
    return getDTypeSize(this._storage.dtype);
  }

  /**
   * Total bytes consumed by the elements of the array
   */
  get nbytes(): number {
    return this.size * this.itemsize;
  }

  /**
   * Fill the array with a scalar value (in-place)
   */
  fill(value: number | bigint): void {
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
   */
  *[Symbol.iterator](): Iterator<NDArrayCore | number | bigint | Complex> {
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
   */
  get(indices: number[]): number | bigint | Complex {
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
   */
  set(indices: number[], value: number | bigint | Complex | { re: number; im: number }): void {
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
   * Get element by flat index
   */
  iget(flatIndex: number): number | bigint | Complex {
    return this._storage.iget(flatIndex);
  }

  /**
   * Set element by flat index
   */
  iset(flatIndex: number, value: number | bigint | Complex): void {
    this._storage.iset(flatIndex, value);
  }

  /**
   * Return a deep copy of the array
   */
  copy(): NDArrayCore {
    return new NDArrayCore(this._storage.copy());
  }

  /**
   * Cast array to a different dtype
   */
  astype(dtype: DType, copy: boolean = true): NDArrayCore {
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
      const isSourceFloat = isFloatDType(currentDtype) || isComplexDType(currentDtype);
      if (isSourceFloat) {
        // Float → BigInt: NaN→0, clamp to int64/uint64 range, then BigInt
        const isSigned = dtype === 'int64';
        const maxVal = isSigned ? BigInt('9223372036854775807') : BigInt('18446744073709551615');
        const minVal = isSigned ? BigInt('-9223372036854775808') : 0n;
        for (let i = 0; i < size; i++) {
          const v = Number(typedOldData[i]);
          if (isNaN(v)) {
            (newData as BigInt64Array | BigUint64Array)[i] = 0n;
          } else if (!isFinite(v) || v >= Number(maxVal)) {
            (newData as BigInt64Array | BigUint64Array)[i] = v < 0 ? minVal : maxVal;
          } else if (v <= Number(minVal)) {
            (newData as BigInt64Array | BigUint64Array)[i] = minVal;
          } else {
            (newData as BigInt64Array | BigUint64Array)[i] = BigInt(Math.trunc(v));
          }
        }
      } else {
        for (let i = 0; i < size; i++) {
          (newData as BigInt64Array | BigUint64Array)[i] = BigInt(
            Math.round(Number(typedOldData[i]))
          );
        }
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
      const needsFloatToInt =
        (isFloatDType(currentDtype) || isComplexDType(currentDtype)) && !isFloatDType(dtype);
      if (needsFloatToInt) {
        // Float → integer: use NumPy-compatible conversion (saturation/truncation)
        for (let i = 0; i < size; i++) {
          (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = floatToInt(
            typedOldData[i]!,
            dtype
          );
        }
      } else {
        for (let i = 0; i < size; i++) {
          (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
        }
      }
    } else {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = typedOldData[i]!;
      }
    }

    const storage = ArrayStorage.fromData(newData, shape, dtype);
    return new NDArrayCore(storage);
  }

  /**
   * Slice the array
   */
  slice(...sliceStrs: string[]): NDArrayCore {
    if (sliceStrs.length === 0) {
      return this;
    }

    if (sliceStrs.length > this.ndim) {
      throw new Error(
        `Too many indices for array: array is ${this.ndim}-dimensional, but ${sliceStrs.length} were indexed`
      );
    }

    const sliceSpecs = sliceStrs.map((str, i) => {
      const parsed = parseSlice(str);
      return normalizeSlice(parsed, this.shape[i]!);
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
      newOffset += spec.start * this._storage.strides[i]!;

      if (spec.step === 0) {
        continue;
      }

      const sliceLength = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
      newShape.push(sliceLength);
      newStrides.push(this._storage.strides[i]! * spec.step);
    }

    const slicedStorage = ArrayStorage.fromData(
      this._storage.data,
      newShape,
      this._storage.dtype,
      newStrides,
      newOffset
    );

    const base = this._base ?? this;
    return new NDArrayCore(slicedStorage, base);
  }

  /**
   * String representation
   */
  toString(): string {
    return array_str(this._storage);
  }

  /**
   * Convert to nested JavaScript array
   */
  toArray(): unknown {
    if (this.ndim === 0) {
      return this._storage.iget(0);
    }

    const shape = this.shape;
    const ndim = shape.length;

    const buildNestedArray = (indices: number[], dim: number): unknown => {
      if (dim === ndim) {
        return this._storage.get(...indices);
      }

      const arr: unknown[] = [];
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
  tolist(): unknown {
    return this.toArray();
  }

  /**
   * Return the raw bytes of the array data
   */
  tobytes(): ArrayBuffer {
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
  item(...args: number[]): number | bigint | Complex {
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
}

// Re-export types
export type { DType, TypedArray };
