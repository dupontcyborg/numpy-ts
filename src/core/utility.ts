/**
 * Utility functions
 *
 * Tree-shakeable standalone functions for array introspection.
 */

import { NDArrayCore, type DType, type TypedArray } from '../common/ndarray-core';
import { Complex } from '../common/complex';

// ============================================================
// Array Properties
// ============================================================

/**
 * Return the number of dimensions of an array.
 * Works with NDArrayCore, scalars, and nested arrays.
 */
export function ndim(a: NDArrayCore | number | bigint | boolean | unknown[] | unknown): number {
  if (a instanceof NDArrayCore) {
    return a.ndim;
  }
  // Scalar
  if (typeof a === 'number' || typeof a === 'bigint' || typeof a === 'boolean') {
    return 0;
  }
  // Nested array - count dimensions
  if (Array.isArray(a)) {
    let dims = 0;
    let current: unknown = a;
    while (Array.isArray(current) && current.length > 0) {
      dims++;
      current = current[0];
    }
    return dims;
  }
  return 0;
}

/**
 * Return the shape of an array.
 * Works with NDArrayCore, scalars, and nested arrays.
 */
export function shape(
  a: NDArrayCore | number | bigint | boolean | unknown[] | unknown
): readonly number[] | number[] {
  if (a instanceof NDArrayCore) {
    return a.shape;
  }
  // Scalar
  if (typeof a === 'number' || typeof a === 'bigint' || typeof a === 'boolean') {
    return [];
  }
  // Nested array - compute shape
  if (Array.isArray(a)) {
    const result: number[] = [];
    let current: unknown = a;
    while (Array.isArray(current) && current.length > 0) {
      result.push(current.length);
      current = current[0];
    }
    return result;
  }
  return [];
}

/**
 * Return the number of elements in an array.
 * Works with NDArrayCore, scalars, and nested arrays.
 */
export function size(a: NDArrayCore | number | bigint | boolean | unknown[] | unknown): number {
  if (a instanceof NDArrayCore) {
    return a.size;
  }
  // Scalar
  if (typeof a === 'number' || typeof a === 'bigint' || typeof a === 'boolean') {
    return 1;
  }
  // Nested array - compute size from shape
  if (Array.isArray(a)) {
    const shapeArr = shape(a);
    return shapeArr.reduce((acc, dim) => acc * dim, 1);
  }
  return 1;
}

// ============================================================
// Data Access
// ============================================================

export function item(a: NDArrayCore, ...args: number[]): number | bigint | boolean | Complex {
  const storage = a.storage;
  const shapeArr = a.shape;

  if (args.length === 0) {
    if (a.size !== 1) {
      throw new Error('can only convert an array of size 1 to a scalar');
    }
    // Fast path: direct data access for non-complex contiguous
    if (storage.isCContiguous) {
      return storage.data[storage.offset] as number | bigint | boolean | Complex;
    }
    return storage.iget(0) as number | bigint | boolean | Complex;
  }

  if (args.length === 1) {
    if (storage.isCContiguous) {
      return storage.data[storage.offset + args[0]!] as number | bigint | boolean | Complex;
    }
    return storage.iget(args[0]!) as number | bigint | boolean | Complex;
  }

  // Multi-index access
  if (args.length !== shapeArr.length) {
    throw new Error('incorrect number of indices for array');
  }

  return storage.get(...args) as number | bigint | boolean | Complex;
}

export function tolist(a: NDArrayCore): unknown {
  const shapeArr = a.shape;
  const storage = a.storage;
  const ndimVal = shapeArr.length;

  if (ndimVal === 0) {
    return storage.iget(0);
  }

  // Fast path for contiguous arrays: direct data access
  if (storage.isCContiguous) {
    const data = storage.data;
    const off = storage.offset;

    if (ndimVal === 1) {
      const len = shapeArr[0]!;
      const result: unknown[] = new Array(len);
      for (let i = 0; i < len; i++) {
        result[i] = data[off + i];
      }
      return result;
    }

    // Multi-dim contiguous: compute strides and use direct data[off + linearIdx]
    const strides: number[] = new Array(ndimVal);
    let stride = 1;
    for (let d = ndimVal - 1; d >= 0; d--) {
      strides[d] = stride;
      stride *= shapeArr[d]!;
    }

    function buildListFast(baseOffset: number, dim: number): unknown {
      const dimSize = shapeArr[dim]!;
      const dimStride = strides[dim]!;
      if (dim === ndimVal - 1) {
        const result: unknown[] = new Array(dimSize);
        for (let i = 0; i < dimSize; i++) {
          result[i] = data[off + baseOffset + i];
        }
        return result;
      }
      const result: unknown[] = new Array(dimSize);
      for (let i = 0; i < dimSize; i++) {
        result[i] = buildListFast(baseOffset + i * dimStride, dim + 1);
      }
      return result;
    }

    return buildListFast(0, 0);
  }

  // Slow path for non-contiguous views
  if (ndimVal === 1) {
    const result: unknown[] = [];
    for (let i = 0; i < shapeArr[0]!; i++) {
      result.push(storage.iget(i));
    }
    return result;
  }

  function buildList(indices: number[], dim: number): unknown {
    if (dim === ndimVal) {
      return storage.get(...indices);
    }

    const result: unknown[] = [];
    for (let i = 0; i < shapeArr[dim]!; i++) {
      indices[dim] = i;
      result.push(buildList(indices, dim + 1));
    }
    return result;
  }

  return buildList(new Array(ndimVal), 0);
}

export function tobytes(a: NDArrayCore, order: 'C' | 'F' = 'C'): Uint8Array {
  const storage = a.storage;

  if (order === 'F') {
    console.warn('tobytes with order="F" not fully implemented, returning C-order');
  }

  const data = storage.data;
  const bytesPerElement = data.BYTES_PER_ELEMENT;

  if (storage.isCContiguous) {
    // Contiguous: return the exact byte range for this view
    const byteOffset = data.byteOffset + storage.offset * bytesPerElement;
    const byteLength = a.size * bytesPerElement;
    return new Uint8Array(data.buffer, byteOffset, byteLength);
  }

  // Non-contiguous: materialize into a contiguous copy first
  const copy = a.copy();
  const copyData = copy.data;
  return new Uint8Array(copyData.buffer, copyData.byteOffset, a.size * bytesPerElement);
}

export function byteswap(a: NDArrayCore, inplace: boolean = false): NDArrayCore {
  const data = a.data;
  const bytesPerElement = (data as TypedArray).BYTES_PER_ELEMENT;

  if (bytesPerElement === 1) {
    return inplace ? a : a.copy();
  }

  const result = inplace ? a : a.copy();
  const resultData = result.data;

  const uint8View = new Uint8Array(resultData.buffer, resultData.byteOffset, resultData.byteLength);

  for (let i = 0; i < resultData.length; i++) {
    const offset = i * bytesPerElement;
    for (let j = 0; j < bytesPerElement / 2; j++) {
      const temp = uint8View[offset + j]!;
      uint8View[offset + j] = uint8View[offset + bytesPerElement - 1 - j]!;
      uint8View[offset + bytesPerElement - 1 - j] = temp;
    }
  }

  return result;
}

export function view(a: NDArrayCore, dtype?: DType): NDArrayCore {
  if (!dtype || dtype === a.dtype) {
    return a.copy();
  }
  throw new Error('view with different dtype not fully implemented');
}

export function tofile(
  _a: NDArrayCore,
  _file: string,
  _sep: string = '',
  _format: string = ''
): void {
  throw new Error(
    'tofile requires Node.js file system access. Use serializeNpy for portable serialization.'
  );
}

export function fill(a: NDArrayCore, value: number | bigint | boolean | Complex): void {
  const storage = a.storage;
  const dtype = a.dtype;
  const sz = a.size;

  if (value instanceof Complex) {
    const isComplex = dtype === 'complex64' || dtype === 'complex128';
    if (!isComplex) {
      throw new Error('Cannot fill non-complex array with complex value');
    }
    if (storage.isCContiguous) {
      const data = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < sz; i++) {
        data[(off + i) * 2] = value.re;
        data[(off + i) * 2 + 1] = value.im;
      }
    } else {
      for (let i = 0; i < sz; i++) {
        storage.iset(i, value);
      }
    }
  } else if (typeof value === 'bigint') {
    if (storage.isCContiguous) {
      const data = storage.data as BigInt64Array | BigUint64Array;
      data.fill(value, storage.offset, storage.offset + sz);
    } else {
      for (let i = 0; i < sz; i++) {
        storage.iset(i, value);
      }
    }
  } else {
    const numValue = typeof value === 'boolean' ? (value ? 1 : 0) : value;
    if (storage.isCContiguous) {
      (storage.data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(
        numValue,
        storage.offset,
        storage.offset + sz
      );
    } else {
      for (let i = 0; i < sz; i++) {
        storage.iset(i, numValue);
      }
    }
  }
}
