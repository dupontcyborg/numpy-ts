/**
 * Utility functions
 *
 * Tree-shakeable standalone functions for array introspection.
 */

import { NDArrayCore, type DType, type TypedArray } from '../core/ndarray-core';
import { Complex } from '../core/complex';


// ============================================================
// Array Properties
// ============================================================

export function ndim(a: NDArrayCore): number {
  return a.ndim;
}

export function shape(a: NDArrayCore): readonly number[] {
  return a.shape;
}

export function size(a: NDArrayCore): number {
  return a.size;
}

// ============================================================
// Data Access
// ============================================================

export function item(a: NDArrayCore, ...args: number[]): number | bigint | boolean | Complex {
  const data = a.data;
  const shapeArr = a.shape;

  if (args.length === 0) {
    if (a.size !== 1) {
      throw new Error('can only convert an array of size 1 to a scalar');
    }
    return data[0] as number | bigint | boolean | Complex;
  }

  if (args.length === 1) {
    return data[args[0]!] as number | bigint | boolean | Complex;
  }

  // Multi-index access
  if (args.length !== shapeArr.length) {
    throw new Error('incorrect number of indices for array');
  }

  let flatIdx = 0;
  let stride = 1;
  for (let i = shapeArr.length - 1; i >= 0; i--) {
    flatIdx += args[i]! * stride;
    stride *= shapeArr[i]!;
  }

  return data[flatIdx] as number | bigint | boolean | Complex;
}

export function tolist(a: NDArrayCore): unknown {
  const shapeArr = [...a.shape];
  const data = a.data;

  if (shapeArr.length === 0) {
    return data[0];
  }

  if (shapeArr.length === 1) {
    return Array.from(data as unknown as ArrayLike<number>);
  }

  function buildList(offset: number, dim: number): unknown[] {
    if (dim === shapeArr.length - 1) {
      const result: unknown[] = [];
      for (let i = 0; i < shapeArr[dim]!; i++) {
        result.push(data[offset + i]);
      }
      return result;
    }

    const result: unknown[] = [];
    const stride = shapeArr.slice(dim + 1).reduce((acc, b) => acc * b, 1);
    for (let i = 0; i < shapeArr[dim]!; i++) {
      result.push(buildList(offset + i * stride, dim + 1));
    }
    return result;
  }

  return buildList(0, 0);
}

export function tobytes(a: NDArrayCore, order: 'C' | 'F' = 'C'): Uint8Array {
  const data = a.data;

  if (order === 'F') {
    console.warn('tobytes with order="F" not fully implemented, returning C-order');
  }

  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
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
  const data = a.data;
  const dtype = a.dtype;

  if (value instanceof Complex) {
    const isComplex = dtype === 'complex64' || dtype === 'complex128';
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      for (let i = 0; i < a.size; i++) {
        complexData[i * 2] = value.re;
        complexData[i * 2 + 1] = value.im;
      }
    } else {
      throw new Error('Cannot fill non-complex array with complex value');
    }
  } else if (typeof value === 'bigint') {
    (data as BigInt64Array | BigUint64Array).fill(value);
  } else if (typeof value === 'boolean') {
    (data as Uint8Array).fill(value ? 1 : 0);
  } else {
    (data as Float64Array).fill(value);
  }
}
