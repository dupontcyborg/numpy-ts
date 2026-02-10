/**
 * Additional array creation functions
 *
 * Tree-shakeable standalone functions that wrap the underlying ops.
 */

import { NDArrayCore, type DType } from '../core/ndarray-core';
import { ArrayStorage } from '../core/storage';
import { zeros, array, asarray } from '../creation';
import { getTypedArrayConstructor, DEFAULT_DTYPE } from '../core/dtype';

// Helper to convert ArrayStorage to NDArrayCore
function fromStorage(storage: ArrayStorage): NDArrayCore {
  return new NDArrayCore(storage);
}

// Helper: flatten an NDArrayCore to 1D
function flattenCore(a: NDArrayCore): NDArrayCore {
  const data = a.data;
  const storage = ArrayStorage.fromData(
    data.slice() as typeof data,
    [data.length],
    a.dtype as DType
  );
  return fromStorage(storage);
}

// ============================================================
// Array Conversion
// ============================================================

export function asanyarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  return asarray(a as NDArrayCore, dtype);
}

export function ascontiguousarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  // In our implementation, arrays are always contiguous
  return arr.copy();
}

export function asfortranarray(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  // Note: We don't actually support Fortran order, return C-order copy
  return arr.copy();
}

export function asarray_chkfinite(a: NDArrayCore | unknown, dtype?: DType): NDArrayCore {
  const arr = asarray(a as NDArrayCore, dtype);
  const data = arr.data;

  for (let i = 0; i < data.length; i++) {
    const val = data[i] as number;
    if (!Number.isFinite(val)) {
      throw new Error('array must not contain infs or NaNs');
    }
  }

  return arr;
}

export function require(
  a: NDArrayCore,
  dtype?: DType,
  _requirements?: string | string[]
): NDArrayCore {
  let result = a;
  if (dtype && dtype !== a.dtype) {
    result = result.astype(dtype);
  }
  // Requirements like 'C', 'F', 'A', 'W', 'O', 'E' are mostly no-ops in our implementation
  return result;
}

// ============================================================
// Diagonal and Triangular
// ============================================================

export function diag(v: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = v.shape;
  const data = v.data;
  const dtype = v.dtype as DType;

  if (shape.length === 1) {
    // Create a 2D array with v as diagonal
    const dim0 = shape[0]!;
    const n = dim0 + Math.abs(k);
    const result = zeros([n, n], dtype);
    const resultData = result.data;

    for (let i = 0; i < dim0; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row >= 0 && row < n && col >= 0 && col < n) {
        (resultData as Float64Array)[row * n + col] = data[i] as number;
      }
    }
    return result;
  } else if (shape.length === 2) {
    // Extract diagonal from 2D array
    const rows = shape[0]!;
    const cols = shape[1]!;
    const diagLen = Math.min(
      k >= 0 ? Math.min(rows, cols - k) : Math.min(rows + k, cols),
      Math.max(0, k >= 0 ? cols - k : rows + k)
    );

    if (diagLen <= 0) {
      return array([], dtype);
    }

    const resultArr: number[] = [];
    for (let i = 0; i < diagLen; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        resultArr.push(data[row * cols + col] as number);
      }
    }
    return array(resultArr, dtype);
  }

  throw new Error('Input must be 1-D or 2-D');
}

export function diagflat(v: NDArrayCore, k: number = 0): NDArrayCore {
  // Flatten v first, then create diagonal matrix
  const flat = flattenCore(v);
  return diag(flat, k);
}

export function tri(
  N: number,
  M?: number,
  k: number = 0,
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  const cols = M ?? N;
  const result = zeros([N, cols], dtype);
  const data = result.data;

  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= Math.min(i + k, cols - 1); j++) {
      if (j >= 0) {
        (data as Float64Array)[i * cols + j] = 1;
      }
    }
  }

  return result;
}

export function tril(m: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = m.shape;
  if (shape.length < 2) {
    throw new Error('Input must be at least 2-D');
  }

  const result = m.copy();
  const data = result.data;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;
  const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
  const matrixSize = rows * cols;

  for (let b = 0; b < batchSize; b++) {
    const offset = b * matrixSize;
    for (let i = 0; i < rows; i++) {
      for (let j = i + k + 1; j < cols; j++) {
        (data as Float64Array)[offset + i * cols + j] = 0;
      }
    }
  }

  return result;
}

export function triu(m: NDArrayCore, k: number = 0): NDArrayCore {
  const shape = m.shape;
  if (shape.length < 2) {
    throw new Error('Input must be at least 2-D');
  }

  const result = m.copy();
  const data = result.data;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;
  const batchSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
  const matrixSize = rows * cols;

  for (let b = 0; b < batchSize; b++) {
    const offset = b * matrixSize;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < Math.min(i + k, cols); j++) {
        (data as Float64Array)[offset + i * cols + j] = 0;
      }
    }
  }

  return result;
}

export function vander(x: NDArrayCore, N?: number, increasing: boolean = false): NDArrayCore {
  const n = x.size;
  const cols = N ?? n;
  const data = x.data;
  const result = zeros([n, cols], x.dtype as DType);
  const resultData = result.data;

  for (let i = 0; i < n; i++) {
    const val = data[i] as number;
    for (let j = 0; j < cols; j++) {
      const exp = increasing ? j : cols - 1 - j;
      (resultData as Float64Array)[i * cols + j] = Math.pow(val, exp);
    }
  }

  return result;
}

// ============================================================
// From Data Sources
// ============================================================

export function frombuffer(
  buffer: ArrayBuffer | TypedArray,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  offset: number = 0
): NDArrayCore {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  let data: TypedArray;
  if (buffer instanceof ArrayBuffer) {
    const byteOffset = offset * Constructor.BYTES_PER_ELEMENT;
    const length =
      count < 0 ? (buffer.byteLength - byteOffset) / Constructor.BYTES_PER_ELEMENT : count;
    data = new Constructor(buffer, byteOffset, length);
  } else {
    const start = offset;
    const end = count < 0 ? buffer.length : offset + count;
    // Extract values from source buffer and create new typed array
    const sliced = Array.from(buffer.slice(start, end) as ArrayLike<number | bigint>);
    data = new Constructor(sliced.length);
    for (let i = 0; i < sliced.length; i++) {
      (data as unknown as number[])[i] = sliced[i] as number;
    }
  }

  const storage = ArrayStorage.fromData(data, [data.length], dtype);
  return fromStorage(storage);
}

type TypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | BigInt64Array
  | BigUint64Array;

export function fromfunction(
  func: (...indices: number[]) => number,
  shape: number[],
  dtype: DType = DEFAULT_DTYPE
): NDArrayCore {
  const size = shape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  const data = new Constructor(size);
  const strides: number[] = [];
  let strideVal = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(strideVal);
    strideVal *= shape[i]!;
  }

  for (let flatIdx = 0; flatIdx < size; flatIdx++) {
    const indices: number[] = [];
    let remaining = flatIdx;
    for (let i = 0; i < shape.length; i++) {
      indices.push(Math.floor(remaining / strides[i]!));
      remaining = remaining % strides[i]!;
    }
    (data as Float64Array)[flatIdx] = func(...indices);
  }

  const storage = ArrayStorage.fromData(data, shape, dtype);
  return fromStorage(storage);
}

export function fromiter(
  iter: Iterable<number>,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1
): NDArrayCore {
  const values: number[] = [];
  let i = 0;
  for (const val of iter) {
    if (count >= 0 && i >= count) break;
    values.push(val);
    i++;
  }
  return array(values, dtype);
}

export function fromstring(
  string: string,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  sep: string = ''
): NDArrayCore {
  if (sep === '') {
    throw new Error('fromstring with binary data not supported, use frombuffer');
  }

  const parts = string.split(sep).filter((s) => s.trim() !== '');
  const values = parts.map((s) => parseFloat(s.trim()));
  const actualValues = count >= 0 ? values.slice(0, count) : values;

  return array(actualValues, dtype);
}

export function fromfile(
  _file: string,
  _dtype: DType = DEFAULT_DTYPE,
  _count: number = -1,
  _sep: string = ''
): NDArrayCore {
  throw new Error('fromfile requires Node.js file system access');
}

export function meshgrid(...arrays: NDArrayCore[]): NDArrayCore[] {
  if (arrays.length === 0) return [];
  if (arrays.length === 1) return [arrays[0]!.copy()];

  const shapes = arrays.map((a) => a.size);
  const outputShape = [...shapes];

  const results: NDArrayCore[] = [];

  for (let dim = 0; dim < arrays.length; dim++) {
    const arr = arrays[dim]!;
    const data = arr.data;
    const result = zeros(outputShape, arr.dtype as DType);
    const resultData = result.data;

    // Calculate strides for output
    const strides: number[] = [];
    let strideVal = 1;
    for (let i = outputShape.length - 1; i >= 0; i--) {
      strides.unshift(strideVal);
      strideVal *= outputShape[i]!;
    }

    const totalSize = outputShape.reduce((a, b) => a * b, 1);
    for (let flatIdx = 0; flatIdx < totalSize; flatIdx++) {
      // Get the index along this dimension
      const idx = Math.floor(flatIdx / strides[dim]!) % shapes[dim]!;
      (resultData as Float64Array)[flatIdx] = data[idx] as number;
    }

    results.push(result);
  }

  return results;
}
