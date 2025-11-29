/**
 * NPY file parser
 *
 * Parses NumPy .npy files (both v1 and v2/v3 formats) into NDArray objects.
 */

import { NDArray } from '../../core/ndarray';
import { ArrayStorage } from '../../core/storage';
import { getTypedArrayConstructor, isBigIntDType, type DType } from '../../core/dtype';
import {
  NPY_MAGIC,
  parseDescriptor,
  InvalidNpyError,
  type NpyHeader,
  type NpyMetadata,
} from './format';

/**
 * Parse an NPY file from a Uint8Array or ArrayBuffer
 *
 * @param buffer - The NPY file contents
 * @returns An NDArray containing the parsed data
 * @throws InvalidNpyError if the file format is invalid
 * @throws UnsupportedDTypeError if the dtype is not supported
 */
export function parseNpy(buffer: ArrayBuffer | Uint8Array): NDArray {
  const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
  const metadata = parseNpyHeader(bytes);
  return parseNpyData(bytes, metadata);
}

/**
 * Parse just the NPY header without reading the data
 *
 * @param bytes - The NPY file bytes
 * @returns Parsed metadata including version, header, and data offset
 */
export function parseNpyHeader(bytes: Uint8Array): NpyMetadata {
  // Check minimum size
  if (bytes.length < 10) {
    throw new InvalidNpyError('File too small to be a valid NPY file');
  }

  // Verify magic number
  for (let i = 0; i < NPY_MAGIC.length; i++) {
    if (bytes[i] !== NPY_MAGIC[i]) {
      throw new InvalidNpyError('Invalid NPY magic number');
    }
  }

  // Read version
  const major = bytes[6]!;
  const minor = bytes[7]!;

  if (major !== 1 && major !== 2 && major !== 3) {
    throw new InvalidNpyError(`Unsupported NPY version: ${major}.${minor}`);
  }

  // Read header length
  let headerLen: number;
  let headerStart: number;

  if (major === 1) {
    // v1.0: 2-byte little-endian header length
    headerLen = bytes[8]! | (bytes[9]! << 8);
    headerStart = 10;
  } else {
    // v2.0 and v3.0: 4-byte little-endian header length
    headerLen = bytes[8]! | (bytes[9]! << 8) | (bytes[10]! << 16) | (bytes[11]! << 24);
    headerStart = 12;
  }

  // Read header string
  const headerEnd = headerStart + headerLen;
  if (bytes.length < headerEnd) {
    throw new InvalidNpyError('File truncated: header extends beyond file');
  }

  const headerBytes = bytes.slice(headerStart, headerEnd);
  const headerStr = new TextDecoder('utf-8').decode(headerBytes).trim();

  // Parse header dictionary
  const header = parseHeaderDict(headerStr);

  return {
    version: { major, minor },
    header,
    dataOffset: headerEnd,
  };
}

/**
 * Parse the data section of an NPY file given parsed metadata
 */
export function parseNpyData(bytes: Uint8Array, metadata: NpyMetadata): NDArray {
  const { header, dataOffset } = metadata;

  // Parse dtype descriptor
  const { dtype, needsByteSwap, itemsize } = parseDescriptor(header.descr);

  // Calculate expected data size
  const numElements = header.shape.reduce((a, b) => a * b, 1);
  const expectedBytes = numElements * itemsize;
  const actualBytes = bytes.length - dataOffset;

  if (actualBytes < expectedBytes) {
    throw new InvalidNpyError(
      `File truncated: expected ${expectedBytes} bytes of data, got ${actualBytes}`
    );
  }

  // Extract data buffer - create a copy to ensure we have a plain ArrayBuffer
  const dataBuffer = new ArrayBuffer(expectedBytes);
  const dataView = new Uint8Array(dataBuffer);
  dataView.set(bytes.subarray(dataOffset, dataOffset + expectedBytes));

  // Create typed array from data
  const typedData = createTypedArray(dataBuffer, dtype, numElements, needsByteSwap, itemsize);

  // Handle Fortran order (column-major)
  // NumPy stores data in row-major (C order) by default
  // If fortran_order is true, we need to adjust
  let shape = header.shape;
  let storage: ArrayStorage;

  if (header.fortran_order && shape.length > 1) {
    // For Fortran order, we can either:
    // 1. Transpose the shape and data (requires copy)
    // 2. Use column-major strides (creates a view)
    // We'll transpose to convert to C-order for consistency
    const reversedShape = [...shape].reverse();
    const tempStorage = ArrayStorage.fromData(typedData, reversedShape, dtype);

    // Transpose to get correct C-order layout
    storage = transposeStorage(tempStorage, reversedShape);
    shape = header.shape; // Use original shape after transpose
  } else {
    storage = ArrayStorage.fromData(typedData, [...shape], dtype);
  }

  return new NDArray(storage);
}

/**
 * Parse the Python dictionary header string
 */
function parseHeaderDict(headerStr: string): NpyHeader {
  // Header is a Python dict literal like:
  // {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }

  // Simple regex-based parser for the specific format
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);

  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new InvalidNpyError(`Failed to parse NPY header: ${headerStr}`);
  }

  const descr = descrMatch[1]!;
  const fortran_order = fortranMatch[1] === 'True';

  // Parse shape tuple
  const shapeStr = shapeMatch[1]!.trim();
  let shape: number[];

  if (shapeStr === '') {
    // Scalar: shape is ()
    shape = [];
  } else {
    // Parse comma-separated integers
    shape = shapeStr
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s !== '')
      .map((s) => {
        const n = parseInt(s, 10);
        if (isNaN(n)) {
          throw new InvalidNpyError(`Invalid shape value: ${s}`);
        }
        return n;
      });
  }

  return { descr, fortran_order, shape };
}

/**
 * Create a typed array from raw bytes with optional byte swapping
 */
function createTypedArray(
  buffer: ArrayBuffer,
  dtype: DType,
  numElements: number,
  needsByteSwap: boolean,
  itemsize: number
):
  | Float64Array
  | Float32Array
  | BigInt64Array
  | Int32Array
  | Int16Array
  | Int8Array
  | BigUint64Array
  | Uint32Array
  | Uint16Array
  | Uint8Array {
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new InvalidNpyError(`Cannot create array for dtype: ${dtype}`);
  }

  if (!needsByteSwap) {
    // Fast path: no byte swapping needed
    return new Constructor(buffer, 0, numElements);
  }

  // Slow path: need to byte swap
  const bytes = new Uint8Array(buffer);
  const swapped = new Uint8Array(buffer.byteLength);

  for (let i = 0; i < numElements; i++) {
    const start = i * itemsize;
    // Reverse bytes for this element
    for (let j = 0; j < itemsize; j++) {
      swapped[start + j] = bytes[start + itemsize - 1 - j]!;
    }
  }

  return new Constructor(swapped.buffer, 0, numElements);
}

/**
 * Transpose storage to convert from Fortran to C order
 */
function transposeStorage(storage: ArrayStorage, shape: readonly number[]): ArrayStorage {
  const ndim = shape.length;
  const size = storage.size;
  const dtype = storage.dtype;
  const Constructor = getTypedArrayConstructor(dtype);

  if (!Constructor) {
    throw new InvalidNpyError(`Cannot create array for dtype: ${dtype}`);
  }

  const newData = new Constructor(size);
  const newShape = [...shape].reverse();

  // Compute strides for both orderings
  const oldStrides = computeStrides(shape);
  const newStrides = computeStrides(newShape);

  // Copy data with transposition
  const indices = new Array(ndim).fill(0);

  for (let linearIdx = 0; linearIdx < size; linearIdx++) {
    // Get multi-index in old layout
    let remaining = linearIdx;
    for (let i = 0; i < ndim; i++) {
      const dimSize = oldStrides[i]!;
      indices[i] = Math.floor(remaining / dimSize);
      remaining = remaining % dimSize;
    }

    // Compute new linear index (reverse indices for transpose)
    let newLinearIdx = 0;
    for (let i = 0; i < ndim; i++) {
      newLinearIdx += indices[ndim - 1 - i]! * newStrides[i]!;
    }

    // Copy value
    if (isBigIntDType(dtype)) {
      (newData as BigInt64Array | BigUint64Array)[newLinearIdx] = storage.iget(linearIdx) as bigint;
    } else {
      (newData as Exclude<typeof newData, BigInt64Array | BigUint64Array>)[newLinearIdx] =
        storage.iget(linearIdx) as number;
    }
  }

  return ArrayStorage.fromData(newData, newShape, dtype);
}

/**
 * Compute C-order strides for a shape
 */
function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}
