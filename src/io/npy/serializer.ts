/**
 * NPY file serializer
 *
 * Serializes NDArray objects to NumPy .npy format (v3.0).
 * Always writes in little-endian, C-contiguous order.
 *
 * v3.0 is identical to v2.0 but allows UTF-8 in dtype descriptions.
 */

import { NDArray } from '../../core/ndarray';
import { getDTypeSize, isBigIntDType, type DType } from '../../core/dtype';
import { NPY_MAGIC, DTYPE_TO_DESCR, isSystemLittleEndian } from './format';

/**
 * Serialize an NDArray to NPY format (v3.0)
 *
 * @param arr - The NDArray to serialize
 * @returns A Uint8Array containing the NPY file data
 */
export function serializeNpy(arr: NDArray): Uint8Array {
  const shape = arr.shape;
  const dtype = arr.dtype as DType;

  // Build header dictionary string
  const descr = DTYPE_TO_DESCR[dtype];
  const shapeStr =
    shape.length === 0 ? '()' : shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;

  // Python dict format: {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }
  let headerDict = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;

  // Header must be padded to 64-byte alignment (including magic, version, header_len)
  // v3.0 uses 4 bytes for header length (same as v2.0)
  // Total prefix is 6 (magic) + 2 (version) + 4 (header_len) = 12 bytes
  // Header string + newline should make total divisible by 64
  const PREFIX_LEN = 12;
  const totalBeforeData = PREFIX_LEN + headerDict.length + 1; // +1 for trailing newline
  const padding = (64 - (totalBeforeData % 64)) % 64;
  headerDict = headerDict + ' '.repeat(padding) + '\n';

  const headerBytes = new TextEncoder().encode(headerDict);
  const headerLen = headerBytes.length;

  // Calculate data size
  const numElements = arr.size;
  const itemsize = getDTypeSize(dtype);
  const dataSize = numElements * itemsize;

  // Allocate output buffer
  const totalSize = PREFIX_LEN + headerLen + dataSize;
  const output = new Uint8Array(totalSize);

  // Write magic number
  output.set(NPY_MAGIC, 0);

  // Write version (3.0)
  output[6] = 3;
  output[7] = 0;

  // Write header length (4-byte little-endian)
  output[8] = headerLen & 0xff;
  output[9] = (headerLen >> 8) & 0xff;
  output[10] = (headerLen >> 16) & 0xff;
  output[11] = (headerLen >> 24) & 0xff;

  // Write header string
  output.set(headerBytes, PREFIX_LEN);

  // Write data
  const dataOffset = PREFIX_LEN + headerLen;
  writeArrayData(arr, output.subarray(dataOffset), itemsize);

  return output;
}

/**
 * Write array data to the output buffer
 */
function writeArrayData(arr: NDArray, output: Uint8Array, itemsize: number): void {
  const dtype = arr.dtype as DType;
  const size = arr.size;
  const isLittleEndian = isSystemLittleEndian();
  const isBigInt = isBigIntDType(dtype);

  // Get raw data - need to handle non-contiguous arrays
  const storage = arr['_storage']; // Access private member
  const isCContiguous = storage.isCContiguous && storage.offset === 0;

  if (isCContiguous && isLittleEndian) {
    // Fast path: just copy the underlying buffer
    const srcData = storage.data;
    const srcBytes = new Uint8Array(srcData.buffer, srcData.byteOffset, size * itemsize);
    output.set(srcBytes);
  } else {
    // Slow path: element by element copy with potential byte swapping
    const dataView = new DataView(output.buffer, output.byteOffset, output.byteLength);

    for (let i = 0; i < size; i++) {
      const value = storage.iget(i);
      const offset = i * itemsize;

      if (isBigInt) {
        // Write BigInt as little-endian
        writeBigInt64LE(dataView, offset, value as bigint, dtype === 'uint64');
      } else {
        // Write number as little-endian
        writeNumberLE(dataView, offset, value as number, dtype);
      }
    }
  }
}

/**
 * Write a BigInt as little-endian
 */
function writeBigInt64LE(view: DataView, offset: number, value: bigint, unsigned: boolean): void {
  if (unsigned) {
    view.setBigUint64(offset, value, true);
  } else {
    view.setBigInt64(offset, value, true);
  }
}

/**
 * Write a number as little-endian
 */
function writeNumberLE(view: DataView, offset: number, value: number, dtype: DType): void {
  switch (dtype) {
    case 'float64':
      view.setFloat64(offset, value, true);
      break;
    case 'float32':
      view.setFloat32(offset, value, true);
      break;
    case 'int32':
      view.setInt32(offset, value, true);
      break;
    case 'int16':
      view.setInt16(offset, value, true);
      break;
    case 'int8':
      view.setInt8(offset, value);
      break;
    case 'uint32':
      view.setUint32(offset, value, true);
      break;
    case 'uint16':
      view.setUint16(offset, value, true);
      break;
    case 'uint8':
    case 'bool':
      view.setUint8(offset, value);
      break;
    default:
      throw new Error(`Unsupported dtype for serialization: ${dtype}`);
  }
}
