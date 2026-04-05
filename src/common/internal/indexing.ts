/**
 * Indexing utilities for array operations
 * @internal
 */

/**
 * Compute row-major strides for a given shape
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

/**
 * Precompute stride-based offsets for iterating over an axis reduction.
 *
 * Returns { baseOffsets, axisStride } where:
 * - baseOffsets[outerIdx] is the buffer offset of the first element along the
 *   reduction axis for that output position
 * - axisStride is the stride along the reduction axis in the buffer
 *
 * Inner loop becomes:
 *   let off = baseOffsets[outerIdx];
 *   for (k = 0; k < axisSize; k++, off += axisStride) { data[off] ... }
 *
 * This eliminates per-element allocations and index decomposition.
 */
export function precomputeAxisOffsets(
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  axis: number,
  outerSize: number
): { baseOffsets: Int32Array; axisStride: number } {
  const ndim = shape.length;
  const axisStride = strides[axis]!;

  // Build output shape (shape with axis removed) and corresponding strides
  const outerDims: number[] = [];
  const outerStrides: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (i !== axis) {
      outerDims.push(shape[i]!);
      outerStrides.push(strides[i]!);
    }
  }

  const baseOffsets = new Int32Array(outerSize);
  const outerNdim = outerDims.length;

  if (outerNdim === 0) {
    // Scalar output — single element
    baseOffsets[0] = offset;
  } else if (outerNdim === 1) {
    // 2D input, reducing one axis — fast path
    const s = outerStrides[0]!;
    for (let i = 0; i < outerSize; i++) {
      baseOffsets[i] = offset + i * s;
    }
  } else {
    // General N-D case — decompose outerIdx into multi-index, dot with strides
    // Precompute cumulative products for index decomposition (right-to-left)
    const cumProd = new Int32Array(outerNdim);
    cumProd[outerNdim - 1] = 1;
    for (let i = outerNdim - 2; i >= 0; i--) {
      cumProd[i] = cumProd[i + 1]! * outerDims[i + 1]!;
    }

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let base = offset;
      let remaining = outerIdx;
      for (let d = 0; d < outerNdim; d++) {
        const idx = (remaining / cumProd[d]!) | 0;
        remaining -= idx * cumProd[d]!;
        base += idx * outerStrides[d]!;
      }
      baseOffsets[outerIdx] = base;
    }
  }

  return { baseOffsets, axisStride };
}
/**
 * Replaces '...' with an appropriate number of ':'.
 * If there's not '...', it's implicit at the end.
 *
 *
 * So the result of this function will have length ndim + newaxisCount
 * where newaxisCount is the number of 'newaxis' in the input.
 */
export function expandEllipsis<T>(indices: (T | string)[], ndim: number): (T | string)[] {
  const ellipsisCount = indices.filter((x) => x === '...').length;
  const newAxisCount = indices.filter((x) => x === 'newaxis').length;
  if (ellipsisCount > 1) throw new Error('an index can only have a single ellipsis (...)');
  const otherCount = indices.length - ellipsisCount - newAxisCount;
  const replacements = ndim - otherCount;
  if (replacements < 0)
    throw new Error(
      `Too many indices for array: array is ${ndim}-dimensional, but ${otherCount} were indexed`
    );
  if (replacements === 0 && ellipsisCount === 0) return indices;

  const ellipsisIndex = ellipsisCount === 0 ? indices.length : indices.indexOf('...');
  const result = indices.slice();
  result.splice(ellipsisIndex, ellipsisCount, ...Array(replacements).fill(':'));

  return result;
}
