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
 * Convert multi-index to linear index in row-major order
 */
export function multiIndexToLinear(indices: number[], shape: readonly number[]): number {
  let linearIdx = 0;
  let stride = 1;
  for (let i = indices.length - 1; i >= 0; i--) {
    linearIdx += indices[i]! * stride;
    stride *= shape[i]!;
  }
  return linearIdx;
}

/**
 * Convert outer index and axis index to full multi-index
 * Used in reductions along a specific axis
 *
 * @param outerIdx - Linear index in the reduced (output) array
 * @param axis - The axis being reduced
 * @param axisIdx - Position along the reduction axis
 * @param shape - Original array shape
 * @returns Full multi-index in the original array
 */
export function outerIndexToMultiIndex(
  outerIdx: number,
  axis: number,
  axisIdx: number,
  shape: readonly number[]
): number[] {
  const ndim = shape.length;
  const indices = new Array(ndim);
  const outputShape = Array.from(shape).filter((_, i) => i !== axis);

  // Convert outerIdx to multi-index in the output shape
  let remaining = outerIdx;
  for (let i = outputShape.length - 1; i >= 0; i--) {
    indices[i >= axis ? i + 1 : i] = remaining % outputShape[i]!;
    remaining = Math.floor(remaining / outputShape[i]!);
  }

  // Insert the axis index
  indices[axis] = axisIdx;
  return indices;
}
