/**
 * Broadcasting utilities for NumPy-compatible array operations
 *
 * Implements NumPy broadcasting rules without external dependencies
 */

import { ArrayStorage } from './storage';

/**
 * Check if two or more shapes are broadcast-compatible
 * and compute the resulting output shape
 *
 * @param shapes - Array of shapes to broadcast
 * @returns The broadcast output shape, or null if incompatible
 *
 * @example
 * ```typescript
 * computeBroadcastShape([[3, 4], [4]]);     // [3, 4]
 * computeBroadcastShape([[3, 4], [3, 1]]);  // [3, 4]
 * computeBroadcastShape([[3, 4], [5]]);     // null (incompatible)
 * ```
 */
export function computeBroadcastShape(shapes: readonly number[][]): number[] | null {
  if (shapes.length === 0) {
    return [];
  }

  if (shapes.length === 1) {
    return Array.from(shapes[0]!);
  }

  // Find max number of dimensions
  const maxNdim = Math.max(...shapes.map((s) => s.length));
  const result = new Array(maxNdim);

  for (let i = 0; i < maxNdim; i++) {
    let dim = 1;
    for (const shape of shapes) {
      const shapeIdx = shape.length - maxNdim + i;
      const shapeDim = shapeIdx < 0 ? 1 : shape[shapeIdx]!;

      if (shapeDim === 1) {
        // Can be broadcast
        continue;
      } else if (dim === 1) {
        // First non-1 dimension
        dim = shapeDim;
      } else if (dim !== shapeDim) {
        // Incompatible
        return null;
      }
    }
    result[i] = dim;
  }

  return result;
}

/**
 * Check if two shapes are broadcast-compatible
 *
 * @param shape1 - First shape
 * @param shape2 - Second shape
 * @returns true if shapes can be broadcast together, false otherwise
 *
 * @example
 * ```typescript
 * areBroadcastable([3, 4], [4]);      // true
 * areBroadcastable([3, 4], [3, 1]);   // true
 * areBroadcastable([3, 4], [5]);      // false
 * ```
 */
export function areBroadcastable(shape1: readonly number[], shape2: readonly number[]): boolean {
  return computeBroadcastShape([Array.from(shape1), Array.from(shape2)]) !== null;
}

/**
 * Compute the strides for broadcasting an array to a target shape
 * Returns strides where dimensions that need broadcasting have stride 0
 */
function broadcastStrides(
  shape: readonly number[],
  strides: readonly number[],
  targetShape: readonly number[]
): number[] {
  const ndim = shape.length;
  const targetNdim = targetShape.length;
  const result = new Array(targetNdim).fill(0);

  // Align dimensions from the right
  for (let i = 0; i < ndim; i++) {
    const targetIdx = targetNdim - ndim + i;
    const dim = shape[i]!;
    const targetDim = targetShape[targetIdx]!;

    if (dim === targetDim) {
      // Same size, use original stride
      result[targetIdx] = strides[i]!;
    } else if (dim === 1) {
      // Broadcasting, stride is 0 (repeat along this dimension)
      result[targetIdx] = 0;
    } else {
      // This shouldn't happen if shapes were validated
      throw new Error('Invalid broadcast');
    }
  }

  return result;
}

/**
 * Broadcast an ArrayStorage to a target shape
 * Returns a view with modified strides for broadcasting
 *
 * @param storage - The storage to broadcast
 * @param targetShape - The target shape to broadcast to
 * @returns A new ArrayStorage view with broadcasting strides
 */
export function broadcastTo(storage: ArrayStorage, targetShape: readonly number[]): ArrayStorage {
  const broadcastedStrides = broadcastStrides(storage.shape, storage.strides, targetShape);
  return ArrayStorage.fromData(
    storage.data,
    Array.from(targetShape),
    storage.dtype,
    broadcastedStrides,
    storage.offset
  );
}

/**
 * Broadcast multiple ArrayStorage objects to a common shape
 *
 * Returns views of the input arrays broadcast to the same shape.
 * Views share memory with the original arrays.
 *
 * @param storages - ArrayStorage objects to broadcast
 * @returns Array of broadcast ArrayStorage views
 * @throws Error if arrays have incompatible shapes
 */
export function broadcastArrays(storages: ArrayStorage[]): ArrayStorage[] {
  if (storages.length === 0) {
    return [];
  }

  if (storages.length === 1) {
    return storages;
  }

  // Compute broadcast shape
  const shapes = storages.map((s) => Array.from(s.shape));
  const targetShape = computeBroadcastShape(shapes);

  if (targetShape === null) {
    throw new Error(
      `operands could not be broadcast together with shapes ${shapes.map((s) => JSON.stringify(s)).join(' ')}`
    );
  }

  // Broadcast each storage to the target shape
  return storages.map((s) => broadcastTo(s, targetShape));
}

/**
 * Compute the broadcast shape for multiple shapes without creating arrays.
 * Returns the resulting shape if all shapes are broadcast-compatible.
 *
 * This is the NumPy-compatible function for computing broadcast shape.
 *
 * @param shapes - Variable number of shapes to broadcast
 * @returns The broadcast output shape
 * @throws Error if shapes are not broadcast-compatible
 *
 * @example
 * ```typescript
 * broadcastShapes([3, 4], [4]);      // [3, 4]
 * broadcastShapes([3, 4], [3, 1]);   // [3, 4]
 * broadcastShapes([3, 4], [5]);      // Error
 * ```
 */
export function broadcastShapes(...shapes: readonly number[][]): number[] {
  const result = computeBroadcastShape(shapes);

  if (result === null) {
    const shapeStrs = shapes.map((s) => `(${s.join(',')})`).join(' ');
    throw new Error(
      `shape mismatch: objects cannot be broadcast to a single shape. Mismatch is between ${shapeStrs}`
    );
  }

  return result;
}

/**
 * Generate a descriptive error message for broadcasting failures
 *
 * @param shapes - The incompatible shapes
 * @param operation - The operation being attempted (e.g., 'add', 'multiply')
 * @returns Error message string
 */
export function broadcastErrorMessage(shapes: readonly number[][], operation?: string): string {
  const opStr = operation ? ` for ${operation}` : '';
  const shapeStrs = shapes.map((s) => `(${s.join(',')})`).join(' ');
  return `operands could not be broadcast together${opStr} with shapes ${shapeStrs}`;
}
