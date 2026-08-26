/**
 * Computation backend abstraction
 *
 * Internal module for element-wise and broadcast operations.
 * Provides a swappable backend for different computation strategies.
 *
 * @internal
 */

import { Complex } from '../complex';
import { isBigIntDType, isComplexDType, mathResultDtype, promoteDTypes } from '../dtype';
import { ArrayStorage } from '../storage';
import { wasmCompare } from '../wasm/compare';

/**
 * Compute the broadcast shape of two arrays
 * Returns the shape that results from broadcasting a and b together
 * Throws if shapes are not compatible for broadcasting
 */
export function broadcastShapes(shapeA: readonly number[], shapeB: readonly number[]): number[] {
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;
  const ndim = Math.max(ndimA, ndimB);
  const result = new Array(ndim);

  for (let i = 0; i < ndim; i++) {
    const dimA = i < ndim - ndimA ? 1 : shapeA[i - (ndim - ndimA)]!;
    const dimB = i < ndim - ndimB ? 1 : shapeB[i - (ndim - ndimB)]!;

    if (dimA === dimB) {
      result[i] = dimA;
    } else if (dimA === 1) {
      result[i] = dimB;
    } else if (dimB === 1) {
      result[i] = dimA;
    } else {
      throw new Error(
        `operands could not be broadcast together with shapes ${JSON.stringify(Array.from(shapeA))} ${JSON.stringify(Array.from(shapeB))}`,
      );
    }
  }

  return result;
}

/**
 * Compute the strides for broadcasting an array to a target shape
 * Returns strides where dimensions that need broadcasting have stride 0
 */
function broadcastStrides(
  shape: readonly number[],
  strides: readonly number[],
  targetShape: readonly number[],
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
 * Create a broadcast view of an ArrayStorage
 * The returned storage shares data with the original but has different shape/strides
 */
function broadcastTo(storage: ArrayStorage, targetShape: readonly number[]): ArrayStorage {
  const broadcastedStrides = broadcastStrides(storage.shape, storage.strides, targetShape);
  return ArrayStorage.fromData(
    storage.data,
    Array.from(targetShape),
    storage.dtype,
    broadcastedStrides,
    storage.offset,
  );
}

/**
 * Perform element-wise operation with broadcasting
 *
 * NOTE: This is the slow path for broadcasting/non-contiguous arrays.
 * Fast paths for contiguous arrays are implemented directly in ops/arithmetic.ts
 *
 * @param a - First array storage
 * @param b - Second array storage
 * @param op - Operation to perform (a, b) => result
 * @param opName - Name of operation (for special handling)
 * @returns Result storage
 */
export function elementwiseBinaryOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => number,
  opName: string,
): ArrayStorage {
  // Determine output dtype using NumPy promotion rules
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // FAST PATH: Same shape, both contiguous, non-BigInt types
  // This avoids broadcasting overhead and uses direct array access
  const aShape = a.shape;
  const bShape = b.shape;
  const sameShape = aShape.length === bShape.length && aShape.every((dim, i) => dim === bShape[i]);

  if (
    sameShape &&
    a.isCContiguous &&
    b.isCContiguous &&
    !isBigIntDType(a.dtype) &&
    !isBigIntDType(b.dtype) &&
    !isBigIntDType(resultDtype)
  ) {
    const size = a.size;
    const result = ArrayStorage.empty(Array.from(aShape), resultDtype);
    const resultData = result.data;
    const aOff = a.offset;
    const bOff = b.offset;

    const aData = a.data;
    const bData = b.data;
    if (aOff === 0 && bOff === 0) {
      for (let i = 0; i < size; i++) {
        resultData[i] = op(aData[i] as number, bData[i] as number);
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = op(aData[aOff + i] as number, bData[bOff + i] as number);
      }
    }
    return result;
  }

  // SLOW PATH: Broadcasting or non-contiguous arrays
  // Compute broadcast shape
  const outputShape = broadcastShapes(a.shape, b.shape);

  // Create broadcast views
  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);

  // Create result storage
  const result = ArrayStorage.empty(outputShape, resultDtype);
  const resultData = result.data;
  const size = result.size;

  if (isBigIntDType(resultDtype)) {
    // BigInt arithmetic - no precision loss
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to BigInt - handle case where value is already BigInt
      // Note: Complex values get their real part extracted
      const aNum = aRaw instanceof Complex ? aRaw.re : aRaw;
      const bNum = bRaw instanceof Complex ? bRaw.re : bRaw;
      const aVal = typeof aNum === 'bigint' ? aNum : BigInt(Math.round(aNum as number));
      const bVal = typeof bNum === 'bigint' ? bNum : BigInt(Math.round(bNum as number));

      // Use BigInt operations
      if (opName === 'add') {
        resultTyped[i] = aVal + bVal;
      } else if (opName === 'subtract') {
        resultTyped[i] = aVal - bVal;
      } else if (opName === 'multiply') {
        resultTyped[i] = aVal * bVal;
      } else if (opName === 'divide') {
        resultTyped[i] = aVal / bVal;
      } else {
        resultTyped[i] = BigInt(Math.round(op(Number(aVal), Number(bVal))));
      }
    }
  } else {
    // Regular numeric types (including float dtypes)
    // Need to convert BigInt values to Number if mixing dtypes
    const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

    for (let i = 0; i < size; i++) {
      const aRaw = aBroadcast.iget(i);
      const bRaw = bBroadcast.iget(i);

      // Convert to Number if needed (handles BigInt → float promotion)
      const aVal = needsConversion && typeof aRaw === 'bigint' ? Number(aRaw) : Number(aRaw);
      const bVal = needsConversion && typeof bRaw === 'bigint' ? Number(bRaw) : Number(bRaw);

      resultData[i] = op(aVal, bVal);
    }
  }

  return result;
}

/**
 * Perform element-wise comparison with broadcasting
 * Returns boolean array (dtype: 'bool', stored as Uint8Array)
 */
// --- Comparison fast-path loops, one function per TypedArray type ---
//
// The bodies are identical and the duplication is deliberate. A single generic
// loop reading `aData[i]` sees every TypedArray type used anywhere in the
// program; past a few types V8 abandons the inline cache on that load and
// *every* dtype pays. Measured on a [100x100] compare: 9.9us for int8 alone,
// 58.2us for the same int8 call once six other dtypes had run — and float64
// degraded just as badly. One function per type keeps each load monomorphic
// (58.2us -> 6.9us). The `k` switch is free: it compiles to a jump and the
// comparison is inlined rather than reached through a closure.

function cmpF64(
  a: Float64Array,
  b: Float64Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpF32(
  a: Float32Array,
  b: Float32Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpI32(
  a: Int32Array,
  b: Int32Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpU32(
  a: Uint32Array,
  b: Uint32Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpI16(
  a: Int16Array,
  b: Int16Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpU16(
  a: Uint16Array,
  b: Uint16Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpI8(
  a: Int8Array,
  b: Int8Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpU8(
  a: Uint8Array,
  b: Uint8Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

function cmpF16(
  a: Float16Array,
  b: Float16Array,
  o: Uint8Array,
  ao: number,
  bo: number,
  n: number,
  k: ComparisonKind,
): void {
  switch (k) {
    case 'eq':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i]! ? 1 : 0;
      return;
    case 'ne':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i]! ? 1 : 0;
      return;
    case 'lt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i]! ? 1 : 0;
      return;
    case 'le':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i]! ? 1 : 0;
      return;
    case 'gt':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i]! ? 1 : 0;
      return;
    case 'ge':
      for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i]! ? 1 : 0;
      return;
  }
}

/** Element-type constructor -> specialised loop. One lookup per call. */
const CMP_LOOPS = new Map<
  unknown,
  (a: never, b: never, o: Uint8Array, ao: number, bo: number, n: number, k: ComparisonKind) => void
>([
  [Float64Array, cmpF64 as never],
  [Float32Array, cmpF32 as never],
  [Int32Array, cmpI32 as never],
  [Uint32Array, cmpU32 as never],
  [Int16Array, cmpI16 as never],
  [Uint16Array, cmpU16 as never],
  [Int8Array, cmpI8 as never],
  [Uint8Array, cmpU8 as never],
]);

// float16 only exists on engines that ship Float16Array; elsewhere the storage
// is a Float32Array and the entry above already covers it.
if (typeof Float16Array !== 'undefined') {
  CMP_LOOPS.set(Float16Array, cmpF16 as never);
}

/** Exact BigInt comparison, avoiding the precision loss of Number(). */
function compareBigInt(a: bigint, b: bigint, k: ComparisonKind): boolean {
  switch (k) {
    case 'eq':
      return a === b;
    case 'ne':
      return a !== b;
    case 'lt':
      return a < b;
    case 'le':
      return a <= b;
    case 'gt':
      return a > b;
    case 'ge':
      return a >= b;
  }
}

/**
 * Which comparison the caller wants, so the fast path can run a dedicated loop.
 *
 * Passing a closure instead makes the per-element call site megamorphic once
 * more than one comparison operator is used in a process: measured 9us for the
 * first operator and ~52us for every one after it, on the same data. Naming the
 * operator lets each loop keep its own monomorphic site.
 */
export type ComparisonKind = 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge';

export function elementwiseComparisonOp(
  a: ArrayStorage,
  b: ArrayStorage,
  op: (a: number, b: number) => boolean,
  kind?: ComparisonKind,
): ArrayStorage {
  // Compute broadcast shape
  const outputShape = broadcastShapes(a.shape, b.shape);

  // FAST PATH: same shape, both contiguous, non-BigInt — mirrors the arithmetic
  // fast path above. Avoids building broadcast views and the per-element
  // iget()/Number() work, which costs far more than the comparison itself:
  // multi-dimensional index arithmetic per element made a [100x100] compare
  // ~22x slower than a direct typed-array loop.
  const aShape = a.shape;
  const bShape = b.shape;

  // WASM kernel first, and deliberately outside the guard below: it handles
  // int64/uint64 natively, comparing true 64-bit values. Every JS path here
  // funnels BigInt through Number(), which silently collapses values above
  // 2^53 — equal(2^53, 2^53+1) returned true. The kernel is both faster and
  // more correct, so it gets first refusal.
  if (kind) {
    const viaWasm = wasmCompare(kind, a, b);
    if (viaWasm) return viaWasm;
  }

  if (
    aShape.length === bShape.length &&
    aShape.every((dim, i) => dim === bShape[i]) &&
    a.isCContiguous &&
    b.isCContiguous &&
    !isBigIntDType(a.dtype) &&
    !isBigIntDType(b.dtype) &&
    !isComplexDType(a.dtype) &&
    !isComplexDType(b.dtype)
  ) {
    const fastResult = ArrayStorage.empty(Array.from(aShape), 'bool');
    const fastData = fastResult.data as Uint8Array;
    const n = a.size;
    const aData = a.data;
    const bData = b.data;
    const aOff = a.offset;
    const bOff = b.offset;

    // Specialised path: identical element types plus a named comparison, so the
    // loop has a monomorphic load site. Mixed dtypes and untagged callers fall
    // through to the generic loop below.
    if (kind && aData.constructor === bData.constructor) {
      const loop = CMP_LOOPS.get(aData.constructor);
      if (loop) {
        (
          loop as unknown as (
            a: unknown,
            b: unknown,
            o: Uint8Array,
            ao: number,
            bo: number,
            n: number,
            k: ComparisonKind,
          ) => void
        )(aData, bData, fastData, aOff, bOff, n, kind);
        return fastResult;
      }
    }

    for (let i = 0; i < n; i++) {
      fastData[i] = op(aData[aOff + i] as number, bData[bOff + i] as number) ? 1 : 0;
    }
    return fastResult;
  }

  // Create broadcast views
  const aBroadcast = broadcastTo(a, outputShape);
  const bBroadcast = broadcastTo(b, outputShape);

  // Create result array with bool dtype
  const result = ArrayStorage.empty(outputShape, 'bool');
  const resultData = result.data as Uint8Array;
  const size = result.size;

  // Check if we need to convert BigInt to Number for comparison
  const needsConversion = isBigIntDType(a.dtype) || isBigIntDType(b.dtype);

  // Perform element-wise comparison.
  //
  // When both sides are BigInt, compare them directly: routing through
  // Number() loses precision above 2^53 and reports distinct int64 values as
  // equal. Mixed BigInt/float still converts, matching NumPy's promotion to
  // float64 for those combinations.
  for (let i = 0; i < size; i++) {
    const aRaw = aBroadcast.iget(i);
    const bRaw = bBroadcast.iget(i);

    if (needsConversion && kind && typeof aRaw === 'bigint' && typeof bRaw === 'bigint') {
      resultData[i] = compareBigInt(aRaw, bRaw, kind) ? 1 : 0;
      continue;
    }

    resultData[i] = op(Number(aRaw), Number(bRaw)) ? 1 : 0;
  }

  return result;
}

/**
 * Perform element-wise unary operation
 *
 * @param a - Input array storage
 * @param op - Operation to perform (x) => result
 * @param preserveDtype - If true, preserve input dtype; if false, promote to float64 (default: true)
 * @returns Result storage
 */
export function elementwiseUnaryOp(
  a: ArrayStorage,
  op: (x: number) => number,
  preserveDtype = true,
): ArrayStorage {
  const dtype = a.dtype;
  const shape = Array.from(a.shape);
  const size = a.size;

  // Determine output dtype
  // Math operations like sqrt use NumPy's type promotion (int8→float16, int16→float32, etc.)
  const resultDtype = preserveDtype ? dtype : mathResultDtype(dtype);

  // Create result storage
  const result = ArrayStorage.empty(shape, resultDtype);
  const resultData = result.data;
  const inputData = a.data;
  const off = a.offset;

  const contiguous = a.isCContiguous;

  if (isBigIntDType(dtype)) {
    // BigInt input - convert to Number for operation, then convert back if preserving dtype
    if (isBigIntDType(resultDtype)) {
      const resultTyped = resultData as BigInt64Array | BigUint64Array;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          const val = Number(inputData[off + i]!);
          resultTyped[i] = BigInt(Math.round(op(val)));
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultTyped[i] = BigInt(Math.round(op(Number(a.iget(i)))));
        }
      }
    } else {
      // BigInt input, float output
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          resultData[i] = op(Number(inputData[off + i]!));
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = op(Number(a.iget(i)));
        }
      }
    }
  } else {
    // Regular numeric types
    if (contiguous) {
      if (off === 0) {
        for (let i = 0; i < size; i++) {
          resultData[i] = op(Number(inputData[i]!));
        }
      } else {
        for (let i = 0; i < size; i++) {
          resultData[i] = op(Number(inputData[off + i]!));
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        resultData[i] = op(Number(a.iget(i)));
      }
    }
  }

  return result;
}
