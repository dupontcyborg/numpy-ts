/**
 * Reduction operations (sum, mean, max, min)
 *
 * Pure functions for reducing arrays along axes.
 * @module ops/reduction
 */

import { ArrayStorage } from '../storage';
import {
  hasFloat16,
  isBigIntDType,
  isComplexDType,
  isFloatDType,
  throwIfComplex,
  reductionAccumDtype,
  type DType,
} from '../dtype';
import { computeStrides, precomputeAxisOffsets } from '../internal/indexing';
import { Complex } from '../complex';
import {
  wasmReduceSum,
  wasmReduceSumStrided,
  wasmReduceSumStridedComplex,
  wasmReduceSumComplex,
} from '../wasm/reduce_sum';
import { wasmReduceMax, wasmReduceMaxStrided } from '../wasm/reduce_max';
import { wasmReduceMin, wasmReduceMinStrided } from '../wasm/reduce_min';
import { wasmReduceArgmax, wasmReduceArgmaxStrided } from '../wasm/reduce_argmax';
import { wasmReduceArgmin, wasmReduceArgminStrided } from '../wasm/reduce_argmin';
import { wasmReduceMean, wasmReduceMeanStrided } from '../wasm/reduce_mean';
import { wasmReduceVar } from '../wasm/reduce_var';
import { wasmReduceNansum } from '../wasm/reduce_nansum';
import { wasmReduceNanmin } from '../wasm/reduce_nanmin';
import { wasmReduceNanmax } from '../wasm/reduce_nanmax';
import {
  wasmReduceProd,
  wasmReduceProdStrided,
  wasmReduceProdStridedComplex,
} from '../wasm/reduce_prod';
import { wasmReduceQuantile, wasmReduceQuantileStrided } from '../wasm/reduce_quantile';
import { wasmReduceAny } from '../wasm/reduce_any';
import { wasmReduceAll } from '../wasm/reduce_all';
import { wasmReduceStd } from '../wasm/reduce_std';

// Reusable Float32Array accumulators — writing to f32acc[0] implicitly rounds
// to float32 precision (same as Math.fround) but V8 optimizes typed-array
// stores better than function calls in hot loops.
const f32acc = new Float32Array(2); // [0]=primary, [1]=secondary (for complex im)

// Float16Array accumulator for matching NumPy's float16 reduction precision.
// When native Float16Array is available, accumulating via f16acc[0] rounds to
// float16 at each step — matching NumPy's overflow/precision behavior exactly.
const f16acc: Float16Array | null =
  typeof globalThis.Float16Array !== 'undefined' ? new Float16Array(2) : null;

/**
 * Get the appropriate typed accumulator for a dtype.
 * Returns f16acc for float16 (native only), f32acc for float32, or null for float64/others.
 * Writing to acc[0] implicitly rounds to the dtype's precision at each step.
 */
function getFloatAcc(dtype: DType): Float16Array | Float32Array | null {
  if (dtype === 'float16') return f16acc;
  if (dtype === 'float32') return f32acc;
  return null;
}

/**
 * Round a scalar value to the precision of the given dtype.
 * For float16 (native) and float32, writes to a TypedArray element and reads back.
 * For float64 and fallback float16, returns the value unchanged.
 */
function roundToDtype(value: number, dtype: DType): number {
  const acc = getFloatAcc(dtype);
  if (acc) {
    acc[0] = value;
    return acc[0]!;
  }
  return value;
}

/**
 * Promote narrow integer dtypes for accumulation, matching NumPy behavior.
 * NumPy promotes int8/int16/int32 → int64, uint8/uint16/uint32 → uint64.
 * Since JS doesn't have int64 typed arrays without BigInt overhead,
 * we promote to float64 which has 53 bits of integer precision — sufficient
 * for all practical accumulation sizes.
 */
/** For sum/prod: promote narrow ints to int64 (matches NumPy) */
function intAccumulationDtype(dtype: DType): DType {
  switch (dtype) {
    case 'bool':
    case 'int8':
    case 'int16':
    case 'int32':
      return 'int64';
    case 'uint8':
    case 'uint16':
    case 'uint32':
      return 'uint64';
    default:
      return dtype;
  }
}

/** For mean/std/var: promote all ints to float64 (matches NumPy) */
function floatAccumulationDtype(dtype: DType): DType {
  switch (dtype) {
    case 'bool':
    case 'int8':
    case 'int16':
    case 'int32':
    case 'uint8':
    case 'uint16':
    case 'uint32':
    case 'int64':
    case 'uint64':
      return 'float64';
    default:
      return dtype;
  }
}

function wrapScalarKeepdims(
  scalar: number | bigint | Complex,
  ndim: number,
  dtype: DType
): ArrayStorage {
  const out = ArrayStorage.zeros(Array(ndim).fill(1) as number[], dtype);
  out.iset(0, scalar as number | bigint | Complex);
  return out;
}

/**
 * Sum array elements over a given axis
 */
export function sum(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;

  const off = storage.offset;
  const inputStrides = storage.strides;

  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // WASM fast path for full-array sum (non-complex)
    const wasmSum = wasmReduceSum(storage);
    if (wasmSum !== null) return wasmSum;

    // Sum all elements - return scalar (or Complex for complex arrays)
    if (isComplexDType(dtype)) {
      // WASM fast path for complex global sum
      const wasmCSum = wasmReduceSumComplex(storage);
      if (wasmCSum !== null) return new Complex(wasmCSum[0], wasmCSum[1]);

      let totalRe = 0;
      let totalIm = 0;
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          totalRe += complexData[(off + i) * 2]!;
          totalIm += complexData[(off + i) * 2 + 1]!;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i);
          totalRe += (val as Complex).re;
          totalIm += (val as Complex).im;
        }
      }
      return new Complex(totalRe, totalIm);
    } else if (isBigIntDType(dtype)) {
      let total = BigInt(0);
      if (contiguous) {
        const typedData = data as BigInt64Array | BigUint64Array;
        for (let i = 0; i < size; i++) {
          total += typedData[off + i]!;
        }
      } else {
        for (let i = 0; i < size; i++) {
          total += storage.iget(i) as bigint;
        }
      }
      return Number(total);
    } else if (dtype === 'float32' || (dtype === 'float16' && f16acc)) {
      const acc = getFloatAcc(dtype)!;
      acc[0] = 0;
      if (contiguous) {
        // Bulk-convert Float16Array to Float32Array to avoid per-element f16 overhead
        const f32 =
          dtype === 'float16'
            ? new Float32Array((data as Float16Array).subarray(off, off + size))
            : data;
        const f32off = dtype === 'float16' ? 0 : off;
        for (let i = 0; i < size; i++) {
          acc[0] += Number(f32[f32off + i]!);
        }
      } else {
        for (let i = 0; i < size; i++) {
          acc[0] += Number(storage.iget(i));
        }
      }
      return acc[0]!;
    } else {
      let total = 0;
      if (contiguous) {
        if (off === 0) {
          for (let i = 0; i < size; i++) {
            total += Number(data[i]!);
          }
        } else {
          for (let i = 0; i < size; i++) {
            total += Number(data[off + i]!);
          }
        }
      } else {
        for (let i = 0; i < size; i++) {
          total += Number(storage.iget(i));
        }
      }
      return total;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Promote narrow int dtypes for accumulation (matching NumPy)
  const outDtype = intAccumulationDtype(dtype);

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = sum(storage);
    if (!keepdims) return scalar;
    const out = ArrayStorage.zeros(Array(ndim).fill(1), outDtype);
    out.iset(0, scalar as number | bigint | Complex);
    return out;
  }

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  // WASM strided fast path: accumulate in f64, convert to output dtype after
  if (contiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceSumStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outShape = keepdims ? shape.map((s, i) => (i === normalizedAxis ? 1 : s)) : outputShape;
      if (outDtype === 'float64') {
        // Share the WASM region directly — no copy needed
        const strides: number[] = new Array(outShape.length);
        let s = 1;
        for (let i = outShape.length - 1; i >= 0; i--) {
          strides[i] = s;
          s *= outShape[i]!;
        }
        const shared = ArrayStorage.fromDataShared(
          wasmResult.data,
          outShape,
          outDtype,
          strides,
          0,
          wasmResult.wasmRegion
        );
        wasmResult.dispose();
        return shared;
      }
      try {
        if (outDtype === 'float32') {
          const f32Result = ArrayStorage.empty(outShape, outDtype);
          const f32 = f32Result.data as Float32Array;
          const f64 = wasmResult.data as Float64Array;
          for (let i = 0; i < f64.length; i++) f32[i] = f64[i]!;
          return f32Result;
        }
        // int64/uint64 output — need a result buffer for conversion
        const intResult = ArrayStorage.zeros(outputShape, outDtype);
        const intResultData = intResult.data;
        const f64Data = wasmResult.data as Float64Array;
        for (let i = 0; i < f64Data.length; i++)
          intResultData[i] =
            intResultData instanceof BigInt64Array || intResultData instanceof BigUint64Array
              ? (BigInt(Math.round(f64Data[i]!)) as bigint)
              : f64Data[i]!;
        return keepdims ? ArrayStorage.fromData(intResultData, outShape, outDtype) : intResult;
      } finally {
        wasmResult.dispose();
      }
    }
  }

  // Create result storage with promoted dtype
  const result = ArrayStorage.zeros(outputShape, outDtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplexDType(dtype)) {
    // WASM fast path for complex sum along axis
    if (contiguous) {
      const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
      const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
      const wasmResult = wasmReduceSumStridedComplex(storage, wasmOuter, axisSize, innerSize);
      if (wasmResult) {
        const outShape = keepdims
          ? shape.map((s, i) => (i === normalizedAxis ? 1 : s))
          : outputShape;
        const shared = ArrayStorage.fromDataShared(
          wasmResult.data,
          outShape,
          dtype,
          computeStrides(outShape),
          0,
          wasmResult.wasmRegion
        );
        wasmResult.dispose();
        result.dispose(); // free the pre-allocated JS fallback result
        return shared;
      }
    }

    // Complex sum along axis — JS fallback
    const complexData = data as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;

    if (dtype === 'complex64') {
      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        f32acc[0] = 0;
        f32acc[1] = 0;
        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          f32acc[0] += complexData[bufIdx * 2]!;
          f32acc[1] += complexData[bufIdx * 2 + 1]!;
          bufIdx += axisStr;
        }
        resultComplex[outerIdx * 2] = f32acc[0]!;
        resultComplex[outerIdx * 2 + 1] = f32acc[1]!;
      }
    } else {
      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        let sumRe = 0;
        let sumIm = 0;
        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          sumRe += complexData[bufIdx * 2]!;
          sumIm += complexData[bufIdx * 2 + 1]!;
          bufIdx += axisStr;
        }
        resultComplex[outerIdx * 2] = sumRe;
        resultComplex[outerIdx * 2 + 1] = sumIm;
      }
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = BigInt(0);
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        sumVal += typedData[bufIdx]!;
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = sumVal;
    }
  } else if (isBigIntDType(outDtype)) {
    // Input is narrow int (int8/16/32, uint8/16/32) promoted to int64/uint64.
    // Accumulate as Number (53-bit precision, sufficient for practical axis sizes)
    // then convert to BigInt once — avoids per-element BigInt(Number()) overhead.
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        sumVal += Number(data[bufIdx]!);
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = BigInt(Math.round(sumVal));
    }
  } else if (dtype === 'float32' || (dtype === 'float16' && f16acc)) {
    const acc = getFloatAcc(dtype)!;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      acc[0] = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        acc[0] += Number(data[bufIdx]!);
        bufIdx += axisStr;
      }
      resultData[outerIdx] = acc[0]!;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumVal = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        sumVal += Number(data[bufIdx]!);
        bufIdx += axisStr;
      }
      resultData[outerIdx] = sumVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, outDtype);
  }

  return result;
}

/**
 * Compute the arithmetic mean along the specified axis
 * Note: mean() returns float64 for integer dtypes, matching NumPy behavior
 * For complex arrays, returns complex mean
 */
export function mean(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;

  if (axis === undefined) {
    // WASM fast path for full-array mean (non-complex)
    const wasmResult = wasmReduceMean(storage);
    if (wasmResult !== null) return wasmResult;

    // NumPy computes mean in higher precision to avoid overflow, then casts back.
    // For float16/float32: accumulate in float64, divide, then round to input dtype.
    if (dtype === 'float16' || dtype === 'float32') {
      let total = 0;
      const off2 = storage.offset;
      if (storage.isCContiguous) {
        for (let i = 0; i < storage.size; i++) {
          total += Number(storage.data[off2 + i]);
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          total += Number(storage.iget(i));
        }
      }
      return roundToDtype(total / storage.size, dtype);
    }

    const sumResult = sum(storage);
    if (sumResult instanceof Complex) {
      return new Complex(sumResult.re / storage.size, sumResult.im / storage.size);
    }
    return (sumResult as number) / storage.size;
  }

  // Normalize negative axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = shape.length + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${shape.length}`);
  }

  // WASM strided fast path for contiguous non-complex mean
  if (storage.isCContiguous && !isComplexDType(dtype)) {
    const axisSize = shape[normalizedAxis]!;
    const outputShape = keepdims
      ? shape.map((s, i) => (i === normalizedAxis ? 1 : s))
      : Array.from(shape).filter((_, i) => i !== normalizedAxis);
    const outerSize = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceMeanStrided(storage, outerSize, axisSize, innerSize);
    if (wasmResult) {
      // NumPy mean: float16→float16, float32→float32, int→float64
      const outDtype = isFloatDType(dtype) ? dtype : 'float64';
      if (outDtype === 'float64') {
        const strides: number[] = new Array(outputShape.length);
        let s = 1;
        for (let i = outputShape.length - 1; i >= 0; i--) {
          strides[i] = s;
          s *= outputShape[i]!;
        }
        const shared = ArrayStorage.fromDataShared(
          wasmResult.data,
          outputShape,
          'float64',
          strides,
          0,
          wasmResult.wasmRegion
        );
        wasmResult.dispose();
        return shared;
      }
      // float16/float32: cast f64 output back to input dtype via .set()
      const outStorage = ArrayStorage.empty(outputShape, outDtype);
      (outStorage.data as Float32Array | Float16Array).set(wasmResult.data as Float64Array);
      wasmResult.dispose();
      return outStorage;
    }
  }

  // For float16/float32: compute sum in float64 to avoid overflow, then round.
  // NumPy's mean uses higher-precision accumulation internally.
  let sumStorage: ArrayStorage;
  if (dtype === 'float16' || dtype === 'float32') {
    // Convert to float64, sum, divide, convert back
    const f64Storage = ArrayStorage.zeros(Array.from(shape), 'float64');
    const srcData = storage.data;
    const dstData = f64Storage.data as Float64Array;
    const off2 = storage.offset;
    if (storage.isCContiguous) {
      const src =
        dtype === 'float16' && hasFloat16
          ? new Float32Array((srcData as Float16Array).subarray(off2, off2 + storage.size))
          : srcData;
      const srcOff = dtype === 'float16' && hasFloat16 ? 0 : off2;
      for (let i = 0; i < storage.size; i++) dstData[i] = Number(src[srcOff + i]);
    } else {
      for (let i = 0; i < storage.size; i++) dstData[i] = Number(storage.iget(i));
    }
    const f64Sum = sum(f64Storage, normalizedAxis, keepdims);
    f64Storage.dispose();
    if (typeof f64Sum === 'number') {
      return roundToDtype(f64Sum / shape[normalizedAxis]!, dtype);
    }
    sumStorage = f64Sum as ArrayStorage;
    try {
      const divisor = shape[normalizedAxis]!;
      const resultData = sumStorage.data as Float64Array;
      const outResult = ArrayStorage.empty(Array.from(sumStorage.shape), dtype);
      const outData = outResult.data;
      for (let i = 0; i < resultData.length; i++) {
        (outData as Float64Array)[i] = resultData[i]! / divisor;
      }
      return outResult;
    } finally {
      sumStorage.dispose();
    }
  }

  const sumResult = sum(storage, axis, keepdims);
  if (typeof sumResult === 'number') {
    return sumResult / shape[normalizedAxis]!;
  }
  if (sumResult instanceof Complex) {
    return new Complex(
      sumResult.re / shape[normalizedAxis]!,
      sumResult.im / shape[normalizedAxis]!
    );
  }

  try {
    // Divide by the size of the reduced axis
    const divisor = shape[normalizedAxis]!;

    // For complex dtypes, mean stays complex
    // For integer dtypes, mean returns float64 (matching NumPy behavior)
    const resultDtype = floatAccumulationDtype(dtype);

    const result = ArrayStorage.zeros(Array.from(sumResult.shape), resultDtype);
    const resultData = result.data;
    const sumData = sumResult.data;
    const sumDtype = sumResult.dtype as DType;

    if (isComplexDType(dtype)) {
      // Complex: divide both real and imaginary parts
      const sumComplex = sumData as Float64Array | Float32Array;
      const resultComplex = resultData as Float64Array | Float32Array;
      const size = sumResult.size;
      for (let i = 0; i < size; i++) {
        resultComplex[i * 2] = sumComplex[i * 2]! / divisor;
        resultComplex[i * 2 + 1] = sumComplex[i * 2 + 1]! / divisor;
      }
    } else if (isBigIntDType(sumDtype)) {
      // Convert BigInt sum results to float for mean
      const sumTyped = sumData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < resultData.length; i++) {
        resultData[i] = Number(sumTyped[i]!) / divisor;
      }
    } else {
      for (let i = 0; i < resultData.length; i++) {
        resultData[i] = Number(sumData[i]!) / divisor;
      }
    }

    return result;
  } finally {
    sumResult.dispose();
  }
}

/**
 * Return the maximum along a given axis
 */
export function max(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Complex max uses lexicographic ordering (real first, then imaginary)
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      if (size === 0) {
        throw new Error('max of empty array');
      }

      let maxRe = complexData[off * 2]!;
      let maxIm = complexData[off * 2 + 1]!;

      for (let i = 1; i < size; i++) {
        const re = complexData[(off + i) * 2]!;
        const im = complexData[(off + i) * 2 + 1]!;

        // Check for NaN propagation
        if (isNaN(re) || isNaN(im)) {
          return new Complex(NaN, NaN);
        }

        // Lexicographic comparison: real first, then imaginary
        if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      // Check initial value for NaN
      if (isNaN(maxRe) || isNaN(maxIm)) {
        return new Complex(NaN, NaN);
      }
      return new Complex(maxRe, maxIm);
    }

    // Max along axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = max(storage) as Complex;
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar, ndim, dtype);
    }

    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let maxRe = complexData[bufIdx * 2]!;
      let maxIm = complexData[bufIdx * 2 + 1]!;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        bufIdx += axisStr;

        if (isNaN(re) || isNaN(im)) {
          maxRe = NaN;
          maxIm = NaN;
          break;
        }
        if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      resultData[outerIdx * 2] = maxRe;
      resultData[outerIdx * 2 + 1] = maxIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  if (axis === undefined) {
    // WASM fast path for full-array max (non-complex)
    const wasmResult = wasmReduceMax(storage);
    if (wasmResult !== null) return wasmResult;

    // Max of all elements - return scalar
    if (size === 0) {
      throw new Error('max of empty array');
    }

    if (storage.isCContiguous) {
      let maxVal = data[off]!;
      if (off === 0) {
        for (let i = 1; i < size; i++) {
          if (data[i]! > maxVal) {
            maxVal = data[i]!;
          }
        }
      } else {
        for (let i = 1; i < size; i++) {
          if (data[off + i]! > maxVal) {
            maxVal = data[off + i]!;
          }
        }
      }
      return Number(maxVal);
    } else {
      let maxVal = storage.iget(0);
      for (let i = 1; i < size; i++) {
        const val = storage.iget(i);
        if (val > maxVal) {
          maxVal = val;
        }
      }
      return Number(maxVal);
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = max(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | bigint, ndim, dtype);
  }

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  // WASM strided fast path for max (output dtype matches input dtype)
  if (storage.isCContiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceMaxStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outShape = keepdims ? shape.map((s, i) => (i === normalizedAxis ? 1 : s)) : outputShape;
      const strides: number[] = new Array(outShape.length);
      let s = 1;
      for (let i = outShape.length - 1; i >= 0; i--) {
        strides[i] = s;
        s *= outShape[i]!;
      }
      const shared = ArrayStorage.fromDataShared(
        wasmResult.data,
        outShape,
        dtype,
        strides,
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return shared;
    }
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let maxVal = typedData[bufIdx]!;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const val = typedData[bufIdx]!;
        if (val > maxVal) {
          maxVal = val;
        }
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = maxVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const val = Number(data[bufIdx]!);
        if (val > maxVal) {
          maxVal = val;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = maxVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Product array elements over a given axis
 */
export function prod(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // WASM fast path for full-array product (non-complex)
    const wasmProd = wasmReduceProd(storage);
    if (wasmProd !== null) return wasmProd;

    // Product of all elements - return scalar (or Complex for complex arrays)
    if (isComplexDType(dtype)) {
      let prodRe = 1;
      let prodIm = 0;
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          const re = val.re;
          const im = val.im;
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
        }
      }
      return new Complex(prodRe, prodIm);
    } else if (isBigIntDType(dtype)) {
      let product = BigInt(1);
      if (contiguous) {
        const typedData = data as BigInt64Array | BigUint64Array;
        for (let i = 0; i < size; i++) {
          product *= typedData[off + i]!;
        }
      } else {
        for (let i = 0; i < size; i++) {
          product *= storage.iget(i) as bigint;
        }
      }
      return Number(product);
    } else if (dtype === 'float32' || (dtype === 'float16' && f16acc)) {
      const acc = getFloatAcc(dtype)!;
      acc[0] = 1;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          acc[0] *= Number(data[off + i]!);
        }
      } else {
        for (let i = 0; i < size; i++) {
          acc[0] *= Number(storage.iget(i));
        }
      }
      return acc[0]!;
    } else {
      let product = 1;
      if (contiguous) {
        if (off === 0) {
          for (let i = 0; i < size; i++) {
            product *= Number(data[i]!);
          }
        } else {
          for (let i = 0; i < size; i++) {
            product *= Number(data[off + i]!);
          }
        }
      } else {
        for (let i = 0; i < size; i++) {
          product *= Number(storage.iget(i));
        }
      }
      return product;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Promote narrow int dtypes for accumulation (matching NumPy)
  const outDtype = intAccumulationDtype(dtype);

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = prod(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | bigint | Complex, ndim, outDtype);
  }

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  // WASM strided fast path for prod
  if (contiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceProdStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outShape = keepdims ? shape.map((s, i) => (i === normalizedAxis ? 1 : s)) : outputShape;
      const strides: number[] = new Array(outShape.length);
      let s = 1;
      for (let i = outShape.length - 1; i >= 0; i--) {
        strides[i] = s;
        s *= outShape[i]!;
      }
      const shared = ArrayStorage.fromDataShared(
        wasmResult.data,
        outShape,
        outDtype,
        strides,
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return shared;
    }
  }

  // Create result storage with promoted dtype
  const result = ArrayStorage.zeros(outputShape, outDtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplexDType(dtype)) {
    // WASM fast path for complex prod along axis
    if (contiguous) {
      const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
      const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
      const wasmResult = wasmReduceProdStridedComplex(storage, wasmOuter, axisSize, innerSize);
      if (wasmResult) {
        const outShape = keepdims
          ? shape.map((s, i) => (i === normalizedAxis ? 1 : s))
          : outputShape;
        const shared = ArrayStorage.fromDataShared(
          wasmResult.data,
          outShape,
          dtype,
          computeStrides(outShape),
          0,
          wasmResult.wasmRegion
        );
        wasmResult.dispose();
        result.dispose(); // free the pre-allocated JS fallback result
        return shared;
      }
    }

    // Complex product along axis — JS fallback
    const complexData = data as Float64Array | Float32Array;
    const resultComplex = resultData as Float64Array | Float32Array;

    if (dtype === 'complex64') {
      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        f32acc[0] = 1; // re
        f32acc[1] = 0; // im
        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          const re = complexData[bufIdx * 2]!;
          const im = complexData[bufIdx * 2 + 1]!;
          const prevRe: number = f32acc[0]!;
          const prevIm: number = f32acc[1]!;
          f32acc[0] = prevRe * re - prevIm * im;
          f32acc[1] = prevRe * im + prevIm * re;
          bufIdx += axisStr;
        }
        resultComplex[outerIdx * 2] = f32acc[0]!;
        resultComplex[outerIdx * 2 + 1] = f32acc[1]!;
      }
    } else {
      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        let prodRe = 1;
        let prodIm = 0;
        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          const re = complexData[bufIdx * 2]!;
          const im = complexData[bufIdx * 2 + 1]!;
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
          bufIdx += axisStr;
        }
        resultComplex[outerIdx * 2] = prodRe;
        resultComplex[outerIdx * 2 + 1] = prodIm;
      }
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = BigInt(1);
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        prodVal *= typedData[bufIdx]!;
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = prodVal;
    }
  } else if (isBigIntDType(outDtype)) {
    // Input is narrow int promoted to int64/uint64
    const resultTyped = resultData as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = BigInt(1);
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        prodVal *= BigInt(Number(data[bufIdx]!));
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = prodVal;
    }
  } else if (dtype === 'float32' || (dtype === 'float16' && f16acc)) {
    const acc = getFloatAcc(dtype)!;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      acc[0] = 1;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        acc[0] *= Number(data[bufIdx]!);
        bufIdx += axisStr;
      }
      resultData[outerIdx] = acc[0]!;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let prodVal = 1;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        prodVal *= Number(data[bufIdx]!);
        bufIdx += axisStr;
      }
      resultData[outerIdx] = prodVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, outDtype);
  }

  return result;
}

/**
 * Return the minimum along a given axis
 */
export function min(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Complex min uses lexicographic ordering (real first, then imaginary)
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      if (size === 0) {
        throw new Error('min of empty array');
      }

      let minRe = complexData[off * 2]!;
      let minIm = complexData[off * 2 + 1]!;

      for (let i = 1; i < size; i++) {
        const re = complexData[(off + i) * 2]!;
        const im = complexData[(off + i) * 2 + 1]!;

        // Check for NaN propagation
        if (isNaN(re) || isNaN(im)) {
          return new Complex(NaN, NaN);
        }

        // Lexicographic comparison: real first, then imaginary
        if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      // Check initial value for NaN
      if (isNaN(minRe) || isNaN(minIm)) {
        return new Complex(NaN, NaN);
      }
      return new Complex(minRe, minIm);
    }

    // Min along axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = min(storage) as Complex;
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar, ndim, dtype);
    }

    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;
    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let minRe = complexData[bufIdx * 2]!;
      let minIm = complexData[bufIdx * 2 + 1]!;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        bufIdx += axisStr;

        if (isNaN(re) || isNaN(im)) {
          minRe = NaN;
          minIm = NaN;
          break;
        }
        if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      resultData[outerIdx * 2] = minRe;
      resultData[outerIdx * 2 + 1] = minIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  if (axis === undefined) {
    // WASM fast path for full-array min (non-complex)
    const wasmResult = wasmReduceMin(storage);
    if (wasmResult !== null) return wasmResult;

    // Min of all elements - return scalar
    if (size === 0) {
      throw new Error('min of empty array');
    }

    if (storage.isCContiguous) {
      let minVal = data[off]!;
      if (off === 0) {
        for (let i = 1; i < size; i++) {
          if (data[i]! < minVal) {
            minVal = data[i]!;
          }
        }
      } else {
        for (let i = 1; i < size; i++) {
          if (data[off + i]! < minVal) {
            minVal = data[off + i]!;
          }
        }
      }
      return Number(minVal);
    } else {
      let minVal = storage.iget(0);
      for (let i = 1; i < size; i++) {
        const val = storage.iget(i);
        if (val < minVal) {
          minVal = val;
        }
      }
      return Number(minVal);
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = min(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | bigint, ndim, dtype);
  }

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  // WASM strided fast path for min (output dtype matches input dtype)
  if (storage.isCContiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceMinStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outShape = keepdims ? shape.map((s, i) => (i === normalizedAxis ? 1 : s)) : outputShape;
      const strides: number[] = new Array(outShape.length);
      let s = 1;
      for (let i = outShape.length - 1; i >= 0; i--) {
        strides[i] = s;
        s *= outShape[i]!;
      }
      const shared = ArrayStorage.fromDataShared(
        wasmResult.data,
        outShape,
        dtype,
        strides,
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return shared;
    }
  }

  // Create result storage
  const result = ArrayStorage.zeros(outputShape, dtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;
    const resultTyped = resultData as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let minVal = typedData[bufIdx]!;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const val = typedData[bufIdx]!;
        if (val < minVal) {
          minVal = val;
        }
        bufIdx += axisStr;
      }
      resultTyped[outerIdx] = minVal;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const val = Number(data[bufIdx]!);
        if (val < minVal) {
          minVal = val;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = minVal;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
  }

  return result;
}

/**
 * Return the indices of the minimum values along a given axis
 */
/**
 * Compare two complex numbers using lexicographic ordering (real first, then imaginary)
 * Returns -1 if a < b, 0 if a == b, 1 if a > b
 */
function complexCompare(aRe: number, aIm: number, bRe: number, bIm: number): number {
  if (aRe < bRe) return -1;
  if (aRe > bRe) return 1;
  // Real parts are equal, compare imaginary parts
  if (aIm < bIm) return -1;
  if (aIm > bIm) return 1;
  return 0;
}

export function argmin(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // WASM fast path for full-array argmin (non-complex)
    const wasmResult = wasmReduceArgmin(storage);
    if (wasmResult !== null) return wasmResult;

    if (size === 0) {
      throw new Error('argmin of empty array');
    }

    if (isComplex) {
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        let minRe = complexData[off * 2]!;
        let minIm = complexData[off * 2 + 1]!;
        let minIdx = 0;
        for (let i = 1; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (complexCompare(re, im, minRe, minIm) < 0) {
            minRe = re;
            minIm = im;
            minIdx = i;
          }
        }
        return minIdx;
      } else {
        const first = storage.iget(0) as Complex;
        let minRe = first.re;
        let minIm = first.im;
        let minIdx = 0;
        for (let i = 1; i < size; i++) {
          const val = storage.iget(i) as Complex;
          if (complexCompare(val.re, val.im, minRe, minIm) < 0) {
            minRe = val.re;
            minIm = val.im;
            minIdx = i;
          }
        }
        return minIdx;
      }
    }

    if (contiguous) {
      let minVal = data[off]!;
      let minIdx = 0;
      if (off === 0) {
        for (let i = 1; i < size; i++) {
          if (data[i]! < minVal) {
            minVal = data[i]!;
            minIdx = i;
          }
        }
      } else {
        for (let i = 1; i < size; i++) {
          if (data[off + i]! < minVal) {
            minVal = data[off + i]!;
            minIdx = i;
          }
        }
      }
      return minIdx;
    } else {
      let minVal = storage.iget(0);
      let minIdx = 0;
      for (let i = 1; i < size; i++) {
        const val = storage.iget(i);
        if (val < minVal) {
          minVal = val;
          minIdx = i;
        }
      }
      return minIdx;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmin(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const axisSize = shape[normalizedAxis]!;

  // WASM strided fast path for argmin (output is always int32)
  if (storage.isCContiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceArgminStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outStrides: number[] = new Array(outputShape.length);
      let s = 1;
      for (let i = outputShape.length - 1; i >= 0; i--) {
        outStrides[i] = s;
        s *= outputShape[i]!;
      }
      const shared = ArrayStorage.fromDataShared(
        wasmResult.data,
        outputShape,
        'int32',
        outStrides,
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return shared;
    }
  }

  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let minRe = complexData[bufIdx * 2]!;
      let minIm = complexData[bufIdx * 2 + 1]!;
      let minAxisIdx = 0;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = minAxisIdx;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let minVal = typedData[bufIdx]!;
      let minAxisIdx = 0;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const val = typedData[bufIdx]!;
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = minAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minVal = Infinity;
      let minAxisIdx = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const val = Number(data[bufIdx]!);
        if (val < minVal) {
          minVal = val;
          minAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = minAxisIdx;
    }
  }

  return result;
}

/**
 * Return the indices of the maximum values along a given axis
 */
export function argmax(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // WASM fast path for full-array argmax (non-complex)
    const wasmResult = wasmReduceArgmax(storage);
    if (wasmResult !== null) return wasmResult;

    if (size === 0) {
      throw new Error('argmax of empty array');
    }

    if (isComplex) {
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        let maxRe = complexData[off * 2]!;
        let maxIm = complexData[off * 2 + 1]!;
        let maxIdx = 0;
        for (let i = 1; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (complexCompare(re, im, maxRe, maxIm) > 0) {
            maxRe = re;
            maxIm = im;
            maxIdx = i;
          }
        }
        return maxIdx;
      } else {
        const first = storage.iget(0) as Complex;
        let maxRe = first.re;
        let maxIm = first.im;
        let maxIdx = 0;
        for (let i = 1; i < size; i++) {
          const val = storage.iget(i) as Complex;
          if (complexCompare(val.re, val.im, maxRe, maxIm) > 0) {
            maxRe = val.re;
            maxIm = val.im;
            maxIdx = i;
          }
        }
        return maxIdx;
      }
    }

    if (contiguous) {
      let maxVal = data[off]!;
      let maxIdx = 0;
      for (let i = 1; i < size; i++) {
        if (data[off + i]! > maxVal) {
          maxVal = data[off + i]!;
          maxIdx = i;
        }
      }
      return maxIdx;
    } else {
      let maxVal = storage.iget(0);
      let maxIdx = 0;
      for (let i = 1; i < size; i++) {
        const val = storage.iget(i);
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
      return maxIdx;
    }
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // Result is scalar
    return argmax(storage);
  }

  // Create result storage with int32 dtype (indices are always integers)
  const axisSize = shape[normalizedAxis]!;

  // WASM strided fast path for argmax (output is always int32)
  if (storage.isCContiguous && !isComplexDType(dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceArgmaxStrided(storage, wasmOuter, axisSize, innerSize);
    if (wasmResult) {
      const outStrides: number[] = new Array(outputShape.length);
      let s = 1;
      for (let i = outputShape.length - 1; i >= 0; i--) {
        outStrides[i] = s;
        s *= outputShape[i]!;
      }
      const shared = ArrayStorage.fromDataShared(
        wasmResult.data,
        outputShape,
        'int32',
        outStrides,
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return shared;
    }
  }

  const result = ArrayStorage.zeros(outputShape, 'int32');
  const resultData = result.data;

  // Perform reduction along axis
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let maxRe = complexData[bufIdx * 2]!;
      let maxIm = complexData[bufIdx * 2 + 1]!;
      let maxAxisIdx = 0;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  } else if (isBigIntDType(dtype)) {
    const typedData = data as BigInt64Array | BigUint64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let bufIdx = baseOffsets[outerIdx]!;
      let maxVal = typedData[bufIdx]!;
      let maxAxisIdx = 0;
      bufIdx += axisStr;

      for (let axisIdx = 1; axisIdx < axisSize; axisIdx++) {
        const val = typedData[bufIdx]!;
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxVal = -Infinity;
      let maxAxisIdx = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const val = Number(data[bufIdx]!);
        if (val > maxVal) {
          maxVal = val;
          maxAxisIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = maxAxisIdx;
    }
  }

  return result;
}

/**
 * Compute the variance along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute variance
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 *
 * For complex arrays: Var(X) = E[|X - E[X]|²] where |z|² = re² + im²
 * Returns real values (float64) for both real and complex input
 */
export function variance(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Compute mean for var.
  // NumPy's var does NOT promote f16→f32 for its internal mean (unlike standalone mean()),
  // so f16 sum can overflow → inf mean → inf var. We must match this behavior.
  let meanResult: ArrayStorage | number | Complex;
  // Compute mean. For f16 variance, NumPy's var does NOT promote its internal mean
  // to f32 (unlike standalone mean()). NumPy computes sum(x)/N where sum overflows in f16.
  // We compute the f64 sum, cast to f16 (which overflows like NumPy), then divide.
  if (dtype === 'float16' && axis === undefined && f16acc) {
    const sumResult = sum(storage);
    f16acc[0] = sumResult as number; // cast sum to f16 (overflow → inf)
    f16acc[0] /= size; // inf/N = inf, matching NumPy
    meanResult = f16acc[0]!;
  } else {
    meanResult = mean(storage, axis, keepdims);
  }

  const contiguous = storage.isCContiguous;

  if (axis === undefined) {
    // WASM fast path for full-array variance (non-complex)
    const wasmResult = wasmReduceVar(storage);
    if (wasmResult !== null) return wasmResult;

    // Variance of all elements - return scalar
    if (isComplexDType(dtype)) {
      const meanComplex = meanResult as Complex;
      let sumSqDiff = 0;

      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          const diffRe = re - meanComplex.re;
          const diffIm = im - meanComplex.im;
          sumSqDiff += diffRe * diffRe + diffIm * diffIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          const diffRe = val.re - meanComplex.re;
          const diffIm = val.im - meanComplex.im;
          sumSqDiff += diffRe * diffRe + diffIm * diffIm;
        }
      }

      return sumSqDiff / (size - ddof);
    }

    const meanVal = meanResult as number;
    const acc = getFloatAcc(dtype);
    if (acc) {
      // float16/float32: accumulate in native precision to match NumPy overflow behavior
      acc[0] = 0;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          const diff = Number(data[off + i]!) - meanVal;
          acc[0] += diff * diff;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const diff = Number(storage.iget(i)) - meanVal;
          acc[0] += diff * diff;
        }
      }
      acc[0] /= size - ddof;
      return acc[0]!;
    }
    let sumSqDiff = 0;

    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const diff = Number(data[off + i]!) - meanVal;
        sumSqDiff += diff * diff;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const diff = Number(storage.iget(i)) - meanVal;
        sumSqDiff += diff * diff;
      }
    }

    return sumSqDiff / (size - ddof);
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const meanArray = meanResult as ArrayStorage;

  try {
    const meanData = meanArray.data;

    // Compute output shape (same as mean's output shape)
    const outputShape = keepdims
      ? meanArray.shape
      : Array.from(shape).filter((_, i) => i !== normalizedAxis);

    // Result is always float64 for variance (even for complex input)
    const result = ArrayStorage.zeros(Array.from(outputShape), 'float64');
    const resultData = result.data;

    const outerSize = outputShape.reduce((a, b) => a * b, 1);

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    if (isComplexDType(dtype)) {
      // Complex variance along axis: Var(X) = E[|X - μ|²]
      const complexData = data as Float64Array | Float32Array;
      const meanComplex = meanData as Float64Array | Float32Array;

      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        let sumSqDiff = 0;
        const meanRe = meanComplex[outerIdx * 2]!;
        const meanIm = meanComplex[outerIdx * 2 + 1]!;

        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          const re = complexData[bufIdx * 2]!;
          const im = complexData[bufIdx * 2 + 1]!;
          // |z - μ|² = (re - μ.re)² + (im - μ.im)²
          const diffRe = re - meanRe;
          const diffIm = im - meanIm;
          sumSqDiff += diffRe * diffRe + diffIm * diffIm;
          bufIdx += axisStr;
        }

        resultData[outerIdx] = sumSqDiff / (axisSize - ddof);
      }
    } else {
      const acc = getFloatAcc(dtype);
      if (acc) {
        // float16/float32: accumulate in native precision, store in input dtype
        result.dispose(); // won't use the float64 result buffer
        const outResult = ArrayStorage.empty(Array.from(outputShape), dtype);
        const outData = outResult.data;
        for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
          acc[0] = 0;
          const meanVal = Number(meanData[outerIdx]!);
          let bufIdx = baseOffsets[outerIdx]!;
          for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
            const diff = Number(data[bufIdx]!) - meanVal;
            acc[0] += diff * diff;
            bufIdx += axisStr;
          }
          acc[0] /= axisSize - ddof;
          (outData as Float16Array)[outerIdx] = acc[0]!;
        }
        return outResult;
      }
      // Real variance for each position (float64)
      for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
        let sumSqDiff = 0;
        const meanVal = Number(meanData[outerIdx]!);

        let bufIdx = baseOffsets[outerIdx]!;
        for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
          const diff = Number(data[bufIdx]!) - meanVal;
          sumSqDiff += diff * diff;
          bufIdx += axisStr;
        }

        resultData[outerIdx] = sumSqDiff / (axisSize - ddof);
      }
    }

    return result;
  } finally {
    meanArray.dispose();
  }
}

/**
 * Compute the standard deviation along the specified axis
 * @param storage - Input array storage
 * @param axis - Axis along which to compute std
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - Keep dimensions (default: false)
 *
 * For complex arrays: returns sqrt(Var(X)) where Var(X) = E[|X - E[X]|²]
 * Returns real values (float64) for both real and complex input
 */
export function std(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  // WASM fast path for full-array std (ddof=0 only — WASM kernel uses population std)
  if (axis === undefined && ddof === 0 && !keepdims) {
    const wasmResult = wasmReduceStd(storage);
    if (wasmResult !== null) return wasmResult;
  }

  // variance() handles complex arrays - returns real values
  const varResult = variance(storage, axis, ddof, keepdims);

  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }

  // Apply sqrt element-wise
  try {
    const result = ArrayStorage.zeros(Array.from(varResult.shape), 'float64');
    const varData = varResult.data;
    const resultData = result.data;

    for (let i = 0; i < varData.length; i++) {
      resultData[i] = Math.sqrt(Number(varData[i]!));
    }

    return result;
  } finally {
    varResult.dispose();
  }
}

/**
 * Test whether all array elements along a given axis evaluate to True
 */
export function all(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  const isComplex = isComplexDType(storage.dtype);

  if (axis === undefined) {
    // WASM fast path for full-array all
    const wasmAll = wasmReduceAll(storage);
    if (wasmAll !== null) return wasmAll === 1;

    // Test all elements
    if (isComplex) {
      // Complex: element is truthy if re != 0 OR im != 0
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (re === 0 && im === 0) return false;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          if (val.re === 0 && val.im === 0) return false;
        }
      }
      return true;
    }

    if (contiguous) {
      for (let i = 0; i < size; i++) {
        if (!data[off + i]) {
          return false;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        if (!storage.iget(i)) {
          return false;
        }
      }
    }
    return true;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = all(storage);
    if (!keepdims) return scalar;
    const out = ArrayStorage.zeros(Array(ndim).fill(1), 'bool');
    out.iset(0, scalar ? 1 : 0);
    return out;
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let allTrue = true;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (re === 0 && im === 0) {
          allTrue = false;
          break;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = allTrue ? 1 : 0;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let allTrue = true;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        if (!data[bufIdx]) {
          allTrue = false;
          break;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = allTrue ? 1 : 0;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}

/**
 * Test whether any array elements along a given axis evaluate to True
 */
export function any(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | boolean {
  const shape = storage.shape;
  const ndim = shape.length;
  const size = storage.size;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  const isComplex = isComplexDType(storage.dtype);

  if (axis === undefined) {
    // WASM fast path for full-array any
    const wasmAny = wasmReduceAny(storage);
    if (wasmAny !== null) return wasmAny === 1;

    // Test all elements
    if (isComplex) {
      // Complex: element is truthy if re != 0 OR im != 0
      if (contiguous) {
        const complexData = data as Float64Array | Float32Array;
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (re !== 0 || im !== 0) return true;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          if (val.re !== 0 || val.im !== 0) return true;
        }
      }
      return false;
    }

    if (contiguous) {
      for (let i = 0; i < size; i++) {
        if (data[off + i]) {
          return true;
        }
      }
    } else {
      for (let i = 0; i < size; i++) {
        if (storage.iget(i)) {
          return true;
        }
      }
    }
    return false;
  }

  // Validate and normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = any(storage);
    if (!keepdims) return scalar;
    const out = ArrayStorage.zeros(Array(ndim).fill(1), 'bool');
    out.iset(0, scalar ? 1 : 0);
    return out;
  }

  // Create result storage with bool dtype
  const result = ArrayStorage.zeros(outputShape, 'bool');
  const resultData = result.data as Uint8Array;

  // Perform reduction along axis
  const axisSize = shape[normalizedAxis]!;
  const outerSize = outputShape.reduce((a, b) => a * b, 1);

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let anyTrue = false;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (re !== 0 || im !== 0) {
          anyTrue = true;
          break;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = anyTrue ? 1 : 0;
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let anyTrue = false;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        if (data[bufIdx]) {
          anyTrue = true;
          break;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = anyTrue ? 1 : 0;
    }
  }

  // Handle keepdims
  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'bool');
  }

  return result;
}

/**
 * Return cumulative sum of elements along a given axis
 * For complex arrays: returns complex cumulative sum
 */
export function cumsum(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(dtype)) {
    // Complex cumsum
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumsum
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let sumRe = 0;
      let sumIm = 0;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          sumRe += complexData[(off + i) * 2]!;
          sumIm += complexData[(off + i) * 2 + 1]!;
          resultData[i * 2] = sumRe;
          resultData[i * 2 + 1] = sumIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          sumRe += val.re;
          sumIm += val.im;
          resultData[i * 2] = sumRe;
          resultData[i * 2 + 1] = sumIm;
        }
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Precompute offsets for outer iteration
    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);
    const { baseOffsets: inBase, axisStride: inStride } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );
    const outStrides = computeStrides(shape);
    const { baseOffsets: outBase, axisStride: outStride } = precomputeAxisOffsets(
      shape,
      outStrides,
      0,
      normalizedAxis,
      outerSize
    );

    // Perform cumsum along axis
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let sumRe = 0;
      let sumIm = 0;
      for (let k = 0; k < axisSize; k++) {
        sumRe += complexData[inIdx * 2]!;
        sumIm += complexData[inIdx * 2 + 1]!;
        resultData[outIdx * 2] = sumRe;
        resultData[outIdx * 2 + 1] = sumIm;
        inIdx += inStride;
        outIdx += outStride;
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumsum
    const size = storage.size;
    const acc = getFloatAcc(dtype);
    if (acc) {
      // float16/float32: accumulate and store in the input dtype to match NumPy
      const result = ArrayStorage.empty([size], dtype);
      const resultData = result.data;
      acc[0] = 0;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          acc[0] += Number(data[off + i]);
          resultData[i] = acc[0]!;
        }
      } else {
        for (let i = 0; i < size; i++) {
          acc[0] += Number(storage.iget(i));
          resultData[i] = acc[0]!;
        }
      }
      return result;
    }
    const accumDtype = reductionAccumDtype(dtype);
    if (isBigIntDType(dtype)) {
      // Input is already BigInt (int64/uint64) — must use BigInt arithmetic
      const result = ArrayStorage.empty([size], accumDtype);
      const resultData = result.data as BigInt64Array | BigUint64Array;
      let sum = 0n;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          sum += data[off + i] as bigint;
          resultData[i] = sum;
        }
      } else {
        for (let i = 0; i < size; i++) {
          sum += storage.iget(i) as bigint;
          resultData[i] = sum;
        }
      }
      return result;
    }
    if (isBigIntDType(accumDtype)) {
      // Non-bigint input promoted to int64/uint64 output — accumulate as Number (fast)
      const result = ArrayStorage.empty([size], accumDtype);
      const resultData = result.data as BigInt64Array | BigUint64Array;
      let sum = 0;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          sum += data[off + i] as number;
          resultData[i] = BigInt(sum);
        }
      } else {
        for (let i = 0; i < size; i++) {
          sum += Number(storage.iget(i));
          resultData[i] = BigInt(sum);
        }
      }
      return result;
    }
    const result = ArrayStorage.empty([size], accumDtype);
    const resultData = result.data as Float64Array;
    let sum = 0;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        sum += Number(data[off + i]);
        resultData[i] = sum;
      }
    } else {
      for (let i = 0; i < size; i++) {
        sum += Number(storage.iget(i));
        resultData[i] = sum;
      }
    }
    return result;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape, using NumPy accumulation dtype
  const axisAcc = getFloatAcc(dtype);
  const outDtype2 = axisAcc ? dtype : reductionAccumDtype(dtype);
  const result = ArrayStorage.empty([...shape], outDtype2);
  const resultData = result.data;
  const axisSize = shape[normalizedAxis]!;

  // Precompute offsets for outer iteration
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);
  const { baseOffsets: inBase, axisStride: inStride } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );
  const outStrides = computeStrides(shape);
  const { baseOffsets: outBase, axisStride: outStride } = precomputeAxisOffsets(
    shape,
    outStrides,
    0,
    normalizedAxis,
    outerSize
  );

  // Perform cumsum along axis
  if (axisAcc) {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      axisAcc[0] = 0;
      for (let k = 0; k < axisSize; k++) {
        axisAcc[0] += Number(data[inIdx]);
        resultData[outIdx] = axisAcc[0]!;
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  } else if (isBigIntDType(dtype)) {
    // Input is already BigInt — use BigInt arithmetic
    const bigResultData = resultData as unknown as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let sum = 0n;
      for (let k = 0; k < axisSize; k++) {
        sum += data[inIdx] as bigint;
        bigResultData[outIdx] = sum;
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  } else if (isBigIntDType(outDtype2)) {
    // Non-bigint input promoted to int64/uint64 output — accumulate as Number
    const bigResultData = resultData as unknown as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let sum = 0;
      for (let k = 0; k < axisSize; k++) {
        sum += data[inIdx] as number;
        bigResultData[outIdx] = BigInt(sum);
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let sum = 0;
      for (let k = 0; k < axisSize; k++) {
        sum += Number(data[inIdx]);
        resultData[outIdx] = sum;
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  }

  return result;
}

/**
 * Return cumulative product of elements along a given axis
 * For complex arrays: returns complex cumulative product
 * Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 */
export function cumprod(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;
  const contiguous = storage.isCContiguous;

  if (isComplexDType(dtype)) {
    // Complex cumprod
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumprod
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let prodRe = 1;
      let prodIm = 0;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
          resultData[i * 2] = prodRe;
          resultData[i * 2 + 1] = prodIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const val = storage.iget(i) as Complex;
          const re = val.re;
          const im = val.im;
          const newRe = prodRe * re - prodIm * im;
          const newIm = prodRe * im + prodIm * re;
          prodRe = newRe;
          prodIm = newIm;
          resultData[i * 2] = prodRe;
          resultData[i * 2 + 1] = prodIm;
        }
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Precompute offsets for outer iteration
    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);
    const { baseOffsets: inBase, axisStride: inStride } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );
    const outStrides = computeStrides(shape);
    const { baseOffsets: outBase, axisStride: outStride } = precomputeAxisOffsets(
      shape,
      outStrides,
      0,
      normalizedAxis,
      outerSize
    );

    // Perform cumprod along axis
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let prodRe = 1;
      let prodIm = 0;
      for (let k = 0; k < axisSize; k++) {
        const re = complexData[inIdx * 2]!;
        const im = complexData[inIdx * 2 + 1]!;
        const newRe = prodRe * re - prodIm * im;
        const newIm = prodRe * im + prodIm * re;
        prodRe = newRe;
        prodIm = newIm;
        resultData[outIdx * 2] = prodRe;
        resultData[outIdx * 2 + 1] = prodIm;
        inIdx += inStride;
        outIdx += outStride;
      }
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // Flatten and cumprod
    const size = storage.size;
    const accumDtype = reductionAccumDtype(dtype);
    if (isBigIntDType(dtype)) {
      // Input is already BigInt (int64/uint64) — must use BigInt arithmetic
      const result = ArrayStorage.empty([size], accumDtype);
      const resultData = result.data as BigInt64Array | BigUint64Array;
      let prod = 1n;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          prod *= data[off + i] as bigint;
          resultData[i] = prod;
        }
      } else {
        for (let i = 0; i < size; i++) {
          prod *= storage.iget(i) as bigint;
          resultData[i] = prod;
        }
      }
      return result;
    }
    if (isBigIntDType(accumDtype)) {
      // Non-bigint input promoted to int64/uint64 output — accumulate as Number (fast)
      const result = ArrayStorage.empty([size], accumDtype);
      const resultData = result.data as BigInt64Array | BigUint64Array;
      let prod = 1;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          prod *= data[off + i] as number;
          resultData[i] = BigInt(prod);
        }
      } else {
        for (let i = 0; i < size; i++) {
          prod *= Number(storage.iget(i));
          resultData[i] = BigInt(prod);
        }
      }
      return result;
    }
    const result = ArrayStorage.empty([size], accumDtype);
    const resultData = result.data as Float64Array;
    let prod = 1;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        prod *= Number(data[off + i]);
        resultData[i] = prod;
      }
    } else {
      for (let i = 0; i < size; i++) {
        prod *= Number(storage.iget(i));
        resultData[i] = prod;
      }
    }
    return result;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape
  const cumprodAccumDtype = reductionAccumDtype(dtype);
  const result = ArrayStorage.empty([...shape], cumprodAccumDtype);
  const resultData = result.data;
  const axisSize = shape[normalizedAxis]!;

  // Precompute offsets for outer iteration
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  const outerSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);
  const { baseOffsets: inBase, axisStride: inStride } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );
  const outStrides = computeStrides(shape);
  const { baseOffsets: outBase, axisStride: outStride } = precomputeAxisOffsets(
    shape,
    outStrides,
    0,
    normalizedAxis,
    outerSize
  );

  // Perform cumprod along axis
  if (isBigIntDType(dtype)) {
    // Input is already BigInt — use BigInt arithmetic
    const bigResultData = resultData as unknown as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let prod = 1n;
      for (let k = 0; k < axisSize; k++) {
        prod *= data[inIdx] as bigint;
        bigResultData[outIdx] = prod;
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  } else if (isBigIntDType(cumprodAccumDtype)) {
    // Non-bigint input promoted to int64/uint64 output — accumulate as Number
    const bigResultData = resultData as unknown as BigInt64Array | BigUint64Array;
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let prod = 1;
      for (let k = 0; k < axisSize; k++) {
        prod *= data[inIdx] as number;
        bigResultData[outIdx] = BigInt(prod);
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  } else {
    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let inIdx = inBase[outerIdx]!;
      let outIdx = outBase[outerIdx]!;
      let prod = 1;
      for (let k = 0; k < axisSize; k++) {
        prod *= Number(data[inIdx]);
        resultData[outIdx] = prod;
        inIdx += inStride;
        outIdx += outStride;
      }
    }
  }

  return result;
}

/**
 * Peak to peak (maximum - minimum) value along a given axis
 */
export function ptp(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;

  // NumPy rejects bool ptp: subtract is not defined for booleans
  if (dtype === 'bool') {
    throw new TypeError(
      `ufunc 'subtract' not supported for boolean dtype. The '-' operator is not supported for booleans, use 'bitwise_xor' instead.`
    );
  }

  // Complex ptp: max - min using lexicographic ordering
  if (isComplexDType(dtype)) {
    const maxResult = max(storage, axis, keepdims) as Complex | ArrayStorage;
    const minResult = min(storage, axis, keepdims) as Complex | ArrayStorage;

    if (maxResult instanceof Complex && minResult instanceof Complex) {
      return new Complex(maxResult.re - minResult.re, maxResult.im - minResult.im);
    }

    // Both are arrays, subtract element-wise
    const maxStorage = maxResult as ArrayStorage;
    const minStorage = minResult as ArrayStorage;
    try {
      const maxData = maxStorage.data as Float64Array | Float32Array;
      const minData = minStorage.data as Float64Array | Float32Array;
      const result = ArrayStorage.empty([...maxStorage.shape], dtype);
      const resultData = result.data as Float64Array;

      for (let i = 0; i < maxStorage.size; i++) {
        resultData[i * 2] = maxData[i * 2]! - minData[i * 2]!;
        resultData[i * 2 + 1] = maxData[i * 2 + 1]! - minData[i * 2 + 1]!;
      }

      return result;
    } finally {
      maxStorage.dispose();
      minStorage.dispose();
    }
  }

  const maxResult = max(storage, axis, keepdims);
  const minResult = min(storage, axis, keepdims);

  if (typeof maxResult === 'number' && typeof minResult === 'number') {
    // For integer dtypes, wrap the subtraction in the input dtype (matching NumPy)
    const diff = maxResult - minResult;
    if (
      dtype === 'int8' ||
      dtype === 'int16' ||
      dtype === 'int32' ||
      dtype === 'uint8' ||
      dtype === 'uint16' ||
      dtype === 'uint32'
    ) {
      const wrap = ArrayStorage.zeros([1], dtype);
      try {
        wrap.iset(0, diff);
        return Number(wrap.iget(0));
      } finally {
        wrap.dispose();
      }
    }
    return diff;
  }

  // Both are arrays, subtract element-wise — use input dtype for wrapping (matching NumPy)
  const maxStorage = maxResult as ArrayStorage;
  const minStorage = minResult as ArrayStorage;
  try {
    const maxData = maxStorage.data;
    const minData = minStorage.data;
    const result = ArrayStorage.zeros([...maxStorage.shape], dtype);
    const resultData = result.data;

    for (let i = 0; i < maxStorage.size; i++) {
      resultData[i] = Number(maxData[i]) - Number(minData[i]);
    }

    return result;
  } finally {
    maxStorage.dispose();
    minStorage.dispose();
  }
}

/**
 * Complex median helper: sorts complex values lexicographically (real first,
 * then imaginary), takes the middle element(s) and averages if even count.
 * TODO: move this to WASM
 */
function _complexMedian(
  storage: ArrayStorage,
  axis: number | undefined,
  keepdims: boolean,
  dropNaN: boolean
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;

  // Scalar reduction (no axis or 1D)
  if (axis === undefined || ndim === 1) {
    // Collect all complex values
    const size = storage.size;
    const pairs: [number, number][] = [];
    if (storage.isCContiguous) {
      const srcData = storage.data as Float64Array | Float32Array;
      const off = storage.offset;
      for (let i = 0; i < size; i++) {
        const re = srcData[(off + i) * 2]!;
        const im = srcData[(off + i) * 2 + 1]!;
        if (dropNaN && (isNaN(re) || isNaN(im))) continue;
        pairs.push([re, im]);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = storage.iget(i) as Complex;
        if (dropNaN && (isNaN(val.re) || isNaN(val.im))) continue;
        pairs.push([val.re, val.im]);
      }
    }

    // Sort lexicographically (real first, then imaginary)
    pairs.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));

    const n = pairs.length;
    if (n === 0) return new Complex(NaN, NaN);

    const mid = Math.floor(n / 2);
    if (n % 2 === 1) {
      return new Complex(pairs[mid]![0], pairs[mid]![1]);
    }
    // Average of two middle elements
    return new Complex(
      (pairs[mid - 1]![0] + pairs[mid]![0]) / 2,
      (pairs[mid - 1]![1] + pairs[mid]![1]) / 2
    );
  }

  // Axis reduction
  let normalizedAxis = axis;
  if (normalizedAxis < 0) normalizedAxis += ndim;
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    // 1D input reduced to scalar — recurse with axis=undefined to use the flat path
    const scalar = _complexMedian(storage, undefined, false, dropNaN);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number, ndim, dtype);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, dtype);
  const resultData = result.data as Float64Array | Float32Array;

  // For each output position, gather the axis slice, sort, and take median
  const innerStride = Array.from(shape)
    .slice(normalizedAxis + 1)
    .reduce((a, b) => a * b, 1);
  const outerStride = axisSize * innerStride;

  for (let outer = 0; outer < outerSize; outer++) {
    const outerIdx = Math.floor(outer / innerStride);
    const innerIdx = outer % innerStride;
    const baseOffset = outerIdx * outerStride + innerIdx;

    const pairs: [number, number][] = [];
    for (let k = 0; k < axisSize; k++) {
      const flatIdx = baseOffset + k * innerStride;
      const val = storage.iget(flatIdx) as Complex;
      if (dropNaN && (isNaN(val.re) || isNaN(val.im))) continue;
      pairs.push([val.re, val.im]);
    }

    pairs.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));

    const n = pairs.length;
    const mid = Math.floor(n / 2);
    if (n === 0) {
      resultData[outer * 2] = NaN;
      resultData[outer * 2 + 1] = NaN;
    } else if (n % 2 === 1) {
      resultData[outer * 2] = pairs[mid]![0];
      resultData[outer * 2 + 1] = pairs[mid]![1];
    } else {
      resultData[outer * 2] = (pairs[mid - 1]![0] + pairs[mid]![0]) / 2;
      resultData[outer * 2 + 1] = (pairs[mid - 1]![1] + pairs[mid]![1]) / 2;
    }
  }

  if (keepdims) {
    const kdShape = Array.from(shape);
    kdShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(result.data, kdShape, dtype);
  }

  return result;
}

/**
 * Compute the median along the specified axis
 */
export function median(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  // Bool median: promote to float64 first (NumPy median succeeds for bool,
  // even though quantile rejects bool due to subtract)
  if (storage.dtype === 'bool') {
    const f64 = ArrayStorage.empty(Array.from(storage.shape), 'float64');
    const srcData = storage.data;
    const dstData = f64.data as Float64Array;
    const off = storage.offset;
    if (storage.isCContiguous) {
      for (let i = 0; i < storage.size; i++) dstData[i] = Number(srcData[off + i]);
    } else {
      for (let i = 0; i < storage.size; i++) dstData[i] = Number(storage.iget(i));
    }
    try {
      return quantile(f64, 0.5, axis, keepdims);
    } finally {
      f64.dispose();
    }
  }

  // Complex median: sort lexicographically, take middle element(s)
  if (isComplexDType(storage.dtype)) {
    return _complexMedian(storage, axis, keepdims, false);
  }

  return quantile(storage, 0.5, axis, keepdims);
}

/**
 * Compute the q-th percentile of data along specified axis
 */
export function percentile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  return quantile(storage, q / 100, axis, keepdims);
}

/**
 * Compute the q-th quantile of data along specified axis
 */
export function quantile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  throwIfComplex(storage.dtype, 'quantile', 'Complex numbers are not orderable.');
  // NumPy rejects bool quantile: linear interpolation requires subtract, unsupported for bool
  if (storage.dtype === 'bool') {
    throw new TypeError(
      `ufunc 'subtract' not supported for boolean dtype. The '-' operator is not supported for booleans, use 'bitwise_xor' instead.`
    );
  }
  if (q < 0 || q > 1) {
    throw new Error('Quantile must be between 0 and 1');
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (axis === undefined) {
    // WASM fast path for full-array quantile
    const wasmQ = wasmReduceQuantile(storage, q);
    if (wasmQ !== null) return wasmQ;

    // Compute quantile over all elements
    const values: number[] = [];
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        values.push(Number(data[off + i]));
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        values.push(Number(storage.iget(i)));
      }
    }
    values.sort((a, b) => a - b);

    const n = values.length;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      return values[lower]!;
    }

    // Linear interpolation
    const frac = idx - lower;
    return values[lower]! * (1 - frac) + values[upper]! * frac;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = quantile(storage, q) as number;
    if (!keepdims) return scalar;
    const out = ArrayStorage.zeros(Array(ndim).fill(1), 'float64');
    out.iset(0, scalar);
    return out;
  }

  const axisSize = shape[normalizedAxis]!;

  // WASM fast path for strided quantile
  if (storage.isCContiguous && !isComplexDType(storage.dtype)) {
    const wasmOuter = shape.slice(0, normalizedAxis).reduce((a, b) => a * b, 1);
    const innerSize = shape.slice(normalizedAxis + 1).reduce((a, b) => a * b, 1);
    const wasmResult = wasmReduceQuantileStrided(storage, wasmOuter, axisSize, innerSize, q);
    if (wasmResult) {
      if (keepdims) {
        const keepdimsShape = [...shape];
        keepdimsShape[normalizedAxis] = 1;
        const shared = ArrayStorage.fromDataShared(
          wasmResult.data,
          keepdimsShape,
          'float64',
          computeStrides(keepdimsShape),
          0,
          wasmResult.wasmRegion
        );
        wasmResult.dispose();
        return shared;
      }
      // Reshape to outputShape
      const reshaped = ArrayStorage.fromDataShared(
        wasmResult.data,
        outputShape,
        'float64',
        computeStrides(outputShape),
        0,
        wasmResult.wasmRegion
      );
      wasmResult.dispose();
      return reshaped;
    }
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  // Reuse a single typed array for sorting — avoids per-column JS array allocation
  const sortBuf = new Float64Array(axisSize);

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // Collect values along axis into reusable buffer
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      sortBuf[axisIdx] = Number(data[bufIdx]);
      bufIdx += axisStr;
    }
    sortBuf.sort(); // Float64Array.sort() — no comparator needed, handles NaN

    const n = axisSize;
    const values = sortBuf;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      resultData[outerIdx] = values[lower]!;
    } else {
      // Linear interpolation
      const frac = idx - lower;
      resultData[outerIdx] = values[lower]! * (1 - frac) + values[upper]! * frac;
    }
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute the weighted average along the specified axis
 */
export function average(
  storage: ArrayStorage,
  axis?: number,
  weights?: ArrayStorage,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype as DType;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (weights === undefined) {
    // Unweighted average is just mean
    return mean(storage, axis, keepdims);
  }

  if (isComplexDType(dtype)) {
    // Complex weighted average: sum(w_i * z_i) / sum(w_i)
    const complexData = data as Float64Array | Float32Array;
    const weightData = weights.data;
    const wOff = weights.offset;

    if (axis === undefined) {
      // Compute weighted average over all elements
      let sumRe = 0;
      let sumIm = 0;
      let sumWeights = 0;

      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const w = Number(weightData[wOff + (i % weights.size)]);
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          sumRe += re * w;
          sumIm += im * w;
          sumWeights += w;
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const w = Number(weightData[wOff + (i % weights.size)]);
          const val = storage.iget(i) as Complex;
          sumRe += val.re * w;
          sumIm += val.im * w;
          sumWeights += w;
        }
      }

      if (sumWeights === 0) {
        return new Complex(NaN, NaN);
      }
      return new Complex(sumRe / sumWeights, sumIm / sumWeights);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Compute output shape
    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = average(storage, undefined, weights);
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar as Complex, ndim, dtype);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.zeros(outputShape, dtype);
    const resultData = result.data as Float64Array | Float32Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let sumRe = 0;
      let sumIm = 0;
      let sumWeights = 0;

      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const w = Number(weightData[wOff + (axisIdx % weights.size)]);
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        sumRe += re * w;
        sumIm += im * w;
        sumWeights += w;
        bufIdx += axisStr;
      }

      if (sumWeights === 0) {
        resultData[outerIdx * 2] = NaN;
        resultData[outerIdx * 2 + 1] = NaN;
      } else {
        resultData[outerIdx * 2] = sumRe / sumWeights;
        resultData[outerIdx * 2 + 1] = sumIm / sumWeights;
      }
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  // Non-complex path
  const wOff2 = weights.offset;
  if (axis === undefined) {
    // Compute weighted average over all elements
    let sumWeightedValues = 0;
    let sumWeights = 0;
    const weightData = weights.data;

    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const w = Number(weightData[wOff2 + (i % weights.size)]);
        sumWeightedValues += Number(data[off + i]) * w;
        sumWeights += w;
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const w = Number(weightData[wOff2 + (i % weights.size)]);
        sumWeightedValues += Number(storage.iget(i)) * w;
        sumWeights += w;
      }
    }

    return sumWeights === 0 ? NaN : roundToDtype(sumWeightedValues / sumWeights, dtype);
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Compute output shape
  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = average(storage, undefined, weights);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number, ndim, 'float64');
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const weightData = weights.data;
  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let sumWeightedValues = 0;
    let sumWeights = 0;

    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const w = Number(weightData[wOff2 + (axisIdx % weights.size)]);
      sumWeightedValues += Number(data[bufIdx]) * w;
      sumWeights += w;
      bufIdx += axisStr;
    }

    resultData[outerIdx] = sumWeights === 0 ? NaN : sumWeightedValues / sumWeights;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

// ============================================================================
// NaN-aware reduction functions
// ============================================================================

/**
 * Return sum of elements, treating NaNs as zero
 */
/**
 * Check if a complex number is NaN (either part is NaN)
 */
function complexIsNaN(re: number, im: number): boolean {
  return isNaN(re) || isNaN(im);
}

export function nansum(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Integer types can't have NaN — delegate to regular sum (which has WASM)
  if (!isComplex && !isFloatDType(dtype)) {
    return sum(storage, axis, keepdims);
  }

  if (axis === undefined) {
    // WASM fast path for float nansum
    const wasmResult = wasmReduceNansum(storage);
    if (wasmResult !== null) return wasmResult;

    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 0;
      let totalIm = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
          }
        }
      }
      return new Complex(totalRe, totalIm);
    }

    const acc = getFloatAcc(dtype as DType);
    if (acc) {
      acc[0] = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const val = Number(data[off + i]);
          if (!isNaN(val)) {
            acc[0] += val;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const val = Number(storage.iget(i));
          if (!isNaN(val)) {
            acc[0] += val;
          }
        }
      }
      return acc[0]!;
    }

    let total = 0;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          total += val;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          total += val;
        }
      }
    }
    return total;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nansum(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | Complex, ndim, dtype as DType);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const result = ArrayStorage.empty(outputShape, dtype);
    const resultData = result.data as Float64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 0;
      let totalIm = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx * 2] = totalRe;
      resultData[outerIdx * 2 + 1] = totalIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 0;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        total += val;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = total;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return product of elements, treating NaNs as one
 */
export function nanprod(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);

  // Integer types can't have NaN — delegate to regular prod (which has WASM)
  if (!isComplex && !isFloatDType(dtype)) {
    return prod(storage, axis, keepdims);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (axis === undefined) {
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 1;
      let totalIm = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            const newRe = totalRe * re - totalIm * im;
            const newIm = totalRe * im + totalIm * re;
            totalRe = newRe;
            totalIm = newIm;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            const newRe = totalRe * re - totalIm * im;
            const newIm = totalRe * im + totalIm * re;
            totalRe = newRe;
            totalIm = newIm;
          }
        }
      }
      return new Complex(totalRe, totalIm);
    }

    let total = 1;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          total *= val;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          total *= val;
        }
      }
    }
    return total;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanprod(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | Complex, ndim, dtype as DType);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const result = ArrayStorage.empty(outputShape, dtype);
    const resultData = result.data as Float64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 1;
      let totalIm = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          const newRe = totalRe * re - totalIm * im;
          const newIm = totalRe * im + totalIm * re;
          totalRe = newRe;
          totalIm = newIm;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx * 2] = totalRe;
      resultData[outerIdx * 2 + 1] = totalIm;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 1;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        total *= val;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = total;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute mean ignoring NaN values
 */
export function nanmean(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const isComplex = isComplexDType(dtype);

  // Integer types can't have NaN — delegate to regular mean (which has WASM)
  if (!isComplex && !isFloatDType(dtype)) {
    return mean(storage, axis, keepdims);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (axis === undefined) {
    if (isComplex) {
      const complexData = data as Float64Array | Float32Array;
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
            count++;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
            count++;
          }
        }
      }
      return count === 0 ? new Complex(NaN, NaN) : new Complex(totalRe / count, totalIm / count);
    }

    // nanmean accumulates in the input dtype's precision (like NumPy)
    const acc = getFloatAcc(dtype as DType);
    let count = 0;
    const contiguous = storage.isCContiguous;
    if (acc) {
      acc[0] = 0;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const val = Number(data[off + i]);
          if (!isNaN(val)) {
            acc[0] += val;
            count++;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const val = Number(storage.iget(i));
          if (!isNaN(val)) {
            acc[0] += val;
            count++;
          }
        }
      }
      if (count === 0) return NaN;
      acc[0] /= count;
      return acc[0]!;
    }
    let total = 0;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          total += val;
          count++;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          total += val;
          count++;
        }
      }
    }
    return count === 0 ? NaN : total / count;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanmean(storage);
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar as number | Complex, ndim, dtype as DType);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  if (isComplex) {
    const complexData = data as Float64Array | Float32Array;
    const result = ArrayStorage.empty(outputShape, dtype);
    const resultData = result.data as Float64Array;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
        bufIdx += axisStr;
      }
      if (count === 0) {
        resultData[outerIdx * 2] = NaN;
        resultData[outerIdx * 2 + 1] = NaN;
      } else {
        resultData[outerIdx * 2] = totalRe / count;
        resultData[outerIdx * 2 + 1] = totalIm / count;
      }
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }
    return result;
  }

  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let total = 0;
    let count = 0;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = count === 0 ? NaN : total / count;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute variance ignoring NaN values
 */
export function nanvar(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  const dtype = storage.dtype as DType;

  // Integer types can't have NaN — delegate to regular variance (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return variance(storage, axis, ddof, keepdims);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (isComplexDType(dtype)) {
    // Complex nanvar: Var(X) = E[|X - μ|²] where |z|² = re² + im²
    // Returns real values (float64)
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      // First pass: compute mean ignoring NaN
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
            count++;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            totalRe += re;
            totalIm += im;
            count++;
          }
        }
      }
      if (count - ddof <= 0) return NaN;
      const meanRe = totalRe / count;
      const meanIm = totalIm / count;

      // Second pass: compute sum of squared deviations
      let sumSq = 0;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            const diffRe = re - meanRe;
            const diffIm = im - meanIm;
            sumSq += diffRe * diffRe + diffIm * diffIm;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            const diffRe = re - meanRe;
            const diffIm = im - meanIm;
            sumSq += diffRe * diffRe + diffIm * diffIm;
          }
        }
      }
      return sumSq / (count - ddof);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = nanvar(storage, undefined, ddof) as number;
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar, ndim, 'float64');
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.empty(outputShape, 'float64');
    const resultData = result.data as Float64Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // First pass: compute mean ignoring NaN
      let totalRe = 0;
      let totalIm = 0;
      let count = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          totalRe += re;
          totalIm += im;
          count++;
        }
        bufIdx += axisStr;
      }

      if (count - ddof <= 0) {
        resultData[outerIdx] = NaN;
        continue;
      }

      const meanRe = totalRe / count;
      const meanIm = totalIm / count;

      // Second pass: compute sum of squared deviations
      let sumSq = 0;
      bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im)) {
          const diffRe = re - meanRe;
          const diffIm = im - meanIm;
          sumSq += diffRe * diffRe + diffIm * diffIm;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = sumSq / (count - ddof);
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    // First pass: compute mean
    let total = 0;
    let count = 0;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          total += val;
          count++;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          total += val;
          count++;
        }
      }
    }
    if (count - ddof <= 0) return NaN;
    const meanVal = total / count;

    // Second pass: compute sum of squared deviations
    let sumSq = 0;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          sumSq += (val - meanVal) ** 2;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          sumSq += (val - meanVal) ** 2;
        }
      }
    }
    return sumSq / (count - ddof);
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanvar(storage, undefined, ddof) as number;
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar, ndim, 'float64');
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // First pass: compute mean
    let total = 0;
    let count = 0;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        total += val;
        count++;
      }
      bufIdx += axisStr;
    }

    if (count - ddof <= 0) {
      resultData[outerIdx] = NaN;
      continue;
    }

    const meanVal = total / count;

    // Second pass: compute sum of squared deviations
    let sumSq = 0;
    bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        sumSq += (val - meanVal) ** 2;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = sumSq / (count - ddof);
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute standard deviation ignoring NaN values
 * For complex arrays: returns sqrt of variance (always real values)
 */
export function nanstd(
  storage: ArrayStorage,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): ArrayStorage | number {
  // nanvar handles complex arrays and returns real values
  const varResult = nanvar(storage, axis, ddof, keepdims);
  if (typeof varResult === 'number') {
    return Math.sqrt(varResult);
  }
  const varStorage = varResult as ArrayStorage;
  const result = ArrayStorage.empty([...varStorage.shape], 'float64');
  const resultData = result.data as Float64Array;
  for (let i = 0; i < varStorage.size; i++) {
    resultData[i] = Math.sqrt(Number(varStorage.data[i]));
  }
  return result;
}

/**
 * Return minimum ignoring NaN values
 */
export function nanmin(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Integer types can't have NaN — delegate to regular min (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return min(storage, axis, keepdims);
  }

  // Complex nanmin uses lexicographic ordering, skipping NaN values
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let minRe = Infinity;
      let minIm = Infinity;
      let foundNonNaN = false;
      const contiguous = storage.isCContiguous;

      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;

          if (isNaN(re) || isNaN(im)) {
            continue;
          }

          if (!foundNonNaN) {
            minRe = re;
            minIm = im;
            foundNonNaN = true;
          } else if (re < minRe || (re === minRe && im < minIm)) {
            minRe = re;
            minIm = im;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;

          if (isNaN(re) || isNaN(im)) {
            continue;
          }

          if (!foundNonNaN) {
            minRe = re;
            minIm = im;
            foundNonNaN = true;
          } else if (re < minRe || (re === minRe && im < minIm)) {
            minRe = re;
            minIm = im;
          }
        }
      }
      return foundNonNaN ? new Complex(minRe, minIm) : new Complex(NaN, NaN);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = nanmin(storage) as Complex;
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar, ndim, dtype);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.empty(outputShape, dtype);
    const resultData = result.data as Float64Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minRe = Infinity;
      let minIm = Infinity;
      let foundNonNaN = false;

      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        bufIdx += axisStr;

        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          minRe = re;
          minIm = im;
          foundNonNaN = true;
        } else if (re < minRe || (re === minRe && im < minIm)) {
          minRe = re;
          minIm = im;
        }
      }
      resultData[outerIdx * 2] = foundNonNaN ? minRe : NaN;
      resultData[outerIdx * 2 + 1] = foundNonNaN ? minIm : NaN;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  if (axis === undefined) {
    // WASM fast path for full-array nanmin (non-complex)
    const wasmResult = wasmReduceNanmin(storage);
    if (wasmResult !== null) return wasmResult;

    let minVal = Infinity;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val) && val < minVal) {
          minVal = val;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val) && val < minVal) {
          minVal = val;
        }
      }
    }
    return minVal === Infinity ? NaN : minVal;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanmin(storage) as number;
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar, ndim, 'float64');
  }

  // nanmin preserves input dtype (like NumPy)
  const outDtype = isFloatDType(dtype) ? dtype : 'float64';
  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, outDtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let minVal = Infinity;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = minVal === Infinity ? NaN : minVal;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return maximum ignoring NaN values
 */
export function nanmax(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  const dtype = storage.dtype;
  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  // Integer types can't have NaN — delegate to regular max (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return max(storage, axis, keepdims);
  }

  // Complex nanmax uses lexicographic ordering, skipping NaN values
  if (isComplexDType(dtype)) {
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let foundNonNaN = false;
      const contiguous = storage.isCContiguous;

      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;

          if (isNaN(re) || isNaN(im)) {
            continue;
          }

          if (!foundNonNaN) {
            maxRe = re;
            maxIm = im;
            foundNonNaN = true;
          } else if (re > maxRe || (re === maxRe && im > maxIm)) {
            maxRe = re;
            maxIm = im;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;

          if (isNaN(re) || isNaN(im)) {
            continue;
          }

          if (!foundNonNaN) {
            maxRe = re;
            maxIm = im;
            foundNonNaN = true;
          } else if (re > maxRe || (re === maxRe && im > maxIm)) {
            maxRe = re;
            maxIm = im;
          }
        }
      }
      return foundNonNaN ? new Complex(maxRe, maxIm) : new Complex(NaN, NaN);
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      const scalar = nanmax(storage) as Complex;
      if (!keepdims) return scalar;
      return wrapScalarKeepdims(scalar, ndim, dtype);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.empty(outputShape, dtype);
    const resultData = result.data as Float64Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let foundNonNaN = false;

      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        bufIdx += axisStr;

        if (isNaN(re) || isNaN(im)) {
          continue;
        }

        if (!foundNonNaN) {
          maxRe = re;
          maxIm = im;
          foundNonNaN = true;
        } else if (re > maxRe || (re === maxRe && im > maxIm)) {
          maxRe = re;
          maxIm = im;
        }
      }
      resultData[outerIdx * 2] = foundNonNaN ? maxRe : NaN;
      resultData[outerIdx * 2 + 1] = foundNonNaN ? maxIm : NaN;
    }

    if (keepdims) {
      const keepdimsShape = [...shape];
      keepdimsShape[normalizedAxis] = 1;
      return ArrayStorage.fromData(resultData, keepdimsShape, dtype);
    }

    return result;
  }

  if (axis === undefined) {
    // WASM fast path for full-array nanmax (non-complex)
    const wasmResult = wasmReduceNanmax(storage);
    if (wasmResult !== null) return wasmResult;

    let maxVal = -Infinity;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val) && val > maxVal) {
          maxVal = val;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val) && val > maxVal) {
          maxVal = val;
        }
      }
    }
    return maxVal === -Infinity ? NaN : maxVal;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanmax(storage) as number;
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar, ndim, 'float64');
  }

  // nanmax preserves input dtype (like NumPy)
  const outDtype = isFloatDType(dtype) ? dtype : 'float64';
  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, outDtype);
  const resultData = result.data;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let maxVal = -Infinity;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = maxVal === -Infinity ? NaN : maxVal;
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Return indices of minimum value, ignoring NaNs
 */
export function nanargmin(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype as DType;

  // Integer types can't have NaN — delegate to regular argmin (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return argmin(storage, axis);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (isComplexDType(dtype)) {
    // Complex nanargmin using lexicographic ordering, skipping NaN values
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let minRe = Infinity;
      let minIm = Infinity;
      let minIdx = -1;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im) && complexCompare(re, im, minRe, minIm) < 0) {
            minRe = re;
            minIm = im;
            minIdx = i;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im) && complexCompare(re, im, minRe, minIm) < 0) {
            minRe = re;
            minIm = im;
            minIdx = i;
          }
        }
      }
      return minIdx;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanargmin(storage);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.empty(outputShape, 'int32');
    const resultData = result.data as Int32Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let minRe = Infinity;
      let minIm = Infinity;
      let minIdx = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, minRe, minIm) < 0) {
          minRe = re;
          minIm = im;
          minIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = minIdx;
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    let minVal = Infinity;
    let minIdx = -1;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val) && val < minVal) {
          minVal = val;
          minIdx = i;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val) && val < minVal) {
          minVal = val;
          minIdx = i;
        }
      }
    }
    return minIdx;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanargmin(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, 'int32');
  const resultData = result.data as Int32Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let minVal = Infinity;
    let minIdx = 0;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val) && val < minVal) {
        minVal = val;
        minIdx = axisIdx;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = minIdx;
  }

  return result;
}

/**
 * Return indices of maximum value, ignoring NaNs
 * For complex arrays: uses lexicographic ordering (real first, then imaginary)
 */
export function nanargmax(storage: ArrayStorage, axis?: number): ArrayStorage | number {
  const dtype = storage.dtype as DType;

  // Integer types can't have NaN — delegate to regular argmax (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return argmax(storage, axis);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (isComplexDType(dtype)) {
    // Complex nanargmax using lexicographic ordering, skipping NaN values
    const complexData = data as Float64Array | Float32Array;

    if (axis === undefined) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let maxIdx = -1;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < storage.size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im) && complexCompare(re, im, maxRe, maxIm) > 0) {
            maxRe = re;
            maxIm = im;
            maxIdx = i;
          }
        }
      } else {
        for (let i = 0; i < storage.size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im) && complexCompare(re, im, maxRe, maxIm) > 0) {
            maxRe = re;
            maxIm = im;
            maxIdx = i;
          }
        }
      }
      return maxIdx;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
    if (outputShape.length === 0) {
      return nanargmax(storage);
    }

    const outerSize = outputShape.reduce((a, b) => a * b, 1);
    const axisSize = shape[normalizedAxis]!;
    const result = ArrayStorage.empty(outputShape, 'int32');
    const resultData = result.data as Int32Array;

    const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
      shape,
      inputStrides,
      off,
      normalizedAxis,
      outerSize
    );

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      let maxRe = -Infinity;
      let maxIm = -Infinity;
      let maxIdx = 0;
      let bufIdx = baseOffsets[outerIdx]!;
      for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        const re = complexData[bufIdx * 2]!;
        const im = complexData[bufIdx * 2 + 1]!;
        if (!complexIsNaN(re, im) && complexCompare(re, im, maxRe, maxIm) > 0) {
          maxRe = re;
          maxIm = im;
          maxIdx = axisIdx;
        }
        bufIdx += axisStr;
      }
      resultData[outerIdx] = maxIdx;
    }

    return result;
  }

  // Non-complex path
  if (axis === undefined) {
    let maxVal = -Infinity;
    let maxIdx = -1;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val) && val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val) && val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
    }
    return maxIdx;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    return nanargmax(storage);
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, 'int32');
  const resultData = result.data as Int32Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val) && val > maxVal) {
        maxVal = val;
        maxIdx = axisIdx;
      }
      bufIdx += axisStr;
    }
    resultData[outerIdx] = maxIdx;
  }

  return result;
}

/**
 * Return cumulative sum, treating NaNs as zero
 */
export function nancumsum(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;

  // Integer types can't have NaN — delegate to regular cumsum (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return cumsum(storage, axis);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;

  if (isComplexDType(dtype)) {
    // Complex nancumsum - treat NaN values as 0
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumsum
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let sumRe = 0;
      let sumIm = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            sumRe += re;
            sumIm += im;
          }
          resultData[i * 2] = sumRe;
          resultData[i * 2 + 1] = sumIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            sumRe += re;
            sumIm += im;
          }
          resultData[i * 2] = sumRe;
          resultData[i * 2 + 1] = sumIm;
        }
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumsum along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;
    const contiguousAxis = storage.isCContiguous;

    if (contiguousAxis) {
      for (let i = 0; i < totalSize; i++) {
        const re = complexData[(off + i) * 2]!;
        const im = complexData[(off + i) * 2 + 1]!;
        const axisPos = Math.floor(i / axisStride) % axisSize;
        const isNan = complexIsNaN(re, im);

        if (axisPos === 0) {
          resultData[i * 2] = isNan ? 0 : re;
          resultData[i * 2 + 1] = isNan ? 0 : im;
        } else {
          resultData[i * 2] = resultData[(i - axisStride) * 2]! + (isNan ? 0 : re);
          resultData[i * 2 + 1] = resultData[(i - axisStride) * 2 + 1]! + (isNan ? 0 : im);
        }
      }
    } else {
      for (let i = 0; i < totalSize; i++) {
        const v = storage.iget(i);
        const re = (v as Complex).re;
        const im = (v as Complex).im;
        const axisPos = Math.floor(i / axisStride) % axisSize;
        const isNan = complexIsNaN(re, im);

        if (axisPos === 0) {
          resultData[i * 2] = isNan ? 0 : re;
          resultData[i * 2 + 1] = isNan ? 0 : im;
        } else {
          resultData[i * 2] = resultData[(i - axisStride) * 2]! + (isNan ? 0 : re);
          resultData[i * 2 + 1] = resultData[(i - axisStride) * 2 + 1]! + (isNan ? 0 : im);
        }
      }
    }

    return result;
  }

  // Non-complex path — preserve float dtype
  if (axis === undefined) {
    // Flatten and cumsum
    const size = storage.size;
    const result = ArrayStorage.empty([size], dtype);
    const resultData = result.data as Float64Array;
    let sum = 0;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          sum += val;
        }
        resultData[i] = sum;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          sum += val;
        }
        resultData[i] = sum;
      }
    }
    return result;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape — preserve float dtype
  const result = ArrayStorage.empty([...shape], dtype);
  const resultData = result.data as Float64Array;
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumsum along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;
  const contiguousAxis = storage.isCContiguous;

  if (contiguousAxis) {
    for (let i = 0; i < totalSize; i++) {
      const val = Number(data[off + i]);
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i] = isNaN(val) ? 0 : val;
      } else {
        resultData[i] = resultData[i - axisStride]! + (isNaN(val) ? 0 : val);
      }
    }
  } else {
    for (let i = 0; i < totalSize; i++) {
      const val = Number(storage.iget(i));
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i] = isNaN(val) ? 0 : val;
      } else {
        resultData[i] = resultData[i - axisStride]! + (isNaN(val) ? 0 : val);
      }
    }
  }

  return result;
}

/**
 * Return cumulative product, treating NaNs as one
 * For complex arrays: NaN values (either part NaN) are treated as 1+0i
 */
export function nancumprod(storage: ArrayStorage, axis?: number): ArrayStorage {
  const dtype = storage.dtype as DType;

  // Integer types can't have NaN — delegate to regular cumprod (which has WASM)
  if (!isComplexDType(dtype) && !isFloatDType(dtype)) {
    return cumprod(storage, axis);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;

  if (isComplexDType(dtype)) {
    // Complex nancumprod - treat NaN values as 1+0i
    const complexData = data as Float64Array | Float32Array;
    const size = storage.size;

    if (axis === undefined) {
      // Flatten and cumprod
      const result = ArrayStorage.zeros([size], dtype);
      const resultData = result.data as Float64Array | Float32Array;
      let prodRe = 1;
      let prodIm = 0;
      const contiguous = storage.isCContiguous;
      if (contiguous) {
        for (let i = 0; i < size; i++) {
          const re = complexData[(off + i) * 2]!;
          const im = complexData[(off + i) * 2 + 1]!;
          if (!complexIsNaN(re, im)) {
            const newRe = prodRe * re - prodIm * im;
            const newIm = prodRe * im + prodIm * re;
            prodRe = newRe;
            prodIm = newIm;
          }
          resultData[i * 2] = prodRe;
          resultData[i * 2 + 1] = prodIm;
        }
      } else {
        for (let i = 0; i < size; i++) {
          const v = storage.iget(i);
          const re = (v as Complex).re;
          const im = (v as Complex).im;
          if (!complexIsNaN(re, im)) {
            const newRe = prodRe * re - prodIm * im;
            const newIm = prodRe * im + prodIm * re;
            prodRe = newRe;
            prodIm = newIm;
          }
          resultData[i * 2] = prodRe;
          resultData[i * 2 + 1] = prodIm;
        }
      }
      return result;
    }

    // Normalize axis
    let normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis = ndim + normalizedAxis;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    // Create result with same shape
    const result = ArrayStorage.zeros([...shape], dtype);
    const resultData = result.data as Float64Array | Float32Array;
    const axisSize = shape[normalizedAxis]!;

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= shape[i]!;
    }

    // Perform cumprod along axis
    const totalSize = storage.size;
    const axisStride = strides[normalizedAxis]!;
    const contiguousAxis = storage.isCContiguous;

    if (contiguousAxis) {
      for (let i = 0; i < totalSize; i++) {
        const re = complexData[(off + i) * 2]!;
        const im = complexData[(off + i) * 2 + 1]!;
        const axisPos = Math.floor(i / axisStride) % axisSize;
        const isNan = complexIsNaN(re, im);

        if (axisPos === 0) {
          resultData[i * 2] = isNan ? 1 : re;
          resultData[i * 2 + 1] = isNan ? 0 : im;
        } else {
          const prevRe = resultData[(i - axisStride) * 2]!;
          const prevIm = resultData[(i - axisStride) * 2 + 1]!;
          if (isNan) {
            resultData[i * 2] = prevRe;
            resultData[i * 2 + 1] = prevIm;
          } else {
            resultData[i * 2] = prevRe * re - prevIm * im;
            resultData[i * 2 + 1] = prevRe * im + prevIm * re;
          }
        }
      }
    } else {
      for (let i = 0; i < totalSize; i++) {
        const v = storage.iget(i);
        const re = (v as Complex).re;
        const im = (v as Complex).im;
        const axisPos = Math.floor(i / axisStride) % axisSize;
        const isNan = complexIsNaN(re, im);

        if (axisPos === 0) {
          resultData[i * 2] = isNan ? 1 : re;
          resultData[i * 2 + 1] = isNan ? 0 : im;
        } else {
          const prevRe = resultData[(i - axisStride) * 2]!;
          const prevIm = resultData[(i - axisStride) * 2 + 1]!;
          if (isNan) {
            resultData[i * 2] = prevRe;
            resultData[i * 2 + 1] = prevIm;
          } else {
            resultData[i * 2] = prevRe * re - prevIm * im;
            resultData[i * 2 + 1] = prevRe * im + prevIm * re;
          }
        }
      }
    }

    return result;
  }

  // Non-complex path — preserve float dtype
  if (axis === undefined) {
    // Flatten and cumprod
    const size = storage.size;
    const result = ArrayStorage.empty([size], dtype);
    const resultData = result.data as Float64Array;
    let prod = 1;
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          prod *= val;
        }
        resultData[i] = prod;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          prod *= val;
        }
        resultData[i] = prod;
      }
    }
    return result;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  // Create result with same shape — preserve float dtype
  const result = ArrayStorage.empty([...shape], dtype);
  const resultData = result.data as Float64Array;
  const axisSize = shape[normalizedAxis]!;

  // Calculate strides
  const strides: number[] = [];
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    strides.unshift(stride);
    stride *= shape[i]!;
  }

  // Perform cumprod along axis
  const totalSize = storage.size;
  const axisStride = strides[normalizedAxis]!;
  const contiguousAxis = storage.isCContiguous;

  if (contiguousAxis) {
    for (let i = 0; i < totalSize; i++) {
      const val = Number(data[off + i]);
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i] = isNaN(val) ? 1 : val;
      } else {
        resultData[i] = resultData[i - axisStride]! * (isNaN(val) ? 1 : val);
      }
    }
  } else {
    for (let i = 0; i < totalSize; i++) {
      const val = Number(storage.iget(i));
      const axisPos = Math.floor(i / axisStride) % axisSize;

      if (axisPos === 0) {
        resultData[i] = isNaN(val) ? 1 : val;
      } else {
        resultData[i] = resultData[i - axisStride]! * (isNaN(val) ? 1 : val);
      }
    }
  }

  return result;
}

/**
 * Compute median ignoring NaN values
 */
export function nanmedian(
  storage: ArrayStorage,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number | Complex {
  if (isComplexDType(storage.dtype)) {
    return _complexMedian(storage, axis, keepdims, true);
  }

  // Integer types can't have NaN — delegate to regular median (which has WASM)
  if (!isFloatDType(storage.dtype)) {
    return median(storage, axis, keepdims);
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (axis === undefined) {
    // Collect non-NaN values
    const values: number[] = [];
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          values.push(val);
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          values.push(val);
        }
      }
    }

    if (values.length === 0) return NaN;

    values.sort((a, b) => a - b);
    const n = values.length;
    const mid = Math.floor(n / 2);

    if (n % 2 === 0) {
      return (values[mid - 1]! + values[mid]!) / 2;
    }
    return values[mid]!;
  }

  // Normalize axis
  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanmedian(storage) as number;
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar, ndim, 'float64');
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    // Collect non-NaN values along axis
    const values: number[] = [];
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        values.push(val);
      }
      bufIdx += axisStr;
    }

    if (values.length === 0) {
      resultData[outerIdx] = NaN;
      continue;
    }

    values.sort((a, b) => a - b);
    const n = values.length;
    const mid = Math.floor(n / 2);

    if (n % 2 === 0) {
      resultData[outerIdx] = (values[mid - 1]! + values[mid]!) / 2;
    } else {
      resultData[outerIdx] = values[mid]!;
    }
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute the q-th quantile of data along specified axis, ignoring NaNs
 */
export function nanquantile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  throwIfComplex(storage.dtype, 'nanquantile', 'Complex numbers are not orderable.');
  if (storage.dtype === 'bool') {
    throw new TypeError(
      `ufunc 'subtract' not supported for boolean dtype. The '-' operator is not supported for booleans, use 'bitwise_xor' instead.`
    );
  }

  // Integer types can't have NaN — delegate to regular quantile (which has WASM)
  if (!isFloatDType(storage.dtype)) {
    return quantile(storage, q, axis, keepdims);
  }

  if (q < 0 || q > 1) {
    throw new Error('Quantile must be between 0 and 1');
  }

  const shape = storage.shape;
  const ndim = shape.length;
  const data = storage.data;
  const off = storage.offset;
  const inputStrides = storage.strides;

  if (axis === undefined) {
    const values: number[] = [];
    const contiguous = storage.isCContiguous;
    if (contiguous) {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(data[off + i]);
        if (!isNaN(val)) {
          values.push(val);
        }
      }
    } else {
      for (let i = 0; i < storage.size; i++) {
        const val = Number(storage.iget(i));
        if (!isNaN(val)) {
          values.push(val);
        }
      }
    }

    if (values.length === 0) {
      return NaN;
    }

    values.sort((a, b) => a - b);
    const n = values.length;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      return values[lower]!;
    }
    const frac = idx - lower;
    return values[lower]! * (1 - frac) + values[upper]! * frac;
  }

  let normalizedAxis = axis;
  if (normalizedAxis < 0) {
    normalizedAxis = ndim + normalizedAxis;
  }
  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const outputShape = Array.from(shape).filter((_, i) => i !== normalizedAxis);
  if (outputShape.length === 0) {
    const scalar = nanquantile(storage, q) as number;
    if (!keepdims) return scalar;
    return wrapScalarKeepdims(scalar, ndim, 'float64');
  }

  const outerSize = outputShape.reduce((a, b) => a * b, 1);
  const axisSize = shape[normalizedAxis]!;
  const result = ArrayStorage.empty(outputShape, 'float64');
  const resultData = result.data as Float64Array;

  const { baseOffsets, axisStride: axisStr } = precomputeAxisOffsets(
    shape,
    inputStrides,
    off,
    normalizedAxis,
    outerSize
  );

  for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
    const values: number[] = [];
    let bufIdx = baseOffsets[outerIdx]!;
    for (let axisIdx = 0; axisIdx < axisSize; axisIdx++) {
      const val = Number(data[bufIdx]);
      if (!isNaN(val)) {
        values.push(val);
      }
      bufIdx += axisStr;
    }

    if (values.length === 0) {
      resultData[outerIdx] = NaN;
      continue;
    }

    values.sort((a, b) => a - b);
    const n = values.length;
    const idx = q * (n - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);

    if (lower === upper) {
      resultData[outerIdx] = values[lower]!;
    } else {
      const frac = idx - lower;
      resultData[outerIdx] = values[lower]! * (1 - frac) + values[upper]! * frac;
    }
  }

  if (keepdims) {
    const keepdimsShape = [...shape];
    keepdimsShape[normalizedAxis] = 1;
    return ArrayStorage.fromData(resultData, keepdimsShape, 'float64');
  }

  return result;
}

/**
 * Compute the q-th percentile of data along specified axis, ignoring NaNs
 */
export function nanpercentile(
  storage: ArrayStorage,
  q: number,
  axis?: number,
  keepdims: boolean = false
): ArrayStorage | number {
  return nanquantile(storage, q / 100, axis, keepdims);
}
