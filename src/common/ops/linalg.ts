/**
 * Linear algebra operations
 *
 * Pure functions for matrix operations (matmul, etc.).
 * @module ops/linalg
 */

import { ArrayStorage } from '../storage';
import {
  promoteDTypes,
  isComplexDType,
  isBigIntDType,
  hasFloat16,
  type DType,
  type TypedArray,
} from '../dtype';
import { Complex } from '../complex';
import { conj as conjStorage } from './complex';
import { wasmMatmul } from '../wasm/matmul';
import { wasmSvdValues } from '../wasm/svd';
import { wasmVectorNorm2 } from '../wasm/vector_norm';
import { wasmInner } from '../wasm/inner';
import { wasmDot1D } from '../wasm/dot';
import { wasmMatvec } from '../wasm/matvec';
import { wasmVecmat } from '../wasm/vecmat';
import { wasmLuFactor, wasmLuInv, wasmLuSolve } from '../wasm/lu';
import { wasmOuter } from '../wasm/outer';
import { wasmVecdot } from '../wasm/vecdot';
import { wasmVdotComplex } from '../wasm/vdot';
import { wasmKron } from '../wasm/kron';
import { wasmCross } from '../wasm/cross';
import { wasmQr } from '../wasm/qr';
import { wasmCholesky, wasmCholeskyF32 } from '../wasm/cholesky';
import { wasmSvd } from '../wasm/svd';
import * as shapeOps from './shape';

/** Match NumPy: reject float16 for linalg decomposition/solve ops. */
function throwIfFloat16(dtype: DType): void {
  if (dtype === 'float16') {
    throw new TypeError(`array type float16 is unsupported in linalg`);
  }
}

// 1-element typed-array accumulators for wrapping integer arithmetic.
// Writing to acc[0] implicitly truncates/wraps to the dtype's range,
// matching NumPy's native accumulation behavior.
const _i32acc = new Int32Array(1);
const _u32acc = new Uint32Array(1);
const _i16acc = new Int16Array(1);
const _u16acc = new Uint16Array(1);
const _i8acc = new Int8Array(1);
const _u8acc = new Uint8Array(1);

type IntAcc = Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;

const _intAccMap: Partial<Record<DType, IntAcc>> = {
  int32: _i32acc,
  uint32: _u32acc,
  int16: _i16acc,
  uint16: _u16acc,
  int8: _i8acc,
  uint8: _u8acc,
};

/** Returns a 1-element wrapping accumulator for narrow int dtypes, or null for float/bigint. */
function getIntAcc(dtype: DType): IntAcc | null {
  return _intAccMap[dtype] ?? null;
}

/**
 * Get the absolute value of a value that may be Complex.
 * For complex: |a+bi| = sqrt(a²+b²)
 * For real: Math.abs(val)
 */
function absValue(val: number | bigint | Complex): number {
  if (val instanceof Complex) {
    return val.abs();
  }
  return Math.abs(Number(val));
}

/**
 * Extract the real part from a value. For Complex, returns .re. For others, Number().
 * Use this instead of Number() when converting values that might be Complex to real numbers.
 */
function realPart(val: number | bigint | Complex): number {
  if (val instanceof Complex) {
    return val.re;
  }
  return Number(val);
}

/**
 * Helper to multiply two values that may be Complex
 * Returns Complex if either input is Complex, number otherwise
 */
function multiplyValues(
  a: number | bigint | Complex,
  b: number | bigint | Complex
): number | bigint | Complex {
  if (a instanceof Complex || b instanceof Complex) {
    const aComplex = a instanceof Complex ? a : new Complex(Number(a), 0);
    const bComplex = b instanceof Complex ? b : new Complex(Number(b), 0);
    return aComplex.mul(bComplex);
  }
  if (typeof a === 'bigint' && typeof b === 'bigint') {
    return a * b;
  }
  return Number(a) * Number(b);
}

/**
 * Hot loop for dot() general ND case: non-complex, contiguous numeric arrays.
 * Extracted into a small function so V8 TurboFan can optimize it independently.
 * @internal
 */
function dotContiguousNumeric(
  aData: TypedArray,
  aOff: number,
  aOuterSize: number,
  bData: TypedArray,
  bOff: number,
  bOuterSize: number,
  bLastDim: number,
  contractionDim: number,
  resultData: TypedArray
): void {
  for (let i = 0; i < aOuterSize; i++) {
    for (let j = 0; j < bOuterSize; j++) {
      for (let k = 0; k < bLastDim; k++) {
        let sum = 0;
        for (let m = 0; m < contractionDim; m++) {
          sum +=
            (aData[aOff + i * contractionDim + m] as number) *
            (bData[bOff + j * contractionDim * bLastDim + m * bLastDim + k] as number);
        }
        resultData[i * bOuterSize * bLastDim + j * bLastDim + k] = sum;
      }
    }
  }
}

/**
 * Hot loop for inner() general case: non-complex, contiguous numeric arrays.
 * Extracted into a small function so V8 TurboFan can optimize it independently.
 * @internal
 */
function innerContiguousNumeric(
  aData: TypedArray,
  aOff: number,
  aOuterSize: number,
  aDim: number,
  bData: TypedArray,
  bOff: number,
  bOuterSize: number,
  bDim: number,
  contractionDim: number,
  resultData: TypedArray
): void {
  for (let i = 0; i < aOuterSize; i++) {
    for (let j = 0; j < bOuterSize; j++) {
      let sum = 0;
      for (let k = 0; k < contractionDim; k++) {
        const aFlatIdx = aDim === 1 ? k : i * contractionDim + k;
        const bFlatIdx = bDim === 1 ? k : j * contractionDim + k;
        sum += (aData[aOff + aFlatIdx] as number) * (bData[bOff + bFlatIdx] as number);
      }
      const idx = aOuterSize === 1 ? j : i * bOuterSize + j;
      resultData[idx] = sum;
    }
  }
}

/**
 * BLAS-like types for matrix operations
 */
type Transpose = 'no-transpose' | 'transpose';

/**
 * Double-precision general matrix multiply (DGEMM)
 *
 * Performs: C = alpha * op(A) * op(B)
 *
 * Row-major layout only. Supports transpose and no-transpose operations.
 * Uses specialized loops for each transpose case to avoid function call overhead.
 *
 * @internal
 */
function dgemm(
  transA: Transpose,
  transB: Transpose,
  M: number, // rows of op(A) and C
  N: number, // cols of op(B) and C
  K: number, // cols of op(A) and rows of op(B)
  alpha: number, // scalar alpha
  A: Float64Array, // matrix A
  lda: number, // leading dimension of A
  B: Float64Array, // matrix B
  ldb: number, // leading dimension of B
  C: Float64Array, // matrix C (output)
  ldc: number // leading dimension of C
): void {
  // Zero out result matrix
  for (let i = 0; i < M * N; i++) {
    C[i] = 0;
  }

  // Select specialized loop based on transpose modes
  const transposeA = transA === 'transpose';
  const transposeB = transB === 'transpose';

  if (!transposeA && !transposeB) {
    // No transpose (most common case)
    // C[i,j] = sum_k A[i,k] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = alpha * sum;
      }
    }
  } else if (transposeA && !transposeB) {
    // A transposed
    // C[i,j] = sum_k A[k,i] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = alpha * sum;
      }
    }
  } else if (!transposeA && transposeB) {
    // B transposed
    // C[i,j] = sum_k A[i,k] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = alpha * sum;
      }
    }
  } else {
    // Both transposed
    // C[i,j] = sum_k A[k,i] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = alpha * sum;
      }
    }
  }
}

/**
 * Dot product of two arrays (fully NumPy-compatible)
 *
 * Behavior depends on input dimensions:
 * - 0D · 0D: Multiply scalars → scalar
 * - 0D · ND or ND · 0D: Element-wise multiply → ND
 * - 1D · 1D: Inner product → scalar
 * - 2D · 2D: Matrix multiplication → 2D
 * - 2D · 1D: Matrix-vector product → 1D
 * - 1D · 2D: Vector-matrix product → 1D
 * - ND · 1D (N≥2): Sum product over last axis of a → (N-1)D
 * - 1D · ND (N≥2): Sum product over first axis of b → (N-1)D
 * - ND · MD (N,M≥2): Sum product over last axis of a and second-to-last of b → (N+M-2)D
 *
 * For 2D·2D, prefer using matmul() instead.
 */
export function dot(a: ArrayStorage, b: ArrayStorage): ArrayStorage | number | bigint | Complex {
  const aDim = a.ndim;
  const bDim = b.ndim;
  const isComplex = isComplexDType(a.dtype) || isComplexDType(b.dtype);

  // Case 0: Scalar (0D) cases - treat as multiplication
  if (aDim === 0 || bDim === 0) {
    // Get scalar values
    const aVal = aDim === 0 ? a.get() : null;
    const bVal = bDim === 0 ? b.get() : null;

    if (aDim === 0 && bDim === 0) {
      // Both scalars: multiply them
      return multiplyValues(aVal!, bVal!);
    } else if (aDim === 0) {
      // a is scalar, b is array: scalar * array (element-wise)
      const resultDtype = promoteDTypes(a.dtype, b.dtype);
      const result = ArrayStorage.zeros([...b.shape], resultDtype);
      // Helper to convert flat index to multi-dimensional indices
      const flatToIndices = (flatIdx: number, shape: readonly number[]): number[] => {
        const indices: number[] = new Array(shape.length);
        let remaining = flatIdx;
        for (let d = shape.length - 1; d >= 0; d--) {
          indices[d] = remaining % shape[d]!;
          remaining = Math.floor(remaining / shape[d]!);
        }
        return indices;
      };
      for (let i = 0; i < b.size; i++) {
        const indices = flatToIndices(i, b.shape);
        const bData = b.get(...indices);
        result.set(indices, multiplyValues(aVal!, bData));
      }
      return result;
    } else {
      // b is scalar, a is array: array * scalar (element-wise)
      const resultDtype = promoteDTypes(a.dtype, b.dtype);
      const result = ArrayStorage.zeros([...a.shape], resultDtype);
      // Helper to convert flat index to multi-dimensional indices
      const flatToIndices = (flatIdx: number, shape: readonly number[]): number[] => {
        const indices: number[] = new Array(shape.length);
        let remaining = flatIdx;
        for (let d = shape.length - 1; d >= 0; d--) {
          indices[d] = remaining % shape[d]!;
          remaining = Math.floor(remaining / shape[d]!);
        }
        return indices;
      };
      for (let i = 0; i < a.size; i++) {
        const indices = flatToIndices(i, a.shape);
        const aData = a.get(...indices);
        result.set(indices, multiplyValues(aData, bVal!));
      }
      return result;
    }
  }

  // Case 1: Both 1D -> scalar (inner product)
  if (aDim === 1 && bDim === 1) {
    if (a.shape[0] !== b.shape[0]) {
      throw new Error(`dot: incompatible shapes (${a.shape[0]},) and (${b.shape[0]},)`);
    }

    // Try WASM-accelerated 1D dot
    const wasmDotResult = wasmDot1D(a, b);
    if (wasmDotResult !== null) return wasmDotResult;

    const n = a.shape[0]!;

    if (isComplex) {
      let sumRe = 0;
      let sumIm = 0;
      for (let i = 0; i < n; i++) {
        const aVal = a.get(i);
        const bVal = b.get(i);
        const prod = multiplyValues(aVal, bVal);
        if (prod instanceof Complex) {
          sumRe += prod.re;
          sumIm += prod.im;
        } else {
          sumRe += Number(prod);
        }
      }
      return new Complex(sumRe, sumIm);
    }

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    // BigInt accumulation for int64/uint64
    if (isBigIntDType(resultDtype)) {
      let sum = BigInt(0);
      for (let i = 0; i < n; i++) {
        sum += BigInt(a.get(i) as number | bigint) * BigInt(b.get(i) as number | bigint);
      }
      return sum;
    }
    const acc = getIntAcc(resultDtype);
    if (acc) {
      acc[0] = 0;
      for (let i = 0; i < n; i++) {
        acc[0] += Number(a.get(i)) * Number(b.get(i));
      }
      return acc[0]!;
    }
    // Accumulate in the result dtype's precision (float16/float32/float64)
    if (resultDtype === 'float16' && hasFloat16) {
      const f16 = new Float16Array(1);
      f16[0] = 0;
      for (let i = 0; i < n; i++) {
        f16[0] += Number(a.get(i)) * Number(b.get(i));
      }
      return Number(f16[0]!);
    }
    if (resultDtype === 'float32') {
      const f32 = new Float32Array(1);
      f32[0] = 0;
      for (let i = 0; i < n; i++) {
        f32[0] += Number(a.get(i)) * Number(b.get(i));
      }
      return f32[0]!;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const aVal = a.get(i);
      const bVal = b.get(i);
      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }
    return sum;
  }

  // Case 2: Both 2D -> matrix multiplication (delegate to matmul)
  if (aDim === 2 && bDim === 2) {
    return matmul(a, b);
  }

  // Case 3: 2D · 1D -> matrix-vector product (returns 1D)
  if (aDim === 2 && bDim === 1) {
    const [m, k] = a.shape;
    const n = b.shape[0]!;
    if (k !== n) {
      throw new Error(`dot: incompatible shapes (${m},${k}) and (${n},)`);
    }

    // Try WASM-accelerated matvec
    const wasmMvResult = wasmMatvec(a, b);
    if (wasmMvResult) return wasmMvResult;

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([m!], resultDtype);

    if (isComplex) {
      for (let i = 0; i < m!; i++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let j = 0; j < k!; j++) {
          const aVal = a.get(i, j);
          const bVal = b.get(j);
          const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
          const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
          sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
          sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
        }
        result.set([i], new Complex(sumRe, sumIm));
      }
    } else if (isBigIntDType(resultDtype)) {
      for (let i = 0; i < m!; i++) {
        let sum = 0n;
        for (let j = 0; j < k!; j++) {
          sum += BigInt(a.get(i, j) as number | bigint) * BigInt(b.get(j) as number | bigint);
        }
        result.set([i], sum);
      }
    } else {
      const dotAcc = getIntAcc(resultDtype);
      for (let i = 0; i < m!; i++) {
        if (dotAcc) {
          dotAcc[0] = 0;
          for (let j = 0; j < k!; j++) {
            dotAcc[0] += Number(a.get(i, j)) * Number(b.get(j));
          }
          result.set([i], dotAcc[0]!);
        } else {
          let sum = 0;
          for (let j = 0; j < k!; j++) {
            sum += Number(a.get(i, j)) * Number(b.get(j));
          }
          result.set([i], sum);
        }
      }
    }

    return result;
  }

  // Case 4: 1D · 2D -> vector-matrix product (returns 1D)
  if (aDim === 1 && bDim === 2) {
    const m = a.shape[0]!;
    const [k, n] = b.shape;
    if (m !== k) {
      throw new Error(`dot: incompatible shapes (${m},) and (${k},${n})`);
    }

    // Try WASM-accelerated vecmat
    const wasmVmResult = wasmVecmat(a, b);
    if (wasmVmResult) return wasmVmResult;

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([n!], resultDtype);

    if (isComplex) {
      for (let j = 0; j < n!; j++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let i = 0; i < m; i++) {
          const aVal = a.get(i);
          const bVal = b.get(i, j);
          const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
          const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
          sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
          sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
        }
        result.set([j], new Complex(sumRe, sumIm));
      }
    } else if (isBigIntDType(resultDtype)) {
      for (let j = 0; j < n!; j++) {
        let sum = 0n;
        for (let i = 0; i < m; i++) {
          sum += BigInt(a.get(i) as number | bigint) * BigInt(b.get(i, j) as number | bigint);
        }
        result.set([j], sum);
      }
    } else {
      const dotAcc2 = getIntAcc(resultDtype);
      for (let j = 0; j < n!; j++) {
        if (dotAcc2) {
          dotAcc2[0] = 0;
          for (let i = 0; i < m; i++) {
            dotAcc2[0] += Number(a.get(i)) * Number(b.get(i, j));
          }
          result.set([j], dotAcc2[0]!);
        } else {
          let sum = 0;
          for (let i = 0; i < m; i++) {
            sum += Number(a.get(i)) * Number(b.get(i, j));
          }
          result.set([j], sum);
        }
      }
    }

    return result;
  }

  // Case 5: ND · 1D (N > 2) -> sum product over last axis, result is (N-1)D
  if (aDim > 2 && bDim === 1) {
    const lastDimA = a.shape[aDim - 1]!;
    const bSize = b.shape[0]!;
    if (lastDimA !== bSize) {
      throw new Error(`dot: incompatible shapes ${JSON.stringify(a.shape)} and (${bSize},)`);
    }

    const resultShape = [...a.shape.slice(0, -1)];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    if (isComplex) {
      for (let i = 0; i < resultSize; i++) {
        let sumRe = 0;
        let sumIm = 0;
        let temp = i;
        const resultIdx: number[] = [];
        for (let d = resultShape.length - 1; d >= 0; d--) {
          resultIdx[d] = temp % resultShape[d]!;
          temp = Math.floor(temp / resultShape[d]!);
        }

        for (let k = 0; k < lastDimA; k++) {
          const aIdx = [...resultIdx, k];
          const aVal = a.get(...aIdx);
          const bVal = b.get(k);
          const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
          const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
          sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
          sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
        }
        result.set(resultIdx, new Complex(sumRe, sumIm));
      }
    } else {
      const ndDot1DAcc = getIntAcc(resultDtype);
      for (let i = 0; i < resultSize; i++) {
        let temp = i;
        const resultIdx: number[] = [];
        for (let d = resultShape.length - 1; d >= 0; d--) {
          resultIdx[d] = temp % resultShape[d]!;
          temp = Math.floor(temp / resultShape[d]!);
        }

        if (ndDot1DAcc) {
          ndDot1DAcc[0] = 0;
          for (let k = 0; k < lastDimA; k++) {
            const aIdx = [...resultIdx, k];
            ndDot1DAcc[0] += Number(a.get(...aIdx)) * Number(b.get(k));
          }
          result.set(resultIdx, ndDot1DAcc[0]!);
        } else {
          let sum = 0;
          for (let k = 0; k < lastDimA; k++) {
            const aIdx = [...resultIdx, k];
            const aVal = a.get(...aIdx);
            const bVal = b.get(k);
            if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
              sum = Number(sum) + Number(aVal * bVal);
            } else {
              sum += Number(aVal) * Number(bVal);
            }
          }
          result.set(resultIdx, sum);
        }
      }
    }

    return result;
  }

  // Case 6: 1D · ND (N > 2) -> sum product over SECOND axis of b
  if (aDim === 1 && bDim > 2) {
    const aSize = a.shape[0]!;
    const contractAxisB = 1;
    const contractDimB = b.shape[contractAxisB]!;

    if (aSize !== contractDimB) {
      throw new Error(`dot: incompatible shapes (${aSize},) and ${JSON.stringify(b.shape)}`);
    }

    const resultShape = [...b.shape.slice(0, contractAxisB), ...b.shape.slice(contractAxisB + 1)];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
    if (isComplex) {
      for (let i = 0; i < resultSize; i++) {
        let temp = i;
        const resultIdx: number[] = [];
        for (let d = resultShape.length - 1; d >= 0; d--) {
          resultIdx[d] = temp % resultShape[d]!;
          temp = Math.floor(temp / resultShape[d]!);
        }

        const bIdxBefore = resultIdx.slice(0, contractAxisB);
        const bIdxAfter = resultIdx.slice(contractAxisB);

        let sumRe = 0;
        let sumIm = 0;
        for (let k = 0; k < aSize; k++) {
          const aVal = a.get(k);
          const bIdx = [...bIdxBefore, k, ...bIdxAfter];
          const bVal = b.get(...bIdx);
          const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
          const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
          sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
          sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
        }
        result.set(resultIdx, new Complex(sumRe, sumIm));
      }
    } else {
      const dot1dNdAcc = getIntAcc(resultDtype);
      for (let i = 0; i < resultSize; i++) {
        let temp = i;
        const resultIdx: number[] = [];
        for (let d = resultShape.length - 1; d >= 0; d--) {
          resultIdx[d] = temp % resultShape[d]!;
          temp = Math.floor(temp / resultShape[d]!);
        }

        const bIdxBefore = resultIdx.slice(0, contractAxisB);
        const bIdxAfter = resultIdx.slice(contractAxisB);

        if (dot1dNdAcc) {
          dot1dNdAcc[0] = 0;
          for (let k = 0; k < aSize; k++) {
            const bIdx = [...bIdxBefore, k, ...bIdxAfter];
            dot1dNdAcc[0] += Number(a.get(k)) * Number(b.get(...bIdx));
          }
          result.set(resultIdx, dot1dNdAcc[0]!);
        } else {
          let sum = 0;
          for (let k = 0; k < aSize; k++) {
            const aVal = a.get(k);
            const bIdx = [...bIdxBefore, k, ...bIdxAfter];
            const bVal = b.get(...bIdx);
            if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
              sum = Number(sum) + Number(aVal * bVal);
            } else {
              sum += Number(aVal) * Number(bVal);
            }
          }
          result.set(resultIdx, sum);
        }
      }
    }

    return result;
  }

  // Case 7: ND · MD (N,M ≥ 2, not both 2) -> general tensor contraction
  if (aDim >= 2 && bDim >= 2 && !(aDim === 2 && bDim === 2)) {
    const lastDimA = a.shape[aDim - 1]!;
    const secondLastDimB = b.shape[bDim - 2]!;

    if (lastDimA !== secondLastDimB) {
      throw new Error(
        `dot: incompatible shapes ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)}`
      );
    }

    const resultShape = [...a.shape.slice(0, -1), ...b.shape.slice(0, -2), b.shape[bDim - 1]!];
    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros(resultShape, resultDtype);

    const aOuterSize = a.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
    const bOuterSize = b.shape.slice(0, -2).reduce((acc, dim) => acc * dim, 1);
    const bLastDim = b.shape[bDim - 1]!;
    const contractionDim = lastDimA;

    if (isComplex) {
      for (let i = 0; i < aOuterSize; i++) {
        for (let j = 0; j < bOuterSize; j++) {
          for (let k = 0; k < bLastDim; k++) {
            let sumRe = 0;
            let sumIm = 0;
            for (let m = 0; m < contractionDim; m++) {
              // Use get with multi-dim indices for proper Complex extraction
              const aMultiIdx: number[] = [];
              let tempA = i;
              for (let d = a.shape.length - 2; d >= 0; d--) {
                aMultiIdx.unshift(tempA % a.shape[d]!);
                tempA = Math.floor(tempA / a.shape[d]!);
              }
              aMultiIdx.push(m);
              const aVal = a.get(...aMultiIdx);

              const bMultiIdx: number[] = [];
              let tempB = j;
              for (let d = b.shape.length - 3; d >= 0; d--) {
                bMultiIdx.unshift(tempB % b.shape[d]!);
                tempB = Math.floor(tempB / b.shape[d]!);
              }
              bMultiIdx.push(m, k);
              const bVal = b.get(...bMultiIdx);

              const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
              const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
              sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
              sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
            }

            const resultIdx = i * bOuterSize * bLastDim + j * bLastDim + k;
            const resultData = result.data as Float64Array;
            resultData[resultIdx * 2] = sumRe;
            resultData[resultIdx * 2 + 1] = sumIm;
          }
        }
      }
    } else {
      if (
        a.isCContiguous &&
        b.isCContiguous &&
        !isBigIntDType(a.dtype) &&
        !isBigIntDType(b.dtype) &&
        !getIntAcc(resultDtype)
      ) {
        // Fast path: contiguous numeric arrays - extracted for V8 optimization
        // Excludes narrow int types (int8/int16 etc.) which need wrapping accumulators
        dotContiguousNumeric(
          a.data,
          a.offset,
          aOuterSize,
          b.data,
          b.offset,
          bOuterSize,
          bLastDim,
          contractionDim,
          result.data
        );
      } else {
        // General fallback: non-contiguous or bigint arrays
        const ndMdAcc = getIntAcc(resultDtype);
        for (let i = 0; i < aOuterSize; i++) {
          for (let j = 0; j < bOuterSize; j++) {
            for (let k = 0; k < bLastDim; k++) {
              const resultIdx = i * bOuterSize * bLastDim + j * bLastDim + k;
              if (ndMdAcc) {
                ndMdAcc[0] = 0;
                for (let m = 0; m < contractionDim; m++) {
                  const aFlatIdx = i * contractionDim + m;
                  const bFlatIdx = j * contractionDim * bLastDim + m * bLastDim + k;
                  ndMdAcc[0] += Number(a.iget(aFlatIdx)) * Number(b.iget(bFlatIdx));
                }
                result.data[resultIdx] = ndMdAcc[0]!;
              } else {
                let sum = 0;
                for (let m = 0; m < contractionDim; m++) {
                  const aFlatIdx = i * contractionDim + m;
                  const bFlatIdx = j * contractionDim * bLastDim + m * bLastDim + k;
                  const aVal = a.iget(aFlatIdx);
                  const bVal = b.iget(bFlatIdx);

                  if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
                    sum = Number(sum) + Number(aVal * bVal);
                  } else {
                    sum += Number(aVal) * Number(bVal);
                  }
                }
                result.data[resultIdx] = sum;
              }
            }
          }
        }
      }
    }

    return result;
  }

  // Should never reach here - all cases covered
  throw new Error(`dot: unexpected combination of dimensions ${aDim}D · ${bDim}D`);
}

/**
 * Core 2D-only matrix multiplication (optimized with DGEMM).
 * Called by the public matmul() after handling 1D promotion and ND batching.
 * @internal
 */
function matmul2D(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const [m = 0, k = 0] = a.shape;
  const [k2 = 0, n = 0] = b.shape;

  if (k !== k2) {
    throw new Error(`matmul shape mismatch: (${m},${k}) @ (${k2},${n})`);
  }

  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  if (isComplexDType(resultDtype)) {
    const aIsComplex = isComplexDType(a.dtype);
    const bIsComplex = isComplexDType(b.dtype);
    const result = ArrayStorage.zeros([m, n], resultDtype);
    const resultData = result.data as Float64Array;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let l = 0; l < k; l++) {
          const aRaw = a.iget(i * k + l);
          const bRaw = b.iget(l * n + j);
          const aRe = aIsComplex ? (aRaw as Complex).re : Number(aRaw);
          const aIm = aIsComplex ? (aRaw as Complex).im : 0;
          const bRe = bIsComplex ? (bRaw as Complex).re : Number(bRaw);
          const bIm = bIsComplex ? (bRaw as Complex).im : 0;
          sumRe += aRe * bRe - aIm * bIm;
          sumIm += aRe * bIm + aIm * bRe;
        }
        const idx = i * n + j;
        resultData[idx * 2] = sumRe;
        resultData[idx * 2 + 1] = sumIm;
      }
    }
    return result;
  }

  // Integer matmul: compute directly in the target integer type (wraps on overflow, like NumPy)
  if (resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool') {
    const result = ArrayStorage.zeros([m, n], resultDtype);
    const rData = result.data;
    const aOff = a.offset;
    const bOff = b.offset;
    const [aStrideR = 0, aStrideC = 0] = a.strides;
    const [bStrideR = 0, bStrideC = 0] = b.strides;
    if (isBigIntDType(resultDtype)) {
      const aD = a.data as BigInt64Array | BigUint64Array;
      const bD = b.data as BigInt64Array | BigUint64Array;
      const rD = rData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0n;
          for (let l = 0; l < k; l++) {
            sum +=
              aD[aOff + i * aStrideR + l * aStrideC]! * bD[bOff + l * bStrideR + j * bStrideC]!;
          }
          rD[i * n + j] = sum;
        }
      }
    } else {
      const aD = a.data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      const bD = b.data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      const rD = rData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let l = 0; l < k; l++) {
            sum +=
              aD[aOff + i * aStrideR + l * aStrideC]! * bD[bOff + l * bStrideR + j * bStrideC]!;
          }
          // Assigning to a TypedArray truncates/wraps automatically
          rD[i * n + j] = sum;
        }
      }
    }
    return result;
  }

  const outputDtype = resultDtype;

  if (outputDtype !== 'float64' && outputDtype !== 'float32' && outputDtype !== 'float16') {
    throw new Error(`matmul currently only supports float64/float32/float16, got ${outputDtype}`);
  }

  const toF64 = (storage: typeof a): Float64Array => {
    if (storage.dtype === 'float64') return storage.data as Float64Array;
    if (storage.dtype === 'float32' || storage.dtype === 'float16')
      return Float64Array.from(storage.data as Float32Array);
    return Float64Array.from(Array.from(storage.data as ArrayLike<number>).map(Number));
  };
  let aData = toF64(a);
  let bData = toF64(b);

  if (a.offset > 0) {
    aData = aData.subarray(a.offset) as Float64Array;
  }
  if (b.offset > 0) {
    bData = bData.subarray(b.offset) as Float64Array;
  }

  const [aStrideRow = 0, aStrideCol = 0] = a.strides;
  const [bStrideRow = 0, bStrideCol = 0] = b.strides;

  const aIsTransposed = aStrideCol > aStrideRow;
  const bIsTransposed = bStrideCol > bStrideRow;

  const transA: Transpose = aIsTransposed ? 'transpose' : 'no-transpose';
  const transB: Transpose = bIsTransposed ? 'transpose' : 'no-transpose';

  let lda: number;
  let ldb: number;

  if (aIsTransposed) {
    lda = aStrideCol;
  } else {
    lda = aStrideRow;
  }

  if (bIsTransposed) {
    ldb = bStrideCol;
  } else {
    ldb = bStrideRow;
  }

  const result = ArrayStorage.zeros([m, n], 'float64');

  dgemm(transA, transB, m, n, k, 1.0, aData, lda, bData, ldb, result.data as Float64Array, n);

  if (outputDtype === 'float32' || outputDtype === 'float16') {
    const out = ArrayStorage.zeros([m, n], outputDtype);
    const src = result.data as Float64Array;
    const dst = out.data;
    for (let i = 0; i < src.length; i++) (dst as Float16Array)[i] = src[i]!;
    result.dispose();
    return out;
  }

  return result;
}

/**
 * Broadcast two batch-shape arrays together (NumPy rules).
 * @internal
 */
function broadcastBatchShapes(shapeA: number[], shapeB: number[]): number[] {
  const ndim = Math.max(shapeA.length, shapeB.length);
  const result: number[] = new Array(ndim);
  for (let i = 0; i < ndim; i++) {
    const ai = shapeA[shapeA.length - ndim + i] ?? 1;
    const bi = shapeB[shapeB.length - ndim + i] ?? 1;
    if (ai !== bi && ai !== 1 && bi !== 1) {
      throw new Error(
        `matmul: cannot broadcast batch shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)}`
      );
    }
    result[i] = Math.max(ai, bi);
  }
  return result;
}

/** Flat index → multi-index for given shape. @internal */
function flatToBatchMultiIndex(flatIdx: number, shape: number[]): number[] {
  const idx: number[] = new Array(shape.length);
  let remaining = flatIdx;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = remaining % shape[i]!;
    remaining = Math.floor(remaining / shape[i]!);
  }
  return idx;
}

/**
 * Multi-index → flat index, applying broadcast clamping (size-1 dims clamp to 0).
 * multiIdx is right-aligned to shape length.
 * @internal
 */
function batchMultiIndexToFlat(multiIdx: number[], shape: number[]): number {
  const ndim = shape.length;
  let flat = 0;
  for (let i = 0; i < ndim; i++) {
    const mi = multiIdx.length - ndim + i;
    const rawIdx = mi >= 0 ? multiIdx[mi]! : 0;
    const idx = shape[i] === 1 ? 0 : rawIdx;
    flat = flat * shape[i]! + idx;
  }
  return flat;
}

/**
 * Ensure array data is a contiguous float64 buffer with no offset.
 * @internal
 */
function toContiguousFloat64(a: ArrayStorage): Float64Array {
  if (a.isCContiguous && a.offset === 0 && a.dtype === 'float64') {
    return a.data as Float64Array;
  }
  const result = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) result[i] = Number(a.iget(i));
  return result;
}

/**
 * Extract a contiguous 2D slice from an ND array at a given batch index.
 * @internal
 */
function extract2DSlice(
  a: ArrayStorage,
  batchIdx: number,
  rows: number,
  cols: number
): ArrayStorage {
  const ndim = a.ndim;
  const sliceSize = rows * cols;
  const isComplex = isComplexDType(a.dtype);
  const isBigInt = isBigIntDType(a.dtype);
  const factor = isComplex ? 2 : 1;
  const result = ArrayStorage.empty([rows, cols], a.dtype);
  const sliceData = result.data;

  // Compute offset into the flat contiguous data for this batch
  if (a.isCContiguous) {
    const srcOff = (a.offset + batchIdx * sliceSize) * factor;
    if (isBigInt) {
      const src = a.data as BigInt64Array | BigUint64Array;
      const dst = sliceData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < sliceSize * factor; i++) dst[i] = src[srcOff + i]!;
    } else {
      const src = a.data as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      const dst = sliceData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < sliceSize * factor; i++) dst[i] = src[srcOff + i]!;
    }
  } else {
    // Non-contiguous: use iget for each element
    const baseLinear = batchIdx * sliceSize;
    for (let i = 0; i < sliceSize; i++) {
      // Compute multi-index from the flattened batch + 2D position
      const linearIdx = baseLinear + i;
      // For non-contiguous, we need to map through strides
      let remaining = linearIdx;
      let physIdx = a.offset;
      for (let d = ndim - 1; d >= 0; d--) {
        const dimSize = a.shape[d]!;
        physIdx += (remaining % dimSize) * a.strides[d]!;
        remaining = Math.floor(remaining / dimSize);
      }
      if (isComplex) {
        const src = a.data as Float64Array | Float32Array;
        const dst = sliceData as Float64Array | Float32Array;
        dst[i * 2] = src[physIdx * 2]!;
        dst[i * 2 + 1] = src[physIdx * 2 + 1]!;
      } else if (isBigInt) {
        (sliceData as BigInt64Array | BigUint64Array)[i] = (
          a.data as BigInt64Array | BigUint64Array
        )[physIdx]!;
      } else {
        (sliceData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = (
          a.data as Exclude<TypedArray, BigInt64Array | BigUint64Array>
        )[physIdx]!;
      }
    }
  }

  return result;
}

/**
 * Matrix multiplication - fully NumPy-compatible.
 *
 * Behavior by input dimensions:
 * - 0D: raises ValueError (at least 1-D required)
 * - 1D @ 1D: inner product → 0D scalar array
 * - 2D @ 1D: matrix-vector → 1D
 * - 1D @ 2D: vector-matrix → 1D
 * - 2D @ 2D: matrix multiplication → 2D (optimized with DGEMM)
 * - ND @ ND (N≥3): batched matrix multiply, batch dims broadcast → ND
 */
export function matmul(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  // Dispatch 1D cases to dedicated WASM kernels (faster than promoting to 2D matmul)
  if (a.ndim === 1 && b.ndim === 1) {
    // 1D · 1D → scalar (wrapped in 0D storage for matmul)
    const dotResult = wasmDot1D(a, b);
    if (dotResult !== null) {
      const dt = promoteDTypes(a.dtype, b.dtype);
      const scalar = ArrayStorage.zeros([], dt);
      if (dotResult instanceof Complex) {
        (scalar.data as Float64Array | Float32Array)[0] = dotResult.re;
        (scalar.data as Float64Array | Float32Array)[1] = dotResult.im;
      } else {
        scalar.data[0] = dotResult;
      }
      return scalar;
    }
  } else if (a.ndim >= 2 && b.ndim === 1) {
    // 2D · 1D → matvec (result shape removes last dim of a)
    if (a.ndim === 2) {
      const mvResult = wasmMatvec(a, b);
      if (mvResult) return mvResult;
    }
  } else if (a.ndim === 1 && b.ndim >= 2) {
    // 1D · 2D → vecmat (result shape removes first dim of b)
    if (b.ndim === 2) {
      const vmResult = wasmVecmat(a, b);
      if (vmResult) return vmResult;
    }
  }

  // Try WASM acceleration for 2D+ matmul (returns null if it can't handle this case)
  const wasmResult = wasmMatmul(a, b);
  if (wasmResult) return wasmResult;

  // JS fallback
  if (a.ndim === 0 || b.ndim === 0) {
    throw new Error(
      `matmul: Input operand does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires at least 1-D)`
    );
  }

  const aWas1D = a.ndim === 1;
  const bWas1D = b.ndim === 1;

  // Promote 1D inputs to 2D (1D a → row vector, 1D b → column vector)
  const aProc: ArrayStorage = aWas1D ? shapeOps.reshape(a, [1, a.shape[0]!]) : a;
  const bProc: ArrayStorage = bWas1D ? shapeOps.reshape(b, [b.shape[0]!, 1]) : b;

  const aNdim = aProc.ndim;
  const bNdim = bProc.ndim;
  const M = aProc.shape[aNdim - 2]!;
  const K = aProc.shape[aNdim - 1]!;
  const K2 = bProc.shape[bNdim - 2]!;
  const N = bProc.shape[bNdim - 1]!;

  if (K !== K2) {
    throw new Error(
      `matmul: shape mismatch: (...,${M},${K}) @ (...,${K2},${N}): inner dimensions must match`
    );
  }

  // Fast path: pure 2D×2D — use optimized DGEMM
  if (aNdim === 2 && bNdim === 2) {
    const result2D = matmul2D(aProc, bProc);
    if (aWas1D && bWas1D) return shapeOps.reshape(result2D, []); // → 0D
    if (aWas1D) return shapeOps.reshape(result2D, [N]); // (1,N) → (N,)
    if (bWas1D) return shapeOps.reshape(result2D, [M]); // (M,1) → (M,)
    return result2D;
  }

  // ND batched matrix multiplication — dispatch each 2D slice to matmul2D
  // which handles all dtypes (int, bigint, float, complex)
  const aBatch = Array.from(aProc.shape).slice(0, aNdim - 2);
  const bBatch = Array.from(bProc.shape).slice(0, bNdim - 2);
  const batchShape = broadcastBatchShapes(aBatch, bBatch);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

  const resultDtype = promoteDTypes(aProc.dtype, bProc.dtype);
  const slices: ArrayStorage[] = [];

  for (let bi = 0; bi < batchSize; bi++) {
    const batchIdx = flatToBatchMultiIndex(bi, batchShape);
    const aFlatBatch = batchMultiIndexToFlat(batchIdx, aBatch);
    const bFlatBatch = batchMultiIndexToFlat(batchIdx, bBatch);

    // Extract 2D slices
    const aSlice = extract2DSlice(aProc, aFlatBatch, M, K);
    const bSlice = extract2DSlice(bProc, bFlatBatch, K, N);
    slices.push(matmul2D(aSlice, bSlice));
  }

  // Concatenate all 2D results into ND output
  const elementsPerSlice = M * N;
  const isComplex = isComplexDType(resultDtype);
  const isBigInt = isBigIntDType(resultDtype);
  const factor = isComplex ? 2 : 1;
  const outShape = [...batchShape, M, N];
  const result = ArrayStorage.empty(outShape, resultDtype);
  const resultData = result.data;

  for (let bi = 0; bi < batchSize; bi++) {
    const slice = slices[bi]!;
    const srcData = slice.data;
    const dstOff = bi * elementsPerSlice * factor;
    if (isBigInt) {
      const src = srcData as BigInt64Array | BigUint64Array;
      const dst = resultData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < elementsPerSlice; i++) dst[dstOff + i] = src[i]!;
    } else {
      const src = srcData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      const dst = resultData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < elementsPerSlice * factor; i++) dst[dstOff + i] = src[i]!;
    }
  }

  if (aWas1D && bWas1D) return shapeOps.reshape(result, [...batchShape]);
  if (aWas1D) return shapeOps.reshape(result, [...batchShape, N]);
  if (bWas1D) return shapeOps.reshape(result, [...batchShape, M]);
  return result;
}

/**
 * Sum along the diagonal of a 2D array
 *
 * Computes the trace (sum of diagonal elements) of a matrix.
 * For non-square matrices, sums along the diagonal up to min(rows, cols).
 *
 * @param a - Input 2D array
 * @returns Sum of diagonal elements
 */
export function trace(
  a: ArrayStorage,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): ArrayStorage | number | bigint | Complex {
  if (a.ndim < 2) {
    throw new Error(`trace requires at least 2D array, got ${a.ndim}D`);
  }

  // For 2D arrays, return a scalar (original fast path)
  if (a.ndim === 2) {
    const ax1 = axis1 < 0 ? a.ndim + axis1 : axis1;
    const ax2 = axis2 < 0 ? a.ndim + axis2 : axis2;
    const rows = a.shape[ax1]!;
    const cols = a.shape[ax2]!;
    const diagLen = Math.min(rows, cols) - Math.max(0, offset);
    if (diagLen <= 0) return isComplexDType(a.dtype) ? new Complex(0, 0) : 0;

    if (isComplexDType(a.dtype)) {
      let sumRe = 0;
      let sumIm = 0;
      for (let i = 0; i < diagLen; i++) {
        const idx0 = offset >= 0 ? i : i - offset;
        const idx1 = offset >= 0 ? i + offset : i;
        const indices: number[] = [0, 0];
        indices[ax1] = idx0;
        indices[ax2] = idx1;
        const val = a.get(...indices) as Complex;
        sumRe += val.re;
        sumIm += val.im;
      }
      return new Complex(sumRe, sumIm);
    }

    // Float16/float32 accumulator for matching NumPy precision
    if (a.dtype === 'float16' && hasFloat16) {
      const f16 = new Float16Array(1);
      f16[0] = 0;
      for (let i = 0; i < diagLen; i++) {
        const idx0 = offset >= 0 ? i : i - offset;
        const idx1 = offset >= 0 ? i + offset : i;
        const indices: number[] = [0, 0];
        indices[ax1] = idx0;
        indices[ax2] = idx1;
        f16[0] += Number(a.get(...indices));
      }
      return Number(f16[0]!);
    }
    if (a.dtype === 'float32') {
      const f32 = new Float32Array(1);
      f32[0] = 0;
      for (let i = 0; i < diagLen; i++) {
        const idx0 = offset >= 0 ? i : i - offset;
        const idx1 = offset >= 0 ? i + offset : i;
        const indices: number[] = [0, 0];
        indices[ax1] = idx0;
        indices[ax2] = idx1;
        f32[0] += Number(a.get(...indices));
      }
      return f32[0]!;
    }
    let sum: number | bigint = 0;
    for (let i = 0; i < diagLen; i++) {
      const idx0 = offset >= 0 ? i : i - offset;
      const idx1 = offset >= 0 ? i + offset : i;
      const indices: number[] = [0, 0];
      indices[ax1] = idx0;
      indices[ax2] = idx1;
      const val = a.get(...indices);
      if (typeof val === 'bigint') {
        sum = (typeof sum === 'bigint' ? sum : BigInt(sum)) + val;
      } else {
        sum = (typeof sum === 'bigint' ? Number(sum) : sum) + (val as number);
      }
    }
    return sum;
  }

  // ND case (ndim > 2): compute trace along axis1/axis2 for each combination
  // of the remaining axes. Result shape = shape with axis1 and axis2 removed.
  const ndim = a.ndim;
  const ax1 = ((axis1 % ndim) + ndim) % ndim;
  const ax2 = ((axis2 % ndim) + ndim) % ndim;
  if (ax1 === ax2) throw new Error('trace: axis1 and axis2 must be different');

  const diagAxis1Size = a.shape[ax1]!;
  const diagAxis2Size = a.shape[ax2]!;
  const diagLen = Math.min(diagAxis1Size, diagAxis2Size) - Math.max(0, offset);

  // Output shape: all axes except ax1 and ax2
  const outShape = Array.from(a.shape).filter((_, i) => i !== ax1 && i !== ax2);
  const outSize = outShape.reduce((acc, d) => acc * d, 1);

  const result = ArrayStorage.zeros(outShape.length > 0 ? outShape : [1], a.dtype);

  if (diagLen <= 0) return result.shape.length === 0 ? 0 : result;

  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    // Convert outFlat to multi-index in outShape
    const outIdx: number[] = new Array(outShape.length);
    let rem = outFlat;
    for (let i = outShape.length - 1; i >= 0; i--) {
      outIdx[i] = rem % outShape[i]!;
      rem = Math.floor(rem / outShape[i]!);
    }

    // Build full index mapping: outIdx → input index (skip ax1, ax2)
    let traceSum: number | bigint = 0;
    let traceRe = 0;
    let traceIm = 0;
    const isComplex = isComplexDType(a.dtype);

    for (let d = 0; d < diagLen; d++) {
      const d0 = offset >= 0 ? d : d - offset;
      const d1 = offset >= 0 ? d + offset : d;

      // Reconstruct full index
      const fullIdx: number[] = new Array(ndim);
      let outPos = 0;
      for (let i = 0; i < ndim; i++) {
        if (i === ax1) {
          fullIdx[i] = d0;
        } else if (i === ax2) {
          fullIdx[i] = d1;
        } else {
          fullIdx[i] = outIdx[outPos++]!;
        }
      }
      const val = a.get(...fullIdx);
      if (isComplex) {
        traceRe += (val as Complex).re;
        traceIm += (val as Complex).im;
      } else if (typeof val === 'bigint') {
        traceSum = (typeof traceSum === 'bigint' ? traceSum : BigInt(traceSum)) + val;
      } else {
        traceSum = (typeof traceSum === 'bigint' ? Number(traceSum) : traceSum) + (val as number);
      }
    }

    if (isComplex) {
      result.iset(outFlat, new Complex(traceRe, traceIm));
    } else {
      result.iset(outFlat, typeof traceSum === 'bigint' ? Number(traceSum) : traceSum);
    }
  }

  return result;
}

/**
 * Permute the dimensions of an array
 *
 * Standalone version of NDArray.transpose() method.
 * Returns a view with axes permuted.
 *
 * @param a - Input array
 * @param axes - Optional permutation of axes (defaults to reverse order)
 * @returns Transposed view
 */
export function transpose(a: ArrayStorage, axes?: number[]): ArrayStorage {
  return shapeOps.transpose(a, axes);
}

/**
 * Inner product of two arrays
 *
 * Computes sum product over the LAST axes of both a and b.
 * - 1D · 1D: Same as dot (ordinary inner product) → scalar
 * - ND · MD: Contracts last dimension of each → (*a.shape[:-1], *b.shape[:-1])
 *
 * Different from dot: always uses last axis of BOTH arrays.
 *
 * @param a - First array
 * @param b - Second array
 * @returns Inner product result
 */
export function inner(a: ArrayStorage, b: ArrayStorage): ArrayStorage | number | bigint | Complex {
  const aDim = a.ndim;
  const bDim = b.ndim;
  const isComplex = isComplexDType(a.dtype) || isComplexDType(b.dtype);

  // 0D case: treat as scalar multiplication
  if (aDim === 0 || bDim === 0) {
    return dot(a, b);
  }

  // Last dimensions must match
  const aLastDim = a.shape[aDim - 1]!;
  const bLastDim = b.shape[bDim - 1]!;

  if (aLastDim !== bLastDim) {
    throw new Error(
      `inner: incompatible shapes - last dimensions ${aLastDim} and ${bLastDim} don't match`
    );
  }

  // Try WASM-accelerated path (handles 1D·1D scalar and ND·MD)
  const wasmResult = wasmInner(a, b);
  if (wasmResult !== null) {
    return wasmResult;
  }

  // Special case: both 1D -> scalar (JS fallback for small arrays / complex / unsupported dtype)
  if (aDim === 1 && bDim === 1) {
    return dot(a, b) as number | Complex;
  }

  // General case: result shape is a.shape[:-1] + b.shape[:-1]
  const resultShape = [...a.shape.slice(0, -1), ...b.shape.slice(0, -1)];
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros(resultShape, resultDtype);

  const aOuterSize = aDim === 1 ? 1 : a.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
  const bOuterSize = bDim === 1 ? 1 : b.shape.slice(0, -1).reduce((acc, dim) => acc * dim, 1);
  const contractionDim = aLastDim;

  // Compute: result[i, j] = sum_k a[i, k] * b[j, k]
  // Separate complex and non-complex paths for performance
  if (isComplex) {
    // Complex path
    for (let i = 0; i < aOuterSize; i++) {
      for (let j = 0; j < bOuterSize; j++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let k = 0; k < contractionDim; k++) {
          let aVal: number | bigint | Complex;
          let bVal: number | bigint | Complex;

          // Convert flat index to multi-dim for get()
          if (aDim === 1) {
            aVal = a.get(k);
          } else {
            const aMultiIdx: number[] = [];
            let tempA = i;
            const aOuterShape = a.shape.slice(0, -1);
            for (let d = aOuterShape.length - 1; d >= 0; d--) {
              aMultiIdx.unshift(tempA % aOuterShape[d]!);
              tempA = Math.floor(tempA / aOuterShape[d]!);
            }
            aMultiIdx.push(k);
            aVal = a.get(...aMultiIdx);
          }

          if (bDim === 1) {
            bVal = b.get(k);
          } else {
            const bMultiIdx: number[] = [];
            let tempB = j;
            const bOuterShape = b.shape.slice(0, -1);
            for (let d = bOuterShape.length - 1; d >= 0; d--) {
              bMultiIdx.unshift(tempB % bOuterShape[d]!);
              tempB = Math.floor(tempB / bOuterShape[d]!);
            }
            bMultiIdx.push(k);
            bVal = b.get(...bMultiIdx);
          }

          // Complex multiplication: (aRe + aIm*i) * (bRe + bIm*i)
          const aComplex = aVal instanceof Complex ? aVal : new Complex(Number(aVal), 0);
          const bComplex = bVal instanceof Complex ? bVal : new Complex(Number(bVal), 0);
          sumRe += aComplex.re * bComplex.re - aComplex.im * bComplex.im;
          sumIm += aComplex.re * bComplex.im + aComplex.im * bComplex.re;
        }

        // Set result
        if (resultShape.length === 0) {
          return new Complex(sumRe, sumIm);
        }
        const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
        const resultData = result.data as Float64Array;
        resultData[resultIdx * 2] = sumRe;
        resultData[resultIdx * 2 + 1] = sumIm;
      }
    }
  } else {
    // Non-complex fast path
    const innerAcc = getIntAcc(resultDtype);
    if (
      a.isCContiguous &&
      b.isCContiguous &&
      !isBigIntDType(a.dtype) &&
      !isBigIntDType(b.dtype) &&
      !innerAcc
    ) {
      // Fast path: contiguous numeric arrays - extracted for V8 optimization
      // Excludes narrow int types (int8/int16 etc.) which need wrapping accumulators
      if (resultShape.length === 0) {
        const aData = a.data;
        const bData = b.data;
        const aOff = a.offset;
        const bOff = b.offset;
        let sum = 0;
        for (let k = 0; k < contractionDim; k++) {
          sum += (aData[aOff + k] as number) * (bData[bOff + k] as number);
        }
        return sum;
      }
      innerContiguousNumeric(
        a.data,
        a.offset,
        aOuterSize,
        aDim,
        b.data,
        b.offset,
        bOuterSize,
        bDim,
        contractionDim,
        result.data
      );
    } else {
      // General fallback: non-contiguous or bigint arrays
      for (let i = 0; i < aOuterSize; i++) {
        for (let j = 0; j < bOuterSize; j++) {
          if (innerAcc) {
            innerAcc[0] = 0;
            for (let k = 0; k < contractionDim; k++) {
              const aFlatIdx = aDim === 1 ? k : i * contractionDim + k;
              const bFlatIdx = bDim === 1 ? k : j * contractionDim + k;
              innerAcc[0] += Number(a.iget(aFlatIdx)) * Number(b.iget(bFlatIdx));
            }
            if (resultShape.length === 0) {
              return innerAcc[0]!;
            }
            const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
            result.data[resultIdx] = innerAcc[0]!;
          } else if (isBigIntDType(resultDtype)) {
            let sum = 0n;
            for (let k = 0; k < contractionDim; k++) {
              const aFlatIdx = aDim === 1 ? k : i * contractionDim + k;
              const bFlatIdx = bDim === 1 ? k : j * contractionDim + k;
              sum += BigInt(a.iget(aFlatIdx) as bigint) * BigInt(b.iget(bFlatIdx) as bigint);
            }
            if (resultShape.length === 0) {
              return sum;
            }
            const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
            (result.data as BigInt64Array | BigUint64Array)[resultIdx] = sum;
          } else {
            let sum = 0;
            for (let k = 0; k < contractionDim; k++) {
              const aFlatIdx = aDim === 1 ? k : i * contractionDim + k;
              const bFlatIdx = bDim === 1 ? k : j * contractionDim + k;
              sum += Number(a.iget(aFlatIdx)) * Number(b.iget(bFlatIdx));
            }
            if (resultShape.length === 0) {
              return sum;
            }
            const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
            result.data[resultIdx] = sum;
          }
        }
      }
    }
  }

  return result;
}

/**
 * Outer product of two vectors
 *
 * Computes out[i, j] = a[i] * b[j]
 * Input arrays are flattened if not 1D.
 *
 * @param a - First input (flattened to 1D)
 * @param b - Second input (flattened to 1D)
 * @returns 2D array of shape (a.size, b.size)
 */
export function outer(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  // Flatten inputs to 1D
  const aFlat = a.ndim === 1 ? a : shapeOps.ravel(a);
  const bFlat = b.ndim === 1 ? b : shapeOps.ravel(b);

  // Try WASM-accelerated outer product
  const wasmResult = wasmOuter(aFlat, bFlat);
  if (wasmResult) return wasmResult;

  const m = aFlat.size;
  const n = bFlat.size;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros([m, n], resultDtype);

  // Float16Array optimization: bulk-convert inputs to Float32Array for faster per-element access
  if (resultDtype === 'float16' && hasFloat16 && aFlat.isCContiguous && bFlat.isCContiguous) {
    const f32A = new Float32Array(
      (aFlat.data as Float16Array).subarray(aFlat.offset, aFlat.offset + m)
    );
    const f32B = new Float32Array(
      (bFlat.data as Float16Array).subarray(bFlat.offset, bFlat.offset + n)
    );
    const f32Out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      const aVal = f32A[i]!;
      const base = i * n;
      for (let j = 0; j < n; j++) {
        f32Out[base + j] = aVal * f32B[j]!;
      }
    }
    (result.data as Float16Array).set(f32Out);
    return result;
  }

  // Compute outer product: result[i,j] = a[i] * b[j]
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aVal = aFlat.get(i);
      const bVal = bFlat.get(j);

      // Use multiplyValues to handle complex numbers properly
      const product = multiplyValues(aVal, bVal);
      result.set([i, j], product);
    }
  }

  return result;
}

/**
 * Tensor dot product along specified axes
 *
 * Computes sum product over specified axes.
 *
 * @param a - First array
 * @param b - Second array
 * @param axes - Axes to contract:
 *   - Integer N: Contract last N axes of a with first N of b
 *   - [a_axes, b_axes]: Contract specified axes
 * @returns Tensor dot product
 */
export function tensordot(
  a: ArrayStorage,
  b: ArrayStorage,
  axes: number | [number[], number[]]
): ArrayStorage | number | bigint | Complex {
  let aAxes: number[];
  let bAxes: number[];

  if (typeof axes === 'number') {
    // Contract last N axes of a with first N of b
    const N = axes;
    if (N < 0) {
      throw new Error('tensordot: axes must be non-negative');
    }
    if (N > a.ndim || N > b.ndim) {
      throw new Error('tensordot: axes exceeds array dimensions');
    }

    // Last N axes of a
    aAxes = Array.from({ length: N }, (_, i) => a.ndim - N + i);
    // First N axes of b
    bAxes = Array.from({ length: N }, (_, i) => i);
  } else {
    [aAxes, bAxes] = axes;
    if (aAxes.length !== bAxes.length) {
      throw new Error('tensordot: axes lists must have same length');
    }
  }

  // Validate axes and check dimension compatibility
  for (let i = 0; i < aAxes.length; i++) {
    const aAxis = aAxes[i]!;
    const bAxis = bAxes[i]!;
    if (aAxis < 0 || aAxis >= a.ndim || bAxis < 0 || bAxis >= b.ndim) {
      throw new Error('tensordot: axis out of bounds');
    }
    if (a.shape[aAxis] !== b.shape[bAxis]) {
      throw new Error(
        `tensordot: shape mismatch on axes ${aAxis} and ${bAxis}: ${a.shape[aAxis]} != ${b.shape[bAxis]}`
      );
    }
  }

  // Separate axes into contracted and free axes
  const aFreeAxes: number[] = [];
  const bFreeAxes: number[] = [];

  for (let i = 0; i < a.ndim; i++) {
    if (!aAxes.includes(i)) {
      aFreeAxes.push(i);
    }
  }
  for (let i = 0; i < b.ndim; i++) {
    if (!bAxes.includes(i)) {
      bFreeAxes.push(i);
    }
  }

  // Build result shape: free axes of a + free axes of b
  const resultShape = [
    ...aFreeAxes.map((ax) => a.shape[ax]!),
    ...bFreeAxes.map((ax) => b.shape[ax]!),
  ];

  // Determine if we're working with complex numbers
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const isComplex = isComplexDType(resultDtype);

  // Helper to get real and imaginary parts
  const getReIm = (val: number | bigint | Complex): { re: number; im: number } => {
    if (val instanceof Complex) {
      return { re: val.re, im: val.im };
    }
    return { re: Number(val), im: 0 };
  };

  // Special case: no free axes (full contraction) -> scalar result
  if (resultShape.length === 0) {
    let sumRe = 0;
    let sumIm = 0;
    // Iterate over all combinations of contracted axes
    const contractSize = aAxes.map((ax) => a.shape[ax]!).reduce((acc, dim) => acc * dim, 1);

    for (let i = 0; i < contractSize; i++) {
      // Convert flat index to contracted indices
      let temp = i;
      const contractedIdx: number[] = new Array(aAxes.length);
      for (let j = aAxes.length - 1; j >= 0; j--) {
        const ax = aAxes[j]!;
        contractedIdx[j] = temp % a.shape[ax]!;
        temp = Math.floor(temp / a.shape[ax]!);
      }

      // Build full indices for a and b
      const aIdx: number[] = new Array(a.ndim);
      const bIdx: number[] = new Array(b.ndim);

      for (let j = 0; j < aAxes.length; j++) {
        aIdx[aAxes[j]!] = contractedIdx[j]!;
      }
      for (let j = 0; j < bAxes.length; j++) {
        bIdx[bAxes[j]!] = contractedIdx[j]!;
      }

      const aVal = a.get(...aIdx);
      const bVal = b.get(...bIdx);

      if (isComplex) {
        const av = getReIm(aVal);
        const bv = getReIm(bVal);
        // Complex multiplication: (aRe + aIm*i) * (bRe + bIm*i)
        sumRe += av.re * bv.re - av.im * bv.im;
        sumIm += av.re * bv.im + av.im * bv.re;
      } else if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sumRe += Number(aVal * bVal);
      } else {
        sumRe += Number(aVal) * Number(bVal);
      }
    }

    if (isComplex) {
      return new Complex(sumRe, sumIm);
    }
    // Bool tensordot: clamp to 0/1 (NumPy uses logical AND for multiply, OR for add)
    if (resultDtype === 'bool') {
      return sumRe ? 1 : 0;
    }
    return sumRe;
  }

  // General case: with free axes
  const result = ArrayStorage.zeros(resultShape, resultDtype);

  const resultSize = resultShape.reduce((acc, dim) => acc * dim, 1);
  const contractSize = aAxes.map((ax) => a.shape[ax]!).reduce((acc, dim) => acc * dim, 1);

  // Iterate over all result positions
  for (let resIdx = 0; resIdx < resultSize; resIdx++) {
    // Convert flat result index to multi-dimensional
    let temp = resIdx;
    const resultIndices: number[] = [];
    for (let i = resultShape.length - 1; i >= 0; i--) {
      resultIndices[i] = temp % resultShape[i]!;
      temp = Math.floor(temp / resultShape[i]!);
    }

    // Extract indices for a's free axes and b's free axes
    const aFreeIndices = resultIndices.slice(0, aFreeAxes.length);
    const bFreeIndices = resultIndices.slice(aFreeAxes.length);

    let sumRe = 0;
    let sumIm = 0;

    // Sum over all contracted axes
    for (let c = 0; c < contractSize; c++) {
      // Convert flat contracted index to multi-dimensional
      temp = c;
      const contractedIndices: number[] = [];
      for (let i = aAxes.length - 1; i >= 0; i--) {
        const ax = aAxes[i]!;
        contractedIndices[i] = temp % a.shape[ax]!;
        temp = Math.floor(temp / a.shape[ax]!);
      }

      // Build full indices for a and b
      const aFullIdx: number[] = new Array(a.ndim);
      const bFullIdx: number[] = new Array(b.ndim);

      // Fill in free axes
      for (let i = 0; i < aFreeAxes.length; i++) {
        aFullIdx[aFreeAxes[i]!] = aFreeIndices[i]!;
      }
      for (let i = 0; i < bFreeAxes.length; i++) {
        bFullIdx[bFreeAxes[i]!] = bFreeIndices[i]!;
      }

      // Fill in contracted axes
      for (let i = 0; i < aAxes.length; i++) {
        aFullIdx[aAxes[i]!] = contractedIndices[i]!;
        bFullIdx[bAxes[i]!] = contractedIndices[i]!;
      }

      const aVal = a.get(...aFullIdx);
      const bVal = b.get(...bFullIdx);

      if (isComplex) {
        const av = getReIm(aVal);
        const bv = getReIm(bVal);
        // Complex multiplication
        sumRe += av.re * bv.re - av.im * bv.im;
        sumIm += av.re * bv.im + av.im * bv.re;
      } else if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sumRe += Number(aVal * bVal);
      } else {
        sumRe += Number(aVal) * Number(bVal);
      }
    }

    if (isComplex) {
      result.set(resultIndices, new Complex(sumRe, sumIm));
    } else if (resultDtype === 'bool') {
      result.set(resultIndices, sumRe ? 1 : 0);
    } else {
      result.set(resultIndices, sumRe);
    }
  }

  return result;
}

/**
 * Extract a diagonal or construct a diagonal array.
 *
 * NumPy behavior:
 * - For 2D arrays: extract the k-th diagonal
 * - For ND arrays (N >= 2): extract diagonal from the axes specified
 * - Returns a view when possible, copy otherwise
 *
 * @param a - Input array (must be at least 2D)
 * @param offset - Offset of the diagonal from the main diagonal (default: 0)
 *                 - offset > 0: diagonal above main diagonal
 *                 - offset < 0: diagonal below main diagonal
 * @param axis1 - First axis for ND arrays (default: 0)
 * @param axis2 - Second axis for ND arrays (default: 1)
 * @returns Array containing the diagonal elements
 */
export function diagonal(
  a: ArrayStorage,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): ArrayStorage {
  const shape = a.shape;
  const ndim = shape.length;

  if (ndim < 2) {
    throw new Error('diagonal requires an array of at least two dimensions');
  }

  // Normalize negative axes
  const ax1 = axis1 < 0 ? ndim + axis1 : axis1;
  const ax2 = axis2 < 0 ? ndim + axis2 : axis2;

  if (ax1 < 0 || ax1 >= ndim || ax2 < 0 || ax2 >= ndim) {
    throw new Error('axis out of bounds');
  }

  if (ax1 === ax2) {
    throw new Error('axis1 and axis2 cannot be the same');
  }

  // Get dimensions of the two axes
  const dim1 = shape[ax1]!;
  const dim2 = shape[ax2]!;

  // Calculate diagonal length
  let diagLen: number;
  if (offset >= 0) {
    diagLen = Math.max(0, Math.min(dim1, dim2 - offset));
  } else {
    diagLen = Math.max(0, Math.min(dim1 + offset, dim2));
  }

  // Build output shape: remove axis1 and axis2, append diagLen
  const outShape: number[] = [];
  for (let i = 0; i < ndim; i++) {
    if (i !== ax1 && i !== ax2) {
      outShape.push(shape[i]!);
    }
  }
  outShape.push(diagLen);

  // Fast path: 2D arrays — return a strided view (zero-copy)
  if (ndim === 2 && ax1 === 0 && ax2 === 1) {
    const startOffset = a.offset + (offset >= 0 ? offset * a.strides[1]! : -offset * a.strides[0]!);
    const diagStride = a.strides[0]! + a.strides[1]!;
    return ArrayStorage.fromDataShared(
      a.data,
      [diagLen],
      a.dtype,
      [diagStride],
      startOffset,
      a.wasmRegion
    );
  }

  // General N-D path: element-by-element copy
  const result = ArrayStorage.zeros(outShape, a.dtype);
  const otherDims = shape.filter((_, i) => i !== ax1 && i !== ax2);
  const otherSize = otherDims.reduce((acc, d) => acc * d, 1);

  for (let otherIdx = 0; otherIdx < otherSize; otherIdx++) {
    let temp = otherIdx;
    const otherIndices: number[] = [];
    for (let i = otherDims.length - 1; i >= 0; i--) {
      otherIndices.unshift(temp % otherDims[i]!);
      temp = Math.floor(temp / otherDims[i]!);
    }

    for (let d = 0; d < diagLen; d++) {
      const srcIndices: number[] = new Array(ndim);
      let otherIdx2 = 0;
      for (let i = 0; i < ndim; i++) {
        if (i === ax1) {
          srcIndices[i] = offset >= 0 ? d : d - offset;
        } else if (i === ax2) {
          srcIndices[i] = offset >= 0 ? d + offset : d;
        } else {
          srcIndices[i] = otherIndices[otherIdx2++]!;
        }
      }

      const dstIndices = [...otherIndices, d];
      const value = a.get(...srcIndices);
      result.set(dstIndices, value);
    }
  }

  return result;
}

/**
 * Einstein summation convention
 *
 * Performs tensor contractions and reductions using Einstein summation notation.
 *
 * Examples:
 * - 'ij,jk->ik': matrix multiplication
 * - 'i,i->': dot product (inner product)
 * - 'ij->ji': transpose
 * - 'ii->': trace
 * - 'ij->j': sum over first axis
 * - 'ijk,ikl->ijl': batched matrix multiplication
 *
 * @param subscripts - Einstein summation subscripts (e.g., 'ij,jk->ik')
 * @param operands - Input arrays
 * @returns Result of the Einstein summation
 */
export function einsum(
  subscripts: string,
  ...operands: ArrayStorage[]
): ArrayStorage | number | bigint | Complex {
  // Parse the subscripts
  const arrowMatch = subscripts.indexOf('->');

  let inputSubscripts: string;
  let outputSubscript: string;

  if (arrowMatch === -1) {
    // Implicit output: collect unique indices not repeated
    inputSubscripts = subscripts;
    outputSubscript = inferOutputSubscript(inputSubscripts);
  } else {
    inputSubscripts = subscripts.slice(0, arrowMatch);
    outputSubscript = subscripts.slice(arrowMatch + 2);
  }

  // Parse input subscripts into individual operand subscripts
  const operandSubscripts = inputSubscripts.split(',').map((s) => s.trim());

  if (operandSubscripts.length !== operands.length) {
    throw new Error(
      `einsum: expected ${operandSubscripts.length} operands, got ${operands.length}`
    );
  }

  // Validate subscripts and build index dimension map
  const indexDims = new Map<string, number>();

  for (let i = 0; i < operands.length; i++) {
    const sub = operandSubscripts[i]!;
    const op = operands[i]!;

    if (sub.length !== op.ndim) {
      throw new Error(
        `einsum: operand ${i} has ${op.ndim} dimensions but subscript '${sub}' has ${sub.length} indices`
      );
    }

    for (let j = 0; j < sub.length; j++) {
      const idx = sub[j]!;
      const dim = op.shape[j]!;

      if (indexDims.has(idx)) {
        if (indexDims.get(idx) !== dim) {
          throw new Error(
            `einsum: size mismatch for index '${idx}': ${indexDims.get(idx)} vs ${dim}`
          );
        }
      } else {
        indexDims.set(idx, dim);
      }
    }
  }

  // Validate output subscript
  for (const idx of outputSubscript) {
    if (!indexDims.has(idx)) {
      throw new Error(`einsum: output subscript contains unknown index '${idx}'`);
    }
  }

  // Identify summation indices (in inputs but not in output)
  const outputIndices = new Set(outputSubscript);
  const allInputIndices = new Set<string>();
  for (const sub of operandSubscripts) {
    for (const idx of sub) {
      allInputIndices.add(idx);
    }
  }

  const sumIndices: string[] = [];
  for (const idx of allInputIndices) {
    if (!outputIndices.has(idx)) {
      sumIndices.push(idx);
    }
  }

  // ========================================
  // FAST PATHS: Detect common patterns and delegate to optimized implementations
  // ========================================

  // Pattern: Matrix multiplication "ij,jk->ik" or similar
  if (operands.length === 2 && operandSubscripts.length === 2) {
    const [sub1, sub2] = operandSubscripts;
    const [op1, op2] = operands;

    // Check for matmul pattern: two 2D arrays, one shared index
    if (
      sub1!.length === 2 &&
      sub2!.length === 2 &&
      outputSubscript.length === 2 &&
      op1!.ndim === 2 &&
      op2!.ndim === 2
    ) {
      const [i1, j1] = [sub1![0]!, sub1![1]!];
      const [i2, j2] = [sub2![0]!, sub2![1]!];
      const [o1, o2] = [outputSubscript[0]!, outputSubscript[1]!];

      // Pattern: "ij,jk->ik" (standard matmul)
      if (i1 === o1 && j2 === o2 && j1 === i2 && sumIndices.length === 1 && sumIndices[0] === j1) {
        return matmul(op1!, op2!);
      }

      // Pattern: "ik,kj->ij" (matmul with different index names)
      if (i1 === o1 && j2 === o2 && j1 === i2 && sumIndices.length === 1 && sumIndices[0] === j1) {
        return matmul(op1!, op2!);
      }

      // Pattern: "ji,jk->ik" (transpose A then multiply)
      if (j1 === o1 && j2 === o2 && i1 === i2 && sumIndices.length === 1 && sumIndices[0] === i1) {
        const op1T = transpose(op1!);
        return matmul(op1T, op2!);
      }

      // Pattern: "ij,kj->ik" (transpose B then multiply)
      if (i1 === o1 && i2 === o2 && j1 === j2 && sumIndices.length === 1 && sumIndices[0] === j1) {
        const op2T = transpose(op2!);
        return matmul(op1!, op2T);
      }
    }

    // Check for dot product pattern: two 1D arrays "i,i->" or "i,i->scalar"
    if (
      sub1!.length === 1 &&
      sub2!.length === 1 &&
      sub1 === sub2 &&
      outputSubscript.length === 0 &&
      op1!.ndim === 1 &&
      op2!.ndim === 1
    ) {
      return computeEinsumScalar(operands, operandSubscripts, sumIndices, indexDims);
    }

    // Check for outer product pattern: "i,j->ij"
    if (
      sub1 &&
      sub2 &&
      sub1.length === 1 &&
      sub2.length === 1 &&
      outputSubscript.length === 2 &&
      outputSubscript === sub1 + sub2 &&
      sumIndices.length === 0 &&
      op1!.ndim === 1 &&
      op2!.ndim === 1
    ) {
      return outer(op1!, op2!);
    }
  }

  // Pattern: Single operand trace "ii->"
  if (operands.length === 1 && operandSubscripts[0]!.length === 2 && outputSubscript.length === 0) {
    const sub = operandSubscripts[0]!;
    if (sub[0] === sub[1]) {
      // This is a trace operation
      const op = operands[0]!;
      if (op.ndim === 2) {
        return computeEinsumScalar(operands, operandSubscripts, sumIndices, indexDims);
      }
    }
  }

  // ========================================
  // END FAST PATHS - Fall through to generic implementation
  // ========================================

  // Build output shape
  const outputShape = Array.from(outputSubscript).map((idx) => indexDims.get(idx)!);

  // Special case: scalar output
  if (outputShape.length === 0) {
    return computeEinsumScalar(operands, operandSubscripts, sumIndices, indexDims);
  }

  // Determine result dtype
  let resultDtype = operands[0]!.dtype;
  for (let i = 1; i < operands.length; i++) {
    resultDtype = promoteDTypes(resultDtype, operands[i]!.dtype);
  }

  // Create output array
  const result = ArrayStorage.zeros(outputShape, resultDtype);

  // Compute output size
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  // Compute sum range
  let sumSize = 1;
  for (const idx of sumIndices) {
    sumSize *= indexDims.get(idx)!;
  }

  // Iterate over all output positions
  for (let outIdx = 0; outIdx < outputSize; outIdx++) {
    // Convert flat index to multi-dimensional output index
    const outMultiIdx = flatToMulti(outIdx, outputShape);

    // Build index assignment for output indices
    const indexValues = new Map<string, number>();
    for (let i = 0; i < outputSubscript.length; i++) {
      indexValues.set(outputSubscript[i]!, outMultiIdx[i]!);
    }

    // Sum over summation indices
    let sum = 0;
    for (let sumIdx = 0; sumIdx < sumSize; sumIdx++) {
      // Assign values to summation indices
      let temp = sumIdx;
      for (let i = sumIndices.length - 1; i >= 0; i--) {
        const idx = sumIndices[i]!;
        const dim = indexDims.get(idx)!;
        indexValues.set(idx, temp % dim);
        temp = Math.floor(temp / dim);
      }

      // Compute product of all operand values
      let product = 1;
      for (let i = 0; i < operands.length; i++) {
        const op = operands[i]!;
        const sub = operandSubscripts[i]!;

        // Build operand index
        const opIdx: number[] = [];
        for (const idx of sub) {
          opIdx.push(indexValues.get(idx)!);
        }

        const val = op.get(...opIdx);
        product *= Number(val);
      }

      sum += product;
    }

    result.set(outMultiIdx, sum);
  }

  return result;
}

/**
 * Infer output subscript for implicit einsum notation
 * @private
 */
function inferOutputSubscript(inputSubscripts: string): string {
  // Count occurrences of each index
  const counts = new Map<string, number>();
  const operandSubscripts = inputSubscripts.split(',');

  for (const sub of operandSubscripts) {
    for (const idx of sub.trim()) {
      counts.set(idx, (counts.get(idx) || 0) + 1);
    }
  }

  // Output contains indices that appear exactly once, sorted alphabetically
  const outputIndices: string[] = [];
  for (const [idx, count] of counts) {
    if (count === 1) {
      outputIndices.push(idx);
    }
  }

  return outputIndices.sort().join('');
}

/**
 * Compute einsum result when output is a scalar
 * @private
 */
function computeEinsumScalar(
  operands: ArrayStorage[],
  operandSubscripts: string[],
  sumIndices: string[],
  indexDims: Map<string, number>
): number | Complex {
  // Check if any operand is complex
  let resultIsComplex = false;
  for (const op of operands) {
    if (isComplexDType(op.dtype)) {
      resultIsComplex = true;
      break;
    }
  }

  // Helper to get real and imaginary parts
  const getReIm = (val: number | bigint | Complex): { re: number; im: number } => {
    if (val instanceof Complex) {
      return { re: val.re, im: val.im };
    }
    return { re: Number(val), im: 0 };
  };

  // All indices are summation indices
  let sumSize = 1;
  for (const idx of sumIndices) {
    sumSize *= indexDims.get(idx)!;
  }

  let sumRe = 0;
  let sumIm = 0;

  for (let sumIdx = 0; sumIdx < sumSize; sumIdx++) {
    // Assign values to summation indices
    const indexValues = new Map<string, number>();
    let temp = sumIdx;
    for (let i = sumIndices.length - 1; i >= 0; i--) {
      const idx = sumIndices[i]!;
      const dim = indexDims.get(idx)!;
      indexValues.set(idx, temp % dim);
      temp = Math.floor(temp / dim);
    }

    // Compute product of all operand values
    let prodRe = 1;
    let prodIm = 0;
    for (let i = 0; i < operands.length; i++) {
      const op = operands[i]!;
      const sub = operandSubscripts[i]!;

      // Build operand index
      const opIdx: number[] = [];
      for (const idx of sub) {
        opIdx.push(indexValues.get(idx)!);
      }

      const val = op.get(...opIdx);
      if (resultIsComplex) {
        const v = getReIm(val);
        // Complex multiplication: prod * v
        const newRe = prodRe * v.re - prodIm * v.im;
        const newIm = prodRe * v.im + prodIm * v.re;
        prodRe = newRe;
        prodIm = newIm;
      } else {
        prodRe *= Number(val);
      }
    }

    sumRe += prodRe;
    sumIm += prodIm;
  }

  if (resultIsComplex) {
    return new Complex(sumRe, sumIm);
  }
  // Bool: clamp to 0/1 (NumPy bool arithmetic wraps)
  if (operands.every((op) => op.dtype === 'bool')) return sumRe ? 1 : 0;
  return sumRe;
}

/**
 * Convert flat index to multi-dimensional index
 * @private
 */
function flatToMulti(flatIdx: number, shape: number[]): number[] {
  const result: number[] = new Array(shape.length);
  let temp = flatIdx;

  for (let i = shape.length - 1; i >= 0; i--) {
    result[i] = temp % shape[i]!;
    temp = Math.floor(temp / shape[i]!);
  }

  return result;
}

/**
 * Kronecker product of two arrays.
 *
 * Computes the Kronecker product, a composite array made of blocks of the
 * second array scaled by the elements of the first.
 *
 * NumPy behavior:
 * - If both inputs are vectors (1D), output is also a vector
 * - If both inputs are 2D matrices, output shape is (m1*m2, n1*n2)
 * - General case: broadcasts shapes then computes block product
 *
 * @param a - First input array
 * @param b - Second input array
 * @returns Kronecker product of a and b
 */
export function kron(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const aShape = a.shape;
  const bShape = b.shape;
  const aNdim = aShape.length;
  const bNdim = bShape.length;

  // Promote dtypes
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // WASM fast path for 2D × 2D
  if (aNdim === 2 && bNdim === 2) {
    const wasmResult = wasmKron(a, b);
    if (wasmResult) return wasmResult;
  }

  // Determine output shape
  const ndim = Math.max(aNdim, bNdim);
  const outShape: number[] = new Array(ndim);

  // Pad shapes with ones on the left if needed
  const aPadded: number[] = new Array(ndim).fill(1);
  const bPadded: number[] = new Array(ndim).fill(1);

  for (let i = 0; i < aNdim; i++) {
    aPadded[ndim - aNdim + i] = aShape[i]!;
  }
  for (let i = 0; i < bNdim; i++) {
    bPadded[ndim - bNdim + i] = bShape[i]!;
  }

  // Output shape is element-wise product
  for (let i = 0; i < ndim; i++) {
    outShape[i] = aPadded[i]! * bPadded[i]!;
  }

  // Create result array
  const result = ArrayStorage.zeros(outShape, resultDtype);

  // Compute total number of elements in each array
  const aSize = aShape.reduce((acc, d) => acc * d, 1);
  const bSize = bShape.reduce((acc, d) => acc * d, 1);

  // Nested loop approach: for each element in a, scale all of b
  for (let aIdx = 0; aIdx < aSize; aIdx++) {
    // Convert flat index to multi-dimensional index for a
    let temp = aIdx;
    const aIndices: number[] = new Array(aNdim);
    for (let i = aNdim - 1; i >= 0; i--) {
      aIndices[i] = temp % aShape[i]!;
      temp = Math.floor(temp / aShape[i]!);
    }

    // Pad aIndices to match ndim
    const aIndicesPadded: number[] = new Array(ndim).fill(0);
    for (let i = 0; i < aNdim; i++) {
      aIndicesPadded[ndim - aNdim + i] = aIndices[i]!;
    }

    const aVal = a.get(...aIndices);

    // For each element in b
    for (let bIdx = 0; bIdx < bSize; bIdx++) {
      // Convert flat index to multi-dimensional index for b
      let temp2 = bIdx;
      const bIndices: number[] = new Array(bNdim);
      for (let i = bNdim - 1; i >= 0; i--) {
        bIndices[i] = temp2 % bShape[i]!;
        temp2 = Math.floor(temp2 / bShape[i]!);
      }

      // Pad bIndices to match ndim
      const bIndicesPadded: number[] = new Array(ndim).fill(0);
      for (let i = 0; i < bNdim; i++) {
        bIndicesPadded[ndim - bNdim + i] = bIndices[i]!;
      }

      const bVal = b.get(...bIndices);

      // Compute output index: each dimension is aIdx*bDim + bIdx
      const outIndices: number[] = new Array(ndim);
      for (let i = 0; i < ndim; i++) {
        outIndices[i] = aIndicesPadded[i]! * bPadded[i]! + bIndicesPadded[i]!;
      }

      // Compute product and store - use multiplyValues for complex support
      const product = multiplyValues(aVal, bVal);
      result.set(outIndices, product);
    }
  }

  return result;
}

// ========================================
// NUMPY.LINALG MODULE FUNCTIONS
// ========================================

/**
 * Cross product of two vectors.
 *
 * For 3D vectors: returns the cross product vector
 * For 2D vectors: returns the scalar z-component of the cross product
 *
 * @param a - First input array
 * @param b - Second input array
 * @param axisa - Axis of a that defines the vectors (default: -1)
 * @param axisb - Axis of b that defines the vectors (default: -1)
 * @param axisc - Axis of c containing the cross product vectors (default: -1)
 * @param axis - If defined, the axis of a, b and c that defines the vectors
 * @returns Cross product of a and b
 */
export function cross(
  a: ArrayStorage,
  b: ArrayStorage,
  axisa: number = -1,
  axisb: number = -1,
  axisc: number = -1,
  axis?: number
): ArrayStorage | number | bigint | Complex {
  if (a.dtype === 'bool' || b.dtype === 'bool') {
    throw new TypeError(
      `ufunc 'subtract' not supported for boolean dtype. The '-' operator is not supported for booleans, use 'bitwise_xor' instead.`
    );
  }
  // If axis is specified, use it for all
  if (axis !== undefined) {
    axisa = axis;
    axisb = axis;
    axisc = axis;
  }

  // Normalize negative axes
  const normalizeAxis = (ax: number, ndim: number) => (ax < 0 ? ndim + ax : ax);

  const axisA = normalizeAxis(axisa, a.ndim);
  const axisB = normalizeAxis(axisb, b.ndim);

  // Determine output dtype (promote to complex if either input is complex)
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const isComplex = isComplexDType(resultDtype);

  // Helper to get value as number or Complex
  type CrossVal = number | bigint | Complex;

  const getValue = (storage: ArrayStorage, ...indices: number[]): CrossVal => {
    const val = storage.get(...indices);
    if (val instanceof Complex) return val;
    if (typeof val === 'bigint') return val;
    return Number(val);
  };

  // Helper for cross product arithmetic
  const crossMul = (x: CrossVal, y: CrossVal): CrossVal => {
    if (x instanceof Complex || y instanceof Complex) {
      const xc = x instanceof Complex ? x : new Complex(Number(x), 0);
      const yc = y instanceof Complex ? y : new Complex(Number(y), 0);
      return xc.mul(yc);
    }
    if (typeof x === 'bigint' || typeof y === 'bigint') {
      return BigInt(x as bigint) * BigInt(y as bigint);
    }
    return x * y;
  };

  const crossSub = (x: CrossVal, y: CrossVal): CrossVal => {
    if (x instanceof Complex || y instanceof Complex) {
      const xc = x instanceof Complex ? x : new Complex(Number(x), 0);
      const yc = y instanceof Complex ? y : new Complex(Number(y), 0);
      return xc.sub(yc);
    }
    if (typeof x === 'bigint' || typeof y === 'bigint') {
      return BigInt(x as bigint) - BigInt(y as bigint);
    }
    return x - y;
  };

  // Simple case: both are 1D vectors
  if (a.ndim === 1 && b.ndim === 1) {
    const dimA = a.shape[0]!;
    const dimB = b.shape[0]!;

    if (dimA === 3 && dimB === 3) {
      // 3D cross product
      const a0 = getValue(a, 0);
      const a1 = getValue(a, 1);
      const a2 = getValue(a, 2);
      const b0 = getValue(b, 0);
      const b1 = getValue(b, 1);
      const b2 = getValue(b, 2);

      const result = ArrayStorage.zeros([3], resultDtype);
      result.set([0], crossSub(crossMul(a1, b2), crossMul(a2, b1)));
      result.set([1], crossSub(crossMul(a2, b0), crossMul(a0, b2)));
      result.set([2], crossSub(crossMul(a0, b1), crossMul(a1, b0)));
      return result;
    } else if (dimA === 2 && dimB === 2) {
      // 2D cross product (returns scalar or Complex)
      const a0 = getValue(a, 0);
      const a1 = getValue(a, 1);
      const b0 = getValue(b, 0);
      const b1 = getValue(b, 1);
      return crossSub(crossMul(a0, b1), crossMul(a1, b0));
    } else if ((dimA === 2 && dimB === 3) || (dimA === 3 && dimB === 2)) {
      // Mixed 2D/3D - treat 2D as having z=0
      const a0 = getValue(a, 0);
      const a1 = getValue(a, 1);
      const a2 =
        dimA === 3
          ? getValue(a, 2)
          : isComplex
            ? new Complex(0, 0)
            : isBigIntDType(resultDtype)
              ? 0n
              : 0;
      const b0 = getValue(b, 0);
      const b1 = getValue(b, 1);
      const b2 =
        dimB === 3
          ? getValue(b, 2)
          : isComplex
            ? new Complex(0, 0)
            : isBigIntDType(resultDtype)
              ? 0n
              : 0;

      const result = ArrayStorage.zeros([3], resultDtype);
      result.set([0], crossSub(crossMul(a1, b2), crossMul(a2, b1)));
      result.set([1], crossSub(crossMul(a2, b0), crossMul(a0, b2)));
      result.set([2], crossSub(crossMul(a0, b1), crossMul(a1, b0)));
      return result;
    } else {
      throw new Error(`cross: incompatible dimensions for cross product: ${dimA} and ${dimB}`);
    }
  }

  // Handle higher dimensional arrays by iterating over all other axes
  // For simplicity, we'll handle the case where the vector axis is last
  const vectorDimA = a.shape[axisA]!;
  const vectorDimB = b.shape[axisB]!;

  if ((vectorDimA !== 2 && vectorDimA !== 3) || (vectorDimB !== 2 && vectorDimB !== 3)) {
    throw new Error(
      `cross: incompatible dimensions for cross product: ${vectorDimA} and ${vectorDimB}`
    );
  }

  // Determine output shape and vector dimension
  const outputVectorDim = vectorDimA === 2 && vectorDimB === 2 ? 0 : 3;

  // Build output shape (broadcast the non-vector axes)
  const aOtherShape = [...a.shape.slice(0, axisA), ...a.shape.slice(axisA + 1)];
  const bOtherShape = [...b.shape.slice(0, axisB), ...b.shape.slice(axisB + 1)];

  // For now, require same shapes for non-vector axes
  if (aOtherShape.length !== bOtherShape.length) {
    throw new Error('cross: incompatible shapes for cross product');
  }
  for (let i = 0; i < aOtherShape.length; i++) {
    if (aOtherShape[i] !== bOtherShape[i]) {
      throw new Error('cross: incompatible shapes for cross product');
    }
  }

  const otherShape = aOtherShape;

  // WASM fast path: batched 3×3 cross product with last axis as vector axis
  if (vectorDimA === 3 && vectorDimB === 3 && axisA === a.ndim - 1 && axisB === b.ndim - 1) {
    const batchSize = otherShape.reduce((acc, d) => acc * d, 1);
    const wasmResult = wasmCross(a, b, batchSize);
    if (wasmResult) return wasmResult;
  }

  const normalizedAxisC = axisc < 0 ? otherShape.length + 1 + axisc : axisc;

  // Build result shape
  let resultShape: number[];
  if (outputVectorDim === 0) {
    // Scalar output per position
    resultShape = otherShape;
  } else {
    // Insert vector dimension at axisc
    resultShape = [
      ...otherShape.slice(0, normalizedAxisC),
      outputVectorDim,
      ...otherShape.slice(normalizedAxisC),
    ];
  }

  if (resultShape.length === 0) {
    // Both 2D vectors, scalar result (already handled above)
    throw new Error('cross: unexpected scalar result from higher-dimensional input');
  }

  const result = ArrayStorage.zeros(resultShape, resultDtype);

  // Iterate over all "other" positions
  const otherSize = otherShape.reduce((acc, d) => acc * d, 1);

  for (let i = 0; i < otherSize; i++) {
    // Convert flat index to multi-dim
    let temp = i;
    const otherIndices: number[] = [];
    for (let j = otherShape.length - 1; j >= 0; j--) {
      otherIndices[j] = temp % otherShape[j]!;
      temp = Math.floor(temp / otherShape[j]!);
    }

    // Build indices for a and b (insert vector axis)
    const aIndices = [...otherIndices.slice(0, axisA), 0, ...otherIndices.slice(axisA)];
    const bIndices = [...otherIndices.slice(0, axisB), 0, ...otherIndices.slice(axisB)];

    // Extract vector components
    const getA = (idx: number): CrossVal => {
      aIndices[axisA] = idx;
      return getValue(a, ...aIndices);
    };
    const getB = (idx: number): CrossVal => {
      bIndices[axisB] = idx;
      return getValue(b, ...bIndices);
    };

    const a0 = getA(0);
    const a1 = getA(1);
    const a2 =
      vectorDimA === 3
        ? getA(2)
        : isComplex
          ? new Complex(0, 0)
          : isBigIntDType(resultDtype)
            ? 0n
            : 0;
    const b0 = getB(0);
    const b1 = getB(1);
    const b2 =
      vectorDimB === 3
        ? getB(2)
        : isComplex
          ? new Complex(0, 0)
          : isBigIntDType(resultDtype)
            ? 0n
            : 0;

    if (outputVectorDim === 0) {
      // Scalar result
      result.set(otherIndices, crossSub(crossMul(a0, b1), crossMul(a1, b0)));
    } else {
      // Vector result
      const c0 = crossSub(crossMul(a1, b2), crossMul(a2, b1));
      const c1 = crossSub(crossMul(a2, b0), crossMul(a0, b2));
      const c2 = crossSub(crossMul(a0, b1), crossMul(a1, b0));

      const setResult = (idx: number, val: CrossVal) => {
        const resultIndices = [
          ...otherIndices.slice(0, normalizedAxisC),
          idx,
          ...otherIndices.slice(normalizedAxisC),
        ];
        result.set(resultIndices, val);
      };

      setResult(0, c0);
      setResult(1, c1);
      setResult(2, c2);
    }
  }

  return result;
}

/**
 * Vector norm.
 *
 * @param x - Input vector or array
 * @param ord - Order of the norm (default: 2)
 *              - Infinity: max(abs(x))
 *              - -Infinity: min(abs(x))
 *              - 0: sum(x != 0)
 *              - Other: sum(abs(x)^ord)^(1/ord)
 * @param axis - Axis along which to compute (flattened if not specified)
 * @param keepdims - Keep reduced dimensions
 * @returns Vector norm
 */
export function vector_norm(
  x: ArrayStorage,
  ord: number | 'fro' | 'nuc' = 2,
  axis?: number | null,
  keepdims: boolean = false
): ArrayStorage | number {
  // Handle numeric ord only for vector norm
  if (typeof ord !== 'number') {
    throw new Error('vector_norm: ord must be a number');
  }

  // If no axis specified, flatten and compute
  if (axis === undefined || axis === null) {
    const flat = x.ndim === 1 ? x : shapeOps.ravel(x);
    const n = flat.size;

    let result: number;
    const isComplex = isComplexDType(flat.dtype);
    if (ord === Infinity) {
      result = 0;
      for (let i = 0; i < n; i++) {
        result = Math.max(result, absValue(flat.get(i)));
      }
    } else if (ord === -Infinity) {
      result = Infinity;
      for (let i = 0; i < n; i++) {
        result = Math.min(result, absValue(flat.get(i)));
      }
    } else if (ord === 0) {
      result = 0;
      for (let i = 0; i < n; i++) {
        const v = flat.get(i);
        const isZero = v instanceof Complex ? v.re === 0 && v.im === 0 : Number(v) === 0;
        if (!isZero) result++;
      }
    } else if (ord === 1) {
      result = 0;
      for (let i = 0; i < n; i++) {
        result += absValue(flat.get(i));
      }
    } else if (ord === 2) {
      // WASM fast path for L2 norm (real dtypes only)
      const wasmNorm = isComplex ? null : wasmVectorNorm2(flat);
      if (wasmNorm !== null) {
        if (flat !== x) flat.dispose();
        result = wasmNorm;
      } else {
        result = 0;
        for (let i = 0; i < n; i++) {
          const a = absValue(flat.get(i));
          result += a * a;
        }
        result = Math.sqrt(result);
      }
    } else {
      result = 0;
      for (let i = 0; i < n; i++) {
        result += Math.pow(absValue(flat.get(i)), ord);
      }
      result = Math.pow(result, 1 / ord);
    }

    if (keepdims) {
      const shape = new Array(x.ndim).fill(1);
      const out = ArrayStorage.zeros(shape, 'float64');
      out.set(new Array(x.ndim).fill(0), result);
      return out;
    }
    return result;
  }

  // Compute along specified axis
  const ax = axis < 0 ? x.ndim + axis : axis;
  if (ax < 0 || ax >= x.ndim) {
    throw new Error(`vector_norm: axis ${axis} out of bounds for array with ${x.ndim} dimensions`);
  }

  // Build output shape
  const outShape = keepdims
    ? [...x.shape.slice(0, ax), 1, ...x.shape.slice(ax + 1)]
    : [...x.shape.slice(0, ax), ...x.shape.slice(ax + 1)];

  if (outShape.length === 0) {
    // Scalar result
    return vector_norm(x, ord, null, false) as number;
  }

  const result = ArrayStorage.zeros(outShape, 'float64');
  const axisLen = x.shape[ax]!;
  const outSize = outShape.reduce((acc, d) => acc * d, 1);

  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    // Convert flat index to multi-dim output index
    let temp = outIdx;
    const outIndices: number[] = [];
    for (let i = outShape.length - 1; i >= 0; i--) {
      outIndices[i] = temp % outShape[i]!;
      temp = Math.floor(temp / outShape[i]!);
    }

    // Build input indices (insert axis dimension)
    const inIndices = keepdims
      ? [...outIndices.slice(0, ax), 0, ...outIndices.slice(ax + 1)]
      : [...outIndices.slice(0, ax), 0, ...outIndices.slice(ax)];

    // Compute norm along axis
    let normVal: number;
    if (ord === Infinity) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal = Math.max(normVal, absValue(x.get(...inIndices)));
      }
    } else if (ord === -Infinity) {
      normVal = Infinity;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal = Math.min(normVal, absValue(x.get(...inIndices)));
      }
    } else if (ord === 0) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        const v = x.get(...inIndices);
        const isZero = v instanceof Complex ? v.re === 0 && v.im === 0 : Number(v) === 0;
        if (!isZero) normVal++;
      }
    } else if (ord === 1) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal += absValue(x.get(...inIndices));
      }
    } else if (ord === 2) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        const a = absValue(x.get(...inIndices));
        normVal += a * a;
      }
      normVal = Math.sqrt(normVal);
    } else {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal += Math.pow(Math.abs(Number(x.get(...inIndices))), ord);
      }
      normVal = Math.pow(normVal, 1 / ord);
    }

    result.set(outIndices, normVal);
  }

  return result;
}

/**
 * Matrix norm.
 *
 * @param x - Input 2D array
 * @param ord - Order of the norm:
 *              - 'fro': Frobenius norm (default)
 *              - 'nuc': Nuclear norm (sum of singular values)
 *              - 1: Max column sum (max(sum(abs(x), axis=0)))
 *              - -1: Min column sum (min(sum(abs(x), axis=0)))
 *              - 2: Largest singular value
 *              - -2: Smallest singular value
 *              - Infinity: Max row sum (max(sum(abs(x), axis=1)))
 *              - -Infinity: Min row sum (min(sum(abs(x), axis=1)))
 * @param keepdims - Keep reduced dimensions
 * @returns Matrix norm
 */
export function matrix_norm(
  x: ArrayStorage,
  ord: number | 'fro' | 'nuc' = 'fro',
  keepdims: boolean = false
): ArrayStorage | number {
  if (x.ndim < 2) {
    throw new Error(`matrix_norm: input must be at least 2D, got ${x.ndim}D`);
  }

  if (x.ndim > 2) {
    const batchShape = Array.from(x.shape).slice(0, -2);
    const m2 = x.shape[x.ndim - 2]!;
    const n2 = x.shape[x.ndim - 1]!;
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const result = ArrayStorage.empty(batchShape, 'float64');
    const resultData = result.data as Float64Array;
    const xData = toContiguousFloat64(x);
    for (let bi = 0; bi < batchSize; bi++) {
      const off = bi * m2 * n2;
      const slice = ArrayStorage.fromData(xData.slice(off, off + m2 * n2), [m2, n2], 'float64');
      resultData[bi] = matrix_norm(slice, ord, false) as number;
    }
    if (keepdims) {
      const onesShape = [...batchShape, 1, 1] as number[];
      return shapeOps.reshape(result, onesShape);
    }
    return result;
  }

  const [m, n] = x.shape;
  let result: number;

  if (ord === 'fro') {
    // Frobenius norm: sqrt(sum(abs(x)^2))
    result = 0;
    for (let i = 0; i < m!; i++) {
      for (let j = 0; j < n!; j++) {
        const a = absValue(x.get(i, j));
        result += a * a;
      }
    }
    result = Math.sqrt(result);
  } else if (ord === 'nuc') {
    // Nuclear norm: sum of singular values
    const { s } = svdFull(x);
    result = 0;
    for (let i = 0; i < s.size; i++) {
      result += Number(s.get(i));
    }
  } else if (ord === 1) {
    // Max column sum
    result = 0;
    for (let j = 0; j < n!; j++) {
      let colSum = 0;
      for (let i = 0; i < m!; i++) {
        colSum += absValue(x.get(i, j));
      }
      result = Math.max(result, colSum);
    }
  } else if (ord === -1) {
    // Min column sum
    result = Infinity;
    for (let j = 0; j < n!; j++) {
      let colSum = 0;
      for (let i = 0; i < m!; i++) {
        colSum += absValue(x.get(i, j));
      }
      result = Math.min(result, colSum);
    }
  } else if (ord === Infinity) {
    // Max row sum
    result = 0;
    for (let i = 0; i < m!; i++) {
      let rowSum = 0;
      for (let j = 0; j < n!; j++) {
        rowSum += absValue(x.get(i, j));
      }
      result = Math.max(result, rowSum);
    }
  } else if (ord === -Infinity) {
    // Min row sum
    result = Infinity;
    for (let i = 0; i < m!; i++) {
      let rowSum = 0;
      for (let j = 0; j < n!; j++) {
        rowSum += absValue(x.get(i, j));
      }
      result = Math.min(result, rowSum);
    }
  } else if (ord === 2) {
    // Largest singular value
    const { s } = svdFull(x);
    result = Number(s.get(0));
  } else if (ord === -2) {
    // Smallest singular value
    const { s } = svdFull(x);
    result = Number(s.get(s.size - 1));
  } else {
    throw new Error(`matrix_norm: invalid ord value: ${ord}`);
  }

  if (keepdims) {
    const out = ArrayStorage.zeros([1, 1], 'float64');
    out.set([0, 0], result);
    return out;
  }
  return result;
}

/**
 * General norm function (for both vectors and matrices).
 *
 * @param x - Input array
 * @param ord - Order of the norm (default: 'fro' for 2D, 2 for 1D)
 * @param axis - Axis or axes along which to compute
 * @param keepdims - Keep reduced dimensions
 * @returns Norm
 */
export function norm(
  x: ArrayStorage,
  ord: number | 'fro' | 'nuc' | null = null,
  axis: number | [number, number] | null = null,
  keepdims: boolean = false
): ArrayStorage | number {
  // Determine default ord based on axis
  if (ord === null) {
    if (axis === null) {
      // Flatten and compute 2-norm
      return vector_norm(x, 2, null, keepdims);
    } else if (typeof axis === 'number') {
      return vector_norm(x, 2, axis, keepdims);
    } else {
      // axis is [number, number] - fall through to matrix norm with fro
      ord = 'fro';
    }
  }

  // If axis is specified as a tuple, compute matrix norm
  if (Array.isArray(axis)) {
    if (axis.length !== 2) {
      throw new Error('norm: axis must be a 2-tuple for matrix norms');
    }
    const ax0 = axis[0] < 0 ? x.ndim + axis[0] : axis[0];
    const ax1 = axis[1] < 0 ? x.ndim + axis[1] : axis[1];
    const matOrd = (ord ?? 'fro') as 'fro' | 'nuc' | number;

    if (x.ndim === 2) {
      return matrix_norm(x, matOrd, keepdims);
    }

    // ND batch: compute matrix norm along ax0/ax1 for each batch element
    const ndim = x.ndim;
    const batchAxes = Array.from({ length: ndim }, (_, i) => i).filter(
      (i) => i !== ax0 && i !== ax1
    );
    const batchShape = batchAxes.map((i) => x.shape[i]!);
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1;
    const m = x.shape[ax0]!;
    const n = x.shape[ax1]!;

    // Transpose to [...batchAxes, ax0, ax1]
    const perm = [...batchAxes, ax0, ax1];
    const xT = transpose(x, perm);

    const resultData = new Float64Array(batchSize);
    for (let b = 0; b < batchSize; b++) {
      // Convert b to multi-index in batchShape
      const batchIdx: number[] = new Array(batchShape.length);
      let rem = b;
      for (let i = batchShape.length - 1; i >= 0; i--) {
        batchIdx[i] = rem % batchShape[i]!;
        rem = Math.floor(rem / batchShape[i]!);
      }
      // Extract 2D slice
      const slice = ArrayStorage.zeros([m, n], 'float64');
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          slice.set([i, j], Number(xT.get(...batchIdx, i, j)));
        }
      }
      const val = matrix_norm(slice, matOrd, false);
      resultData[b] = typeof val === 'number' ? val : Number(val);
    }

    if (keepdims) {
      const ksShape = Array.from(x.shape);
      ksShape[ax0] = 1;
      ksShape[ax1] = 1;
      return ArrayStorage.fromData(resultData, ksShape, 'float64');
    }
    return batchShape.length === 0
      ? resultData[0]!
      : ArrayStorage.fromData(resultData, batchShape, 'float64');
  }

  // Single axis or no axis - vector norm
  if (x.ndim === 2 && axis === null && (ord === 'fro' || ord === 'nuc')) {
    return matrix_norm(x, ord, keepdims);
  }

  if (typeof ord !== 'number' && ord !== null) {
    throw new Error(`norm: ord '${ord}' not valid for vector norm`);
  }

  return vector_norm(x, ord ?? 2, axis, keepdims);
}

/**
 * QR decomposition using Householder reflections.
 *
 * @param a - Input matrix (m x n)
 * @param mode - 'reduced' (default), 'complete', 'r', or 'raw'
 * @returns { q, r } where A = Q @ R
 */
export function qr(
  a: ArrayStorage,
  mode: 'reduced' | 'complete' | 'r' | 'raw' = 'reduced'
): { q: ArrayStorage; r: ArrayStorage } | ArrayStorage | { h: ArrayStorage; tau: ArrayStorage } {
  throwIfFloat16(a.dtype);
  if (a.ndim > 2) {
    // Batch mode: iterate over leading dims
    const batchShape = a.shape.slice(0, -2);
    const [m, n] = [a.shape[a.ndim - 2]!, a.shape[a.ndim - 1]!];
    const k = Math.min(m, n);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const qCols = mode === 'complete' ? m : k;

    const qOut = ArrayStorage.zeros([...batchShape, m, qCols], 'float64');
    const rOut = ArrayStorage.zeros([...batchShape, qCols, n], 'float64');

    for (let b = 0; b < batchSize; b++) {
      const bIdx = flatToBatchMultiIndex(b, batchShape);
      const slice = ArrayStorage.zeros([m, n], 'float64');
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) slice.set([i, j], realPart(a.get(...bIdx, i, j)));
      const res = qr(slice, mode) as { q: ArrayStorage; r: ArrayStorage };
      for (let i = 0; i < m; i++)
        for (let j = 0; j < qCols; j++) qOut.set([...bIdx, i, j], Number(res.q.get(i, j)));
      for (let i = 0; i < qCols; i++)
        for (let j = 0; j < n; j++) rOut.set([...bIdx, i, j], Number(res.r.get(i, j)));
      slice.dispose();
      res.q.dispose();
      res.r.dispose();
    }
    return { q: qOut, r: rOut };
  }

  if (a.ndim !== 2) {
    throw new Error(`qr: input must be 2D, got ${a.ndim}D`);
  }

  // WASM fast path for 'reduced' mode
  if (mode === 'reduced') {
    const wasmResult = wasmQr(a);
    if (wasmResult) return wasmResult;
  }

  const [m, n] = a.shape;
  const k = Math.min(m!, n!);

  // Copy input to working array (float64)
  // TODO: implement complex Householder QR; currently extracts real parts only
  const R = ArrayStorage.zeros([m!, n!], 'float64');
  for (let i = 0; i < m!; i++) {
    for (let j = 0; j < n!; j++) {
      R.set([i, j], realPart(a.get(i, j)));
    }
  }

  // Store Householder vectors for Q reconstruction
  const householderVectors: number[][] = [];
  const tau: number[] = [];

  // Householder QR
  for (let j = 0; j < k; j++) {
    // Extract column j from row j onwards
    const colLen = m! - j;
    const col: number[] = [];
    for (let i = j; i < m!; i++) {
      col.push(Number(R.get(i, j)));
    }

    // Compute norm of column
    let normCol = 0;
    for (let i = 0; i < colLen; i++) {
      normCol += col[i]! * col[i]!;
    }
    normCol = Math.sqrt(normCol);

    if (normCol < 1e-15) {
      householderVectors.push(col);
      tau.push(0);
      continue;
    }

    // Compute Householder vector
    const sign = col[0]! >= 0 ? 1 : -1;
    const u0 = col[0]! + sign * normCol;
    const v: number[] = [1];
    for (let i = 1; i < colLen; i++) {
      v.push(col[i]! / u0);
    }

    const tauJ = (sign * u0) / normCol;
    tau.push(tauJ);
    householderVectors.push(v);

    // Apply Householder reflection to R[:, j:]
    // R = R - tau * v * (v^T @ R)
    for (let jj = j; jj < n!; jj++) {
      // Compute v^T @ R[:, jj]
      let vTR = 0;
      for (let i = 0; i < colLen; i++) {
        vTR += v[i]! * Number(R.get(j + i, jj));
      }
      // Update R[:, jj]
      for (let i = 0; i < colLen; i++) {
        R.set([j + i, jj], Number(R.get(j + i, jj)) - tauJ * v[i]! * vTR);
      }
    }
  }

  if (mode === 'raw') {
    // Return raw Householder representation
    const h = ArrayStorage.zeros([m!, n!], 'float64');
    for (let i = 0; i < m!; i++) {
      for (let j = 0; j < n!; j++) {
        h.set([i, j], Number(R.get(i, j)));
      }
    }
    R.dispose();
    const tauArr = ArrayStorage.zeros([k], 'float64');
    for (let i = 0; i < k; i++) {
      tauArr.set([i], tau[i]!);
    }
    return { h, tau: tauArr };
  }

  if (mode === 'r') {
    // Return only R (upper triangular)
    const rResult = ArrayStorage.zeros([k, n!], 'float64');
    for (let i = 0; i < k; i++) {
      for (let j = i; j < n!; j++) {
        rResult.set([i, j], Number(R.get(i, j)));
      }
    }
    R.dispose();
    return rResult;
  }

  // Reconstruct Q from Householder vectors
  const qRows = mode === 'complete' ? m! : k;
  const Q = ArrayStorage.zeros([m!, qRows], 'float64');

  // Initialize Q as identity
  for (let i = 0; i < Math.min(m!, qRows); i++) {
    Q.set([i, i], 1);
  }

  // Apply Householder reflections in reverse order
  for (let j = k - 1; j >= 0; j--) {
    const v = householderVectors[j]!;
    const tauJ = tau[j]!;
    const colLen = m! - j;

    // Apply H_j to Q[j:, j:]
    // Q = Q - tau * v * (v^T @ Q)
    for (let jj = j; jj < qRows; jj++) {
      let vTQ = 0;
      for (let i = 0; i < colLen; i++) {
        vTQ += v[i]! * Number(Q.get(j + i, jj));
      }
      for (let i = 0; i < colLen; i++) {
        Q.set([j + i, jj], Number(Q.get(j + i, jj)) - tauJ * v[i]! * vTQ);
      }
    }
  }

  // Extract final Q and R
  const qResult = ArrayStorage.zeros([m!, qRows], 'float64');
  for (let i = 0; i < m!; i++) {
    for (let j = 0; j < qRows; j++) {
      qResult.set([i, j], Number(Q.get(i, j)));
    }
  }
  Q.dispose();

  const rRows = mode === 'complete' ? m! : k;
  const rResult = ArrayStorage.zeros([rRows, n!], 'float64');
  for (let i = 0; i < rRows; i++) {
    for (let j = 0; j < n!; j++) {
      if (j >= i) {
        rResult.set([i, j], Number(R.get(i, j)));
      }
    }
  }
  R.dispose();

  return { q: qResult, r: rResult };
}

/**
 * Cholesky decomposition.
 *
 * Returns the lower triangular matrix L such that A = L @ L^T
 * for a symmetric positive-definite matrix A.
 *
 * @param a - Symmetric positive-definite matrix
 * @param upper - If true, return upper triangular U such that A = U^T @ U
 * @returns Lower (or upper) triangular Cholesky factor
 */
export function cholesky(a: ArrayStorage, upper: boolean = false): ArrayStorage {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`cholesky: input must be at least 2D, got ${a.ndim}D`);
  }

  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) throw new Error(`cholesky: last 2 dimensions must be square, got ${m2}x${n}`);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const result = ArrayStorage.empty([...batchShape, n, n], 'float64');
    const resultData = result.data as Float64Array;
    for (let bi = 0; bi < batchSize; bi++) {
      const bIdx = flatToBatchMultiIndex(bi, batchShape);
      const slice = ArrayStorage.zeros([n, n], 'float64');
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) slice.set([i, j], realPart(a.get(...bIdx, i, j)));
      const r = cholesky(slice, upper);
      resultData.set(toContiguousFloat64(r), bi * n * n);
      slice.dispose();
      r.dispose();
    }
    return result;
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`cholesky: matrix must be square, got ${m}x${n}`);
  }

  // WASM fast path
  const wasmResult = a.dtype === 'float32' ? wasmCholeskyF32(a) : wasmCholesky(a);
  if (wasmResult) {
    if (upper) {
      // Transpose L to get U
      const size = m!;
      const U = ArrayStorage.zeros([size, size], wasmResult.dtype);
      for (let i = 0; i < size; i++) {
        for (let j = i; j < size; j++) {
          U.set([i, j], Number(wasmResult.get(j, i)));
        }
      }
      wasmResult.dispose();
      return U;
    }
    return wasmResult;
  }

  const size = m!;
  const L = ArrayStorage.zeros([size, size], 'float64');

  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;

      if (i === j) {
        // Diagonal elements
        for (let k = 0; k < j; k++) {
          sum += Number(L.get(j, k)) ** 2;
        }
        const val = realPart(a.get(j, j)) - sum;
        if (val < 0) {
          throw new Error('cholesky: matrix is not positive definite');
        }
        L.set([j, j], Math.sqrt(val));
      } else {
        // Off-diagonal elements
        for (let k = 0; k < j; k++) {
          sum += Number(L.get(i, k)) * Number(L.get(j, k));
        }
        const ljj = Number(L.get(j, j));
        if (Math.abs(ljj) < 1e-15) {
          throw new Error('cholesky: matrix is not positive definite');
        }
        L.set([i, j], (realPart(a.get(i, j)) - sum) / ljj);
      }
    }
  }

  if (upper) {
    // Return L^T (upper triangular)
    const U = ArrayStorage.zeros([size, size], 'float64');
    for (let i = 0; i < size; i++) {
      for (let j = i; j < size; j++) {
        U.set([i, j], Number(L.get(j, i)));
      }
    }
    return U;
  }

  return L;
}

/**
 * Singular Value Decomposition (full).
 * Internal helper that returns all components.
 *
 * @param a - Input matrix
 * @returns { u, s, vt } where A = U @ diag(S) @ V^T
 */
function svdFull(a: ArrayStorage): { u: ArrayStorage; s: ArrayStorage; vt: ArrayStorage } {
  if (a.ndim !== 2) {
    throw new Error(`svd: input must be 2D, got ${a.ndim}D`);
  }

  // WASM fast path
  const wasmResult = wasmSvd(a);
  if (wasmResult) return wasmResult;

  const [m, n] = a.shape;
  const smaller = Math.min(m!, n!);

  // For complex, compute A^H @ A (Hermitian product).
  // The result is a real symmetric matrix (Hermitian with real diagonal).
  // For SVD, A^H @ A has eigenvalues sigma^2.
  // V are eigenvectors of A^H @ A, U = A @ V @ S^-1.
  const isComplex = isComplexDType(a.dtype);

  // Compute A^H @ A (or A^T @ A for real)
  // Result is always real symmetric for Hermitian product
  const ATA = ArrayStorage.zeros([n!, n!], 'float64');
  for (let i = 0; i < n!; i++) {
    for (let j = 0; j < n!; j++) {
      let sumRe = 0;
      for (let k = 0; k < m!; k++) {
        const aki = a.get(k, i);
        const akj = a.get(k, j);
        if (isComplex) {
          // conj(A[k,i]) * A[k,j] — only real part needed (Hermitian product is real-symmetric)
          const aiC = aki instanceof Complex ? aki : new Complex(Number(aki), 0);
          const ajC = akj instanceof Complex ? akj : new Complex(Number(akj), 0);
          sumRe += aiC.re * ajC.re + aiC.im * ajC.im;
        } else {
          sumRe += Number(aki) * Number(akj);
        }
      }
      // For Hermitian A^H @ A, the result should be Hermitian.
      // The diagonal is always real. Off-diagonal: ATA[i,j] = conj(ATA[j,i]).
      // We store only real part since eigSymmetric works on real symmetric matrices.
      // This is valid because for A^H @ A, the imaginary parts are antisymmetric
      // and cancel when we symmetrize.
      ATA.set([i, j], sumRe);
    }
  }

  // Get eigendecomposition of A^H @ A (real symmetric)
  const { values: eigVals, vectors: V } = eigSymmetric(ATA);
  ATA.dispose();

  // Sort eigenvalues in descending order
  const indices = Array.from({ length: n! }, (_, i) => i);
  indices.sort((i, j) => eigVals[j]! - eigVals[i]!);

  // Singular values are sqrt of eigenvalues
  const s = ArrayStorage.zeros([smaller], 'float64');
  for (let i = 0; i < smaller; i++) {
    const eigVal = eigVals[indices[i]!]!;
    s.set([i], Math.sqrt(Math.max(0, eigVal)));
  }

  // V^T (sorted) - real eigenvectors
  const vt = ArrayStorage.zeros([n!, n!], 'float64');
  for (let i = 0; i < n!; i++) {
    for (let j = 0; j < n!; j++) {
      vt.set([i, j], V[j]![indices[i]!]!);
    }
  }

  // Compute U = A @ V @ S^-1
  const u = ArrayStorage.zeros([m!, m!], 'float64');
  for (let i = 0; i < m!; i++) {
    for (let j = 0; j < smaller; j++) {
      const sigma = Number(s.get(j));
      if (sigma > 1e-10) {
        let sumRe = 0;
        for (let k = 0; k < n!; k++) {
          const aik = a.get(i, k);
          const vjk = Number(vt.get(j, k));
          if (isComplex) {
            const c = aik instanceof Complex ? aik : new Complex(Number(aik), 0);
            sumRe += c.re * vjk; // For real V, only real part contributes to U
          } else {
            sumRe += Number(aik) * vjk;
          }
        }
        u.set([i, j], sumRe / sigma);
      }
    }
  }

  // Fill remaining columns of U with orthonormal vectors
  if (m! > smaller) {
    // Use Gram-Schmidt to complete U
    for (let j = smaller; j < m!; j++) {
      // Start with a standard basis vector
      const col: number[] = new Array(m!).fill(0);
      col[j] = 1;

      // Orthogonalize against existing columns
      for (let k = 0; k < j; k++) {
        let dotProd = 0;
        for (let i = 0; i < m!; i++) {
          dotProd += col[i]! * Number(u.get(i, k));
        }
        for (let i = 0; i < m!; i++) {
          col[i] = col[i]! - dotProd * Number(u.get(i, k));
        }
      }

      // Normalize
      let norm = 0;
      for (let i = 0; i < m!; i++) {
        norm += col[i]! * col[i]!;
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let i = 0; i < m!; i++) {
          u.set([i, j], col[i]! / norm);
        }
      }
    }
  }

  return { u, s, vt };
}

/**
 * Symmetric matrix eigendecomposition using Jacobi iteration.
 * Returns eigenvalues and eigenvectors.
 *
 * @param a - Symmetric matrix
 * @returns { values, vectors } - eigenvalues array and eigenvector matrix (columns are eigenvectors)
 */
function eigSymmetric(a: ArrayStorage): { values: number[]; vectors: number[][] } {
  const n = a.shape[0]!;
  const maxIter = 100 * n * n;
  const tol = 1e-10;

  // Copy matrix (extract real parts for complex input)
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    A.push([]);
    for (let j = 0; j < n; j++) {
      A[i]!.push(realPart(a.get(i, j)));
    }
  }

  // Initialize eigenvector matrix as identity
  const V: number[][] = [];
  for (let i = 0; i < n; i++) {
    V.push([]);
    for (let j = 0; j < n; j++) {
      V[i]!.push(i === j ? 1 : 0);
    }
  }

  // Jacobi iteration
  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0;
    let p = 0;
    let q = 1;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i]![j]!) > maxVal) {
          maxVal = Math.abs(A[i]![j]!);
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < tol) break;

    // Compute rotation angle
    const app = A[p]![p]!;
    const aqq = A[q]![q]!;
    const apq = A[p]![q]!;

    let theta: number;
    if (Math.abs(app - aqq) < 1e-15) {
      theta = Math.PI / 4;
    } else {
      // Standard Jacobi rotation: tan(2θ) = 2*A[p,q] / (A[q,q] - A[p,p])
      theta = 0.5 * Math.atan2(2 * apq, aqq - app);
    }

    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Apply rotation to A
    const newApp = c * c * app + s * s * aqq - 2 * s * c * apq;
    const newAqq = s * s * app + c * c * aqq + 2 * s * c * apq;

    A[p]![p] = newApp;
    A[q]![q] = newAqq;
    A[p]![q] = 0;
    A[q]![p] = 0;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const aip = A[i]![p]!;
        const aiq = A[i]![q]!;
        A[i]![p] = c * aip - s * aiq;
        A[p]![i] = A[i]![p]!;
        A[i]![q] = s * aip + c * aiq;
        A[q]![i] = A[i]![q]!;
      }
    }

    // Apply rotation to V (eigenvector accumulation)
    for (let i = 0; i < n; i++) {
      const vip = V[i]![p]!;
      const viq = V[i]![q]!;
      V[i]![p] = c * vip - s * viq;
      V[i]![q] = s * vip + c * viq;
    }
  }

  // Extract eigenvalues from diagonal
  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    values.push(A[i]![i]!);
  }

  return { values, vectors: V };
}

/**
 * Singular Value Decomposition.
 *
 * @param a - Input matrix (m x n)
 * @param full_matrices - If true, return full U and V^T, otherwise reduced
 * @param compute_uv - If true, return U, S, V^T; if false, return only S
 * @returns { u, s, vt } or just s depending on compute_uv
 */
export function svd(
  a: ArrayStorage,
  full_matrices: boolean = true,
  compute_uv: boolean = true
): { u: ArrayStorage; s: ArrayStorage; vt: ArrayStorage } | ArrayStorage {
  throwIfFloat16(a.dtype);
  // Batch mode: iterate over leading dims
  if (a.ndim > 2) {
    const batchShape = a.shape.slice(0, -2);
    const [m, n] = [a.shape[a.ndim - 2]!, a.shape[a.ndim - 1]!];
    const k = Math.min(m, n);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

    if (!compute_uv) {
      const sOut = ArrayStorage.zeros([...batchShape, k], 'float64');
      for (let b = 0; b < batchSize; b++) {
        const bIdx = flatToBatchMultiIndex(b, batchShape);
        const slice = ArrayStorage.zeros([m, n], a.dtype);
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++) slice.set([i, j], a.get(...bIdx, i, j));
        const { u, s, vt } = svdFull(slice);
        for (let i = 0; i < k; i++) sOut.set([...bIdx, i], Number(s.get(i)));
        slice.dispose();
        u.dispose();
        s.dispose();
        vt.dispose();
      }
      return sOut;
    }

    const uCols = full_matrices ? m : k;
    const vtRows = full_matrices ? n : k;
    const uOut = ArrayStorage.zeros([...batchShape, m, uCols], 'float64');
    const sOut = ArrayStorage.zeros([...batchShape, k], 'float64');
    const vtOut = ArrayStorage.zeros([...batchShape, vtRows, n], 'float64');

    for (let b = 0; b < batchSize; b++) {
      const bIdx = flatToBatchMultiIndex(b, batchShape);
      const slice = ArrayStorage.zeros([m, n], a.dtype);
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) slice.set([i, j], a.get(...bIdx, i, j));
      const res = svd(slice, full_matrices, true) as {
        u: ArrayStorage;
        s: ArrayStorage;
        vt: ArrayStorage;
      };
      for (let i = 0; i < m; i++)
        for (let j = 0; j < uCols; j++) uOut.set([...bIdx, i, j], Number(res.u.get(i, j)));
      for (let i = 0; i < k; i++) sOut.set([...bIdx, i], Number(res.s.get(i)));
      for (let i = 0; i < vtRows; i++)
        for (let j = 0; j < n; j++) vtOut.set([...bIdx, i, j], Number(res.vt.get(i, j)));
      slice.dispose();
      res.u.dispose();
      res.s.dispose();
      res.vt.dispose();
    }
    return { u: uOut, s: sOut, vt: vtOut };
  }

  const result = svdFull(a);

  if (!compute_uv) {
    result.u.dispose();
    result.vt.dispose();
    return result.s;
  }

  if (!full_matrices) {
    const [m, n] = a.shape;
    const k = Math.min(m!, n!);

    // Reduced U: m x k
    const uReduced = ArrayStorage.zeros([m!, k], 'float64');
    for (let i = 0; i < m!; i++) {
      for (let j = 0; j < k; j++) {
        uReduced.set([i, j], Number(result.u.get(i, j)));
      }
    }
    result.u.dispose();

    // Reduced V^T: k x n
    const vtReduced = ArrayStorage.zeros([k, n!], 'float64');
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n!; j++) {
        vtReduced.set([i, j], Number(result.vt.get(i, j)));
      }
    }
    result.vt.dispose();

    return { u: uReduced, s: result.s, vt: vtReduced };
  }

  return result;
}

/**
 * Compute the determinant of a square matrix.
 *
 * Uses LU decomposition for numerical stability.
 *
 * @param a - Square matrix
 * @returns Determinant
 */
export function det(a: ArrayStorage): number | Complex | ArrayStorage {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`det: input must be at least 2D, got ${a.ndim}D`);
  }

  const isComplex = isComplexDType(a.dtype);

  // Batch case: ndim > 2 → apply det to each 2D slice, return array of scalars
  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) {
      throw new Error(`det: last 2 dimensions must be square, got ${m2}x${n}`);
    }
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

    if (isComplex) {
      const outDtype = a.dtype;
      const result = ArrayStorage.zeros(batchShape, outDtype);
      for (let bi = 0; bi < batchSize; bi++) {
        // Extract 2D slice via .get()
        const batchIdx: number[] = [];
        let rem = bi;
        for (let d = batchShape.length - 1; d >= 0; d--) {
          batchIdx[d] = rem % batchShape[d]!;
          rem = Math.floor(rem / batchShape[d]!);
        }
        const slice = ArrayStorage.zeros([n, n], outDtype);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            slice.set([i, j], a.get(...batchIdx, i, j));
          }
        }
        const d = det(slice) as Complex;
        // Store result using iset
        result.iset(bi, d);
        slice.dispose();
      }
      return result;
    }

    const result = ArrayStorage.empty(batchShape, 'float64');
    const resultData = result.data as Float64Array;
    const aData = toContiguousFloat64(a);

    for (let bi = 0; bi < batchSize; bi++) {
      const off = bi * n * n;
      const slice = ArrayStorage.fromData(aData.slice(off, off + n * n), [n, n], 'float64');
      try {
        resultData[bi] = det(slice) as number;
      } finally {
        slice.dispose();
      }
    }
    return result;
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`det: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  if (size === 0) {
    return isComplex ? new Complex(1, 0) : 1;
  }

  if (isComplex) {
    if (size === 1) {
      return a.get(0, 0) as Complex;
    }
    if (size === 2) {
      const a00 = a.get(0, 0) as Complex;
      const a01 = a.get(0, 1) as Complex;
      const a10 = a.get(1, 0) as Complex;
      const a11 = a.get(1, 1) as Complex;
      return a00.mul(a11).sub(a01.mul(a10));
    }

    // Complex LU decomposition
    const { lu, sign } = luDecomposition(a);
    try {
      const luData = lu.data as Float64Array;
      let re = sign as number;
      let im = 0;
      for (let i = 0; i < size; i++) {
        const idx = (i * size + i) * 2;
        const dRe = luData[idx]!;
        const dIm = luData[idx + 1]!;
        const newRe = re * dRe - im * dIm;
        const newIm = re * dIm + im * dRe;
        re = newRe;
        im = newIm;
      }
      return new Complex(re, im);
    } finally {
      lu.dispose();
    }
  }

  const aData = a.data;

  if (size === 1) {
    return Number(aData[0]);
  }

  if (size === 2) {
    return Number(aData[0]) * Number(aData[3]) - Number(aData[1]) * Number(aData[2]);
  }

  // WASM fast path for f64/f32
  if (a.dtype === 'float64' || a.dtype === 'float32') {
    const factored = wasmLuFactor(a);
    if (factored) {
      const luData = factored.lu.data as Float64Array | Float32Array;
      let result: number = factored.sign;
      for (let i = 0; i < size; i++) result *= luData[i * size + i]!;
      factored.lu.dispose();
      return result;
    }
  }

  // JS fallback: LU decomposition with partial pivoting
  const { lu, sign } = luDecomposition(a);

  try {
    const luData = lu.data as Float64Array;
    let result = sign;
    for (let i = 0; i < size; i++) {
      result *= luData[i * size + i]!;
    }

    return result;
  } finally {
    lu.dispose();
  }
}

/**
 * LU decomposition with partial pivoting - optimized with direct array access.
 *
 * @param a - Input matrix
 * @returns { lu, piv, sign } - Combined L\U matrix, pivot indices, and sign from pivoting
 */
function luDecomposition(a: ArrayStorage): { lu: ArrayStorage; piv: number[]; sign: number } {
  const [m, n] = a.shape;
  const size = m!;
  const cols = n!;
  const isComplex = isComplexDType(a.dtype);

  if (isComplex) {
    return luDecompositionComplex(a, size, cols);
  }

  // Real path: use direct array access for speed
  const lu = ArrayStorage.zeros([size, cols], 'float64');
  const luData = lu.data as Float64Array;
  const aData = a.data;

  // Fast copy
  for (let i = 0; i < size * cols; i++) {
    luData[i] = Number(aData[i]);
  }

  const piv: number[] = Array.from({ length: size }, (_, i) => i);
  let sign = 1;

  for (let k = 0; k < Math.min(size, cols); k++) {
    // Find pivot - direct array access
    let maxVal = Math.abs(luData[k * cols + k]!);
    let maxRow = k;

    for (let i = k + 1; i < size; i++) {
      const val = Math.abs(luData[i * cols + k]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    // Swap rows - direct array access
    if (maxRow !== k) {
      for (let j = 0; j < cols; j++) {
        const temp = luData[k * cols + j]!;
        luData[k * cols + j] = luData[maxRow * cols + j]!;
        luData[maxRow * cols + j] = temp;
      }
      const tempPiv = piv[k]!;
      piv[k] = piv[maxRow]!;
      piv[maxRow] = tempPiv;
      sign = -sign;
    }

    // Eliminate - direct array access
    const pivot = luData[k * cols + k]!;
    if (Math.abs(pivot) > 1e-15) {
      for (let i = k + 1; i < size; i++) {
        const factor = luData[i * cols + k]! / pivot;
        luData[i * cols + k] = factor;
        for (let j = k + 1; j < cols; j++) {
          luData[i * cols + j] = luData[i * cols + j]! - factor * luData[k * cols + j]!;
        }
      }
    }
  }

  return { lu, piv, sign };
}

/**
 * Complex LU decomposition with partial pivoting.
 * Works with interleaved re/im data in the underlying Float64Array.
 * TODO: move this to WASM
 */
function luDecompositionComplex(
  a: ArrayStorage,
  size: number,
  cols: number
): { lu: ArrayStorage; piv: number[]; sign: number } {
  // Use complex128 for the LU result
  const lu = ArrayStorage.zeros([size, cols], 'complex128');
  const luData = lu.data as Float64Array; // interleaved [re, im, re, im, ...]

  // Copy from input: each logical element is 2 floats
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < cols; j++) {
      const val = a.get(i, j);
      const idx = (i * cols + j) * 2;
      if (val instanceof Complex) {
        luData[idx] = val.re;
        luData[idx + 1] = val.im;
      } else {
        luData[idx] = Number(val);
        luData[idx + 1] = 0;
      }
    }
  }

  const piv: number[] = Array.from({ length: size }, (_, i) => i);
  let sign = 1;

  for (let k = 0; k < Math.min(size, cols); k++) {
    // Find pivot by magnitude
    const kIdx = (k * cols + k) * 2;
    let maxVal = Math.sqrt(luData[kIdx]! * luData[kIdx]! + luData[kIdx + 1]! * luData[kIdx + 1]!);
    let maxRow = k;

    for (let i = k + 1; i < size; i++) {
      const idx = (i * cols + k) * 2;
      const mag = Math.sqrt(luData[idx]! * luData[idx]! + luData[idx + 1]! * luData[idx + 1]!);
      if (mag > maxVal) {
        maxVal = mag;
        maxRow = i;
      }
    }

    // Swap rows
    if (maxRow !== k) {
      for (let j = 0; j < cols; j++) {
        const kj = (k * cols + j) * 2;
        const mj = (maxRow * cols + j) * 2;
        const tmpRe = luData[kj]!;
        const tmpIm = luData[kj + 1]!;
        luData[kj] = luData[mj]!;
        luData[kj + 1] = luData[mj + 1]!;
        luData[mj] = tmpRe;
        luData[mj + 1] = tmpIm;
      }
      const tempPiv = piv[k]!;
      piv[k] = piv[maxRow]!;
      piv[maxRow] = tempPiv;
      sign = -sign;
    }

    // Eliminate: complex division and subtraction
    const pivIdx = (k * cols + k) * 2;
    const pivRe = luData[pivIdx]!;
    const pivIm = luData[pivIdx + 1]!;
    const pivMag2 = pivRe * pivRe + pivIm * pivIm;

    if (pivMag2 > 1e-30) {
      for (let i = k + 1; i < size; i++) {
        // factor = lu[i,k] / lu[k,k]  (complex division)
        const ikIdx = (i * cols + k) * 2;
        const aRe = luData[ikIdx]!;
        const aIm = luData[ikIdx + 1]!;
        // (aRe + aIm*i) / (pivRe + pivIm*i) = ((aRe*pivRe + aIm*pivIm) + (aIm*pivRe - aRe*pivIm)*i) / pivMag2
        const fRe = (aRe * pivRe + aIm * pivIm) / pivMag2;
        const fIm = (aIm * pivRe - aRe * pivIm) / pivMag2;
        luData[ikIdx] = fRe;
        luData[ikIdx + 1] = fIm;

        for (let j = k + 1; j < cols; j++) {
          const ijIdx = (i * cols + j) * 2;
          const kjIdx = (k * cols + j) * 2;
          // lu[i,j] -= factor * lu[k,j]  (complex multiply + subtract)
          const ukjRe = luData[kjIdx]!;
          const ukjIm = luData[kjIdx + 1]!;
          luData[ijIdx] = luData[ijIdx]! - (fRe * ukjRe - fIm * ukjIm);
          luData[ijIdx + 1] = luData[ijIdx + 1]! - (fRe * ukjIm + fIm * ukjRe);
        }
      }
    }
  }

  return { lu, piv, sign };
}

/**
 * Compute the matrix inverse - optimized to do LU decomposition once.
 *
 * @param a - Square matrix
 * @returns Inverse matrix
 */
export function inv(a: ArrayStorage): ArrayStorage {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`inv: input must be at least 2D, got ${a.ndim}D`);
  }

  const isComplex = isComplexDType(a.dtype);

  // Batch case: ndim > 2 → apply inv to each 2D slice
  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) {
      throw new Error(`inv: last 2 dimensions must be square, got ${m2}x${n}`);
    }
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

    if (isComplex) {
      const outDtype = a.dtype;
      const result = ArrayStorage.zeros(Array.from(a.shape), outDtype);
      for (let bi = 0; bi < batchSize; bi++) {
        const batchIdx: number[] = [];
        let rem = bi;
        for (let d = batchShape.length - 1; d >= 0; d--) {
          batchIdx[d] = rem % batchShape[d]!;
          rem = Math.floor(rem / batchShape[d]!);
        }
        const slice = ArrayStorage.zeros([n, n], outDtype);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            slice.set([i, j], a.get(...batchIdx, i, j));
          }
        }
        const invSlice = inv(slice);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            result.set([...batchIdx, i, j], invSlice.get(i, j));
          }
        }
        slice.dispose();
        invSlice.dispose();
      }
      return result;
    }

    const aData = toContiguousFloat64(a);
    const result = ArrayStorage.empty(Array.from(a.shape), 'float64');
    const resultData = result.data as Float64Array;

    for (let bi = 0; bi < batchSize; bi++) {
      const off = bi * n * n;
      const slice = ArrayStorage.fromData(aData.slice(off, off + n * n), [n, n], 'float64');
      const invSlice = inv(slice);
      const invData = invSlice.data as Float64Array;
      for (let i = 0; i < n * n; i++) {
        resultData[off + i] = invData[i]!;
      }
      slice.dispose();
      invSlice.dispose();
    }
    return result;
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`inv: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  if (isComplex) {
    return invComplex(a, size);
  }

  // WASM fast path for f64/f32
  if (a.dtype === 'float64' || a.dtype === 'float32') {
    const factored = wasmLuFactor(a);
    if (factored) {
      const { lu: wasmLu, piv: wasmPiv } = factored;
      // Check for singularity: any zero on the LU diagonal
      const luData = wasmLu.data as Float64Array | Float32Array;
      for (let i = 0; i < size; i++) {
        if (Math.abs(luData[i * size + i]!) < 1e-15) {
          wasmLu.dispose();
          throw new Error('inv: singular matrix');
        }
      }
      const result = wasmLuInv(wasmLu, wasmPiv, a.dtype);
      wasmLu.dispose();
      if (result) return result;
    }
  }

  // JS fallback
  const { lu, piv } = luDecomposition(a);
  const luData = lu.data as Float64Array;

  const result = ArrayStorage.zeros([size, size], 'float64');
  const resultData = result.data as Float64Array;

  for (let col = 0; col < size; col++) {
    const y = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let sum = piv[i] === col ? 1 : 0;
      for (let j = 0; j < i; j++) {
        sum -= luData[i * size + j]! * y[j]!;
      }
      y[i] = sum;
    }

    for (let i = size - 1; i >= 0; i--) {
      let sum = y[i]!;
      for (let j = i + 1; j < size; j++) {
        sum -= luData[i * size + j]! * resultData[j * size + col]!;
      }
      const diag = luData[i * size + i]!;
      if (Math.abs(diag) < 1e-15) {
        throw new Error('inv: singular matrix');
      }
      resultData[i * size + col] = sum / diag;
    }
  }

  lu.dispose();
  return result;
}

/**
 * Complex matrix inverse via LU decomposition.
 * Forward/back substitution using interleaved complex data.
 * TODO: move this to WASM
 */
function invComplex(a: ArrayStorage, size: number): ArrayStorage {
  const { lu, piv } = luDecomposition(a);
  const luData = lu.data as Float64Array; // interleaved [re, im, ...]

  // Check singularity
  for (let i = 0; i < size; i++) {
    const idx = (i * size + i) * 2;
    const mag2 = luData[idx]! * luData[idx]! + luData[idx + 1]! * luData[idx + 1]!;
    if (mag2 < 1e-30) {
      lu.dispose();
      throw new Error('inv: singular matrix');
    }
  }

  const outDtype = a.dtype === 'complex64' ? 'complex64' : 'complex128';
  const result = ArrayStorage.zeros([size, size], outDtype);
  const resultData = result.data as Float64Array;

  // For each column of the identity matrix
  for (let col = 0; col < size; col++) {
    // y = forward substitution: L * y = P * e_col
    const yRe = new Float64Array(size);
    const yIm = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let sRe = piv[i] === col ? 1 : 0;
      let sIm = 0;
      for (let j = 0; j < i; j++) {
        const lIdx = (i * size + j) * 2;
        const lRe = luData[lIdx]!;
        const lIm = luData[lIdx + 1]!;
        // s -= L[i,j] * y[j]
        sRe -= lRe * yRe[j]! - lIm * yIm[j]!;
        sIm -= lRe * yIm[j]! + lIm * yRe[j]!;
      }
      yRe[i] = sRe;
      yIm[i] = sIm;
    }

    // x = back substitution: U * x = y
    for (let i = size - 1; i >= 0; i--) {
      let sRe = yRe[i]!;
      let sIm = yIm[i]!;
      for (let j = i + 1; j < size; j++) {
        const uIdx = (i * size + j) * 2;
        const uRe = luData[uIdx]!;
        const uIm = luData[uIdx + 1]!;
        const rIdx = (j * size + col) * 2;
        const xRe = resultData[rIdx]!;
        const xIm = resultData[rIdx + 1]!;
        // s -= U[i,j] * result[j,col]
        sRe -= uRe * xRe - uIm * xIm;
        sIm -= uRe * xIm + uIm * xRe;
      }
      // result[i,col] = s / U[i,i]
      const dIdx = (i * size + i) * 2;
      const dRe = luData[dIdx]!;
      const dIm = luData[dIdx + 1]!;
      const dMag2 = dRe * dRe + dIm * dIm;
      const rIdx = (i * size + col) * 2;
      resultData[rIdx] = (sRe * dRe + sIm * dIm) / dMag2;
      resultData[rIdx + 1] = (sIm * dRe - sRe * dIm) / dMag2;
    }
  }

  lu.dispose();
  return result;
}

/**
 * Solve a linear system A @ x = b for a vector b - optimized with direct array access.
 *
 * @param a - Coefficient matrix
 * @param b - Right-hand side vector
 * @returns Solution vector x
 */
function solveVector(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const [m] = a.shape;
  const size = m!;
  const isComplex = isComplexDType(a.dtype) || isComplexDType(b.dtype);

  if (isComplex) {
    return solveVectorComplex(a, b, size);
  }

  // LU decomposition
  const { lu, piv } = luDecomposition(a);
  try {
    const luData = lu.data as Float64Array;
    const bData = b.data;

    // Apply permutation to b - direct array access
    const pb = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      pb[i] = Number(bData[piv[i]!]);
    }

    // Forward substitution (L @ y = Pb) - direct array access
    const y = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let sum = pb[i]!;
      for (let j = 0; j < i; j++) {
        sum -= luData[i * size + j]! * y[j]!;
      }
      y[i] = sum;
    }

    // Back substitution (U @ x = y) - direct array access
    const x = ArrayStorage.zeros([size], 'float64');
    const xData = x.data as Float64Array;
    for (let i = size - 1; i >= 0; i--) {
      let sum = y[i]!;
      for (let j = i + 1; j < size; j++) {
        sum -= luData[i * size + j]! * xData[j]!;
      }
      const diag = luData[i * size + i]!;
      if (Math.abs(diag) < 1e-15) {
        throw new Error('solve: singular matrix');
      }
      xData[i] = sum / diag;
    }

    return x;
  } finally {
    lu.dispose();
  }
}

/**
 * Complex vector solve: A @ x = b using LU decomposition.
 */
function solveVectorComplex(a: ArrayStorage, b: ArrayStorage, size: number): ArrayStorage {
  const { lu, piv } = luDecomposition(a);
  try {
    const luData = lu.data as Float64Array;

    // Apply permutation to b
    const pbRe = new Float64Array(size);
    const pbIm = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const val = b.get(piv[i]!);
      if (val instanceof Complex) {
        pbRe[i] = val.re;
        pbIm[i] = val.im;
      } else {
        pbRe[i] = Number(val);
        pbIm[i] = 0;
      }
    }

    // Forward substitution (L @ y = Pb)
    const yRe = new Float64Array(size);
    const yIm = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let sRe = pbRe[i]!;
      let sIm = pbIm[i]!;
      for (let j = 0; j < i; j++) {
        const lIdx = (i * size + j) * 2;
        const lRe = luData[lIdx]!;
        const lIm = luData[lIdx + 1]!;
        sRe -= lRe * yRe[j]! - lIm * yIm[j]!;
        sIm -= lRe * yIm[j]! + lIm * yRe[j]!;
      }
      yRe[i] = sRe;
      yIm[i] = sIm;
    }

    // Back substitution (U @ x = y)
    const outDtype = isComplexDType(a.dtype)
      ? a.dtype
      : isComplexDType(b.dtype)
        ? b.dtype
        : 'complex128';
    const x = ArrayStorage.zeros([size], outDtype);
    const xData = x.data as Float64Array;
    for (let i = size - 1; i >= 0; i--) {
      let sRe = yRe[i]!;
      let sIm = yIm[i]!;
      for (let j = i + 1; j < size; j++) {
        const uIdx = (i * size + j) * 2;
        const uRe = luData[uIdx]!;
        const uIm = luData[uIdx + 1]!;
        const xjRe = xData[j * 2]!;
        const xjIm = xData[j * 2 + 1]!;
        sRe -= uRe * xjRe - uIm * xjIm;
        sIm -= uRe * xjIm + uIm * xjRe;
      }
      const dIdx = (i * size + i) * 2;
      const dRe = luData[dIdx]!;
      const dIm = luData[dIdx + 1]!;
      const dMag2 = dRe * dRe + dIm * dIm;
      if (dMag2 < 1e-30) {
        throw new Error('solve: singular matrix');
      }
      xData[i * 2] = (sRe * dRe + sIm * dIm) / dMag2;
      xData[i * 2 + 1] = (sIm * dRe - sRe * dIm) / dMag2;
    }

    return x;
  } finally {
    lu.dispose();
  }
}

/**
 * Solve a linear system A @ x = b.
 *
 * @param a - Coefficient matrix (n x n)
 * @param b - Right-hand side (n,) or (n, k)
 * @returns Solution x with same shape as b
 */
export function solve(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  throwIfFloat16(a.dtype);
  if (a.ndim !== 2) {
    throw new Error(`solve: coefficient matrix must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`solve: coefficient matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // WASM fast path: factor once, solve all columns
  if ((a.dtype === 'float64' || a.dtype === 'float32') && b.isCContiguous) {
    const factored = wasmLuFactor(a);
    if (factored) {
      const { lu: wasmLu, piv: wasmPiv } = factored;
      const workDtype = a.dtype;

      if (b.ndim === 1) {
        if (b.shape[0] !== size) {
          wasmLu.dispose();
          throw new Error(`solve: incompatible shapes (${m},${n}) and (${b.shape[0]},)`);
        }
        const bConverted =
          b.dtype === workDtype
            ? b
            : ArrayStorage.fromData(
                new (workDtype === 'float32' ? Float32Array : Float64Array)(
                  Array.from({ length: size }, (_, i) => Number(b.iget(i)))
                ),
                [size],
                workDtype
              );
        const result = wasmLuSolve(wasmLu, wasmPiv, bConverted, workDtype);
        wasmLu.dispose();
        if (bConverted !== b) bConverted.dispose();
        if (result) return result;
      }

      if (b.ndim === 2) {
        if (b.shape[0] !== size) {
          wasmLu.dispose();
          throw new Error(
            `solve: incompatible shapes (${m},${n}) and (${b.shape[0]},${b.shape[1]})`
          );
        }
        const k = b.shape[1]!;
        const Ctor = workDtype === 'float32' ? Float32Array : Float64Array;
        const result = ArrayStorage.empty([size, k], workDtype);
        const resultData = result.data as Float64Array | Float32Array;
        const bData = b.data;

        for (let j = 0; j < k; j++) {
          // Extract column j into contiguous vector
          const col = new Ctor(size);
          for (let i = 0; i < size; i++) col[i] = Number(bData[b.offset + i * k + j]);
          const colStorage = ArrayStorage.fromData(col, [size], workDtype);
          const xCol = wasmLuSolve(wasmLu, wasmPiv, colStorage, workDtype);
          colStorage.dispose();
          if (xCol) {
            const xData = xCol.data as Float64Array | Float32Array;
            for (let i = 0; i < size; i++) resultData[i * k + j] = xData[i]!;
            xCol.dispose();
          }
        }
        wasmLu.dispose();
        return result;
      }

      wasmLu.dispose();
    }
  }

  // JS fallback
  if (b.ndim === 1) {
    if (b.shape[0] !== size) {
      throw new Error(`solve: incompatible shapes (${m},${n}) and (${b.shape[0]},)`);
    }
    return solveVector(a, b);
  }

  if (b.ndim === 2) {
    if (b.shape[0] !== size) {
      throw new Error(`solve: incompatible shapes (${m},${n}) and (${b.shape[0]},${b.shape[1]})`);
    }

    const isComplex = isComplexDType(a.dtype) || isComplexDType(b.dtype);
    const k = b.shape[1]!;
    const outDtype = isComplex ? (isComplexDType(a.dtype) ? a.dtype : b.dtype) : 'float64';
    const result = ArrayStorage.zeros([size, k], outDtype);

    for (let j = 0; j < k; j++) {
      const bColDtype = isComplexDType(b.dtype) ? b.dtype : isComplex ? 'complex128' : 'float64';
      const bCol = ArrayStorage.zeros([size], bColDtype);
      for (let i = 0; i < size; i++) {
        bCol.set([i], b.get(i, j));
      }

      const xCol = solveVector(a, bCol);

      for (let i = 0; i < size; i++) {
        result.set([i, j], xCol.get(i));
      }

      bCol.dispose();
      xCol.dispose();
    }

    return result;
  }

  throw new Error(`solve: b must be 1D or 2D, got ${b.ndim}D`);
}

/**
 * Compute the least-squares solution to a linear matrix equation.
 *
 * @param a - Coefficient matrix (m x n)
 * @param b - Right-hand side (m,) or (m, k)
 * @param rcond - Cutoff for small singular values (default: machine precision * max(m, n))
 * @returns { x, residuals, rank, s } - Solution, residuals, effective rank, singular values
 */
export function lstsq(
  a: ArrayStorage,
  b: ArrayStorage,
  rcond: number | null = null
): { x: ArrayStorage; residuals: ArrayStorage; rank: number; s: ArrayStorage } {
  throwIfFloat16(a.dtype);
  if (a.ndim !== 2) {
    throw new Error(`lstsq: coefficient matrix must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  const k = Math.min(m!, n!);

  // Handle 1D b
  const b2D = b.ndim === 1 ? shapeOps.reshape(b, [b.size, 1]) : b;
  const nrhs = b2D.shape[1]!;

  if (b2D.shape[0] !== m) {
    throw new Error(`lstsq: incompatible shapes (${m},${n}) and (${b.shape.join(',')})`);
  }

  // SVD for singular values, rank, and x computation
  const { u, s, vt } = svdFull(a);
  try {
    const sData = s.data as Float64Array;
    const uData = u.data as Float64Array;
    const vtData = vt.data as Float64Array;

    // Determine rcond and rank
    const threshold = rcond ?? Math.max(m!, n!) * Number.EPSILON;
    const maxSigma = sData[0]!;
    const cutoff = maxSigma * threshold;
    let rank = 0;
    for (let i = 0; i < k; i++) {
      if (sData[i]! > cutoff) rank++;
    }

    // Compute x = V @ S^+ @ U^T @ b using WASM matmul
    // Build (V @ S^+) as n×k matrix, U^T as k×m, then matmul chains
    const vsInv = ArrayStorage.zeros([n!, k], 'float64');
    const vsInvData = vsInv.data as Float64Array;
    for (let l = 0; l < k; l++) {
      const sigma = sData[l]!;
      if (sigma > cutoff) {
        const invSigma = 1.0 / sigma;
        for (let i = 0; i < n!; i++) {
          vsInvData[i * k + l] = vtData[l * n! + i]! * invSigma;
        }
      }
    }

    // U^T truncated: k × m
    const ut = ArrayStorage.empty([k, m!], 'float64');
    const utData = ut.data as Float64Array;
    for (let l = 0; l < k; l++) {
      for (let j = 0; j < m!; j++) {
        utData[l * m! + j] = uData[j * m! + l]!;
      }
    }

    // x = (V @ S^+) @ (U^T @ b) via two WASM matmuls
    const utb = wasmMatmul(ut, b2D) ?? matmul2D(ut, b2D); // k × nrhs
    let x: ArrayStorage = wasmMatmul(vsInv, utb) ?? matmul2D(vsInv, utb); // n × nrhs
    vsInv.dispose();
    ut.dispose();
    utb.dispose();

    // Compute residuals if m > n (overdetermined) and full rank
    let residuals: ArrayStorage;
    if (m! > n! && rank === n!) {
      residuals = ArrayStorage.empty([nrhs], 'float64');
      const resArr = residuals.data as Float64Array;
      const xForMul = b.ndim === 1 ? shapeOps.reshape(x, [n!, 1]) : x;
      const ax = wasmMatmul(a, xForMul) ?? matmul2D(a, xForMul);
      if (xForMul !== x) xForMul.dispose();
      const axData = ax.data as Float64Array;
      for (let j = 0; j < nrhs; j++) {
        let resSum = 0;
        for (let i = 0; i < m!; i++) {
          const diff = axData[i * nrhs + j]! - Number(b2D.iget(i * nrhs + j));
          resSum += diff * diff;
        }
        resArr[j] = resSum;
      }
      ax.dispose();
    } else {
      residuals = ArrayStorage.zeros([0], 'float64');
    }

    // Reshape x if b was 1D
    const xResult = b.ndim === 1 ? shapeOps.reshape(x, [n!]) : x;
    if (xResult !== x) x.dispose();

    if (b2D !== b) b2D.dispose();
    return { x: xResult, residuals, rank, s };
  } finally {
    u.dispose();
    vt.dispose();
  }
}

/**
 * Compute the condition number of a matrix.
 *
 * @param a - Input matrix
 * @param p - Order of the norm (default: 2, -2, 'fro', or inf)
 * @returns Condition number
 */
export function cond(a: ArrayStorage, p: number | 'fro' | 'nuc' = 2): number {
  throwIfFloat16(a.dtype);
  if (a.ndim !== 2) {
    throw new Error(`cond: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;

  if (p === 2 || p === -2) {
    // Condition number from singular values (values only — no U/V needed)
    const s = svdvals(a);
    try {
      const k = Math.min(m!, n!);
      const maxS = Number(s.get(0));
      const minS = Number(s.get(k - 1));

      if (p === 2) {
        return minS > 0 ? maxS / minS : Infinity;
      } else {
        return maxS > 0 ? minS / maxS : 0;
      }
    } finally {
      s.dispose();
    }
  }

  // For other norms, compute norm(A) * norm(inv(A))
  if (m !== n) {
    throw new Error(`cond: matrix must be square for p=${p}`);
  }

  const normA = matrix_norm(a, p as 'fro' | number) as number;
  const invA = inv(a);
  try {
    const normInvA = matrix_norm(invA, p as 'fro' | number) as number;
    return normA * normInvA;
  } finally {
    invA.dispose();
  }
}

/**
 * Compute the rank of a matrix using SVD.
 *
 * @param a - Input matrix
 * @param tol - Threshold below which singular values are considered zero
 * @returns Matrix rank
 */
export function matrix_rank(a: ArrayStorage, tol?: number): number {
  throwIfFloat16(a.dtype);
  if (a.ndim === 0) {
    return absValue(a.get()) !== 0 ? 1 : 0;
  }

  if (a.ndim === 1) {
    for (let i = 0; i < a.size; i++) {
      if (absValue(a.get(i)) !== 0) return 1;
    }
    return 0;
  }

  if (a.ndim !== 2) {
    throw new Error(`matrix_rank: input must be at most 2D, got ${a.ndim}D`);
  }

  const s = svdvals(a);
  try {
    const maxS = Number(s.get(0));

    // Default tolerance
    const threshold = tol ?? maxS * Math.max(a.shape[0]!, a.shape[1]!) * Number.EPSILON;

    let rank = 0;
    for (let i = 0; i < s.size; i++) {
      if (Number(s.get(i)) > threshold) {
        rank++;
      }
    }

    return rank;
  } finally {
    s.dispose();
  }
}

/**
 * Raise a square matrix to an integer power.
 *
 * @param a - Input square matrix
 * @param n - Integer power (can be negative)
 * @returns Matrix raised to power n
 */
export function matrix_power(a: ArrayStorage, n: number): ArrayStorage {
  if (a.ndim !== 2) {
    throw new Error(`matrix_power: input must be 2D, got ${a.ndim}D`);
  }

  const [m, k] = a.shape;
  if (m !== k) {
    throw new Error(`matrix_power: matrix must be square, got ${m}x${k}`);
  }

  const size = m!;

  if (!Number.isInteger(n)) {
    throw new Error('matrix_power: exponent must be an integer');
  }

  // Preserve complex/bigint/float32 dtypes; all other integer types upcast to float64
  const isComplex = isComplexDType(a.dtype);
  const isBigInt = isBigIntDType(a.dtype);
  const outDtype = isComplex
    ? a.dtype
    : a.dtype === 'float32'
      ? 'float32'
      : isBigInt
        ? a.dtype
        : 'float64';
  const one: number | bigint | Complex = isComplex ? new Complex(1, 0) : isBigInt ? 1n : 1;

  // Handle n = 0: return identity
  if (n === 0) {
    const result = ArrayStorage.zeros([size, size], outDtype);
    for (let i = 0; i < size; i++) {
      result.set([i, i], one);
    }
    return result;
  }

  // Handle negative powers: A^-n = (A^-1)^n
  let base = a;
  let power = n;
  if (n < 0) {
    base = inv(a);
    power = -n;
  }

  // Use binary exponentiation
  let result = ArrayStorage.zeros([size, size], outDtype);
  for (let i = 0; i < size; i++) {
    result.set([i, i], one);
  }

  // Copy base data
  let current: ArrayStorage;
  if (base.isCContiguous && base.dtype === outDtype) {
    current = base.copy();
  } else {
    current = ArrayStorage.zeros([size, size], outDtype);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        current.set([i, j], base.get(i, j));
      }
    }
  }

  try {
    while (power > 0) {
      if (power & 1) {
        const oldResult = result;
        result = matmul(result, current);
        oldResult.dispose();
      }
      power >>= 1;
      if (power) {
        const oldCurrent = current;
        current = matmul(current, current);
        oldCurrent.dispose();
      }
    }

    return result;
  } finally {
    current.dispose();
    if (n < 0) base.dispose(); // base was created by inv()
  }
}

/**
 * Compute the Moore-Penrose pseudo-inverse using SVD.
 *
 * @param a - Input matrix
 * @param rcond - Cutoff for small singular values
 * @returns Pseudo-inverse of a
 */
export function pinv(a: ArrayStorage, rcond: number = 1e-15): ArrayStorage {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`pinv: input must be at least 2D, got ${a.ndim}D`);
  }

  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const m2 = a.shape[a.ndim - 2]!;
    const n2 = a.shape[a.ndim - 1]!;
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const result = ArrayStorage.empty([...batchShape, n2, m2], 'float64');
    const resultData = result.data as Float64Array;
    const aData = toContiguousFloat64(a);
    for (let bi = 0; bi < batchSize; bi++) {
      const off = bi * m2 * n2;
      const slice = ArrayStorage.fromData(aData.slice(off, off + m2 * n2), [m2, n2], 'float64');
      const r = pinv(slice, rcond);
      try {
        resultData.set(toContiguousFloat64(r), bi * n2 * m2);
      } finally {
        slice.dispose();
        r.dispose();
      }
    }
    return result;
  }

  const [m, n] = a.shape;
  const { u, s, vt } = svdFull(a);
  try {
    const k = Math.min(m!, n!);
    const sData = s.data as Float64Array;

    // Determine cutoff
    const maxS = sData[0]!;
    const cutoff = maxS * rcond;

    // Compute pinv = V^T^T @ S^+ @ U^T = (V^T transposed with S^+ scaling) @ U^T
    // Step 1: Build S^+ @ V^T → each row l of vt scaled by 1/s[l] (or 0)
    // Result is k × n, but we want V @ S^+ which is n × k (= vt^T with scaling)
    const vsInv = ArrayStorage.zeros([n!, k], 'float64');
    const vsInvData = vsInv.data as Float64Array;
    for (let l = 0; l < k; l++) {
      const sigma = sData[l]!;
      if (sigma > cutoff) {
        const invSigma = 1.0 / sigma;
        for (let i = 0; i < n!; i++) {
          // V @ S^+ : column l of V (= row l of vt) scaled by 1/sigma
          vsInvData[i * k + l] = (vt.data as Float64Array)[l * n! + i]! * invSigma;
        }
      }
      // else: column stays 0 (already zero-initialized)
    }

    // Step 2: Build U^T (k × m) — take first k rows of u^T (= first k columns of u, transposed)
    const ut = ArrayStorage.empty([k, m!], 'float64');
    const utData = ut.data as Float64Array;
    const uData = u.data as Float64Array;
    for (let l = 0; l < k; l++) {
      for (let j = 0; j < m!; j++) {
        utData[l * m! + j] = uData[j * m! + l]!;
      }
    }

    // Step 3: pinv = vsInv @ ut via WASM matmul (n × k) @ (k × m) → (n × m)
    const result = wasmMatmul(vsInv, ut) ?? matmul2D(vsInv, ut);
    vsInv.dispose();
    ut.dispose();
    return result;
  } finally {
    u.dispose();
    s.dispose();
    vt.dispose();
  }
}

/**
 * Compute eigenvalues and right eigenvectors of a square matrix.
 *
 * For general matrices, uses iterative methods.
 * For symmetric matrices, use eigh for better performance.
 *
 * **Limitation**: Complex eigenvalues are not supported. For non-symmetric matrices,
 * this function returns only the real parts of eigenvalues. If your matrix has
 * complex eigenvalues (e.g., rotation matrices), results will be incorrect.
 * Use eigh() for symmetric matrices where eigenvalues are guaranteed to be real.
 *
 * @param a - Input square matrix
 * @returns { w, v } - Eigenvalues (real only) and eigenvector matrix
 */
export function eig(a: ArrayStorage): { w: ArrayStorage; v: ArrayStorage } {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`eig: input must be at least 2D, got ${a.ndim}D`);
  }

  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) throw new Error(`eig: last 2 dimensions must be square, got ${m2}x${n}`);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const wResult = ArrayStorage.empty([...batchShape, n], 'float64');
    const vResult = ArrayStorage.empty([...batchShape, n, n], 'float64');
    const wData = wResult.data as Float64Array;
    const vData = vResult.data as Float64Array;
    for (let bi = 0; bi < batchSize; bi++) {
      const bIdx = flatToBatchMultiIndex(bi, batchShape);
      const slice = ArrayStorage.zeros([n, n], 'float64');
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) slice.set([i, j], realPart(a.get(...bIdx, i, j)));
      const { w, v } = eig(slice);
      wData.set(toContiguousFloat64(w), bi * n);
      vData.set(toContiguousFloat64(v), bi * n * n);
      slice.dispose();
      w.dispose();
      v.dispose();
    }
    return {
      w: wResult,
      v: vResult,
    };
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`eig: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // Check if symmetric (or Hermitian for complex)
  let isSymmetric = true;
  outer: for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      if (Math.abs(realPart(a.get(i, j)) - realPart(a.get(j, i))) > 1e-10) {
        isSymmetric = false;
        break outer;
      }
    }
  }

  if (isSymmetric) {
    // Use symmetric eigendecomposition (Jacobi method)
    // Symmetric matrices always have real eigenvalues, so this is exact
    const { values, vectors } = eigSymmetric(a);

    const w = ArrayStorage.zeros([size], 'float64');
    const v = ArrayStorage.zeros([size, size], 'float64');

    for (let i = 0; i < size; i++) {
      w.set([i], values[i]!);
      for (let j = 0; j < size; j++) {
        v.set([j, i], vectors[j]![i]!);
      }
    }

    return { w, v };
  }

  // For non-symmetric matrices, use QR iteration (simplified)
  // This is a basic implementation that may not converge for all matrices
  const { values, vectors, hasComplexEigenvalues } = qrEigendecomposition(a);

  // Only warn when complex eigenvalues are detected (real results would be inaccurate)
  if (hasComplexEigenvalues) {
    console.warn(
      'numpy-ts: eig() detected complex eigenvalues which cannot be represented. ' +
        'Results are real approximations and may be inaccurate. ' +
        'For symmetric matrices, use eigh() instead.'
    );
  }

  const w = ArrayStorage.zeros([size], 'float64');
  const v = ArrayStorage.zeros([size, size], 'float64');

  for (let i = 0; i < size; i++) {
    w.set([i], values[i]!);
    for (let j = 0; j < size; j++) {
      v.set([j, i], vectors[j]![i]!);
    }
  }

  return { w, v };
}

/**
 * QR algorithm for eigendecomposition.
 * Simplified version for real matrices.
 *
 * @param a - Input matrix
 * @returns { values, vectors }
 */
function qrEigendecomposition(a: ArrayStorage): {
  values: number[];
  vectors: number[][];
  hasComplexEigenvalues: boolean;
} {
  const n = a.shape[0]!;
  const maxIter = 1000;
  const tol = 1e-10;

  // Copy matrix (extract real parts for complex input)
  let A = ArrayStorage.zeros([n, n], 'float64');
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A.set([i, j], realPart(a.get(i, j)));
    }
  }

  // Initialize eigenvector accumulator as identity
  let V = ArrayStorage.zeros([n, n], 'float64');
  for (let i = 0; i < n; i++) {
    V.set([i, i], 1);
  }

  // QR iteration
  for (let iter = 0; iter < maxIter; iter++) {
    // Check for convergence (off-diagonal elements small)
    let offDiagNorm = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          offDiagNorm += Number(A.get(i, j)) ** 2;
        }
      }
    }
    if (Math.sqrt(offDiagNorm) < tol * n) break;

    // QR decomposition
    const qrResult = qr(A, 'reduced') as { q: ArrayStorage; r: ArrayStorage };
    const Q = qrResult.q;
    const R = qrResult.r;

    // A = R @ Q
    A = matmul(R, Q);

    // V = V @ Q
    V = matmul(V, Q);
  }

  // Detect complex eigenvalues: check for significant off-diagonal elements
  // in 2x2 blocks along the diagonal (indicates complex conjugate pairs)
  let hasComplexEigenvalues = false;
  for (let i = 0; i < n - 1; i++) {
    const subdiag = Math.abs(Number(A.get(i + 1, i)));
    const diag0 = Math.abs(Number(A.get(i, i)));
    const diag1 = Math.abs(Number(A.get(i + 1, i + 1)));
    const scale = Math.max(diag0, diag1, 1e-10);
    if (subdiag / scale > 1e-6) {
      hasComplexEigenvalues = true;
      break;
    }
  }

  // Extract eigenvalues from diagonal
  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    values.push(Number(A.get(i, i)));
  }

  // Convert V to 2D array
  const vectors: number[][] = [];
  for (let i = 0; i < n; i++) {
    vectors.push([]);
    for (let j = 0; j < n; j++) {
      vectors[i]!.push(Number(V.get(i, j)));
    }
  }

  return { values, vectors, hasComplexEigenvalues };
}

/**
 * Compute eigenvalues and eigenvectors of a real symmetric matrix.
 *
 * Note: Named "Hermitian" for NumPy compatibility, but only real symmetric
 * matrices are supported (complex Hermitian matrices require complex dtype support).
 * Symmetric matrices always have real eigenvalues, so results are exact.
 *
 * @param a - Real symmetric matrix
 * @param UPLO - 'L' or 'U' to use lower or upper triangle (default: 'L')
 * @returns { w, v } - Eigenvalues (sorted ascending) and eigenvector matrix
 */
export function eigh(a: ArrayStorage, UPLO: 'L' | 'U' = 'L'): { w: ArrayStorage; v: ArrayStorage } {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`eigh: input must be at least 2D, got ${a.ndim}D`);
  }

  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) throw new Error(`eigh: last 2 dimensions must be square, got ${m2}x${n}`);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const wResult = ArrayStorage.empty([...batchShape, n], 'float64');
    const vResult = ArrayStorage.empty([...batchShape, n, n], 'float64');
    const wData = wResult.data as Float64Array;
    const vData = vResult.data as Float64Array;
    for (let bi = 0; bi < batchSize; bi++) {
      const bIdx = flatToBatchMultiIndex(bi, batchShape);
      const slice = ArrayStorage.zeros([n, n], 'float64');
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) slice.set([i, j], realPart(a.get(...bIdx, i, j)));
      const { w, v } = eigh(slice, UPLO);
      wData.set(toContiguousFloat64(w), bi * n);
      vData.set(toContiguousFloat64(v), bi * n * n);
      slice.dispose();
      w.dispose();
      v.dispose();
    }
    return {
      w: wResult,
      v: vResult,
    };
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`eigh: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // Symmetrize the matrix using specified triangle
  // TODO: complex Hermitian eigendecomp (Lanczos/complex Jacobi); currently extracts real parts only
  const sym = ArrayStorage.zeros([size, size], 'float64');
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (UPLO === 'L') {
        if (i >= j) {
          const val = realPart(a.get(i, j));
          sym.set([i, j], val);
          sym.set([j, i], val);
        }
      } else {
        if (j >= i) {
          const val = realPart(a.get(i, j));
          sym.set([i, j], val);
          sym.set([j, i], val);
        }
      }
    }
  }

  // Use symmetric eigendecomposition
  const { values, vectors } = eigSymmetric(sym);
  sym.dispose();

  // Sort by eigenvalue (ascending)
  const indices = Array.from({ length: size }, (_, i) => i);
  indices.sort((i, j) => values[i]! - values[j]!);

  const w = ArrayStorage.zeros([size], 'float64');
  const v = ArrayStorage.zeros([size, size], 'float64');

  for (let i = 0; i < size; i++) {
    w.set([i], values[indices[i]!]!);
    for (let j = 0; j < size; j++) {
      v.set([j, i], vectors[j]![indices[i]!]!);
    }
  }

  return { w, v };
}

/**
 * Compute eigenvalues of a general square matrix.
 *
 * **Limitation**: Complex eigenvalues are not supported. For non-symmetric matrices,
 * this function returns only real approximations. Use eigvalsh() for symmetric
 * matrices where eigenvalues are guaranteed to be real.
 *
 * @param a - Input square matrix
 * @returns Array of eigenvalues (real only)
 */
export function eigvals(a: ArrayStorage): ArrayStorage {
  throwIfFloat16(a.dtype);
  const { w } = eig(a);
  return w;
}

/**
 * Compute eigenvalues of a real symmetric matrix.
 *
 * Note: Named "Hermitian" for NumPy compatibility, but only real symmetric
 * matrices are supported (complex Hermitian matrices require complex dtype support).
 * Symmetric matrices always have real eigenvalues, so results are exact.
 *
 * @param a - Real symmetric matrix
 * @param UPLO - 'L' or 'U' to use lower or upper triangle
 * @returns Array of eigenvalues (sorted ascending)
 */
export function eigvalsh(a: ArrayStorage, UPLO: 'L' | 'U' = 'L'): ArrayStorage {
  throwIfFloat16(a.dtype);
  const { w } = eigh(a, UPLO);
  return w;
}

/**
 * Return the dot product of two vectors (flattened).
 *
 * Unlike dot(), vdot flattens both inputs before computing the dot product.
 * For complex numbers, vdot uses the complex conjugate of the first argument.
 *
 * @param a - First input array (will be flattened)
 * @param b - Second input array (will be flattened)
 * @returns Scalar dot product
 */
export function vdot(a: ArrayStorage, b: ArrayStorage): number | bigint | Complex {
  // Flatten both arrays
  const aFlat = shapeOps.flatten(a);
  const bFlat = shapeOps.flatten(b);

  try {
    const result = vdotImpl(aFlat, bFlat, a.dtype, b.dtype);
    // Bool: clamp to 0/1 (NumPy bool arithmetic wraps)
    if (a.dtype === 'bool' && b.dtype === 'bool' && typeof result === 'number') {
      return result ? 1 : 0;
    }
    return result;
  } finally {
    aFlat.dispose();
    bFlat.dispose();
  }
}

function vdotImpl(
  aFlat: ArrayStorage,
  bFlat: ArrayStorage,
  aDtype: DType,
  bDtype: DType
): number | bigint | Complex {
  const aSize = aFlat.shape[0]!;
  const bSize = bFlat.shape[0]!;

  if (aSize !== bSize) {
    throw new Error(`vdot: arrays must have same number of elements, got ${aSize} and ${bSize}`);
  }

  const isComplex = isComplexDType(aDtype) || isComplexDType(bDtype);

  // WASM path: real/integer types use dot kernel, complex uses conjugate kernel
  if (!isComplex) {
    const wasmResult = wasmDot1D(aFlat, bFlat);
    if (wasmResult !== null) {
      // Bool: clamp to 0/1 (NumPy bool arithmetic wraps)
      if (aDtype === 'bool' && bDtype === 'bool') return wasmResult ? 1 : 0;
      return wasmResult;
    }
  } else {
    const wasmResult = wasmVdotComplex(aFlat, bFlat);
    if (wasmResult !== null) return wasmResult;
  }

  if (isComplex) {
    let sumRe = 0;
    let sumIm = 0;
    for (let i = 0; i < aSize; i++) {
      const aVal = aFlat.get(i);
      const bVal = bFlat.get(i);
      // Complex conjugate of first argument: conj(a) * b
      const aRe = aVal instanceof Complex ? aVal.re : Number(aVal);
      const aIm = aVal instanceof Complex ? aVal.im : 0;
      const bRe = bVal instanceof Complex ? bVal.re : Number(bVal);
      const bIm = bVal instanceof Complex ? bVal.im : 0;
      // conj(a)*b = (aRe - aIm*i)(bRe + bIm*i) = (aRe*bRe + aIm*bIm) + (aRe*bIm - aIm*bRe)*i
      sumRe += aRe * bRe + aIm * bIm;
      sumIm += -aIm * bRe + aRe * bIm;
    }
    if (Math.abs(sumIm) < 1e-15) {
      return sumRe;
    }
    return new Complex(sumRe, sumIm);
  }

  // Real case
  const vdotResultDtype = promoteDTypes(aDtype, bDtype);
  const vdotAcc = getIntAcc(vdotResultDtype);
  if (vdotAcc) {
    vdotAcc[0] = 0;
    for (let i = 0; i < aSize; i++) {
      vdotAcc[0] += Number(aFlat.get(i)) * Number(bFlat.get(i));
    }
    return vdotAcc[0]!;
  }
  // Float16/float32 accumulator for matching NumPy precision
  if (vdotResultDtype === 'float16' && hasFloat16) {
    const f16 = new Float16Array(1);
    f16[0] = 0;
    for (let i = 0; i < aSize; i++) {
      f16[0] += Number(aFlat.get(i)) * Number(bFlat.get(i));
    }
    return Number(f16[0]!);
  }
  if (vdotResultDtype === 'float32') {
    const f32 = new Float32Array(1);
    f32[0] = 0;
    for (let i = 0; i < aSize; i++) {
      f32[0] += Number(aFlat.get(i)) * Number(bFlat.get(i));
    }
    return f32[0]!;
  }
  let sum: number | bigint = 0;
  for (let i = 0; i < aSize; i++) {
    const aVal = aFlat.get(i);
    const bVal = bFlat.get(i);
    if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
      sum = (typeof sum === 'bigint' ? sum : BigInt(sum)) + aVal * bVal;
    } else {
      sum = (typeof sum === 'bigint' ? Number(sum) : sum) + Number(aVal) * Number(bVal);
    }
  }

  return sum;
}

/**
 * Vector dot product along the last axis.
 *
 * Computes the dot product of vectors along the last axis of both inputs.
 * The last dimensions of a and b must match.
 *
 * @param a - First input array
 * @param b - Second input array
 * @param axis - Axis along which to compute (default: -1, meaning last axis)
 * @returns Result with last dimension removed
 */
export function vecdot(
  a: ArrayStorage,
  b: ArrayStorage,
  axis: number = -1
): ArrayStorage | number | bigint | Complex {
  const aDim = a.ndim;
  const bDim = b.ndim;

  // Normalize axis
  const normalizedAxisA = axis < 0 ? aDim + axis : axis;
  const normalizedAxisB = axis < 0 ? bDim + axis : axis;

  if (normalizedAxisA < 0 || normalizedAxisA >= aDim) {
    throw new Error(`vecdot: axis ${axis} out of bounds for array with ${aDim} dimensions`);
  }
  if (normalizedAxisB < 0 || normalizedAxisB >= bDim) {
    throw new Error(`vecdot: axis ${axis} out of bounds for array with ${bDim} dimensions`);
  }

  const aAxisLen = a.shape[normalizedAxisA]!;
  const bAxisLen = b.shape[normalizedAxisB]!;

  if (aAxisLen !== bAxisLen) {
    throw new Error(`vecdot: axis dimensions must match, got ${aAxisLen} and ${bAxisLen}`);
  }

  // For 1D arrays, compute conj(a) · b directly (can't delegate to dot which doesn't conjugate)
  if (aDim === 1 && bDim === 1) {
    const isComplexInput = isComplexDType(a.dtype) || isComplexDType(b.dtype);
    if (!isComplexInput) {
      return dot(a, b) as number | bigint | Complex;
    }
    const n = a.shape[0]!;
    let sumRe = 0;
    let sumIm = 0;
    for (let i = 0; i < n; i++) {
      const aVal = a.get(i);
      const bVal = b.get(i);
      const aConj = aVal instanceof Complex ? new Complex(aVal.re, -aVal.im) : aVal;
      const prod = multiplyValues(aConj, bVal);
      if (prod instanceof Complex) {
        sumRe += prod.re;
        sumIm += prod.im;
      } else {
        sumRe += Number(prod);
      }
    }
    return new Complex(sumRe, sumIm);
  }

  // Try WASM-accelerated vecdot for 2D arrays with default axis (-1)
  if (aDim === 2 && bDim === 2 && axis === -1) {
    const wasmResult = wasmVecdot(a, b);
    if (wasmResult) return wasmResult;
  }

  // Use einsum for the general case: contract the specified axis
  // For last axis, this is equivalent to 'i...k,j...k->ij...' (removing k)
  // Build subscripts dynamically based on input shapes
  const aShapeWithoutAxis = [
    ...a.shape.slice(0, normalizedAxisA),
    ...a.shape.slice(normalizedAxisA + 1),
  ];
  const bShapeWithoutAxis = [
    ...b.shape.slice(0, normalizedAxisB),
    ...b.shape.slice(normalizedAxisB + 1),
  ];

  // Compute result by iterating over all positions
  // Result shape is broadcast of (aShape without axis) and (bShape without axis)
  const contractDim = aAxisLen;
  const isComplex = isComplexDType(a.dtype) || isComplexDType(b.dtype);
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Simple case: both 1D handled above
  // For higher dims, broadcast and sum over last axis
  // This is essentially np.sum(a * b, axis=-1)
  const resultShape =
    aShapeWithoutAxis.length > bShapeWithoutAxis.length ? aShapeWithoutAxis : bShapeWithoutAxis;

  const vecdotAcc = getIntAcc(resultDtype);

  if (resultShape.length === 0) {
    // Scalar result
    if (vecdotAcc) {
      vecdotAcc[0] = 0;
      for (let k = 0; k < contractDim; k++) {
        vecdotAcc[0] += Number(a.get(k)) * Number(b.get(k));
      }
      return vecdotAcc[0]!;
    }
    let sum: number | bigint | Complex = isComplex
      ? new Complex(0, 0)
      : isBigIntDType(resultDtype)
        ? 0n
        : 0;
    for (let k = 0; k < contractDim; k++) {
      const aVal = a.get(k);
      const bVal = b.get(k);
      // vecdot uses conj(a) * b
      const aConj = aVal instanceof Complex ? new Complex(aVal.re, -aVal.im) : aVal;
      const prod = multiplyValues(aConj, bVal);
      if (sum instanceof Complex || prod instanceof Complex) {
        const sumC: Complex = sum instanceof Complex ? sum : new Complex(Number(sum), 0);
        const prodC: Complex = prod instanceof Complex ? prod : new Complex(Number(prod), 0);
        sum = sumC.add(prodC);
      } else if (typeof sum === 'bigint' || typeof prod === 'bigint') {
        sum = BigInt(sum as number) + BigInt(prod as number);
      } else {
        sum = (sum as number) + (prod as number);
      }
    }
    return sum;
  }

  const result = ArrayStorage.zeros(resultShape, resultDtype);

  // Iterate over all output positions
  const totalOutputSize = resultShape.reduce((acc, dim) => acc * dim, 1);

  for (let flatIdx = 0; flatIdx < totalOutputSize; flatIdx++) {
    // Convert flat index to multi-index
    const multiIdx: number[] = [];
    let temp = flatIdx;
    for (let d = resultShape.length - 1; d >= 0; d--) {
      multiIdx.unshift(temp % resultShape[d]!);
      temp = Math.floor(temp / resultShape[d]!);
    }

    // Build indices for a and b
    const aIdx = [...multiIdx.slice(0, normalizedAxisA), 0, ...multiIdx.slice(normalizedAxisA)];
    const bIdx = [...multiIdx.slice(0, normalizedAxisB), 0, ...multiIdx.slice(normalizedAxisB)];

    if (vecdotAcc) {
      vecdotAcc[0] = 0;
      for (let k = 0; k < contractDim; k++) {
        aIdx[normalizedAxisA] = k;
        bIdx[normalizedAxisB] = k;
        vecdotAcc[0] += Number(a.get(...aIdx)) * Number(b.get(...bIdx));
      }
      result.set(multiIdx, vecdotAcc[0]!);
    } else {
      let sum: number | bigint | Complex = isComplex
        ? new Complex(0, 0)
        : isBigIntDType(resultDtype)
          ? 0n
          : 0;
      for (let k = 0; k < contractDim; k++) {
        aIdx[normalizedAxisA] = k;
        bIdx[normalizedAxisB] = k;
        const aVal = a.get(...aIdx);
        const bVal = b.get(...bIdx);
        // vecdot uses conj(a) * b
        const aConj = aVal instanceof Complex ? new Complex(aVal.re, -aVal.im) : aVal;
        const prod = multiplyValues(aConj, bVal);
        if (sum instanceof Complex || prod instanceof Complex) {
          const sumC: Complex = sum instanceof Complex ? sum : new Complex(Number(sum), 0);
          const prodC: Complex = prod instanceof Complex ? prod : new Complex(Number(prod), 0);
          sum = sumC.add(prodC);
        } else if (typeof sum === 'bigint' || typeof prod === 'bigint') {
          sum = BigInt(sum as number) + BigInt(prod as number);
        } else {
          sum = (sum as number) + (prod as number);
        }
      }
      result.set(multiIdx, sum);
    }
  }

  return result;
}

/**
 * Transpose the last two axes of an array.
 *
 * Equivalent to swapaxes(a, -2, -1) or transpose with axes that swap the last two.
 * For a 2D array, this is the same as transpose.
 *
 * @param a - Input array with at least 2 dimensions
 * @returns Array with last two axes transposed
 */
export function matrix_transpose(a: ArrayStorage): ArrayStorage {
  if (a.ndim < 2) {
    throw new Error(`matrix_transpose: input must have at least 2 dimensions, got ${a.ndim}D`);
  }

  // Build axes that swap the last two
  const axes = Array.from({ length: a.ndim }, (_, i) => i);
  const last = axes.length - 1;
  axes[last] = last - 1;
  axes[last - 1] = last;

  return transpose(a, axes);
}

/**
 * Permute the dimensions of an array.
 *
 * This is an alias for transpose to match the Array API standard.
 *
 * @param a - Input array
 * @param axes - Permutation of axes. If not specified, reverses the axes.
 * @returns Transposed array
 */
export function permute_dims(a: ArrayStorage, axes?: number[]): ArrayStorage {
  return transpose(a, axes);
}

/**
 * Matrix-vector multiplication.
 *
 * Computes the matrix-vector product over the last two axes of x1 and
 * the last axis of x2.
 *
 * @param x1 - First input array (matrix) with shape (..., M, N)
 * @param x2 - Second input array (vector) with shape (..., N)
 * @returns Result with shape (..., M)
 */
export function matvec(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  if (x1.ndim < 2) {
    throw new Error(`matvec: x1 must have at least 2 dimensions, got ${x1.ndim}D`);
  }
  if (x2.ndim < 1) {
    throw new Error(`matvec: x2 must have at least 1 dimension, got ${x2.ndim}D`);
  }

  const m = x1.shape[x1.ndim - 2]!;
  const n1 = x1.shape[x1.ndim - 1]!;
  const n2 = x2.shape[x2.ndim - 1]!;

  if (n1 !== n2) {
    throw new Error(`matvec: last axis of x1 (${n1}) must match last axis of x2 (${n2})`);
  }

  // For simple 2D @ 1D case, use existing dot
  if (x1.ndim === 2 && x2.ndim === 1) {
    return dot(x1, x2) as ArrayStorage;
  }

  // General case: batch matrix-vector multiplication
  const batchShapeX1 = x1.shape.slice(0, -2);
  const batchShapeX2 = x2.shape.slice(0, -1);

  // Broadcast batch dimensions
  const maxBatchDims = Math.max(batchShapeX1.length, batchShapeX2.length);
  const paddedX1 = [...Array(maxBatchDims - batchShapeX1.length).fill(1), ...batchShapeX1];
  const paddedX2 = [...Array(maxBatchDims - batchShapeX2.length).fill(1), ...batchShapeX2];

  const batchShape: number[] = [];
  for (let i = 0; i < maxBatchDims; i++) {
    const d1 = paddedX1[i]!;
    const d2 = paddedX2[i]!;
    if (d1 !== 1 && d2 !== 1 && d1 !== d2) {
      throw new Error(
        `matvec: batch dimensions not broadcastable: ${batchShapeX1} vs ${batchShapeX2}`
      );
    }
    batchShape.push(Math.max(d1, d2));
  }

  const resultShape = [...batchShape, m];
  const resultDtype = promoteDTypes(x1.dtype, x2.dtype);
  const result = ArrayStorage.zeros(resultShape, resultDtype);
  const isComplex = isComplexDType(resultDtype);

  const totalBatch = batchShape.reduce((acc, d) => acc * d, 1);

  for (let batchIdx = 0; batchIdx < totalBatch; batchIdx++) {
    // Convert flat batch index to multi-index
    const batchMultiIdx: number[] = [];
    let temp = batchIdx;
    for (let d = batchShape.length - 1; d >= 0; d--) {
      batchMultiIdx.unshift(temp % batchShape[d]!);
      temp = Math.floor(temp / batchShape[d]!);
    }

    // Map to x1 and x2 batch indices (handle broadcasting)
    const x1BatchIdx = batchMultiIdx.slice(-(batchShapeX1.length || 1)).map((idx, i) => {
      const dim = batchShapeX1[i] ?? 1;
      return dim === 1 ? 0 : idx;
    });
    const x2BatchIdx = batchMultiIdx.slice(-(batchShapeX2.length || 1)).map((idx, i) => {
      const dim = batchShapeX2[i] ?? 1;
      return dim === 1 ? 0 : idx;
    });

    const matvecAcc = getIntAcc(resultDtype);
    for (let i = 0; i < m; i++) {
      if (matvecAcc) {
        matvecAcc[0] = 0;
        for (let j = 0; j < n1; j++) {
          const x1Idx = [...x1BatchIdx, i, j];
          const x2Idx = [...x2BatchIdx, j];
          matvecAcc[0] += Number(x1.get(...x1Idx)) * Number(x2.get(...x2Idx));
        }
        result.set([...batchMultiIdx, i], matvecAcc[0]!);
      } else {
        let sum: number | bigint | Complex = isComplex
          ? new Complex(0, 0)
          : isBigIntDType(resultDtype)
            ? 0n
            : 0;
        for (let j = 0; j < n1; j++) {
          const x1Idx = [...x1BatchIdx, i, j];
          const x2Idx = [...x2BatchIdx, j];
          const x1Val = x1.get(...x1Idx);
          const x2Val = x2.get(...x2Idx);
          const prod = multiplyValues(x1Val, x2Val);
          if (sum instanceof Complex || prod instanceof Complex) {
            const sumC: Complex = sum instanceof Complex ? sum : new Complex(Number(sum), 0);
            const prodC: Complex = prod instanceof Complex ? prod : new Complex(Number(prod), 0);
            sum = sumC.add(prodC);
          } else if (typeof sum === 'bigint' || typeof prod === 'bigint') {
            sum = BigInt(sum as number) + BigInt(prod as number);
          } else {
            sum = (sum as number) + (prod as number);
          }
        }
        result.set([...batchMultiIdx, i], sum);
      }
    }
  }

  return result;
}

/**
 * Vector-matrix multiplication.
 *
 * Computes the vector-matrix product over the last axis of x1 and
 * the second-to-last axis of x2.
 *
 * @param x1 - First input array (vector) with shape (..., M)
 * @param x2 - Second input array (matrix) with shape (..., M, N)
 * @returns Result with shape (..., N)
 */
export function vecmat(x1: ArrayStorage, x2: ArrayStorage): ArrayStorage {
  if (x1.ndim < 1) {
    throw new Error(`vecmat: x1 must have at least 1 dimension, got ${x1.ndim}D`);
  }
  if (x2.ndim < 2) {
    throw new Error(`vecmat: x2 must have at least 2 dimensions, got ${x2.ndim}D`);
  }

  const m1 = x1.shape[x1.ndim - 1]!;
  const m2 = x2.shape[x2.ndim - 2]!;
  const n = x2.shape[x2.ndim - 1]!;

  if (m1 !== m2) {
    throw new Error(`vecmat: last axis of x1 (${m1}) must match second-to-last axis of x2 (${m2})`);
  }

  // For simple 1D @ 2D case, use existing dot (conjugate x1 for complex types)
  if (x1.ndim === 1 && x2.ndim === 2) {
    const x1Conj = isComplexDType(x1.dtype) ? conjStorage(x1) : x1;
    return dot(x1Conj, x2) as ArrayStorage;
  }

  // General case: batch vector-matrix multiplication
  const batchShapeX1 = x1.shape.slice(0, -1);
  const batchShapeX2 = x2.shape.slice(0, -2);

  // Broadcast batch dimensions
  const maxBatchDims = Math.max(batchShapeX1.length, batchShapeX2.length);
  const paddedX1 = [...Array(maxBatchDims - batchShapeX1.length).fill(1), ...batchShapeX1];
  const paddedX2 = [...Array(maxBatchDims - batchShapeX2.length).fill(1), ...batchShapeX2];

  const batchShape: number[] = [];
  for (let i = 0; i < maxBatchDims; i++) {
    const d1 = paddedX1[i]!;
    const d2 = paddedX2[i]!;
    if (d1 !== 1 && d2 !== 1 && d1 !== d2) {
      throw new Error(
        `vecmat: batch dimensions not broadcastable: ${batchShapeX1} vs ${batchShapeX2}`
      );
    }
    batchShape.push(Math.max(d1, d2));
  }

  const resultShape = [...batchShape, n];
  const resultDtype = promoteDTypes(x1.dtype, x2.dtype);
  const result = ArrayStorage.zeros(resultShape, resultDtype);
  const isComplex = isComplexDType(resultDtype);

  const totalBatch = batchShape.reduce((acc, d) => acc * d, 1);

  for (let batchIdx = 0; batchIdx < totalBatch; batchIdx++) {
    // Convert flat batch index to multi-index
    const batchMultiIdx: number[] = [];
    let temp = batchIdx;
    for (let d = batchShape.length - 1; d >= 0; d--) {
      batchMultiIdx.unshift(temp % batchShape[d]!);
      temp = Math.floor(temp / batchShape[d]!);
    }

    // Map to x1 and x2 batch indices (handle broadcasting)
    const x1BatchIdx = batchMultiIdx.slice(-(batchShapeX1.length || 1)).map((idx, i) => {
      const dim = batchShapeX1[i] ?? 1;
      return dim === 1 ? 0 : idx;
    });
    const x2BatchIdx = batchMultiIdx.slice(-(batchShapeX2.length || 1)).map((idx, i) => {
      const dim = batchShapeX2[i] ?? 1;
      return dim === 1 ? 0 : idx;
    });

    const vecmatAcc = getIntAcc(resultDtype);
    for (let j = 0; j < n; j++) {
      if (vecmatAcc) {
        vecmatAcc[0] = 0;
        for (let i = 0; i < m1; i++) {
          const x1Idx = [...x1BatchIdx, i];
          const x2Idx = [...x2BatchIdx, i, j];
          vecmatAcc[0] += Number(x1.get(...x1Idx)) * Number(x2.get(...x2Idx));
        }
        result.set([...batchMultiIdx, j], vecmatAcc[0]!);
      } else {
        let sum: number | bigint | Complex = isComplex
          ? new Complex(0, 0)
          : isBigIntDType(resultDtype)
            ? 0n
            : 0;
        for (let i = 0; i < m1; i++) {
          const x1Idx = [...x1BatchIdx, i];
          const x2Idx = [...x2BatchIdx, i, j];
          let x1Val = x1.get(...x1Idx);
          // vecmat conjugates x1 for complex types (matches NumPy)
          if (x1Val instanceof Complex) {
            x1Val = new Complex(x1Val.re, -x1Val.im);
          }
          const x2Val = x2.get(...x2Idx);
          const prod = multiplyValues(x1Val, x2Val);
          if (sum instanceof Complex || prod instanceof Complex) {
            const sumC: Complex = sum instanceof Complex ? sum : new Complex(Number(sum), 0);
            const prodC: Complex = prod instanceof Complex ? prod : new Complex(Number(prod), 0);
            sum = sumC.add(prodC);
          } else if (typeof sum === 'bigint' || typeof prod === 'bigint') {
            sum = BigInt(sum as number) + BigInt(prod as number);
          } else {
            sum = (sum as number) + (prod as number);
          }
        }
        result.set([...batchMultiIdx, j], sum);
      }
    }
  }

  return result;
}

/**
 * Compute sign and (natural) logarithm of the determinant.
 *
 * Returns (sign, logabsdet) where sign is the sign of the determinant
 * and logabsdet is the natural log of the absolute value of the determinant.
 *
 * This is useful for computing determinants of large matrices where the
 * determinant itself might overflow or underflow.
 *
 * @param a - Square matrix
 * @returns { sign, logabsdet }
 */
export function slogdet(a: ArrayStorage): {
  sign: number | ArrayStorage;
  logabsdet: number | ArrayStorage;
} {
  throwIfFloat16(a.dtype);
  if (a.ndim < 2) {
    throw new Error(`slogdet: input must be at least 2D, got ${a.ndim}D`);
  }

  if (a.ndim > 2) {
    const batchShape = Array.from(a.shape).slice(0, -2);
    const n = a.shape[a.ndim - 1]!;
    const m2 = a.shape[a.ndim - 2]!;
    if (m2 !== n) throw new Error(`slogdet: last 2 dimensions must be square, got ${m2}x${n}`);
    const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
    const signResult = ArrayStorage.empty(batchShape, 'float64');
    const logResult = ArrayStorage.empty(batchShape, 'float64');
    const signData = signResult.data as Float64Array;
    const logData = logResult.data as Float64Array;
    const aData = toContiguousFloat64(a);
    for (let bi = 0; bi < batchSize; bi++) {
      const off = bi * n * n;
      const slice = ArrayStorage.fromData(aData.slice(off, off + n * n), [n, n], 'float64');
      try {
        const { sign, logabsdet } = slogdet(slice) as { sign: number; logabsdet: number };
        signData[bi] = sign;
        logData[bi] = logabsdet;
      } finally {
        slice.dispose();
      }
    }
    return {
      sign: signResult,
      logabsdet: logResult,
    };
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`slogdet: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  if (size === 0) {
    return { sign: 1, logabsdet: 0 }; // Empty matrix has determinant 1
  }

  // WASM fast path for f64/f32
  if (a.dtype === 'float64' || a.dtype === 'float32') {
    const factored = wasmLuFactor(a);
    if (factored) {
      const luData = factored.lu.data as Float64Array | Float32Array;
      let logAbsDet = 0;
      let sign = factored.sign;
      for (let i = 0; i < size; i++) {
        const diagVal = luData[i * size + i]!;
        if (diagVal === 0) {
          factored.lu.dispose();
          return { sign: 0, logabsdet: -Infinity };
        }
        if (diagVal < 0) sign = -sign;
        logAbsDet += Math.log(Math.abs(diagVal));
      }
      factored.lu.dispose();
      return { sign, logabsdet: logAbsDet };
    }
  }

  // JS fallback: LU decomposition with partial pivoting
  const { lu, sign: pivotSign } = luDecomposition(a);

  try {
    const luData = lu.data as Float64Array;
    const isComplex = isComplexDType(lu.dtype);
    let logAbsDet = 0;
    let sign = pivotSign;

    if (isComplex) {
      // For complex LU, diagonal entries are complex.
      // sign = product of (diag[i] / |diag[i]|), logabsdet = sum of log(|diag[i]|)
      // We track sign as a complex number on the unit circle.
      let signRe = pivotSign as number;
      let signIm = 0;
      for (let i = 0; i < size; i++) {
        const idx = (i * size + i) * 2;
        const dRe = luData[idx]!;
        const dIm = luData[idx + 1]!;
        const mag = Math.sqrt(dRe * dRe + dIm * dIm);
        if (mag === 0) {
          return { sign: 0, logabsdet: -Infinity };
        }
        logAbsDet += Math.log(mag);
        // multiply sign by diag[i] / |diag[i]|
        const uRe = dRe / mag;
        const uIm = dIm / mag;
        const newRe = signRe * uRe - signIm * uIm;
        const newIm = signRe * uIm + signIm * uRe;
        signRe = newRe;
        signIm = newIm;
      }
      // For real-valued determinants (Hermitian matrices), sign should be +1 or -1
      // For general complex, sign is on the unit circle
      // Round to nearest integer if very close
      if (Math.abs(signIm) < 1e-10) {
        sign = Math.round(signRe);
      } else {
        // Return sign as-is (real part) - for complex det, sign concept is limited
        sign = signRe;
      }
    } else {
      for (let i = 0; i < size; i++) {
        const diagVal = luData[i * size + i]!;
        if (diagVal === 0) {
          return { sign: 0, logabsdet: -Infinity };
        }
        if (diagVal < 0) {
          sign = -sign;
        }
        logAbsDet += Math.log(Math.abs(diagVal));
      }
    }

    return { sign, logabsdet: logAbsDet };
  } finally {
    lu.dispose();
  }
}

/**
 * Compute singular values of a matrix.
 *
 * This is equivalent to svd(a, compute_uv=False) but more efficient
 * as it doesn't compute the U and V matrices.
 *
 * @param a - Input matrix (m x n)
 * @returns 1D array of singular values in descending order
 */
export function svdvals(a: ArrayStorage): ArrayStorage {
  throwIfFloat16(a.dtype);
  const inputDtype = a.dtype;
  // Fast path: Golub-Kahan (values only, no U/V)
  const wasmResult = wasmSvdValues(a);
  if (wasmResult) {
    // Downcast to input dtype if needed (WASM computes in f64)
    if (inputDtype === 'float32' && wasmResult.dtype === 'float64') {
      const f32 = ArrayStorage.empty(Array.from(wasmResult.shape), 'float32');
      const src = wasmResult.data as Float64Array;
      const dst = f32.data as Float32Array;
      for (let i = 0; i < wasmResult.size; i++) dst[i] = src[i]!;
      wasmResult.dispose();
      return f32;
    }
    return wasmResult;
  }

  // Fallback: full SVD, extract S
  const result = svd(a, true, false);
  return result as ArrayStorage;
}

/**
 * Compute the dot product of two or more arrays in a single function call.
 *
 * Optimizes the order of multiplications to minimize computation.
 * For example, for three arrays A, B, C with shapes (10, 100), (100, 5), (5, 50),
 * it's more efficient to compute (A @ B) @ C than A @ (B @ C).
 *
 * @param arrays - List of arrays to multiply
 * @returns Result of multiplying all arrays
 */
export function multi_dot(arrays: ArrayStorage[]): ArrayStorage {
  if (arrays.length < 2) {
    throw new Error('multi_dot: need at least 2 arrays');
  }

  if (arrays.length === 2) {
    return matmul(arrays[0]!, arrays[1]!);
  }

  // For simplicity, use left-to-right order
  // A proper implementation would use dynamic programming to find optimal order
  // But for now, left-to-right is correct and reasonably efficient
  let result = matmul(arrays[0]!, arrays[1]!);
  for (let i = 2; i < arrays.length; i++) {
    const oldResult = result;
    result = matmul(result, arrays[i]!);
    oldResult.dispose();
  }

  return result;
}

/**
 * Compute the 'inverse' of an N-dimensional array.
 *
 * The inverse is defined such that tensordot(tensorinv(a), a, ind) == I
 * where I is the identity operator.
 *
 * @param a - Input array to invert
 * @param ind - Number of first indices that are involved in the inverse sum (default: 2)
 * @returns Tensor inverse
 */
export function tensorinv(a: ArrayStorage, ind: number = 2): ArrayStorage {
  throwIfFloat16(a.dtype);
  if (ind <= 0) {
    throw new Error(`tensorinv: ind must be positive, got ${ind}`);
  }

  const shape = a.shape;
  const ndim = a.ndim;

  if (ndim < ind) {
    throw new Error(`tensorinv: array has ${ndim} dimensions, ind=${ind} is too large`);
  }

  // Compute product of first ind dimensions
  let prodA = 1;
  for (let i = 0; i < ind; i++) {
    prodA *= shape[i]!;
  }

  // Compute product of remaining dimensions
  let prodB = 1;
  for (let i = ind; i < ndim; i++) {
    prodB *= shape[i]!;
  }

  if (prodA !== prodB) {
    throw new Error(
      `tensorinv: product of first ${ind} dimensions (${prodA}) must equal product of remaining dimensions (${prodB})`
    );
  }

  // Reshape to 2D, invert, then reshape back
  const reshaped = shapeOps.reshape(a, [prodA, prodB]);
  const inverted = inv(reshaped);

  // New shape: remaining dims + first dims
  const newShape = [...shape.slice(ind), ...shape.slice(0, ind)];
  return shapeOps.reshape(inverted, newShape);
}

/**
 * Solve the tensor equation a x = b for x.
 *
 * This is equivalent to solve after reshaping a and b appropriately.
 *
 * @param a - Coefficient tensor
 * @param b - Target tensor
 * @param axes - Axes of a to be summed over in the contraction (default based on b.ndim)
 * @returns Solution tensor x
 */
export function tensorsolve(
  a: ArrayStorage,
  b: ArrayStorage,
  axes?: number[] | null
): ArrayStorage {
  throwIfFloat16(a.dtype);
  const aShape = a.shape;
  const bShape = b.shape;
  const aDim = a.ndim;
  const bDim = b.ndim;

  // Default axes: last b.ndim axes of a
  let axesToSum: number[];
  if (axes === null || axes === undefined) {
    axesToSum = Array.from({ length: bDim }, (_, i) => aDim - bDim + i);
  } else {
    axesToSum = axes.map((ax) => (ax < 0 ? aDim + ax : ax));
  }

  // Move the axes to sum to the end
  const otherAxes: number[] = [];
  for (let i = 0; i < aDim; i++) {
    if (!axesToSum.includes(i)) {
      otherAxes.push(i);
    }
  }
  const newAxes = [...otherAxes, ...axesToSum];
  const aTransposed = transpose(a, newAxes);

  // Compute dimensions
  const sumDims = axesToSum.map((ax) => aShape[ax]!);
  const sumProd = sumDims.reduce((acc, d) => acc * d, 1);
  const otherDims = otherAxes.map((ax) => aShape[ax]!);
  const otherProd = otherDims.reduce((acc, d) => acc * d, 1);

  // Check that dimensions are compatible
  const bProd = bShape.reduce((acc, d) => acc * d, 1);
  if (sumProd !== bProd) {
    throw new Error(
      `tensorsolve: dimensions don't match - sum dimensions product (${sumProd}) != b total elements (${bProd})`
    );
  }
  if (otherProd !== sumProd) {
    throw new Error(
      `tensorsolve: non-square problem - other dimensions product (${otherProd}) != sum dimensions product (${sumProd})`
    );
  }

  // Reshape to 2D and solve
  const aReshaped = shapeOps.reshape(aTransposed, [otherProd, sumProd]);
  const bReshaped = shapeOps.reshape(b, [sumProd]);
  const xFlat = solve(aReshaped, bReshaped);

  // Reshape result to original b shape
  return shapeOps.reshape(xFlat, [...bShape]);
}

/**
 * Evaluate the lowest cost contraction order for an einsum expression
 *
 * This function analyzes an einsum expression and finds an optimized contraction
 * path that minimizes the total number of operations (FLOPs).
 *
 * @param subscripts - Einsum subscript string (e.g., 'ij,jk,kl->il')
 * @param operands - Input arrays (or their shapes)
 * @param optimize - Optimization strategy: 'greedy' (default), 'optimal', or false for no optimization
 * @returns A tuple of [path, string_representation] where path is an array of index pairs
 *
 * @example
 * ```typescript
 * const a = ones([10, 10]);
 * const b = ones([10, 10]);
 * const c = ones([10, 10]);
 * const [path, info] = einsum_path('ij,jk,kl->il', a.storage, b.storage, c.storage);
 * // path: [[0, 1], [0, 1]] - contract first two, then result with third
 * ```
 */
export function einsum_path(
  subscripts: string,
  ...operands: (ArrayStorage | number[])[]
): [Array<[number, number] | number[]>, string] {
  // Parse subscripts
  const arrowMatch = subscripts.indexOf('->');

  let inputSubscripts: string;
  let outputSubscript: string;

  if (arrowMatch === -1) {
    inputSubscripts = subscripts;
    outputSubscript = inferOutputSubscript(inputSubscripts);
  } else {
    inputSubscripts = subscripts.slice(0, arrowMatch);
    outputSubscript = subscripts.slice(arrowMatch + 2);
  }

  const operandSubscripts = inputSubscripts.split(',').map((s) => s.trim());

  if (operandSubscripts.length !== operands.length) {
    throw new Error(
      `einsum_path: expected ${operandSubscripts.length} operands, got ${operands.length}`
    );
  }

  // Get shapes from operands
  const shapes: number[][] = operands.map((op) => {
    if (Array.isArray(op)) {
      return op;
    }
    return Array.from(op.shape);
  });

  // Build index dimension map
  const indexDims = new Map<string, number>();
  for (let i = 0; i < operands.length; i++) {
    const sub = operandSubscripts[i]!;
    const shape = shapes[i]!;

    if (sub.length !== shape.length) {
      throw new Error(
        `einsum_path: operand ${i} has ${shape.length} dimensions but subscript '${sub}' has ${sub.length} indices`
      );
    }

    for (let j = 0; j < sub.length; j++) {
      const idx = sub[j]!;
      const dim = shape[j]!;
      if (indexDims.has(idx) && indexDims.get(idx) !== dim) {
        throw new Error(
          `einsum_path: size mismatch for index '${idx}': ${indexDims.get(idx)} vs ${dim}`
        );
      }
      indexDims.set(idx, dim);
    }
  }

  // Simple path for 1 or 2 operands
  if (operands.length === 1) {
    const path: Array<[number, number] | number[]> = [[0]];
    return [path, buildPathInfo(subscripts, shapes, path, indexDims)];
  }

  if (operands.length === 2) {
    const path: Array<[number, number] | number[]> = [[0, 1]];
    return [path, buildPathInfo(subscripts, shapes, path, indexDims)];
  }

  // Greedy contraction path for 3+ operands
  // This is a simplified greedy algorithm that contracts pairs with smallest intermediate size
  const path: Array<[number, number]> = [];
  const currentSubscripts = [...operandSubscripts];
  const currentShapes = [...shapes];
  const currentIndices = operands.map((_, i) => i);

  while (currentSubscripts.length > 1) {
    let bestI = 0;
    let bestJ = 1;
    let bestCost = Infinity;

    // Find the best pair to contract
    for (let i = 0; i < currentSubscripts.length; i++) {
      for (let j = i + 1; j < currentSubscripts.length; j++) {
        const cost = estimateContractionCost(
          currentSubscripts[i]!,
          currentSubscripts[j]!,
          currentShapes[i]!,
          currentShapes[j]!,
          outputSubscript,
          indexDims
        );

        if (cost < bestCost) {
          bestCost = cost;
          bestI = i;
          bestJ = j;
        }
      }
    }

    // Record the contraction
    path.push([currentIndices[bestI]!, currentIndices[bestJ]!]);

    // Compute the result subscript and shape
    const [newSubscript, newShape] = computeContractionResult(
      currentSubscripts[bestI]!,
      currentSubscripts[bestJ]!,
      currentShapes[bestI]!,
      currentShapes[bestJ]!,
      outputSubscript,
      indexDims
    );

    // Update arrays (remove j first since j > i)
    currentSubscripts.splice(bestJ, 1);
    currentSubscripts.splice(bestI, 1);
    currentShapes.splice(bestJ, 1);
    currentShapes.splice(bestI, 1);
    currentIndices.splice(bestJ, 1);
    currentIndices.splice(bestI, 1);

    // Add the result
    currentSubscripts.push(newSubscript);
    currentShapes.push(newShape);
    currentIndices.push(-1); // Intermediate result marker
  }

  return [path, buildPathInfo(subscripts, shapes, path, indexDims)];
}

/**
 * Estimate the cost of contracting two operands
 * @private
 */
function estimateContractionCost(
  sub1: string,
  sub2: string,
  _shape1: number[],
  _shape2: number[],
  _outputSubscript: string,
  indexDims: Map<string, number>
): number {
  // Find indices that will be summed over (in both operands but not in output)
  const indices1 = new Set(sub1);
  const indices2 = new Set(sub2);

  let size = 1;

  // Product of all index dimensions involved
  for (const idx of indices1) {
    size *= indexDims.get(idx) || 1;
  }
  for (const idx of indices2) {
    if (!indices1.has(idx)) {
      size *= indexDims.get(idx) || 1;
    }
  }

  return size;
}

/**
 * Compute the result subscript and shape after contracting two operands
 * @private
 */
function computeContractionResult(
  sub1: string,
  sub2: string,
  _shape1: number[],
  _shape2: number[],
  outputSubscript: string,
  indexDims: Map<string, number>
): [string, number[]] {
  // Find all indices in both operands
  const allIndices = new Set([...sub1, ...sub2]);

  // Count occurrences
  const counts = new Map<string, number>();
  for (const idx of sub1) {
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  for (const idx of sub2) {
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }

  // Result keeps indices that:
  // 1. Appear in the final output, OR
  // 2. Appear only once in the two operands (will be needed for further contractions)
  const outputIndices = new Set(outputSubscript);
  const resultIndices: string[] = [];

  for (const idx of allIndices) {
    if (outputIndices.has(idx) || counts.get(idx) === 1) {
      resultIndices.push(idx);
    }
  }

  // Sort alphabetically for consistency
  resultIndices.sort();

  const resultShape = resultIndices.map((idx) => indexDims.get(idx)!);

  return [resultIndices.join(''), resultShape];
}

/**
 * Build the path information string
 * @private
 */
function buildPathInfo(
  subscripts: string,
  shapes: number[][],
  path: Array<[number, number] | number[]>,
  _indexDims: Map<string, number>
): string {
  const lines: string[] = [];

  lines.push('  Complete contraction:  ' + subscripts);
  lines.push('         Operand shapes:  ' + shapes.map((s) => `(${s.join(', ')})`).join(', '));
  lines.push('  Contraction path:      ' + JSON.stringify(path));

  // Estimate total FLOPS
  let totalFlops = 0;
  for (const shape of shapes) {
    totalFlops += shape.reduce((a, b) => a * b, 1);
  }

  lines.push('  Estimated FLOPS:       ~' + totalFlops.toExponential(2));

  return lines.join('\n');
}
