/**
 * Linear algebra operations
 *
 * Pure functions for matrix operations (matmul, etc.).
 * @module ops/linalg
 */

import { ArrayStorage } from '../core/storage';
import { promoteDTypes, isComplexDType } from '../core/dtype';
import { Complex } from '../core/complex';
import * as shapeOps from './shape';

/**
 * Helper to multiply two values that may be Complex
 * Returns Complex if either input is Complex, number otherwise
 */
function multiplyValues(
  a: number | bigint | Complex,
  b: number | bigint | Complex
): number | Complex {
  if (a instanceof Complex || b instanceof Complex) {
    const aComplex = a instanceof Complex ? a : new Complex(Number(a), 0);
    const bComplex = b instanceof Complex ? b : new Complex(Number(b), 0);
    return aComplex.mul(bComplex);
  }
  if (typeof a === 'bigint' && typeof b === 'bigint') {
    return Number(a * b);
  }
  return Number(a) * Number(b);
}

/**
 * Helper to add two values that may be Complex
 */
function addValues(
  a: number | Complex,
  b: number | Complex
): number | Complex {
  if (a instanceof Complex || b instanceof Complex) {
    const aComplex = a instanceof Complex ? a : new Complex(a, 0);
    const bComplex = b instanceof Complex ? b : new Complex(b, 0);
    return aComplex.add(bComplex);
  }
  return a + b;
}

/**
 * BLAS-like types for matrix operations
 */
type Layout = 'row-major' | 'column-major';
type Transpose = 'no-transpose' | 'transpose';

/**
 * Double-precision general matrix multiply (DGEMM)
 *
 * Full BLAS-compatible implementation without external dependencies.
 * Performs: C = alpha * op(A) * op(B) + beta * C
 *
 * Supports all combinations of:
 * - Row-major and column-major layouts
 * - Transpose and no-transpose operations
 * - Arbitrary alpha and beta scalars
 *
 * Uses specialized loops for each case to avoid function call overhead.
 *
 * @internal
 */
function dgemm(
  layout: Layout,
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
  beta: number, // scalar beta
  C: Float64Array, // matrix C (output)
  ldc: number // leading dimension of C
): void {
  // Apply beta scaling to C first
  if (beta === 0) {
    for (let i = 0; i < M * N; i++) {
      C[i] = 0;
    }
  } else if (beta !== 1) {
    for (let i = 0; i < M * N; i++) {
      C[i] = (C[i] ?? 0) * beta;
    }
  }

  // Select specialized loop based on layout and transpose modes
  // This avoids function call overhead in the hot loop
  const isRowMajor = layout === 'row-major';
  const transposeA = transA === 'transpose';
  const transposeB = transB === 'transpose';

  if (isRowMajor && !transposeA && !transposeB) {
    // Row-major, no transpose (most common case)
    // C[i,j] = sum_k A[i,k] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && transposeA && !transposeB) {
    // Row-major, A transposed
    // C[i,j] = sum_k A[k,i] * B[k,j]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && !transposeA && transposeB) {
    // Row-major, B transposed
    // C[i,j] = sum_k A[i,k] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (isRowMajor && transposeA && transposeB) {
    // Row-major, both transposed
    // C[i,j] = sum_k A[k,i] * B[j,k]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[i * ldc + j] = (C[i * ldc + j] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && !transposeA && !transposeB) {
    // Column-major, no transpose
    // C[i,j] = sum_k A[i,k] * B[k,j]
    // Column-major: A[i,k] = A[k*lda + i], C[i,j] = C[j*ldc + i]
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && transposeA && !transposeB) {
    // Column-major, A transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[j * ldb + k] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else if (!isRowMajor && !transposeA && transposeB) {
    // Column-major, B transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[k * lda + i] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
      }
    }
  } else {
    // Column-major, both transposed
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += (A[i * lda + k] ?? 0) * (B[k * ldb + j] ?? 0);
        }
        C[j * ldc + i] = (C[j * ldc + i] ?? 0) + alpha * sum;
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
      for (let i = 0; i < b.size; i++) {
        const bData = b.get(i);
        result.set([i], multiplyValues(aVal!, bData));
      }
      return result;
    } else {
      // b is scalar, a is array: array * scalar (element-wise)
      const resultDtype = promoteDTypes(a.dtype, b.dtype);
      const result = ArrayStorage.zeros([...a.shape], resultDtype);
      for (let i = 0; i < a.size; i++) {
        const aData = a.get(i);
        result.set([i], multiplyValues(aData, bVal!));
      }
      return result;
    }
  }

  // Case 1: Both 1D -> scalar (inner product)
  if (aDim === 1 && bDim === 1) {
    if (a.shape[0] !== b.shape[0]) {
      throw new Error(`dot: incompatible shapes (${a.shape[0]},) and (${b.shape[0]},)`);
    }
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
          sumRe += prod;
        }
      }
      return new Complex(sumRe, sumIm);
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

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([m!], resultDtype);

    for (let i = 0; i < m!; i++) {
      let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
      for (let j = 0; j < k!; j++) {
        const aVal = a.get(i, j);
        const bVal = b.get(j);
        const prod = multiplyValues(aVal, bVal);
        sum = addValues(sum, prod);
      }
      result.set([i], sum);
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

    const resultDtype = promoteDTypes(a.dtype, b.dtype);
    const result = ArrayStorage.zeros([n!], resultDtype);

    for (let j = 0; j < n!; j++) {
      let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
      for (let i = 0; i < m; i++) {
        const aVal = a.get(i);
        const bVal = b.get(i, j);
        const prod = multiplyValues(aVal, bVal);
        sum = addValues(sum, prod);
      }
      result.set([j], sum);
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
    for (let i = 0; i < resultSize; i++) {
      let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
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
        const prod = multiplyValues(aVal, bVal);
        sum = addValues(sum, prod);
      }
      result.set(resultIdx, sum);
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
    for (let i = 0; i < resultSize; i++) {
      let temp = i;
      const resultIdx: number[] = [];
      for (let d = resultShape.length - 1; d >= 0; d--) {
        resultIdx[d] = temp % resultShape[d]!;
        temp = Math.floor(temp / resultShape[d]!);
      }

      const bIdxBefore = resultIdx.slice(0, contractAxisB);
      const bIdxAfter = resultIdx.slice(contractAxisB);

      let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
      for (let k = 0; k < aSize; k++) {
        const aVal = a.get(k);
        const bIdx = [...bIdxBefore, k, ...bIdxAfter];
        const bVal = b.get(...bIdx);
        const prod = multiplyValues(aVal, bVal);
        sum = addValues(sum, prod);
      }
      result.set(resultIdx, sum);
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

    for (let i = 0; i < aOuterSize; i++) {
      for (let j = 0; j < bOuterSize; j++) {
        for (let k = 0; k < bLastDim; k++) {
          let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
          for (let m = 0; m < contractionDim; m++) {
            const aIdx = i * contractionDim + m;
            const bIdx = j * contractionDim * bLastDim + m * bLastDim + k;

            // For complex, we need to use get() to properly extract Complex values
            // For non-complex, direct data access is fine
            let aVal: number | bigint | Complex;
            let bVal: number | bigint | Complex;

            if (isComplex) {
              // Use get with multi-dim indices for proper Complex extraction
              const aMultiIdx: number[] = [];
              let tempA = i;
              for (let d = a.shape.length - 2; d >= 0; d--) {
                aMultiIdx.unshift(tempA % a.shape[d]!);
                tempA = Math.floor(tempA / a.shape[d]!);
              }
              aMultiIdx.push(m);
              aVal = a.get(...aMultiIdx);

              const bMultiIdx: number[] = [];
              let tempB = j;
              for (let d = b.shape.length - 3; d >= 0; d--) {
                bMultiIdx.unshift(tempB % b.shape[d]!);
                tempB = Math.floor(tempB / b.shape[d]!);
              }
              bMultiIdx.push(m, k);
              bVal = b.get(...bMultiIdx);
            } else {
              aVal = a.data[aIdx + a.offset] as number | bigint;
              bVal = b.data[bIdx + b.offset] as number | bigint;
            }

            const prod = multiplyValues(aVal, bVal);
            sum = addValues(sum, prod);
          }

          const resultIdx = i * bOuterSize * bLastDim + j * bLastDim + k;
          if (isComplex) {
            const sumComplex = sum as Complex;
            const resultData = result.data as Float64Array;
            resultData[resultIdx * 2] = sumComplex.re;
            resultData[resultIdx * 2 + 1] = sumComplex.im;
          } else {
            result.data[resultIdx] = sum as number;
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
 * Matrix multiplication
 * Requires 2D arrays with compatible shapes
 *
 * Automatically detects transposed/non-contiguous arrays via strides
 * and uses appropriate DGEMM transpose parameters.
 *
 * Note: Currently uses float64 precision for all operations.
 * Integer inputs are promoted to float64 (matching NumPy behavior).
 */
export function matmul(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('matmul requires 2D arrays');
  }

  const [m = 0, k = 0] = a.shape;
  const [k2 = 0, n = 0] = b.shape;

  if (k !== k2) {
    throw new Error(`matmul shape mismatch: (${m},${k}) @ (${k2},${n})`);
  }

  // Determine result dtype (promote inputs, but use float64 for integer types)
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Handle complex matrix multiplication
  if (isComplexDType(resultDtype)) {
    const result = ArrayStorage.zeros([m, n], resultDtype);
    const resultData = result.data as Float64Array;

    // Simple O(m*n*k) matrix multiplication with complex numbers
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let l = 0; l < k; l++) {
          const aVal = a.get(i, l) as Complex;
          const bVal = b.get(l, j) as Complex;
          // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
          const aRe = aVal instanceof Complex ? aVal.re : Number(aVal);
          const aIm = aVal instanceof Complex ? aVal.im : 0;
          const bRe = bVal instanceof Complex ? bVal.re : Number(bVal);
          const bIm = bVal instanceof Complex ? bVal.im : 0;
          sumRe += aRe * bRe - aIm * bIm;
          sumIm += aRe * bIm + aIm * bRe;
        }
        // Store in interleaved format
        const idx = i * n + j;
        resultData[idx * 2] = sumRe;
        resultData[idx * 2 + 1] = sumIm;
      }
    }
    return result;
  }

  const computeDtype =
    resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool'
      ? 'float64'
      : resultDtype;

  // For now, we only support float64 matmul (using dgemm)
  // TODO: Add float32 support using sgemm
  if (computeDtype !== 'float64') {
    throw new Error(`matmul currently only supports float64, got ${computeDtype}`);
  }

  // Convert inputs to Float64Array if needed
  let aData =
    a.dtype === 'float64'
      ? (a.data as Float64Array)
      : Float64Array.from(Array.from(a.data as ArrayLike<number>).map(Number));
  let bData =
    b.dtype === 'float64'
      ? (b.data as Float64Array)
      : Float64Array.from(Array.from(b.data as ArrayLike<number>).map(Number));

  // Handle offset for sliced arrays (views)
  // If the array has an offset, we need to pass the subarray starting from that offset
  if (a.offset > 0) {
    aData = aData.subarray(a.offset) as Float64Array;
  }
  if (b.offset > 0) {
    bData = bData.subarray(b.offset) as Float64Array;
  }

  // Detect array layout from strides
  // Row-major (C-contiguous): row stride > col stride
  // Transposed (F-contiguous or transposed view): col stride > row stride
  const [aStrideRow = 0, aStrideCol = 0] = a.strides;
  const [bStrideRow = 0, bStrideCol = 0] = b.strides;

  // Determine if arrays are effectively transposed
  // For a normal MxK array: strides are [K, 1] (row stride = K cols)
  // For a transposed KxM array (viewed as MxK): strides are [1, M] (col stride > row stride)
  const aIsTransposed = aStrideCol > aStrideRow;
  const bIsTransposed = bStrideCol > bStrideRow;

  const transA: Transpose = aIsTransposed ? 'transpose' : 'no-transpose';
  const transB: Transpose = bIsTransposed ? 'transpose' : 'no-transpose';

  // Determine leading dimensions based on memory layout
  // Leading dimension is the stride of the major dimension in memory
  let lda: number;
  let ldb: number;

  if (aIsTransposed) {
    // Array is stored with columns contiguous (F-order or transposed)
    // The leading dimension is how many elements to skip between columns
    lda = aStrideCol;
  } else {
    // Array is row-major (C-order)
    // The leading dimension is the row stride (number of elements per row)
    lda = aStrideRow;
  }

  if (bIsTransposed) {
    ldb = bStrideCol;
  } else {
    ldb = bStrideRow;
  }

  // Create result array (always row-major)
  const result = ArrayStorage.zeros([m, n], 'float64');

  // Call dgemm with detected transpose flags and leading dimensions
  dgemm(
    'row-major',
    transA,
    transB,
    m,
    n,
    k,
    1.0, // alpha
    aData,
    lda, // leading dimension of a (accounts for actual memory layout)
    bData,
    ldb, // leading dimension of b (accounts for actual memory layout)
    0.0, // beta
    result.data as Float64Array,
    n // ldc (result is always row-major with n cols)
  );

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
export function trace(a: ArrayStorage): number | bigint | Complex {
  if (a.ndim !== 2) {
    throw new Error(`trace requires 2D array, got ${a.ndim}D`);
  }

  const [rows = 0, cols = 0] = a.shape;
  const diagLen = Math.min(rows, cols);

  // Handle complex arrays - return Complex sum
  if (isComplexDType(a.dtype)) {
    let sumRe = 0;
    let sumIm = 0;
    for (let i = 0; i < diagLen; i++) {
      const val = a.get(i, i) as Complex;
      sumRe += val.re;
      sumIm += val.im;
    }
    return new Complex(sumRe, sumIm);
  }

  let sum: number | bigint = 0;

  for (let i = 0; i < diagLen; i++) {
    const val = a.get(i, i);
    if (typeof val === 'bigint') {
      sum = (typeof sum === 'bigint' ? sum : BigInt(sum)) + val;
    } else {
      sum = (typeof sum === 'bigint' ? Number(sum) : sum) + (val as number);
    }
  }

  return sum;
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

  // Last dimensions must match
  const aLastDim = a.shape[aDim - 1]!;
  const bLastDim = b.shape[bDim - 1]!;

  if (aLastDim !== bLastDim) {
    throw new Error(
      `inner: incompatible shapes - last dimensions ${aLastDim} and ${bLastDim} don't match`
    );
  }

  // Special case: both 1D -> scalar
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
  for (let i = 0; i < aOuterSize; i++) {
    for (let j = 0; j < bOuterSize; j++) {
      let sum: number | Complex = isComplex ? new Complex(0, 0) : 0;
      for (let k = 0; k < contractionDim; k++) {
        // For simplicity, use direct index for flat 2D case
        const aFlatIdx = aDim === 1 ? k : i * contractionDim + k;
        const bFlatIdx = bDim === 1 ? k : j * contractionDim + k;

        let aVal: number | bigint | Complex;
        let bVal: number | bigint | Complex;

        if (isComplex) {
          // For complex, need to properly extract values
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
        } else {
          aVal = a.data[aFlatIdx + a.offset] as number | bigint;
          bVal = b.data[bFlatIdx + b.offset] as number | bigint;
        }

        const prod = multiplyValues(aVal, bVal);
        sum = addValues(sum, prod);
      }

      // Set result
      if (resultShape.length === 0) {
        // Scalar result
        return sum;
      }
      const resultIdx = aOuterSize === 1 ? j : i * bOuterSize + j;
      if (isComplex) {
        const sumComplex = sum as Complex;
        const resultData = result.data as Float64Array;
        resultData[resultIdx * 2] = sumComplex.re;
        resultData[resultIdx * 2 + 1] = sumComplex.im;
      } else {
        result.data[resultIdx] = sum as number;
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

  const m = aFlat.size;
  const n = bFlat.size;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const result = ArrayStorage.zeros([m, n], resultDtype);

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
): ArrayStorage | number | bigint {
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

  // Special case: no free axes (full contraction) -> scalar result
  if (resultShape.length === 0) {
    let sum = 0;
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

      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }
    return sum;
  }

  // General case: with free axes
  const resultDtype = promoteDTypes(a.dtype, b.dtype);
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

    let sum = 0;

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

      if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
        sum = Number(sum) + Number(aVal * bVal);
      } else {
        sum += Number(aVal) * Number(bVal);
      }
    }

    result.set(resultIndices, sum);
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

  // Create output array
  const result = ArrayStorage.zeros(outShape, a.dtype);

  // Extract diagonal elements
  // We need to iterate over all combinations of indices for other dimensions
  const otherDims = shape.filter((_, i) => i !== ax1 && i !== ax2);
  const otherSize = otherDims.reduce((acc, d) => acc * d, 1);

  for (let otherIdx = 0; otherIdx < otherSize; otherIdx++) {
    // Convert flat index to multi-dimensional indices for "other" dimensions
    let temp = otherIdx;
    const otherIndices: number[] = [];
    for (let i = otherDims.length - 1; i >= 0; i--) {
      otherIndices.unshift(temp % otherDims[i]!);
      temp = Math.floor(temp / otherDims[i]!);
    }

    // Extract diagonal for this slice
    for (let d = 0; d < diagLen; d++) {
      // Build source indices
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

      // Build destination indices
      const dstIndices = [...otherIndices, d];

      // Copy element
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
): ArrayStorage | number | bigint {
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
): number {
  // All indices are summation indices
  let sumSize = 1;
  for (const idx of sumIndices) {
    sumSize *= indexDims.get(idx)!;
  }

  let sum = 0;

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

  return sum;
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
): ArrayStorage | number {
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

  // Simple case: both are 1D vectors
  if (a.ndim === 1 && b.ndim === 1) {
    const dimA = a.shape[0]!;
    const dimB = b.shape[0]!;

    if (dimA === 3 && dimB === 3) {
      // 3D cross product
      const a0 = Number(a.get(0));
      const a1 = Number(a.get(1));
      const a2 = Number(a.get(2));
      const b0 = Number(b.get(0));
      const b1 = Number(b.get(1));
      const b2 = Number(b.get(2));

      const result = ArrayStorage.zeros([3], 'float64');
      result.set([0], a1 * b2 - a2 * b1);
      result.set([1], a2 * b0 - a0 * b2);
      result.set([2], a0 * b1 - a1 * b0);
      return result;
    } else if (dimA === 2 && dimB === 2) {
      // 2D cross product (returns scalar)
      const a0 = Number(a.get(0));
      const a1 = Number(a.get(1));
      const b0 = Number(b.get(0));
      const b1 = Number(b.get(1));
      return a0 * b1 - a1 * b0;
    } else if ((dimA === 2 && dimB === 3) || (dimA === 3 && dimB === 2)) {
      // Mixed 2D/3D - treat 2D as having z=0
      const a0 = Number(a.get(0));
      const a1 = Number(a.get(1));
      const a2 = dimA === 3 ? Number(a.get(2)) : 0;
      const b0 = Number(b.get(0));
      const b1 = Number(b.get(1));
      const b2 = dimB === 3 ? Number(b.get(2)) : 0;

      const result = ArrayStorage.zeros([3], 'float64');
      result.set([0], a1 * b2 - a2 * b1);
      result.set([1], a2 * b0 - a0 * b2);
      result.set([2], a0 * b1 - a1 * b0);
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

  const result = ArrayStorage.zeros(resultShape, 'float64');

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
    const getA = (idx: number) => {
      aIndices[axisA] = idx;
      return Number(a.get(...aIndices));
    };
    const getB = (idx: number) => {
      bIndices[axisB] = idx;
      return Number(b.get(...bIndices));
    };

    const a0 = getA(0);
    const a1 = getA(1);
    const a2 = vectorDimA === 3 ? getA(2) : 0;
    const b0 = getB(0);
    const b1 = getB(1);
    const b2 = vectorDimB === 3 ? getB(2) : 0;

    if (outputVectorDim === 0) {
      // Scalar result
      result.set(otherIndices, a0 * b1 - a1 * b0);
    } else {
      // Vector result
      const c0 = a1 * b2 - a2 * b1;
      const c1 = a2 * b0 - a0 * b2;
      const c2 = a0 * b1 - a1 * b0;

      const setResult = (idx: number, val: number) => {
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
    if (ord === Infinity) {
      result = 0;
      for (let i = 0; i < n; i++) {
        result = Math.max(result, Math.abs(Number(flat.get(i))));
      }
    } else if (ord === -Infinity) {
      result = Infinity;
      for (let i = 0; i < n; i++) {
        result = Math.min(result, Math.abs(Number(flat.get(i))));
      }
    } else if (ord === 0) {
      result = 0;
      for (let i = 0; i < n; i++) {
        if (Number(flat.get(i)) !== 0) result++;
      }
    } else if (ord === 1) {
      result = 0;
      for (let i = 0; i < n; i++) {
        result += Math.abs(Number(flat.get(i)));
      }
    } else if (ord === 2) {
      result = 0;
      for (let i = 0; i < n; i++) {
        const val = Number(flat.get(i));
        result += val * val;
      }
      result = Math.sqrt(result);
    } else {
      result = 0;
      for (let i = 0; i < n; i++) {
        result += Math.pow(Math.abs(Number(flat.get(i))), ord);
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
        normVal = Math.max(normVal, Math.abs(Number(x.get(...inIndices))));
      }
    } else if (ord === -Infinity) {
      normVal = Infinity;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal = Math.min(normVal, Math.abs(Number(x.get(...inIndices))));
      }
    } else if (ord === 0) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        if (Number(x.get(...inIndices)) !== 0) normVal++;
      }
    } else if (ord === 1) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        normVal += Math.abs(Number(x.get(...inIndices)));
      }
    } else if (ord === 2) {
      normVal = 0;
      for (let i = 0; i < axisLen; i++) {
        inIndices[ax] = i;
        const val = Number(x.get(...inIndices));
        normVal += val * val;
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
  if (x.ndim !== 2) {
    throw new Error(`matrix_norm: input must be 2D, got ${x.ndim}D`);
  }

  const [m, n] = x.shape;
  let result: number;

  if (ord === 'fro') {
    // Frobenius norm: sqrt(sum(abs(x)^2))
    result = 0;
    for (let i = 0; i < m!; i++) {
      for (let j = 0; j < n!; j++) {
        const val = Number(x.get(i, j));
        result += val * val;
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
        colSum += Math.abs(Number(x.get(i, j)));
      }
      result = Math.max(result, colSum);
    }
  } else if (ord === -1) {
    // Min column sum
    result = Infinity;
    for (let j = 0; j < n!; j++) {
      let colSum = 0;
      for (let i = 0; i < m!; i++) {
        colSum += Math.abs(Number(x.get(i, j)));
      }
      result = Math.min(result, colSum);
    }
  } else if (ord === Infinity) {
    // Max row sum
    result = 0;
    for (let i = 0; i < m!; i++) {
      let rowSum = 0;
      for (let j = 0; j < n!; j++) {
        rowSum += Math.abs(Number(x.get(i, j)));
      }
      result = Math.max(result, rowSum);
    }
  } else if (ord === -Infinity) {
    // Min row sum
    result = Infinity;
    for (let i = 0; i < m!; i++) {
      let rowSum = 0;
      for (let j = 0; j < n!; j++) {
        rowSum += Math.abs(Number(x.get(i, j)));
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
      // axis is [number, number] - matrix norm
      return matrix_norm(x, 'fro', keepdims);
    }
  }

  // If axis is specified as a tuple, compute matrix norm
  if (Array.isArray(axis)) {
    if (axis.length !== 2) {
      throw new Error('norm: axis must be a 2-tuple for matrix norms');
    }
    // For now, only support the simple case where the matrix axes are (0, 1) or (-2, -1)
    const ax0 = axis[0] < 0 ? x.ndim + axis[0] : axis[0];
    const ax1 = axis[1] < 0 ? x.ndim + axis[1] : axis[1];

    if (x.ndim !== 2 || (ax0 !== 0 && ax0 !== 1) || (ax1 !== 0 && ax1 !== 1) || ax0 === ax1) {
      throw new Error('norm: complex axis specification not yet supported');
    }

    return matrix_norm(x, ord as 'fro' | 'nuc' | number, keepdims);
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
  if (a.ndim !== 2) {
    throw new Error(`qr: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  const k = Math.min(m!, n!);

  // Copy input to working array (float64)
  const R = ArrayStorage.zeros([m!, n!], 'float64');
  for (let i = 0; i < m!; i++) {
    for (let j = 0; j < n!; j++) {
      R.set([i, j], Number(a.get(i, j)));
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

  const rRows = mode === 'complete' ? m! : k;
  const rResult = ArrayStorage.zeros([rRows, n!], 'float64');
  for (let i = 0; i < rRows; i++) {
    for (let j = 0; j < n!; j++) {
      if (j >= i) {
        rResult.set([i, j], Number(R.get(i, j)));
      }
    }
  }

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
  if (a.ndim !== 2) {
    throw new Error(`cholesky: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`cholesky: matrix must be square, got ${m}x${n}`);
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
        const val = Number(a.get(j, j)) - sum;
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
        L.set([i, j], (Number(a.get(i, j)) - sum) / ljj);
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

  const [m, n] = a.shape;
  const smaller = Math.min(m!, n!);

  // For SVD, we use the approach: A^T @ A has eigenvalues sigma^2
  // and A @ A^T also has eigenvalues sigma^2
  // V are eigenvectors of A^T @ A
  // U are eigenvectors of A @ A^T

  // Compute A^T @ A
  const ATA = ArrayStorage.zeros([n!, n!], 'float64');
  for (let i = 0; i < n!; i++) {
    for (let j = 0; j < n!; j++) {
      let sum = 0;
      for (let k = 0; k < m!; k++) {
        sum += Number(a.get(k, i)) * Number(a.get(k, j));
      }
      ATA.set([i, j], sum);
    }
  }

  // Get eigendecomposition of A^T @ A
  const { values: eigVals, vectors: V } = eigSymmetric(ATA);

  // Sort eigenvalues in descending order
  const indices = Array.from({ length: n! }, (_, i) => i);
  indices.sort((i, j) => eigVals[j]! - eigVals[i]!);

  // Singular values are sqrt of eigenvalues
  const s = ArrayStorage.zeros([smaller], 'float64');
  for (let i = 0; i < smaller; i++) {
    const eigVal = eigVals[indices[i]!]!;
    s.set([i], Math.sqrt(Math.max(0, eigVal)));
  }

  // V^T (sorted)
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
        let sum = 0;
        for (let k = 0; k < n!; k++) {
          sum += Number(a.get(i, k)) * Number(vt.get(j, k));
        }
        u.set([i, j], sum / sigma);
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

  // Copy matrix
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    A.push([]);
    for (let j = 0; j < n; j++) {
      A[i]!.push(Number(a.get(i, j)));
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
  const result = svdFull(a);

  if (!compute_uv) {
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

    // Reduced V^T: k x n
    const vtReduced = ArrayStorage.zeros([k, n!], 'float64');
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n!; j++) {
        vtReduced.set([i, j], Number(result.vt.get(i, j)));
      }
    }

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
export function det(a: ArrayStorage): number {
  if (a.ndim !== 2) {
    throw new Error(`det: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`det: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  if (size === 0) {
    return 1; // Empty matrix has determinant 1
  }

  const aData = a.data;

  if (size === 1) {
    return Number(aData[0]);
  }

  if (size === 2) {
    return Number(aData[0]) * Number(aData[3]) - Number(aData[1]) * Number(aData[2]);
  }

  // LU decomposition with partial pivoting
  const { lu, sign } = luDecomposition(a);

  // Determinant is product of diagonal of U times sign from pivoting
  // Use direct array access for speed
  const luData = lu.data as Float64Array;
  let result = sign;
  for (let i = 0; i < size; i++) {
    result *= luData[i * size + i]!;
  }

  return result;
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

  // Copy matrix - use direct array access for speed
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
 * Compute the matrix inverse - optimized to do LU decomposition once.
 *
 * @param a - Square matrix
 * @returns Inverse matrix
 */
export function inv(a: ArrayStorage): ArrayStorage {
  if (a.ndim !== 2) {
    throw new Error(`inv: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`inv: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // Do LU decomposition once
  const { lu, piv } = luDecomposition(a);
  const luData = lu.data as Float64Array;

  // Solve A @ X = I for all columns at once
  const result = ArrayStorage.zeros([size, size], 'float64');
  const resultData = result.data as Float64Array;

  // Process each column of the identity matrix
  for (let col = 0; col < size; col++) {
    // Forward substitution for column col (L @ y = P @ e_col)
    const y = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      // e_col[piv[i]] is 1 if piv[i] === col, else 0
      let sum = piv[i] === col ? 1 : 0;
      for (let j = 0; j < i; j++) {
        sum -= luData[i * size + j]! * y[j]!;
      }
      y[i] = sum;
    }

    // Back substitution (U @ x = y)
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

  // LU decomposition
  const { lu, piv } = luDecomposition(a);
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
}

/**
 * Solve a linear system A @ x = b.
 *
 * @param a - Coefficient matrix (n x n)
 * @param b - Right-hand side (n,) or (n, k)
 * @returns Solution x with same shape as b
 */
export function solve(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  if (a.ndim !== 2) {
    throw new Error(`solve: coefficient matrix must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`solve: coefficient matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

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

    const k = b.shape[1]!;
    const result = ArrayStorage.zeros([size, k], 'float64');

    // Solve for each column
    for (let j = 0; j < k; j++) {
      // Extract column j of b
      const bCol = ArrayStorage.zeros([size], 'float64');
      for (let i = 0; i < size; i++) {
        bCol.set([i], Number(b.get(i, j)));
      }

      // Solve
      const xCol = solveVector(a, bCol);

      // Copy to result
      for (let i = 0; i < size; i++) {
        result.set([i, j], Number(xCol.get(i)));
      }
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
  if (a.ndim !== 2) {
    throw new Error(`lstsq: coefficient matrix must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;

  // Use SVD to solve least squares
  const { u, s, vt } = svdFull(a);
  const k = Math.min(m!, n!);

  // Determine rcond
  const threshold = rcond ?? Math.max(m!, n!) * Number.EPSILON;
  const maxSigma = Number(s.get(0));
  const cutoff = maxSigma * threshold;

  // Compute effective rank
  let rank = 0;
  for (let i = 0; i < k; i++) {
    if (Number(s.get(i)) > cutoff) {
      rank++;
    }
  }

  // Handle 1D b
  const b2D = b.ndim === 1 ? shapeOps.reshape(b, [b.size, 1]) : b;
  const nrhs = b2D.shape[1]!;

  if (b2D.shape[0] !== m) {
    throw new Error(`lstsq: incompatible shapes (${m},${n}) and (${b.shape.join(',')})`);
  }

  // Compute x = V @ S^+ @ U^T @ b
  // where S^+ is pseudoinverse of S (1/s for s > cutoff, 0 otherwise)
  const x = ArrayStorage.zeros([n!, nrhs], 'float64');

  for (let j = 0; j < nrhs; j++) {
    // Compute U^T @ b[:, j]
    const utb: number[] = new Array(m!).fill(0);
    for (let i = 0; i < m!; i++) {
      for (let l = 0; l < m!; l++) {
        utb[i]! += Number(u.get(l, i)) * Number(b2D.get(l, j));
      }
    }

    // Apply S^+ and V
    for (let i = 0; i < n!; i++) {
      let sum = 0;
      for (let l = 0; l < k; l++) {
        const sigma = Number(s.get(l));
        if (sigma > cutoff) {
          sum += (Number(vt.get(l, i)) * utb[l]!) / sigma;
        }
      }
      x.set([i, j], sum);
    }
  }

  // Compute residuals if m > n (overdetermined)
  let residuals: ArrayStorage;
  if (m! > n!) {
    residuals = ArrayStorage.zeros([nrhs], 'float64');
    for (let j = 0; j < nrhs; j++) {
      // Compute ||A @ x[:, j] - b[:, j]||^2
      let resSum = 0;
      for (let i = 0; i < m!; i++) {
        let axij = 0;
        for (let l = 0; l < n!; l++) {
          axij += Number(a.get(i, l)) * Number(x.get(l, j));
        }
        const diff = axij - Number(b2D.get(i, j));
        resSum += diff * diff;
      }
      residuals.set([j], resSum);
    }
  } else {
    residuals = ArrayStorage.zeros([0], 'float64');
  }

  // Reshape x if b was 1D
  const xResult = b.ndim === 1 ? shapeOps.reshape(x, [n!]) : x;

  return { x: xResult, residuals, rank, s };
}

/**
 * Compute the condition number of a matrix.
 *
 * @param a - Input matrix
 * @param p - Order of the norm (default: 2, -2, 'fro', or inf)
 * @returns Condition number
 */
export function cond(a: ArrayStorage, p: number | 'fro' | 'nuc' = 2): number {
  if (a.ndim !== 2) {
    throw new Error(`cond: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;

  if (p === 2 || p === -2) {
    // Condition number from singular values
    const { s } = svdFull(a);
    const k = Math.min(m!, n!);
    const maxS = Number(s.get(0));
    const minS = Number(s.get(k - 1));

    if (p === 2) {
      return minS > 0 ? maxS / minS : Infinity;
    } else {
      return maxS > 0 ? minS / maxS : 0;
    }
  }

  // For other norms, compute norm(A) * norm(inv(A))
  if (m !== n) {
    throw new Error(`cond: matrix must be square for p=${p}`);
  }

  const normA = matrix_norm(a, p as 'fro' | number) as number;
  const invA = inv(a);
  const normInvA = matrix_norm(invA, p as 'fro' | number) as number;

  return normA * normInvA;
}

/**
 * Compute the rank of a matrix using SVD.
 *
 * @param a - Input matrix
 * @param tol - Threshold below which singular values are considered zero
 * @returns Matrix rank
 */
export function matrix_rank(a: ArrayStorage, tol?: number): number {
  if (a.ndim === 0) {
    return Number(a.get()) !== 0 ? 1 : 0;
  }

  if (a.ndim === 1) {
    for (let i = 0; i < a.size; i++) {
      if (Number(a.get(i)) !== 0) return 1;
    }
    return 0;
  }

  if (a.ndim !== 2) {
    throw new Error(`matrix_rank: input must be at most 2D, got ${a.ndim}D`);
  }

  const { s } = svdFull(a);
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

  // Handle n = 0: return identity
  if (n === 0) {
    const result = ArrayStorage.zeros([size, size], 'float64');
    for (let i = 0; i < size; i++) {
      result.set([i, i], 1);
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
  let result = ArrayStorage.zeros([size, size], 'float64');
  for (let i = 0; i < size; i++) {
    result.set([i, i], 1);
  }

  let current = ArrayStorage.zeros([size, size], 'float64');
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      current.set([i, j], Number(base.get(i, j)));
    }
  }

  while (power > 0) {
    if (power & 1) {
      result = matmul(result, current);
    }
    current = matmul(current, current);
    power >>= 1;
  }

  return result;
}

/**
 * Compute the Moore-Penrose pseudo-inverse using SVD.
 *
 * @param a - Input matrix
 * @param rcond - Cutoff for small singular values
 * @returns Pseudo-inverse of a
 */
export function pinv(a: ArrayStorage, rcond: number = 1e-15): ArrayStorage {
  if (a.ndim !== 2) {
    throw new Error(`pinv: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  const { u, s, vt } = svdFull(a);
  const k = Math.min(m!, n!);

  // Determine cutoff
  const maxS = Number(s.get(0));
  const cutoff = maxS * rcond;

  // Compute V @ S^+ @ U^T
  // S^+ has 1/s for s > cutoff, 0 otherwise
  const result = ArrayStorage.zeros([n!, m!], 'float64');

  for (let i = 0; i < n!; i++) {
    for (let j = 0; j < m!; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) {
        const sigma = Number(s.get(l));
        if (sigma > cutoff) {
          sum += (Number(vt.get(l, i)) * Number(u.get(j, l))) / sigma;
        }
      }
      result.set([i, j], sum);
    }
  }

  return result;
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
  if (a.ndim !== 2) {
    throw new Error(`eig: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`eig: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // Check if symmetric
  let isSymmetric = true;
  outer: for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      if (Math.abs(Number(a.get(i, j)) - Number(a.get(j, i))) > 1e-10) {
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

  // WARNING: Non-symmetric matrices may have complex eigenvalues which we cannot represent.
  // This implementation returns only real approximations and may be inaccurate.
  console.warn(
    'numpy-ts: eig() called on non-symmetric matrix. Complex eigenvalues are not supported; ' +
      'results may be inaccurate. For symmetric matrices, use eigh() instead.'
  );

  // For non-symmetric matrices, use QR iteration (simplified)
  // This is a basic implementation that may not converge for all matrices
  const { values, vectors } = qrEigendecomposition(a);

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
function qrEigendecomposition(a: ArrayStorage): { values: number[]; vectors: number[][] } {
  const n = a.shape[0]!;
  const maxIter = 1000;
  const tol = 1e-10;

  // Copy matrix
  let A = ArrayStorage.zeros([n, n], 'float64');
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A.set([i, j], Number(a.get(i, j)));
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

  return { values, vectors };
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
  if (a.ndim !== 2) {
    throw new Error(`eigh: input must be 2D, got ${a.ndim}D`);
  }

  const [m, n] = a.shape;
  if (m !== n) {
    throw new Error(`eigh: matrix must be square, got ${m}x${n}`);
  }

  const size = m!;

  // Symmetrize the matrix using specified triangle
  const sym = ArrayStorage.zeros([size, size], 'float64');
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (UPLO === 'L') {
        if (i >= j) {
          sym.set([i, j], Number(a.get(i, j)));
          sym.set([j, i], Number(a.get(i, j)));
        }
      } else {
        if (j >= i) {
          sym.set([i, j], Number(a.get(i, j)));
          sym.set([j, i], Number(a.get(i, j)));
        }
      }
    }
  }

  // Use symmetric eigendecomposition
  const { values, vectors } = eigSymmetric(sym);

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
  const { w } = eigh(a, UPLO);
  return w;
}
