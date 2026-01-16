/**
 * Pure TypeScript implementations for benchmarking
 *
 * These are standalone implementations (not using numpy-ts) for fair comparison.
 */

// ============================================================================
// Element-wise Operations
// ============================================================================

/**
 * Element-wise addition: c = a + b
 */
export function add_f64(a: Float64Array, b: Float64Array, c: Float64Array): void {
  const len = a.length;
  for (let i = 0; i < len; i++) {
    c[i] = a[i]! + b[i]!;
  }
}

export function add_f32(a: Float32Array, b: Float32Array, c: Float32Array): void {
  const len = a.length;
  for (let i = 0; i < len; i++) {
    c[i] = a[i]! + b[i]!;
  }
}

/**
 * Element-wise sine: c = sin(a)
 */
export function sin_f64(a: Float64Array, c: Float64Array): void {
  const len = a.length;
  for (let i = 0; i < len; i++) {
    c[i] = Math.sin(a[i]!);
  }
}

export function sin_f32(a: Float32Array, c: Float32Array): void {
  const len = a.length;
  for (let i = 0; i < len; i++) {
    c[i] = Math.fround(Math.sin(a[i]!));
  }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/**
 * Sum reduction
 */
export function sum_f64(a: Float64Array): number {
  let total = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    total += a[i]!;
  }
  return total;
}

export function sum_f32(a: Float32Array): number {
  let total = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    total += a[i]!;
  }
  return total;
}

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix multiplication: C = A * B
 * A is MxK, B is KxN, C is MxN
 * Row-major layout
 */
export function matmul_f64(
  a: Float64Array,
  b: Float64Array,
  c: Float64Array,
  m: number,
  n: number,
  k: number
): void {
  // Zero output
  c.fill(0);

  // Naive triple loop with reordered for better cache access (ikj order)
  for (let i = 0; i < m; i++) {
    for (let kk = 0; kk < k; kk++) {
      const aik = a[i * k + kk]!;
      for (let j = 0; j < n; j++) {
        c[i * n + j] += aik * b[kk * n + j]!;
      }
    }
  }
}

export function matmul_f32(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  m: number,
  n: number,
  k: number
): void {
  // Zero output
  c.fill(0);

  // Naive triple loop with reordered for better cache access (ikj order)
  for (let i = 0; i < m; i++) {
    for (let kk = 0; kk < k; kk++) {
      const aik = a[i * k + kk]!;
      for (let j = 0; j < n; j++) {
        c[i * n + j] += aik * b[kk * n + j]!;
      }
    }
  }
}

/**
 * Blocked matrix multiplication for better cache utilization
 * Block size tuned for typical L1 cache
 */
const BLOCK_SIZE = 64;

export function matmul_blocked_f64(
  a: Float64Array,
  b: Float64Array,
  c: Float64Array,
  m: number,
  n: number,
  k: number
): void {
  // Zero output
  c.fill(0);

  // Blocked multiplication
  for (let i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
    const iMax = Math.min(i0 + BLOCK_SIZE, m);
    for (let k0 = 0; k0 < k; k0 += BLOCK_SIZE) {
      const kMax = Math.min(k0 + BLOCK_SIZE, k);
      for (let j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
        const jMax = Math.min(j0 + BLOCK_SIZE, n);

        // Multiply blocks
        for (let i = i0; i < iMax; i++) {
          for (let kk = k0; kk < kMax; kk++) {
            const aik = a[i * k + kk]!;
            for (let j = j0; j < jMax; j++) {
              c[i * n + j] += aik * b[kk * n + j]!;
            }
          }
        }
      }
    }
  }
}

export function matmul_blocked_f32(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  m: number,
  n: number,
  k: number
): void {
  // Zero output
  c.fill(0);

  // Blocked multiplication
  for (let i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
    const iMax = Math.min(i0 + BLOCK_SIZE, m);
    for (let k0 = 0; k0 < k; k0 += BLOCK_SIZE) {
      const kMax = Math.min(k0 + BLOCK_SIZE, k);
      for (let j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
        const jMax = Math.min(j0 + BLOCK_SIZE, n);

        // Multiply blocks
        for (let i = i0; i < iMax; i++) {
          for (let kk = k0; kk < kMax; kk++) {
            const aik = a[i * k + kk]!;
            for (let j = j0; j < jMax; j++) {
              c[i * n + j] += aik * b[kk * n + j]!;
            }
          }
        }
      }
    }
  }
}
