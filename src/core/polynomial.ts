/**
 * Polynomial functions
 *
 * Tree-shakeable standalone functions for polynomial operations.
 */

import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { array } from './creation';

// Helper to convert to array
function toArray(a: NDArrayCore | number[]): NDArrayCore {
  return a instanceof NDArrayCore ? a : array(a);
}

/**
 * Find the coefficients of a polynomial with given roots
 */
export function poly(seq_of_zeros: NDArrayCore | number[]): NDArrayCore {
  const roots = toArray(seq_of_zeros);
  const data = roots.data;
  const n = roots.size;

  if (n === 0) {
    return array([1]);
  }

  // Start with [1]
  let coeffs = [1];

  // Multiply by (x - root) for each root
  for (let i = 0; i < n; i++) {
    const root = data[i] as number;
    const newCoeffs = new Array(coeffs.length + 1).fill(0);

    for (let j = 0; j < coeffs.length; j++) {
      (newCoeffs[j] as number) += coeffs[j]!;
      (newCoeffs[j + 1] as number) -= coeffs[j]! * root;
    }

    coeffs = newCoeffs;
  }

  return array(coeffs);
}

/**
 * Add two polynomials
 */
export function polyadd(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArrayCore {
  const p1 = toArray(a1);
  const p2 = toArray(a2);
  const d1 = p1.data;
  const d2 = p2.data;

  const maxLen = Math.max(p1.size, p2.size);
  const result = new Array(maxLen).fill(0);

  // Add from the end (lower degree terms)
  for (let i = 0; i < p1.size; i++) {
    result[maxLen - p1.size + i] += d1[i] as number;
  }
  for (let i = 0; i < p2.size; i++) {
    result[maxLen - p2.size + i] += d2[i] as number;
  }

  // Remove leading zeros
  let start = 0;
  while (start < result.length - 1 && result[start] === 0) {
    start++;
  }

  return array(result.slice(start));
}

/**
 * Differentiate a polynomial
 */
export function polyder(p: NDArrayCore | number[], m: number = 1): NDArrayCore {
  let poly = toArray(p);

  for (let k = 0; k < m; k++) {
    const data = poly.data;
    const n = poly.size;

    if (n <= 1) {
      return array([0]);
    }

    const result: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      const power = n - 1 - i;
      result.push((data[i] as number) * power);
    }

    poly = array(result);
  }

  return poly;
}

/**
 * Divide two polynomials
 */
export function polydiv(
  u: NDArrayCore | number[],
  v: NDArrayCore | number[]
): [NDArrayCore, NDArrayCore] {
  const dividend = [...(toArray(u).data as unknown as number[])];
  const divisor = [...(toArray(v).data as unknown as number[])];

  if (divisor.length === 0 || (divisor.length === 1 && divisor[0] === 0)) {
    throw new Error('Division by zero polynomial');
  }

  // Remove leading zeros
  while (dividend.length > 1 && dividend[0] === 0) dividend.shift();
  while (divisor.length > 1 && divisor[0] === 0) divisor.shift();

  if (dividend.length < divisor.length) {
    return [array([0]), array(dividend)];
  }

  const quotient: number[] = [];
  const remainder = [...dividend];

  while (remainder.length >= divisor.length) {
    const coeff = remainder[0]! / divisor[0]!;
    quotient.push(coeff);

    for (let i = 0; i < divisor.length; i++) {
      (remainder[i] as number) -= coeff * divisor[i]!;
    }

    remainder.shift();
  }

  // Remove leading zeros from remainder
  while (remainder.length > 1 && Math.abs(remainder[0]!) < 1e-15) {
    remainder.shift();
  }

  return [
    array(quotient.length > 0 ? quotient : [0]),
    array(remainder.length > 0 ? remainder : [0]),
  ];
}

/**
 * Least squares polynomial fit
 */
export function polyfit(x: NDArrayCore, y: NDArrayCore, deg: number): NDArrayCore {
  const xData = x.data;
  const yData = y.data;
  const n = x.size;

  if (deg >= n) {
    throw new Error('polyfit: degree must be less than number of points');
  }

  // Build Vandermonde matrix
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = deg; j >= 0; j--) {
      row.push(Math.pow(xData[i] as number, j));
    }
    A.push(row);
  }

  // Solve A^T * A * c = A^T * y using normal equations
  const ATA: number[][] = [];
  const ATy: number[] = [];

  for (let i = 0; i <= deg; i++) {
    ATA.push([]);
    for (let j = 0; j <= deg; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += A[k]![i]! * A[k]![j]!;
      }
      ATA[i]!.push(sum);
    }

    let sum = 0;
    for (let k = 0; k < n; k++) {
      sum += A[k]![i]! * (yData[k] as number);
    }
    ATy.push(sum);
  }

  // Gaussian elimination with partial pivoting
  const m = deg + 1;
  const augmented = ATA.map((row, i) => [...row, ATy[i]!]);

  for (let i = 0; i < m; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < m; k++) {
      if (Math.abs(augmented[k]![i]!) > Math.abs(augmented[maxRow]![i]!)) {
        maxRow = k;
      }
    }
    [augmented[i], augmented[maxRow]] = [augmented[maxRow]!, augmented[i]!];

    // Eliminate column
    for (let k = i + 1; k < m; k++) {
      const factor = augmented[k]![i]! / augmented[i]![i]!;
      for (let j = i; j <= m; j++) {
        (augmented[k]![j] as number) -= factor * augmented[i]![j]!;
      }
    }
  }

  // Back substitution
  const coeffs = new Array(m).fill(0) as number[];
  for (let i = m - 1; i >= 0; i--) {
    let sum = augmented[i]![m]!;
    for (let j = i + 1; j < m; j++) {
      sum -= augmented[i]![j]! * coeffs[j]!;
    }
    coeffs[i] = sum / augmented[i]![i]!;
  }

  return array(coeffs);
}

/**
 * Integrate a polynomial
 */
export function polyint(
  p: NDArrayCore | number[],
  m: number = 1,
  k: number | number[] = 0
): NDArrayCore {
  let poly = toArray(p);
  const constants = Array.isArray(k) ? k : [k];

  for (let i = 0; i < m; i++) {
    const data = poly.data;
    const n = poly.size;

    const result: number[] = [];
    for (let j = 0; j < n; j++) {
      const power = n - j;
      result.push((data[j] as number) / power);
    }

    // Add integration constant
    const c = i < constants.length ? constants[i]! : 0;
    result.push(c);

    poly = array(result);
  }

  return poly;
}

/**
 * Multiply two polynomials
 */
export function polymul(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArrayCore {
  const p1 = toArray(a1);
  const p2 = toArray(a2);
  const d1 = p1.data;
  const d2 = p2.data;

  const resultLen = p1.size + p2.size - 1;
  const result = new Array(resultLen).fill(0);

  for (let i = 0; i < p1.size; i++) {
    for (let j = 0; j < p2.size; j++) {
      result[i + j] += (d1[i] as number) * (d2[j] as number);
    }
  }

  return array(result);
}

/**
 * Subtract two polynomials
 */
export function polysub(a1: NDArrayCore | number[], a2: NDArrayCore | number[]): NDArrayCore {
  const p1 = toArray(a1);
  const p2 = toArray(a2);
  const d1 = p1.data;
  const d2 = p2.data;

  const maxLen = Math.max(p1.size, p2.size);
  const result = new Array(maxLen).fill(0);

  // Subtract from the end (lower degree terms)
  for (let i = 0; i < p1.size; i++) {
    result[maxLen - p1.size + i] += d1[i] as number;
  }
  for (let i = 0; i < p2.size; i++) {
    result[maxLen - p2.size + i] -= d2[i] as number;
  }

  // Remove leading zeros
  let start = 0;
  while (start < result.length - 1 && result[start] === 0) {
    start++;
  }

  return array(result.slice(start));
}

/**
 * Evaluate a polynomial at given points
 */
export function polyval(
  p: NDArrayCore | number[],
  x: NDArrayCore | number | number[]
): NDArrayCore | number {
  const poly = toArray(p);
  const coeffs = poly.data;

  if (typeof x === 'number') {
    // Horner's method for single value
    let result = coeffs[0] as number;
    for (let i = 1; i < poly.size; i++) {
      result = result * x + (coeffs[i] as number);
    }
    return result;
  }

  const xArr = x instanceof NDArrayCore ? x : array(x);
  const xData = xArr.data;
  const results: number[] = [];

  for (let j = 0; j < xArr.size; j++) {
    const xVal = xData[j] as number;
    let result = coeffs[0] as number;
    for (let i = 1; i < poly.size; i++) {
      result = result * xVal + (coeffs[i] as number);
    }
    results.push(result);
  }

  return array(results);
}

/**
 * Find the roots of a polynomial.
 *
 * Uses the companion matrix eigenvalue method (same as NumPy).
 * Always returns a complex128 NDArrayCore.
 */
export function roots(p: NDArrayCore | number[]): NDArrayCore {
  const poly = toArray(p);
  const coeffs = [...(poly.data as unknown as number[])];

  // Remove leading zeros
  while (coeffs.length > 1 && coeffs[0] === 0) {
    coeffs.shift();
  }

  // Count and remove trailing zeros (these are zero roots)
  let numZeroRoots = 0;
  while (coeffs.length > 1 && coeffs[coeffs.length - 1] === 0) {
    coeffs.pop();
    numZeroRoots++;
  }

  const n = coeffs.length - 1; // degree of reduced polynomial
  const totalRoots = n + numZeroRoots;

  if (totalRoots === 0) {
    return _makeComplex128Array([], []);
  }

  // Find roots of the reduced polynomial
  let realParts: number[] = [];
  let imagParts: number[] = [];

  if (n === 1) {
    realParts.push(-coeffs[1]! / coeffs[0]!);
    imagParts.push(0);
  } else if (n === 2) {
    const a = coeffs[0]!;
    const b = coeffs[1]!;
    const c = coeffs[2]!;
    const disc = b * b - 4 * a * c;

    if (disc >= 0) {
      const sqrtD = Math.sqrt(disc);
      realParts.push((-b + sqrtD) / (2 * a), (-b - sqrtD) / (2 * a));
      imagParts.push(0, 0);
    } else {
      const sqrtD = Math.sqrt(-disc);
      realParts.push(-b / (2 * a), -b / (2 * a));
      imagParts.push(sqrtD / (2 * a), -sqrtD / (2 * a));
    }
  } else if (n >= 3) {
    const evs = _companionEigenvalues(coeffs, n);
    for (const ev of evs) {
      realParts.push(ev.re);
      imagParts.push(ev.im);
    }
  }

  // Add zero roots
  for (let i = 0; i < numZeroRoots; i++) {
    realParts.push(0);
    imagParts.push(0);
  }

  // Sort by magnitude descending (matching NumPy's eigenvalue ordering)
  const indices = realParts.map((_, i) => i);
  indices.sort((a, b) => {
    const magA = Math.sqrt(realParts[a]! ** 2 + imagParts[a]! ** 2);
    const magB = Math.sqrt(realParts[b]! ** 2 + imagParts[b]! ** 2);
    if (Math.abs(magA - magB) > 1e-10) return magB - magA;
    if (Math.abs(realParts[a]! - realParts[b]!) > 1e-10) return realParts[b]! - realParts[a]!;
    return imagParts[b]! - imagParts[a]!;
  });

  const sortedReal = indices.map((i) => realParts[i]!);
  const sortedImag = indices.map((i) => imagParts[i]!);

  return _makeComplex128Array(sortedReal, sortedImag);
}

/**
 * Create a complex128 NDArrayCore from parallel real/imag arrays.
 */
function _makeComplex128Array(realParts: number[], imagParts: number[]): NDArrayCore {
  const n = realParts.length;
  const data = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    data[2 * i] = realParts[i]!;
    data[2 * i + 1] = imagParts[i]!;
  }
  const storage = ArrayStorage.fromData(data, [n], 'complex128');
  return new NDArrayCore(storage);
}

/**
 * Compute eigenvalues of the companion matrix for polynomial coefficients.
 * The companion matrix is already upper Hessenberg.
 */
function _companionEigenvalues(coeffs: number[], n: number): { re: number; im: number }[] {
  // Build companion matrix (n×n)
  // Row 0: [-c[1]/c[0], -c[2]/c[0], ..., -c[n]/c[0]]
  // Row i (i>0): 1 at column i-1, 0 elsewhere
  const H: number[][] = Array.from({ length: n }, () => new Array(n).fill(0) as number[]);
  const lead = coeffs[0]!;

  for (let j = 0; j < n; j++) {
    H[0]![j] = -coeffs[j + 1]! / lead;
  }
  for (let i = 1; i < n; i++) {
    H[i]![i - 1] = 1;
  }

  return _hessenbergQR(H, n);
}

/**
 * QR iteration on an upper Hessenberg matrix to find all eigenvalues.
 * Uses single-shift Wilkinson QR steps with exceptional shifts.
 */
function _hessenbergQR(H: number[][], n: number): { re: number; im: number }[] {
  const eigenvalues: { re: number; im: number }[] = [];
  const eps = 2.22e-16;
  let nn = n; // size of the active (unreduced) portion
  let totalIter = 0;
  const maxIter = 100 * n;
  let lastNn = n;
  let noDeflationCount = 0;

  while (nn > 0 && totalIter < maxIter) {
    totalIter++;

    if (nn === lastNn) {
      noDeflationCount++;
    } else {
      noDeflationCount = 0;
      lastNn = nn;
    }

    if (nn === 1) {
      eigenvalues.push({ re: H[0]![0]!, im: 0 });
      nn = 0;
      break;
    }

    if (nn === 2) {
      eigenvalues.push(..._eigenvalues2x2(H[0]![0]!, H[0]![1]!, H[1]![0]!, H[1]![1]!));
      nn = 0;
      break;
    }

    // Find start of active unreduced block by scanning from bottom
    let l = nn - 1;
    while (l > 0) {
      const s = Math.abs(H[l - 1]![l - 1]!) + Math.abs(H[l]![l]!);
      const threshold = eps * (s === 0 ? 1 : s);
      if (Math.abs(H[l]![l - 1]!) <= threshold) {
        H[l]![l - 1] = 0;
        break;
      }
      l--;
    }

    const windowSize = nn - l;

    if (windowSize === 1) {
      eigenvalues.push({ re: H[nn - 1]![nn - 1]!, im: 0 });
      nn--;
      continue;
    }

    if (windowSize === 2) {
      eigenvalues.push(
        ..._eigenvalues2x2(
          H[nn - 2]![nn - 2]!,
          H[nn - 2]![nn - 1]!,
          H[nn - 1]![nn - 2]!,
          H[nn - 1]![nn - 1]!
        )
      );
      nn -= 2;
      continue;
    }

    // Compute shift
    let shift: number;
    if (noDeflationCount > 0 && noDeflationCount % 10 === 0) {
      // Exceptional shift to break convergence stalls
      shift = Math.abs(H[nn - 1]![nn - 2]!) + Math.abs(H[nn - 2]![nn - 3]!);
    } else {
      // Wilkinson shift: eigenvalue of bottom-right 2×2 closest to H[nn-1][nn-1]
      const a = H[nn - 2]![nn - 2]!;
      const b = H[nn - 2]![nn - 1]!;
      const c = H[nn - 1]![nn - 2]!;
      const d = H[nn - 1]![nn - 1]!;
      const tr = a + d;
      const det = a * d - b * c;
      const disc = tr * tr - 4 * det;

      if (disc >= 0) {
        const sqrtDisc = Math.sqrt(disc);
        const e1 = (tr + sqrtDisc) / 2;
        const e2 = (tr - sqrtDisc) / 2;
        shift = Math.abs(e1 - d) < Math.abs(e2 - d) ? e1 : e2;
      } else {
        shift = d;
      }
    }

    // Single-shift QR step on active window H[l:nn, l:nn]
    // Apply shift
    for (let i = l; i < nn; i++) {
      H[i]![i] = H[i]![i]! - shift;
    }

    // QR factorization using Givens rotations
    const givensC: number[] = [];
    const givensS: number[] = [];

    for (let i = l; i < nn - 1; i++) {
      const r = Math.hypot(H[i]![i]!, H[i + 1]![i]!);
      const c = r === 0 ? 1 : H[i]![i]! / r;
      const s = r === 0 ? 0 : H[i + 1]![i]! / r;
      givensC.push(c);
      givensS.push(s);

      // Apply G^T from left to rows i and i+1
      for (let j = i; j < nn; j++) {
        const t1 = H[i]![j]!;
        const t2 = H[i + 1]![j]!;
        H[i]![j] = c * t1 + s * t2;
        H[i + 1]![j] = -s * t1 + c * t2;
      }
    }

    // Apply G from right to form RQ
    for (let k = 0; k < givensC.length; k++) {
      const i = l + k;
      const c = givensC[k]!;
      const s = givensS[k]!;
      const maxRow = Math.min(i + 2, nn - 1);

      for (let j = l; j <= maxRow; j++) {
        const t1 = H[j]![i]!;
        const t2 = H[j]![i + 1]!;
        H[j]![i] = c * t1 + s * t2;
        H[j]![i + 1] = -s * t1 + c * t2;
      }
    }

    // Restore shift
    for (let i = l; i < nn; i++) {
      H[i]![i] = H[i]![i]! + shift;
    }
  }

  // If we didn't converge, use diagonal as approximations
  if (nn > 0) {
    for (let i = 0; i < nn; i++) {
      eigenvalues.push({ re: H[i]![i]!, im: 0 });
    }
  }

  return eigenvalues;
}

/**
 * Eigenvalues of a 2×2 matrix [[a, b], [c, d]].
 * Returns complex pair if discriminant is negative.
 */
function _eigenvalues2x2(a: number, b: number, c: number, d: number): { re: number; im: number }[] {
  const tr = a + d;
  const det = a * d - b * c;
  const disc = tr * tr - 4 * det;

  if (disc >= 0) {
    const sqrtDisc = Math.sqrt(disc);
    return [
      { re: (tr + sqrtDisc) / 2, im: 0 },
      { re: (tr - sqrtDisc) / 2, im: 0 },
    ];
  } else {
    const sqrtDisc = Math.sqrt(-disc);
    return [
      { re: tr / 2, im: sqrtDisc / 2 },
      { re: tr / 2, im: -sqrtDisc / 2 },
    ];
  }
}
