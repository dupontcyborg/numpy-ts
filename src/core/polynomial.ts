/**
 * Polynomial functions
 *
 * Tree-shakeable standalone functions for polynomial operations.
 */

import { NDArrayCore } from '../common/ndarray-core';
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
 * Find the roots of a polynomial
 */
export function roots(p: NDArrayCore | number[]): NDArrayCore {
  const poly = toArray(p);
  const coeffs = [...(poly.data as unknown as number[])];

  // Remove leading zeros
  while (coeffs.length > 1 && coeffs[0] === 0) {
    coeffs.shift();
  }

  const n = coeffs.length - 1;

  if (n === 0) {
    return array([]);
  }

  if (n === 1) {
    return array([-coeffs[1]! / coeffs[0]!]);
  }

  if (n === 2) {
    // Quadratic formula
    const a = coeffs[0]!;
    const b = coeffs[1]!;
    const c = coeffs[2]!;
    const discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
      const sqrtD = Math.sqrt(discriminant);
      return array([(-b + sqrtD) / (2 * a), (-b - sqrtD) / (2 * a)]);
    } else {
      // Complex roots - return real parts for now
      // Full complex support would need complex array
      const realPart = -b / (2 * a);
      return array([realPart, realPart]);
    }
  }

  // For higher degree polynomials, use companion matrix eigenvalues
  // Simplified: use Newton-Raphson for real roots
  // This is a basic implementation - full implementation would use eigenvalue decomposition

  const realRoots: number[] = [];

  // Try to find roots using Newton-Raphson from multiple starting points
  const startPoints = [];
  for (let i = -10; i <= 10; i += 0.5) {
    startPoints.push(i);
  }

  for (const start of startPoints) {
    let x = start;
    for (let iter = 0; iter < 100; iter++) {
      // Evaluate polynomial and derivative at x
      let val = coeffs[0]!;
      let deriv = 0;
      for (let i = 1; i < coeffs.length; i++) {
        deriv = deriv * x + val;
        val = val * x + coeffs[i]!;
      }

      if (Math.abs(val) < 1e-10) {
        // Found a root
        // Check if we already have this root
        const isDuplicate = realRoots.some((r) => Math.abs(r - x) < 1e-6);
        if (!isDuplicate) {
          realRoots.push(x);
        }
        break;
      }

      if (Math.abs(deriv) < 1e-15) break;

      const newX = x - val / deriv;
      if (Math.abs(newX - x) < 1e-12) {
        if (Math.abs(val) < 1e-8) {
          const isDuplicate = realRoots.some((r) => Math.abs(r - x) < 1e-6);
          if (!isDuplicate) {
            realRoots.push(x);
          }
        }
        break;
      }
      x = newX;
    }

    if (realRoots.length >= n) break;
  }

  return array(realRoots.sort((a, b) => b - a));
}
