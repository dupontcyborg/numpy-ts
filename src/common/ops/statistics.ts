/**
 * Statistics operations (histogram, bincount, correlate, cov, etc.)
 *
 * Pure functions for statistical analysis.
 * @module ops/statistics
 */

import { Complex } from '../complex';
import {
  type DType,
  getComplexComponentDType,
  hasFloat16,
  isBigIntDType,
  isComplexDType,
  isIntegerDType,
  promoteDTypes,
  type TypedArray,
  throwIfComplex,
} from '../dtype';
import { ArrayStorage } from '../storage';
import { wasmConvolve } from '../wasm/convolve';
import { wasmCorrelate } from '../wasm/correlate';
import { wasmMatmul } from '../wasm/matmul';

/**
 * Extract real parts from a complex array into a float array.
 * complex128 → float64, complex64 → float32.
 */
function extractRealPart(a: ArrayStorage): ArrayStorage {
  const shape = Array.from(a.shape);
  const size = a.size;
  const realDtype = getComplexComponentDType(a.dtype);
  const result = ArrayStorage.empty(shape, realDtype);
  const resultData = result.data as Float64Array | Float32Array;
  const srcData = a.data as Float64Array | Float32Array;
  const off = a.offset;
  for (let i = 0; i < size; i++) {
    resultData[i] = srcData[(off + i) * 2]!;
  }
  return result;
}

/**
 * Count number of occurrences of each value in array of non-negative ints.
 *
 * @param x - Input array (must contain non-negative integers)
 * @param weights - Optional weights, same shape as x
 * @param minlength - Minimum number of bins for output (default: 0)
 * @returns Array of bin counts
 */
export function bincount(
  x: ArrayStorage,
  weights?: ArrayStorage,
  minlength: number = 0,
): ArrayStorage {
  throwIfComplex(x.dtype, 'bincount', 'bincount requires integer input.');
  const xData = x.data;
  const size = x.size;

  // Find maximum value to determine output size
  let maxVal = 0;
  for (let i = 0; i < size; i++) {
    const val = Number(xData[i]);
    if (val < 0 || !Number.isInteger(val)) {
      throw new Error("'x' argument must contain non-negative integers");
    }
    if (val > maxVal) {
      maxVal = val;
    }
  }

  const outputSize = Math.max(maxVal + 1, minlength);

  if (weights !== undefined) {
    // Weighted bincount
    if (weights.size !== size) {
      throw new Error('weights array must have same length as x');
    }
    const weightsData = weights.data;
    const result = ArrayStorage.empty([outputSize], 'float64');
    const resultData = result.data as Float64Array;

    for (let i = 0; i < size; i++) {
      const idx = Number(xData[i]);
      resultData[idx]! += Number(weightsData[i]);
    }

    return result;
  } else {
    // Unweighted bincount
    const result = ArrayStorage.empty([outputSize], 'float64');
    const resultData = result.data as Float64Array;

    for (let i = 0; i < size; i++) {
      const idx = Number(xData[i]);
      resultData[idx]!++;
    }

    return result;
  }
}

/**
 * Return the indices of the bins to which each value in input array belongs.
 *
 * @param x - Input array to be binned
 * @param bins - Array of bins (monotonically increasing or decreasing)
 * @param right - If true, intervals are closed on the right (default: false)
 * @returns Array of bin indices
 */
export function digitize(
  x: ArrayStorage,
  bins: ArrayStorage,
  right: boolean = false,
): ArrayStorage {
  throwIfComplex(x.dtype, 'digitize', 'digitize requires real numbers.');
  throwIfComplex(bins.dtype, 'digitize', 'digitize requires real numbers.');
  const xData = x.data;
  const binsData = bins.data;
  const xSize = x.size;
  const numBins = bins.size;

  const result = ArrayStorage.empty([...x.shape], 'float64');
  const resultData = result.data as Float64Array;

  // Check if bins is monotonically increasing or decreasing
  let increasing = true;
  if (numBins > 1) {
    increasing = Number(binsData[1]) >= Number(binsData[0]);
  }

  for (let i = 0; i < xSize; i++) {
    const val = Number(xData[i]);
    let idx: number;

    if (increasing) {
      // For increasing bins:
      // right=False: bins[i-1] <= x < bins[i], returns i
      // right=True: bins[i-1] < x <= bins[i], returns i
      if (right) {
        // Find first index where bins[idx] >= val
        idx = lowerBound(binsData, numBins, val);
      } else {
        // Find first index where bins[idx] > val
        idx = upperBound(binsData, numBins, val);
      }
    } else {
      // For decreasing bins:
      // right=False: bins[i-1] > x >= bins[i], returns i
      // right=True: bins[i-1] >= x > bins[i], returns i
      // Linear search from beginning to find correct bin
      if (right) {
        // Find first index where bins[idx] < val
        idx = 0;
        while (idx < numBins && Number(binsData[idx]) >= val) {
          idx++;
        }
      } else {
        // Find first index where bins[idx] <= val (where x >= bins[idx])
        idx = 0;
        while (idx < numBins && Number(binsData[idx]) > val) {
          idx++;
        }
      }
    }

    resultData[i] = idx;
  }

  return result;
}

// Binary search helpers
function lowerBound(arr: TypedArray | number[], size: number, val: number): number {
  let lo = 0;
  let hi = size;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (Number(arr[mid]) < val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

function upperBound(arr: TypedArray | number[], size: number, val: number): number {
  let lo = 0;
  let hi = size;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (Number(arr[mid]) <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

type BinStrategy = 'auto' | 'fd' | 'doane' | 'scott' | 'stone' | 'rice' | 'sturges' | 'sqrt';

const BIN_STRATEGIES: readonly string[] = [
  'auto',
  'fd',
  'doane',
  'scott',
  'stone',
  'rice',
  'sturges',
  'sqrt',
];

/** Values in logical order as doubles, so offset and strided views read correctly. */
function realValues(a: ArrayStorage): Float64Array {
  const out = new Float64Array(a.size);
  const src = a.data as ArrayLike<number | bigint>;
  if (a.isCContiguous) {
    for (let i = 0; i < out.length; i++) out[i] = Number(src[a.offset + i]);
  } else {
    for (let i = 0; i < out.length; i++) out[i] = Number(a.iget(i));
  }
  return out;
}

function checkWeightShape(a: ArrayStorage, weights: ArrayStorage | undefined): void {
  if (!weights) return;
  const same =
    a.ndim === weights.ndim && a.shape.every((n: number, i: number) => n === weights.shape[i]);
  if (!same) throw new Error('weights should have the same shape as a.');
}

/** Rounding to the precision edges are computed in: identity for float64. */
function precisionRounder(dtype: DType): (x: number) => number {
  if (dtype === 'float32') return Math.fround;
  if (dtype === 'float16') return typeof Math.f16round === 'function' ? Math.f16round : Math.fround;
  return (x) => x;
}

/**
 * First and last edge for a set of values: the explicit range, else the data extent, with an
 * empty range widened by one half on each side. Non-finite or reversed ranges are rejected.
 */
function outerEdges(
  vals: Float64Array,
  range: readonly [number, number] | undefined,
  rnd: (x: number) => number,
): [number, number] {
  let first: number;
  let last: number;
  if (range) {
    [first, last] = range;
    if (first > last) throw new Error('max must be larger than min in range parameter.');
    if (!(Number.isFinite(first) && Number.isFinite(last))) {
      throw new Error(`supplied range of [${first}, ${last}] is not finite`);
    }
  } else if (vals.length === 0) {
    first = 0;
    last = 1;
  } else {
    first = Infinity;
    last = -Infinity;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i]!;
      if (Number.isNaN(v)) {
        first = NaN;
        last = NaN;
        break;
      }
      if (v < first) first = v;
      if (v > last) last = v;
    }
    if (!(Number.isFinite(first) && Number.isFinite(last))) {
      throw new Error(`autodetected range of [${first}, ${last}] is not finite`);
    }
  }
  if (first === last) {
    first = rnd(first - 0.5);
    last = rnd(last + 0.5);
  }
  return [first, last];
}

/**
 * n + 1 equally spaced edges from first to last, the last one exactly. With a rounder other
 * than the identity every step is rounded to that precision, as NumPy does when the endpoints
 * are scalars of a narrow float dtype.
 */
function linspaceEdges(
  first: number,
  last: number,
  n: number,
  rnd: (x: number) => number,
): number[] {
  const delta = rnd(last - first);
  const step = rnd(delta / n);
  const edges: number[] = new Array(n + 1);
  for (let i = 0; i < n; i++) {
    edges[i] = step === 0 ? rnd(rnd((i / n) * delta) + first) : rnd(rnd(i * step) + first);
  }
  edges[n] = last;
  return edges;
}

function edgeDtypeOf(a: DType): DType {
  return a === 'float32' || a === 'float16' ? a : 'float64';
}

function edgeStorageOf(edges: ArrayLike<number>, dtype: DType): ArrayStorage {
  const out = ArrayStorage.empty([edges.length], dtype);
  const dst = out.data as unknown as number[];
  for (let i = 0; i < edges.length; i++) dst[i] = edges[i]!;
  return out;
}

function checkIncreasing(edges: ArrayLike<number>, what: string): void {
  for (let i = 0; i + 1 < edges.length; i++) {
    if (edges[i]! > edges[i + 1]!) {
      throw new Error(`${what} must increase monotonically, when an array`);
    }
  }
}

function checkBinCount(bins: number): void {
  if (!Number.isInteger(bins)) {
    throw new TypeError(`'${typeof bins}' object cannot be interpreted as an integer`);
  }
  if (bins < 1) throw new Error('`bins` must be positive, when an integer');
}

function stdOf(x: Float64Array): number {
  const n = x.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += x[i]!;
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i]! - mean;
    sq += d * d;
  }
  return Math.sqrt(sq / n);
}

// Linear-method percentile of an ascending array, with the same lerp as NumPy.
function sortedPercentile(sorted: Float64Array, p: number): number {
  const n = sorted.length;
  const q = p / 100;
  const virtual = n * q + (1 + q * -1) - 1;
  const prev = Math.max(0, Math.min(n - 1, Math.floor(virtual)));
  const next = Math.min(n - 1, prev + 1);
  const gamma = virtual - Math.floor(virtual);
  const a = sorted[prev]!;
  const b = sorted[next]!;
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

// Count of x in each of nbins equal bins over [first, last], matching how NumPy places values
// that land on a computed edge.
function uniformCounts(x: Float64Array, first: number, last: number, nbins: number): Float64Array {
  const edges = linspaceEdges(first, last, nbins, (v) => v);
  const counts = new Float64Array(nbins);
  const norm = last - first;
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    let idx = Math.trunc(((v - first) / norm) * nbins);
    if (idx === nbins) idx -= 1;
    if (v < edges[idx]!) idx -= 1;
    else if (v >= edges[idx + 1]! && idx !== nbins - 1) idx += 1;
    counts[idx]! += 1;
  }
  return counts;
}

/**
 * Optimal bin width for a bin-count strategy over x, the values inside [first, last]. Zero means
 * no usable width and gives a single bin.
 */
function strategyWidth(
  method: BinStrategy,
  x: Float64Array,
  sorted: () => Float64Array,
  first: number,
  last: number,
): number {
  const n = x.length;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (x[i]! < min) min = x[i]!;
    if (x[i]! > max) max = x[i]!;
  }
  const ptp = max - min;
  const sturges = () => ptp / (Math.log2(n) + 1.0);
  const fd = () => {
    const s = sorted();
    const iqr = sortedPercentile(s, 75) - sortedPercentile(s, 25);
    return 2.0 * iqr * n ** (-1.0 / 3.0);
  };
  switch (method) {
    case 'sqrt':
      return ptp / Math.sqrt(n);
    case 'sturges':
      return sturges();
    case 'rice':
      return ptp / (2.0 * n ** (1.0 / 3));
    case 'scott':
      return ((24.0 * Math.PI ** 0.5) / n) ** (1.0 / 3.0) * stdOf(x);
    case 'fd':
      return fd();
    case 'auto':
      // The Freedman-Diaconis width is floored at half the sqrt width so a zero IQR or a tight
      // cluster cannot produce a huge bin count.
      return Math.min(Math.max(fd(), ptp / Math.sqrt(n) / 2), sturges());
    case 'doane': {
      if (n > 2) {
        const sg1 = Math.sqrt((6.0 * (n - 2)) / ((n + 1.0) * (n + 3)));
        const sigma = stdOf(x);
        if (sigma > 0.0) {
          let sum = 0;
          for (let i = 0; i < n; i++) sum += x[i]!;
          const mean = sum / n;
          let g = 0;
          for (let i = 0; i < n; i++) g += ((x[i]! - mean) / sigma) ** 3;
          const g1 = g / n;
          return ptp / (1.0 + Math.log2(n) + Math.log2(1.0 + Math.abs(g1) / sg1));
        }
      }
      return 0.0;
    }
    case 'stone': {
      if (n <= 1 || ptp === 0) return 0;
      const upper = Math.max(100, Math.trunc(Math.sqrt(n)));
      let best = 1;
      let bestScore = Infinity;
      for (let nbins = 1; nbins <= upper; nbins++) {
        const counts = uniformCounts(x, first, last, nbins);
        let dot = 0;
        for (let i = 0; i < nbins; i++) {
          const p = counts[i]! / n;
          dot += p * p;
        }
        const score = (2 - (n + 1) * dot) / (ptp / nbins);
        if (score < bestScore) {
          bestScore = score;
          best = nbins;
        }
      }
      return ptp / best;
    }
  }
}

interface ResolvedEdges {
  edges: Float64Array | number[];
  // The caller's own edge array when bins was an array, so its dtype survives.
  source?: ArrayStorage;
  dtype: DType;
}

/**
 * Bin edges for one set of values from an edge count, a strategy name or an explicit edge array,
 * following NumPy's histogram_bin_edges.
 */
function resolveBinEdges(
  a: ArrayStorage,
  vals: Float64Array,
  bins: number | BinStrategy | ArrayStorage,
  range: readonly [number, number] | undefined,
  weights: ArrayStorage | undefined,
): ResolvedEdges {
  checkWeightShape(a, weights);
  if (typeof bins !== 'number' && typeof bins !== 'string') {
    if (bins.ndim !== 1) throw new Error('`bins` must be 1d, when an array');
    const edges = realValues(bins);
    checkIncreasing(edges, '`bins`');
    return { edges, source: bins, dtype: bins.dtype };
  }

  const dtype = edgeDtypeOf(a.dtype);
  const rnd = precisionRounder(dtype);
  // Edges taken from the data are narrow-float scalars, so NumPy steps through them in that
  // precision. An explicit range is plain Python floats and is computed in float64.
  const stepRnd = range ? (x: number) => x : rnd;

  let nBins: number;
  let first: number;
  let last: number;
  if (typeof bins === 'string') {
    if (!BIN_STRATEGIES.includes(bins)) {
      throw new Error(`'${bins}' is not a valid estimator for \`bins\``);
    }
    if (weights) {
      throw new TypeError(
        'Automated estimation of the number of bins is not supported for weighted data',
      );
    }
    [first, last] = outerEdges(vals, range, rnd);
    let x = vals;
    if (range) x = vals.filter((v) => v >= first && v <= last);
    if (x.length === 0) {
      nBins = 1;
    } else {
      let cached: Float64Array | undefined;
      const sorted = () => {
        cached ??= Float64Array.from(x).sort();
        return cached;
      };
      let width = strategyWidth(bins, x, sorted, first, last);
      if (width) {
        if (isIntegerDType(a.dtype) && width < 1) width = 1;
        nBins = Math.ceil((last - first) / width);
      } else {
        nBins = 1;
      }
    }
  } else {
    checkBinCount(bins);
    nBins = bins;
    [first, last] = outerEdges(vals, range, rnd);
  }
  const edges = linspaceEdges(first, last, nBins, stepRnd);
  for (let i = 0; i + 1 < edges.length; i++) {
    if (edges[i]! >= edges[i + 1]!) {
      throw new Error(`Too many bins for data range. Cannot create ${nBins} finite-sized bins.`);
    }
  }
  return { edges, dtype };
}

function toEdgeStorage(r: ResolvedEdges): ArrayStorage {
  return r.source ? r.source.copy() : edgeStorageOf(r.edges, r.dtype);
}

/**
 * Edges for one dimension of histogram2d or histogramdd: an edge count over the data extent or
 * explicit edges. Unlike the 1-D histogram these never accept a strategy name, and NumPy builds
 * count-based edges with linspace, so an explicit range gives float64 and data-derived edges keep
 * a narrow float dtype.
 */
function resolveDimEdges(
  vals: Float64Array,
  dtype: DType,
  bins: number | ArrayStorage,
  range: readonly [number, number] | undefined,
  what: string,
): ResolvedEdges {
  if (typeof bins !== 'number') {
    if (bins.ndim !== 1) throw new Error(`${what} must be 1d, when an array`);
    const edges = realValues(bins);
    checkIncreasing(edges, what);
    return { edges, source: bins, dtype: bins.dtype };
  }
  checkBinCount(bins);
  const edgeDtype = range ? 'float64' : edgeDtypeOf(dtype);
  const rnd = precisionRounder(edgeDtype);
  const [first, last] = outerEdges(vals, range, rnd);
  return { edges: linspaceEdges(first, last, bins, rnd), dtype: edgeDtype };
}

/**
 * Compute the histogram of a set of data. Bin edges follow histogram_bin_edges, so bins may be a
 * count, a strategy name or an explicit edge array; counts are floats.
 *
 * @param a - Input data (flattened if not 1D)
 * @param bins - Number of bins (default: 10), a strategy name, or an array of bin edges
 * @param range - Lower and upper range of bins. If not provided, uses [a.min(), a.max()]
 * @param density - If true, return probability density function (default: false)
 * @param weights - Optional weights for each data point, same shape as a
 * @returns Tuple of [hist, bin_edges]
 */
export function histogram(
  a: ArrayStorage,
  bins: number | BinStrategy | ArrayStorage = 10,
  range?: [number, number],
  density: boolean = false,
  weights?: ArrayStorage,
): { hist: ArrayStorage; bin_edges: ArrayStorage } {
  // Complex input: use real part (matches NumPy behavior)
  if (isComplexDType(a.dtype)) {
    const realPart = extractRealPart(a);
    const result = histogram(realPart, bins, range, density, weights);
    realPart.dispose();
    return result;
  }
  if (typeof bins !== 'number' && typeof bins !== 'string' && isComplexDType(bins.dtype)) {
    const realBins = extractRealPart(bins);
    const result = histogram(a, realBins, range, density, weights);
    realBins.dispose();
    return result;
  }
  const vals = realValues(a);
  const weightVals = weights ? realValues(weights) : undefined;
  const resolved = resolveBinEdges(a, vals, bins, range, weights);
  const binEdges = resolved.edges;

  const numBins = Math.max(0, binEdges.length - 1);
  const histStorage = ArrayStorage.zeros([numBins], 'float64');
  const hist = histStorage.data as Float64Array;

  for (let i = 0; i < vals.length; i++) {
    const val = vals[i]!;
    const w = weightVals ? weightVals[i]! : 1;

    let binIdx = upperBound(binEdges, binEdges.length, val) - 1;
    if (binIdx < 0) continue;
    if (binIdx >= numBins) {
      // The last bin includes its right edge.
      if (val === binEdges[numBins]) {
        binIdx = numBins - 1;
      } else {
        continue;
      }
    }
    hist[binIdx]! += w;
  }

  if (density) {
    let totalWeight = 0;
    for (let i = 0; i < numBins; i++) {
      totalWeight += hist[i]!;
    }
    for (let i = 0; i < numBins; i++) {
      const binWidth = binEdges[i + 1]! - binEdges[i]!;
      hist[i]! = hist[i]! / (totalWeight * binWidth);
    }
  }

  return { hist: histStorage, bin_edges: toEdgeStorage(resolved) };
}

/**
 * Compute the bi-dimensional histogram of two data samples.
 *
 * @param x - Array of x coordinates
 * @param y - Array of y coordinates (must have same length as x)
 * @param bins - Number of bins or [nx, ny] or [x_edges, y_edges]
 * @param range - [[xmin, xmax], [ymin, ymax]]
 * @param density - If true, return probability density function
 * @param weights - Optional weights for each data point
 * @returns Object with hist, x_edges, y_edges
 */
export function histogram2d(
  x: ArrayStorage,
  y: ArrayStorage,
  bins: number | [number, number] | [ArrayStorage, ArrayStorage] = 10,
  range?: [[number, number], [number, number]],
  density: boolean = false,
  weights?: ArrayStorage,
): { hist: ArrayStorage; x_edges: ArrayStorage; y_edges: ArrayStorage } {
  throwIfComplex(x.dtype, 'histogram2d', 'histogram2d requires real numbers.');
  throwIfComplex(y.dtype, 'histogram2d', 'histogram2d requires real numbers.');
  const size = x.size;
  if (y.size !== size) {
    throw new Error('x and y must have the same length');
  }
  const xVals = realValues(x);
  const yVals = realValues(y);

  let nxBins: number | ArrayStorage;
  let nyBins: number | ArrayStorage;
  if (typeof bins === 'number') {
    nxBins = bins;
    nyBins = bins;
  } else if (Array.isArray(bins) && bins.length === 2) {
    [nxBins, nyBins] = bins as [number | ArrayStorage, number | ArrayStorage];
  } else {
    nxBins = 10;
    nyBins = 10;
  }

  const xr = resolveDimEdges(xVals, x.dtype, nxBins, range?.[0], '`bins[0]`');
  const yr = resolveDimEdges(yVals, y.dtype, nyBins, range?.[1], '`bins[1]`');
  const xEdges = xr.edges;
  const yEdges = yr.edges;

  const nx = Math.max(0, xEdges.length - 1);
  const ny = Math.max(0, yEdges.length - 1);
  const histStorage = ArrayStorage.zeros([nx, ny], 'float64');
  const hist = histStorage.data as Float64Array;
  const weightVals = weights ? realValues(weights) : undefined;

  for (let i = 0; i < size; i++) {
    const xVal = xVals[i]!;
    const yVal = yVals[i]!;
    const w = weightVals ? weightVals[i]! : 1;

    let xIdx = upperBound(xEdges, xEdges.length, xVal) - 1;
    let yIdx = upperBound(yEdges, yEdges.length, yVal) - 1;

    if (xIdx < 0 || xIdx >= nx) {
      if (xVal === xEdges[nx] && xIdx === nx) {
        xIdx = nx - 1;
      } else {
        continue;
      }
    }
    if (yIdx < 0 || yIdx >= ny) {
      if (yVal === yEdges[ny] && yIdx === ny) {
        yIdx = ny - 1;
      } else {
        continue;
      }
    }

    hist[xIdx * ny + yIdx]! += w;
  }

  if (density) {
    let totalWeight = 0;
    for (let i = 0; i < hist.length; i++) {
      totalWeight += hist[i]!;
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const area = (xEdges[i + 1]! - xEdges[i]!) * (yEdges[j + 1]! - yEdges[j]!);
        hist[i * ny + j]! = hist[i * ny + j]! / (totalWeight * area);
      }
    }
  }

  return { hist: histStorage, x_edges: toEdgeStorage(xr), y_edges: toEdgeStorage(yr) };
}

/**
 * Compute the multidimensional histogram of some data.
 *
 * @param sample - Array of shape (N, D) where N is number of samples and D is number of dimensions
 * @param bins - Number of bins for all axes, or array of bin counts per axis
 * @param range - Array of [min, max] for each dimension
 * @param density - If true, return probability density function
 * @param weights - Optional weights for each sample
 * @returns Object with hist and edges (array of edge arrays)
 */
export function histogramdd(
  sample: ArrayStorage,
  bins: number | number[] = 10,
  range?: [number, number][],
  density: boolean = false,
  weights?: ArrayStorage,
): { hist: ArrayStorage; edges: ArrayStorage[] } {
  throwIfComplex(sample.dtype, 'histogramdd', 'histogramdd requires real numbers.');
  const shape = sample.shape;

  let N: number, D: number;
  if (shape.length === 1) {
    N = shape[0]!;
    D = 1;
  } else if (shape.length === 2) {
    N = shape[0]!;
    D = shape[1]!;
  } else {
    throw new Error('sample must be 1D or 2D array');
  }
  const data = realValues(sample);

  let binCounts: number[];
  if (typeof bins === 'number') {
    binCounts = new Array(D).fill(bins);
  } else {
    binCounts = bins;
    if (binCounts.length !== D) {
      throw new Error('bins array length must match number of dimensions');
    }
  }

  const resolved: ResolvedEdges[] = [];
  for (let d = 0; d < D; d++) {
    const column = new Float64Array(N);
    for (let i = 0; i < N; i++) column[i] = data[i * D + d]!;
    resolved.push(resolveDimEdges(column, sample.dtype, binCounts[d]!, range?.[d], `bins[${d}]`));
  }
  const allEdges = resolved.map((r) => r.edges);

  const histShape = allEdges.map((e) => Math.max(0, e.length - 1));
  const histSize = histShape.reduce((a, b) => a * b, 1);
  const histStorage = ArrayStorage.zeros(histShape, 'float64');
  const hist = histStorage.data as Float64Array;

  const strides: number[] = new Array(D);
  strides[D - 1] = 1;
  for (let d = D - 2; d >= 0; d--) {
    strides[d] = strides[d + 1]! * histShape[d + 1]!;
  }

  const weightVals = weights ? realValues(weights) : undefined;

  for (let i = 0; i < N; i++) {
    const w = weightVals ? weightVals[i]! : 1;
    let histIdx = 0;
    let outOfBounds = false;

    for (let d = 0; d < D; d++) {
      const val = data[i * D + d]!;
      const edges = allEdges[d]!;
      const numBins = histShape[d]!;

      let binIdx = upperBound(edges, edges.length, val) - 1;

      if (binIdx < 0 || binIdx >= numBins) {
        if (val === edges[numBins] && binIdx === numBins) {
          binIdx = numBins - 1;
        } else {
          outOfBounds = true;
          break;
        }
      }

      histIdx += binIdx * strides[d]!;
    }

    if (!outOfBounds) {
      hist[histIdx]! += w;
    }
  }

  if (density) {
    let totalWeight = 0;
    for (let i = 0; i < histSize; i++) {
      totalWeight += hist[i]!;
    }

    for (let i = 0; i < histSize; i++) {
      let volume = 1;
      for (let d = 0; d < D; d++) {
        const binIdx = Math.floor(i / strides[d]!) % histShape[d]!;
        const edges = allEdges[d]!;
        volume *= edges[binIdx + 1]! - edges[binIdx]!;
      }
      hist[i]! = hist[i]! / (totalWeight * volume);
    }
  }

  return { hist: histStorage, edges: resolved.map(toEdgeStorage) };
}

/**
 * Cross-correlation of two 1-dimensional sequences.
 *
 * @param a - First input sequence
 * @param v - Second input sequence
 * @param mode - 'full', 'same', or 'valid' (default: 'full')
 * @returns Cross-correlation of a and v
 */
export function correlate(
  a: ArrayStorage,
  v: ArrayStorage,
  mode: 'full' | 'same' | 'valid' = 'valid',
): ArrayStorage {
  const aData = a.data;
  const vData = v.data;
  const aLen = a.size;
  const vLen = v.size;
  const isComplex = isComplexDType(a.dtype) || isComplexDType(v.dtype);

  // Compute full correlation first
  const fullLen = aLen + vLen - 1;

  if (isComplex) {
    // Complex cross-correlation: c[k] = sum_n(a[n] * conj(v[n-k]))
    const fullRe = new Float64Array(fullLen);
    const fullIm = new Float64Array(fullLen);
    const aComplex = isComplexDType(a.dtype);
    const vComplex = isComplexDType(v.dtype);

    for (let k = 0; k < fullLen; k++) {
      let sumRe = 0;
      let sumIm = 0;
      const offset = k - vLen + 1;

      for (let n = 0; n < aLen; n++) {
        const vIdx = n - offset;
        if (vIdx >= 0 && vIdx < vLen) {
          let aRe: number, aIm: number, vRe: number, vIm: number;

          if (aComplex) {
            aRe = (aData as Float64Array | Float32Array)[n * 2]!;
            aIm = (aData as Float64Array | Float32Array)[n * 2 + 1]!;
          } else {
            aRe = Number(aData[n]!);
            aIm = 0;
          }

          if (vComplex) {
            vRe = (vData as Float64Array | Float32Array)[vIdx * 2]!;
            vIm = (vData as Float64Array | Float32Array)[vIdx * 2 + 1]!;
          } else {
            vRe = Number(vData[vIdx]!);
            vIm = 0;
          }

          // a * conj(v) = (aRe + aIm*i) * (vRe - vIm*i)
          // = (aRe*vRe + aIm*vIm) + (aIm*vRe - aRe*vIm)*i
          sumRe += aRe * vRe + aIm * vIm;
          sumIm += aIm * vRe - aRe * vIm;
        }
      }

      fullRe[k] = sumRe;
      fullIm[k] = sumIm;
    }

    // Pack into interleaved format and return based on mode
    // Promote complex64+complex64 → complex64, mixed → complex128 (matches NumPy)
    const outComplexDtype = promoteDTypes(a.dtype as DType, v.dtype as DType);
    const packResult = (re: Float64Array, im: Float64Array, len: number, start: number = 0) => {
      const storage = ArrayStorage.empty([len], outComplexDtype);
      const result = storage.data as Float64Array | Float32Array;
      for (let i = 0; i < len; i++) {
        result[i * 2] = re[start + i]!;
        result[i * 2 + 1] = im[start + i]!;
      }
      return storage;
    };

    if (mode === 'full') {
      return packResult(fullRe, fullIm, fullLen);
    } else if (mode === 'same') {
      const start = Math.floor((fullLen - aLen) / 2);
      return packResult(fullRe, fullIm, aLen, start);
    } else {
      const validLen = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
      const start = Math.min(aLen, vLen) - 1;
      return packResult(fullRe, fullIm, validLen, start);
    }
  }

  // Try WASM fast path for real arrays
  const wasmFull = wasmCorrelate(a, v);
  if (wasmFull) {
    if (mode === 'full') return wasmFull;
    const wasmData = wasmFull.data;
    if (mode === 'same') {
      const start = Math.floor((fullLen - aLen) / 2);
      const sameResult = ArrayStorage.empty([aLen], wasmFull.dtype);
      const result = sameResult.data as Float64Array;
      for (let i = 0; i < aLen; i++) result[i] = wasmData[start + i] as number;
      return sameResult;
    } else {
      const validLen = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
      const start = Math.min(aLen, vLen) - 1;
      const validResult = ArrayStorage.empty([validLen], wasmFull.dtype);
      const result = validResult.data as Float64Array;
      for (let i = 0; i < validLen; i++) result[i] = wasmData[start + i] as number;
      return validResult;
    }
  }

  // Real arrays — JS fallback (preserve input dtype, matching NumPy)
  const outDtype = promoteDTypes(a.dtype as DType, v.dtype as DType);
  const fullStorage = ArrayStorage.empty([fullLen], outDtype);
  const fullResult = fullStorage.data;

  if (isBigIntDType(outDtype)) {
    const aBig = aData as BigInt64Array | BigUint64Array;
    const vBig = vData as BigInt64Array | BigUint64Array;
    const out = fullResult as BigInt64Array | BigUint64Array;
    for (let k = 0; k < fullLen; k++) {
      let sum = 0n;
      const offset = k - vLen + 1;
      for (let n = 0; n < aLen; n++) {
        const vIdx = n - offset;
        if (vIdx >= 0 && vIdx < vLen) {
          sum += aBig[n]! * vBig[vIdx]!;
        }
      }
      out[k] = sum;
    }
  } else {
    for (let k = 0; k < fullLen; k++) {
      let sum = 0;
      const offset = k - vLen + 1;
      for (let n = 0; n < aLen; n++) {
        const vIdx = n - offset;
        if (vIdx >= 0 && vIdx < vLen) {
          sum += Number(aData[n]) * Number(vData[vIdx]);
        }
      }
      fullResult[k] = sum;
    }
  }

  // Return based on mode
  if (mode === 'full') {
    return fullStorage;
  } else if (mode === 'same') {
    const start = Math.floor((fullLen - aLen) / 2);
    const sameStorage = ArrayStorage.empty([aLen], outDtype);
    const result = sameStorage.data;
    for (let i = 0; i < aLen; i++) {
      result[i] = fullResult[start + i]!;
    }
    fullStorage.dispose();
    return sameStorage;
  } else {
    const validLen = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
    const start = Math.min(aLen, vLen) - 1;
    const validStorage = ArrayStorage.empty([validLen], outDtype);
    const result = validStorage.data;
    for (let i = 0; i < validLen; i++) {
      result[i] = fullResult[start + i]!;
    }
    fullStorage.dispose();
    return validStorage;
  }
}

/**
 * Discrete, linear convolution of two one-dimensional sequences.
 *
 * @param a - First input sequence
 * @param v - Second input sequence
 * @param mode - 'full', 'same', or 'valid' (default: 'full')
 * @returns Convolution of a and v
 */
export function convolve(
  a: ArrayStorage,
  v: ArrayStorage,
  mode: 'full' | 'same' | 'valid' = 'full',
): ArrayStorage {
  const vLen = v.size;
  const aLen = a.size;
  const vComplex = isComplexDType(v.dtype);

  // Try WASM fast path for non-complex
  if (!vComplex && !isComplexDType(a.dtype)) {
    const wasmFull = wasmConvolve(a, v);
    if (wasmFull) {
      const fullLen = aLen + vLen - 1;
      if (mode === 'full') return wasmFull;
      const wasmData = wasmFull.data;
      // 'full' hands wasmFull straight back; the other two modes copy a window
      // out of it and must release it, or every call leaks the whole buffer.
      if (mode === 'same') {
        const start = Math.floor((fullLen - aLen) / 2);
        const sameResult = ArrayStorage.empty([aLen], wasmFull.dtype);
        const result = sameResult.data as Float64Array;
        for (let i = 0; i < aLen; i++) result[i] = wasmData[start + i] as number;
        wasmFull.dispose();
        return sameResult;
      } else {
        const validLen = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
        const start = Math.min(aLen, vLen) - 1;
        const validResult = ArrayStorage.empty([validLen], wasmFull.dtype);
        const result = validResult.data as Float64Array;
        for (let i = 0; i < validLen; i++) result[i] = wasmData[start + i] as number;
        wasmFull.dispose();
        return validResult;
      }
    }
  }

  // Convolution is correlation with v reversed (without conjugation)
  const vData = v.data;

  // Create reversed v
  let vReversedStorage: ArrayStorage;

  if (vComplex) {
    const isC64 = v.dtype === 'complex64';
    const vReversed = isC64 ? new Float32Array(vLen * 2) : new Float64Array(vLen * 2);
    for (let i = 0; i < vLen; i++) {
      const srcIdx = vLen - 1 - i;
      vReversed[i * 2] = (vData as Float64Array | Float32Array)[srcIdx * 2]!;
      vReversed[i * 2 + 1] = (vData as Float64Array | Float32Array)[srcIdx * 2 + 1]!;
    }
    vReversedStorage = ArrayStorage.fromData(vReversed, [vLen], v.dtype);
  } else {
    // Preserve dtype for reversed v so correlate output matches input dtype
    vReversedStorage = ArrayStorage.empty([vLen], v.dtype);
    const vRevData = vReversedStorage.data;
    for (let i = 0; i < vLen; i++) {
      vRevData[i] = vData[vLen - 1 - i]!;
    }
  }

  // For convolution, we need correlation without conjugation
  // So we manually conjugate vReversed to cancel the conjugation in correlate
  if (vComplex) {
    const data = vReversedStorage.data as Float64Array | Float32Array;
    for (let i = 0; i < vLen; i++) {
      data[i * 2 + 1] = -data[i * 2 + 1]!; // Negate imaginary to cancel conjugation
    }
  }

  // correlate builds its own output, so the reversed copy is a temporary and
  // leaks on every call that reaches this path if it is not released.
  const convolved = correlate(a, vReversedStorage, mode);
  vReversedStorage.dispose();
  return convolved;
}

/**
 * Estimate a covariance matrix.
 *
 * @param m - Input array (1D or 2D). Each row represents a variable, columns are observations.
 * @param y - Optional second array (for 2 variable case)
 * @param rowvar - If true, each row is a variable (default: true)
 * @param bias - If true, use N for normalization; if false, use N-1 (default: false)
 * @param ddof - Delta degrees of freedom (overrides bias if provided)
 * @returns Covariance matrix
 */
export function cov(
  m: ArrayStorage,
  y?: ArrayStorage,
  rowvar: boolean = true,
  bias: boolean = false,
  ddof?: number,
): ArrayStorage {
  const mShape = m.shape;
  // Bulk-convert float16 to float64 upfront to avoid per-element f16→f64 overhead
  const mData =
    m.dtype === 'float16'
      ? (() => {
          const f = new Float64Array(m.size);
          f.set(m.data.subarray(m.offset, m.offset + m.size) as unknown as ArrayLike<number>);
          return f;
        })()
      : m.data;
  const isComplex = isComplexDType(m.dtype) || (y !== undefined && isComplexDType(y.dtype));

  // Determine effective ddof
  let effectiveDdof: number;
  if (ddof !== undefined) {
    effectiveDdof = ddof;
  } else {
    effectiveDdof = bias ? 0 : 1;
  }

  // Handle 1D case
  if (mShape.length === 1) {
    if (y !== undefined) {
      // Two 1D arrays - compute 2x2 covariance matrix
      const yData = y.data;
      const n = m.size;
      const mIsComplex = isComplexDType(m.dtype);
      const yIsComplex = isComplexDType(y.dtype);

      if (y.size !== n) {
        throw new Error('m and y must have same length');
      }

      if (isComplex) {
        // Complex 1D case with y
        // Compute complex means
        let mMeanRe = 0,
          mMeanIm = 0,
          yMeanRe = 0,
          yMeanIm = 0;
        for (let i = 0; i < n; i++) {
          if (mIsComplex) {
            mMeanRe += (mData as Float64Array)[i * 2]!;
            mMeanIm += (mData as Float64Array)[i * 2 + 1]!;
          } else {
            mMeanRe += Number(mData[i]);
          }
          if (yIsComplex) {
            yMeanRe += (yData as Float64Array)[i * 2]!;
            yMeanIm += (yData as Float64Array)[i * 2 + 1]!;
          } else {
            yMeanRe += Number(yData[i]);
          }
        }
        mMeanRe /= n;
        mMeanIm /= n;
        yMeanRe /= n;
        yMeanIm /= n;

        // Compute covariances: cov[i,j] = E[(Xi - μi) * conj(Xj - μj)]
        let varMRe = 0,
          varMIm = 0;
        let varYRe = 0,
          varYIm = 0;
        let covMYRe = 0,
          covMYIm = 0;
        let covYMRe = 0,
          covYMIm = 0;

        for (let i = 0; i < n; i++) {
          let dmRe: number, dmIm: number, dyRe: number, dyIm: number;
          if (mIsComplex) {
            dmRe = (mData as Float64Array)[i * 2]! - mMeanRe;
            dmIm = (mData as Float64Array)[i * 2 + 1]! - mMeanIm;
          } else {
            dmRe = Number(mData[i]) - mMeanRe;
            dmIm = 0;
          }
          if (yIsComplex) {
            dyRe = (yData as Float64Array)[i * 2]! - yMeanRe;
            dyIm = (yData as Float64Array)[i * 2 + 1]! - yMeanIm;
          } else {
            dyRe = Number(yData[i]) - yMeanRe;
            dyIm = 0;
          }

          // dm * conj(dm) = |dm|^2 (real)
          varMRe += dmRe * dmRe + dmIm * dmIm;
          // dy * conj(dy) = |dy|^2 (real)
          varYRe += dyRe * dyRe + dyIm * dyIm;
          // dm * conj(dy) = (dmRe + dmIm*i) * (dyRe - dyIm*i)
          covMYRe += dmRe * dyRe + dmIm * dyIm;
          covMYIm += dmIm * dyRe - dmRe * dyIm;
          // dy * conj(dm) = (dyRe + dyIm*i) * (dmRe - dmIm*i)
          covYMRe += dyRe * dmRe + dyIm * dmIm;
          covYMIm += dyIm * dmRe - dyRe * dmIm;
        }

        const divisor = n - effectiveDdof;
        if (divisor <= 0) {
          const nanStorage = ArrayStorage.empty([2, 2], 'complex128');
          (nanStorage.data as Float64Array).fill(NaN);
          return nanStorage;
        }

        varMRe /= divisor;
        varYRe /= divisor;
        covMYRe /= divisor;
        covMYIm /= divisor;
        covYMRe /= divisor;
        covYMIm /= divisor;

        // Pack into interleaved format: [[varM, covMY], [covYM, varY]]
        const resultStorage = ArrayStorage.empty([2, 2], 'complex128');
        const result = resultStorage.data as Float64Array;
        result[0] = varMRe;
        result[1] = varMIm; // [0,0]
        result[2] = covMYRe;
        result[3] = covMYIm; // [0,1]
        result[4] = covYMRe;
        result[5] = covYMIm; // [1,0]
        result[6] = varYRe;
        result[7] = varYIm; // [1,1]
        return resultStorage;
      }

      // Real 1D case with y
      // Compute means
      let mMean = 0,
        yMean = 0;
      for (let i = 0; i < n; i++) {
        mMean += Number(mData[i]);
        yMean += Number(yData[i]);
      }
      mMean /= n;
      yMean /= n;

      // Compute covariances
      let varM = 0,
        varY = 0,
        covMY = 0;
      for (let i = 0; i < n; i++) {
        const dm = Number(mData[i]) - mMean;
        const dy = Number(yData[i]) - yMean;
        varM += dm * dm;
        varY += dy * dy;
        covMY += dm * dy;
      }

      const divisor = n - effectiveDdof;
      if (divisor <= 0) {
        return ArrayStorage.fromData(new Float64Array([NaN, NaN, NaN, NaN]), [2, 2], 'float64');
      }

      varM /= divisor;
      varY /= divisor;
      covMY /= divisor;

      return ArrayStorage.fromData(new Float64Array([varM, covMY, covMY, varY]), [2, 2], 'float64');
    } else {
      // Single 1D array - compute scalar variance (0-d array like NumPy)
      const n = m.size;

      if (isComplex) {
        // Complex variance: E[|X - μ|^2]
        let meanRe = 0,
          meanIm = 0;
        for (let i = 0; i < n; i++) {
          meanRe += (mData as Float64Array)[i * 2]!;
          meanIm += (mData as Float64Array)[i * 2 + 1]!;
        }
        meanRe /= n;
        meanIm /= n;

        let variance = 0;
        for (let i = 0; i < n; i++) {
          const dRe = (mData as Float64Array)[i * 2]! - meanRe;
          const dIm = (mData as Float64Array)[i * 2 + 1]! - meanIm;
          variance += dRe * dRe + dIm * dIm;
        }

        const divisor = n - effectiveDdof;
        if (divisor <= 0) {
          return ArrayStorage.fromData(new Float64Array([NaN, 0]), [], 'complex128');
        }

        variance /= divisor;
        return ArrayStorage.fromData(new Float64Array([variance, 0]), [], 'complex128');
      }

      // Real 1D case
      let mean = 0;
      for (let i = 0; i < n; i++) {
        mean += Number(mData[i]);
      }
      mean /= n;

      let variance = 0;
      for (let i = 0; i < n; i++) {
        const d = Number(mData[i]) - mean;
        variance += d * d;
      }

      const divisor = n - effectiveDdof;
      if (divisor <= 0) {
        return ArrayStorage.fromData(new Float64Array([NaN]), [], 'float64');
      }

      variance /= divisor;

      return ArrayStorage.fromData(new Float64Array([variance]), [], 'float64');
    }
  }

  // 2D case
  let numVars: number;
  let numObs: number;

  if (rowvar) {
    numVars = mShape[0]!;
    numObs = mShape[1]!;
  } else {
    numVars = mShape[1]!;
    numObs = mShape[0]!;
  }

  const divisor = numObs - effectiveDdof;

  if (isComplex) {
    // Complex 2D case
    // Compute complex means for each variable
    const meansRe = new Float64Array(numVars);
    const meansIm = new Float64Array(numVars);
    for (let i = 0; i < numVars; i++) {
      let sumRe = 0,
        sumIm = 0;
      for (let j = 0; j < numObs; j++) {
        const idx = rowvar ? i * numObs + j : j * numVars + i;
        sumRe += (mData as Float64Array)[idx * 2]!;
        sumIm += (mData as Float64Array)[idx * 2 + 1]!;
      }
      meansRe[i] = sumRe / numObs;
      meansIm[i] = sumIm / numObs;
    }

    // Compute complex covariance matrix (Hermitian: cov[i,j] = conj(cov[j,i]))
    const covStorage = ArrayStorage.empty([numVars, numVars], 'complex128');
    const covMatrix = covStorage.data as Float64Array;

    if (divisor <= 0) {
      covMatrix.fill(NaN);
      return covStorage;
    }

    for (let i = 0; i < numVars; i++) {
      for (let j = 0; j < numVars; j++) {
        let sumRe = 0,
          sumIm = 0;
        for (let k = 0; k < numObs; k++) {
          const idxI = rowvar ? i * numObs + k : k * numVars + i;
          const idxJ = rowvar ? j * numObs + k : k * numVars + j;
          const diRe = (mData as Float64Array)[idxI * 2]! - meansRe[i]!;
          const diIm = (mData as Float64Array)[idxI * 2 + 1]! - meansIm[i]!;
          const djRe = (mData as Float64Array)[idxJ * 2]! - meansRe[j]!;
          const djIm = (mData as Float64Array)[idxJ * 2 + 1]! - meansIm[j]!;
          // di * conj(dj) = (diRe + diIm*i) * (djRe - djIm*i)
          // = (diRe*djRe + diIm*djIm) + (diIm*djRe - diRe*djIm)*i
          sumRe += diRe * djRe + diIm * djIm;
          sumIm += diIm * djRe - diRe * djIm;
        }
        const outIdx = (i * numVars + j) * 2;
        covMatrix[outIdx] = sumRe / divisor;
        covMatrix[outIdx + 1] = sumIm / divisor;
      }
    }

    return covStorage;
  }

  // Real 2D case — center data, then cov = centered @ centered^T / divisor.
  // Always use float64 for intermediate computation to match NumPy's precision.

  // Step 1: Build centered matrix in WASM [numVars × numObs]
  const centeredStorage = ArrayStorage.empty([numVars, numObs], 'float64');
  const centered = centeredStorage.data as Float64Array;
  for (let i = 0; i < numVars; i++) {
    let sum = 0;
    for (let j = 0; j < numObs; j++) {
      const idx = rowvar ? i * numObs + j : j * numVars + i;
      sum += Number(mData[idx]);
    }
    const mean = sum / numObs;
    const row = i * numObs;
    for (let j = 0; j < numObs; j++) {
      const idx = rowvar ? i * numObs + j : j * numVars + i;
      centered[row + j] = Number(mData[idx]) - mean;
    }
  }

  if (divisor <= 0) {
    centeredStorage.dispose();
    const covStorage = ArrayStorage.empty([numVars, numVars], 'float64');
    (covStorage.data as Float64Array).fill(NaN);
    return covStorage;
  }

  // Step 2: cov = centered @ centeredT / divisor — use WASM matmul
  // Create explicit transpose (contiguous) for WASM matmul
  const centeredTStorage = ArrayStorage.empty([numObs, numVars], 'float64');
  const centeredT = centeredTStorage.data as Float64Array;
  for (let i = 0; i < numVars; i++) {
    for (let j = 0; j < numObs; j++) {
      centeredT[j * numVars + i] = centered[i * numObs + j]!;
    }
  }

  const product = wasmMatmul(centeredStorage, centeredTStorage);
  centeredStorage.dispose();
  centeredTStorage.dispose();

  if (product) {
    // Scale by 1/divisor in-place
    const covMatrix = product.data as Float64Array;
    const invDiv = 1.0 / divisor;
    for (let i = 0; i < numVars * numVars; i++) covMatrix[i] = covMatrix[i]! * invDiv;
    return product;
  }

  // JS fallback (shouldn't happen for float64 contiguous arrays)
  const covStorage = ArrayStorage.empty([numVars, numVars], 'float64');
  const covMatrix = covStorage.data as Float64Array;
  const invDiv = 1.0 / divisor;
  for (let i = 0; i < numVars; i++) {
    const rowI = i * numObs;
    for (let j = i; j < numVars; j++) {
      const rowJ = j * numObs;
      let sum = 0;
      for (let k = 0; k < numObs; k++) {
        sum += centered[rowI + k]! * centered[rowJ + k]!;
      }
      const covVal = sum * invDiv;
      covMatrix[i * numVars + j] = covVal;
      covMatrix[j * numVars + i] = covVal;
    }
  }
  return covStorage;
}

/**
 * Return Pearson product-moment correlation coefficients.
 *
 * @param x - Input array (1D or 2D)
 * @param y - Optional second array (for 2 variable case)
 * @param rowvar - If true, each row is a variable (default: true)
 * @returns Correlation coefficient matrix
 */
export function corrcoef(x: ArrayStorage, y?: ArrayStorage, rowvar: boolean = true): ArrayStorage {
  const isComplex = isComplexDType(x.dtype) || (y !== undefined && isComplexDType(y.dtype));

  // Handle 1D array without y - return scalar 1.0 (0-d array like NumPy)
  if (x.shape.length === 1 && y === undefined) {
    if (isComplex) {
      return ArrayStorage.fromData(new Float64Array([1, 0]), [], 'complex128');
    }
    return ArrayStorage.fromData(new Float64Array([1]), [], 'float64');
  }

  // Compute covariance matrix
  const covMatrix = cov(x, y, rowvar, false);

  try {
    const covData = covMatrix.data;
    const shape = covMatrix.shape;
    const n = shape[0]!;

    if (isComplex) {
      // Complex correlation coefficients
      // corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
      // Diagonal elements (variances) are real, so sqrt(var_i * var_j) is real
      const corrStorage = ArrayStorage.empty([n, n], 'complex128');
      const corrData = corrStorage.data as Float64Array;

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          // Get cov[i,j] (complex)
          const covIJRe = (covData as Float64Array)[(i * n + j) * 2]!;
          const covIJIm = (covData as Float64Array)[(i * n + j) * 2 + 1]!;
          // Variances are real (imaginary part is 0 for diagonal elements)
          const varI = (covData as Float64Array)[(i * n + i) * 2]!;
          const varJ = (covData as Float64Array)[(j * n + j) * 2]!;

          const outIdx = (i * n + j) * 2;
          if (varI <= 0 || varJ <= 0) {
            corrData[outIdx] = NaN;
            corrData[outIdx + 1] = NaN;
          } else {
            const divisor = Math.sqrt(varI * varJ);
            corrData[outIdx] = covIJRe / divisor;
            corrData[outIdx + 1] = covIJIm / divisor;
          }
        }
      }

      return corrStorage;
    }

    // Real correlation coefficients
    // corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
    const corrStorage = ArrayStorage.empty([n, n], 'float64');
    const corrData = corrStorage.data as Float64Array;
    const cData = covData as Float64Array;

    // Precompute 1/sqrt(variance) for each variable
    const invStd = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = cData[i * n + i]!;
      invStd[i] = v > 0 ? 1.0 / Math.sqrt(v) : NaN;
    }

    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const val = cData[i * n + j]! * invStd[i]! * invStd[j]!;
        corrData[i * n + j] = val;
        corrData[j * n + i] = val;
      }
    }

    return corrStorage;
  } finally {
    covMatrix.dispose();
  }
}

/**
 * Compute the edges of the bins for a histogram, without computing the histogram itself.
 * Strategy names estimate the bin count from the data and cannot be combined with weights; an
 * explicit edge array is returned as a copy with its own dtype.
 *
 * @param a - Input data (flattened if not 1D)
 * @param bins - Number of bins (default: 10), a strategy name, or an array of bin edges
 * @param range - Lower and upper range of bins. If not provided, uses [a.min(), a.max()]
 * @param weights - Optional weights, same shape as a; only validated
 * @returns Array of bin edges (length = bins + 1)
 */
export function histogram_bin_edges(
  a: ArrayStorage,
  bins: number | BinStrategy | ArrayStorage = 10,
  range?: [number, number],
  weights?: ArrayStorage,
): ArrayStorage {
  throwIfComplex(a.dtype, 'histogram_bin_edges', 'histogram_bin_edges requires real numbers.');
  return toEdgeStorage(resolveBinEdges(a, realValues(a), bins, range, weights));
}

/**
 * Trapezoidal integration for complex arrays.
 * Integrates complex values, producing a complex result.
 */
function trapezoidComplex(
  y: ArrayStorage,
  x: ArrayStorage | undefined,
  dx: number,
  axis: number,
): ArrayStorage | Complex {
  const yShape = Array.from(y.shape);
  const ndim = yShape.length;

  if (axis < 0) axis = ndim + axis;
  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = yShape[axis]!;
  if (axisSize < 2) {
    throw new Error('trapezoid requires at least 2 samples along axis');
  }

  let xValues: Float64Array;
  if (x !== undefined) {
    if (isComplexDType(x.dtype)) {
      throw new Error('trapezoid requires real x values.');
    }
    if (x.size !== axisSize) {
      throw new Error(`x array size (${x.size}) must match y axis size (${axisSize})`);
    }
    const xData = x.data;
    xValues = new Float64Array(axisSize);
    for (let i = 0; i < axisSize; i++) {
      xValues[i] = Number(xData[i]);
    }
  } else {
    xValues = new Float64Array(axisSize);
    for (let i = 0; i < axisSize; i++) {
      xValues[i] = i * dx;
    }
  }

  const yData = y.data as Float64Array | Float32Array;
  const yOff = y.offset;

  // 1D case
  if (ndim === 1) {
    let integralRe = 0;
    let integralIm = 0;
    for (let i = 0; i < axisSize - 1; i++) {
      const y0Re = yData[(yOff + i) * 2]!;
      const y0Im = yData[(yOff + i) * 2 + 1]!;
      const y1Re = yData[(yOff + i + 1) * 2]!;
      const y1Im = yData[(yOff + i + 1) * 2 + 1]!;
      const dx_i = xValues[i + 1]! - xValues[i]!;
      integralRe += 0.5 * (y0Re + y1Re) * dx_i;
      integralIm += 0.5 * (y0Im + y1Im) * dx_i;
    }
    return new Complex(integralRe, integralIm);
  }

  // N-D case
  const outShape = [...yShape];
  outShape.splice(axis, 1);
  const outSize = outShape.reduce((a, b) => a * b, 1);

  const resultStorage = ArrayStorage.empty(outShape, y.dtype);
  const resultData = resultStorage.data as Float64Array | Float32Array;

  const yStrides: number[] = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    yStrides[i] = stride;
    stride *= yShape[i]!;
  }

  const outStrides: number[] = new Array(outShape.length);
  stride = 1;
  for (let i = outShape.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= outShape[i]!;
  }

  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    const outCoords: number[] = [];
    let remaining = outIdx;
    for (let d = 0; d < outShape.length; d++) {
      const coord = Math.floor(remaining / outStrides[d]!);
      remaining %= outStrides[d]!;
      outCoords.push(coord);
    }

    const inCoordsBase: number[] = [];
    let outD = 0;
    for (let d = 0; d < ndim; d++) {
      if (d === axis) {
        inCoordsBase.push(0);
      } else {
        inCoordsBase.push(outCoords[outD]!);
        outD++;
      }
    }

    let integralRe = 0;
    let integralIm = 0;
    for (let i = 0; i < axisSize - 1; i++) {
      inCoordsBase[axis] = i;
      let idx0 = 0;
      for (let d = 0; d < ndim; d++) {
        idx0 += inCoordsBase[d]! * yStrides[d]!;
      }
      inCoordsBase[axis] = i + 1;
      let idx1 = 0;
      for (let d = 0; d < ndim; d++) {
        idx1 += inCoordsBase[d]! * yStrides[d]!;
      }
      const dx_i = xValues[i + 1]! - xValues[i]!;
      integralRe +=
        0.5 * (Number(yData[(yOff + idx0) * 2]!) + Number(yData[(yOff + idx1) * 2]!)) * dx_i;
      integralIm +=
        0.5 *
        (Number(yData[(yOff + idx0) * 2 + 1]!) + Number(yData[(yOff + idx1) * 2 + 1]!)) *
        dx_i;
    }
    resultData[outIdx * 2] = integralRe;
    resultData[outIdx * 2 + 1] = integralIm;
  }

  if (outShape.length === 0) {
    const re = Number(resultData[0]!);
    const im = Number(resultData[1]!);
    resultStorage.dispose();
    return new Complex(re, im);
  }

  return resultStorage;
}

/**
 * Integrate along the given axis using the composite trapezoidal rule.
 *
 * @param y - Input array to integrate
 * @param x - Optional sample points corresponding to y values. If not provided, spacing is assumed to be 1.
 * @param dx - Spacing between sample points when x is not given (default: 1.0)
 * @param axis - The axis along which to integrate (default: -1, meaning last axis)
 * @returns Definite integral approximated using the composite trapezoidal rule
 */
export function trapezoid(
  y: ArrayStorage,
  x?: ArrayStorage,
  dx: number = 1.0,
  axis: number = -1,
): ArrayStorage | number | Complex {
  // Complex y: integrate complex values using trapezoidal rule
  if (isComplexDType(y.dtype)) {
    return trapezoidComplex(y, x, dx, axis);
  }
  if (x !== undefined) {
    throwIfComplex(x.dtype, 'trapezoid', 'trapezoid requires real x values.');
  }

  const yShape = Array.from(y.shape);
  const ndim = yShape.length;

  // Handle negative axis
  if (axis < 0) {
    axis = ndim + axis;
  }

  if (axis < 0 || axis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = yShape[axis]!;

  if (axisSize < 2) {
    throw new Error('trapezoid requires at least 2 samples along axis');
  }

  // Get sample points
  let xValues: Float64Array;
  if (x !== undefined) {
    if (x.size !== axisSize) {
      throw new Error(`x array size (${x.size}) must match y axis size (${axisSize})`);
    }
    const xData = x.data;
    xValues = new Float64Array(axisSize);
    for (let i = 0; i < axisSize; i++) {
      xValues[i] = Number(xData[i]);
    }
  } else {
    // Create evenly spaced x values
    xValues = new Float64Array(axisSize);
    for (let i = 0; i < axisSize; i++) {
      xValues[i] = i * dx;
    }
  }

  // Compute output shape (remove the integration axis)
  const outShape = [...yShape];
  outShape.splice(axis, 1);

  // Handle 1D case
  if (ndim === 1) {
    const yData = y.data;
    const yDtype = y.dtype;
    if (yDtype === 'float16' && hasFloat16) {
      const f16 = new Float16Array(1);
      f16[0] = 0;
      for (let i = 0; i < axisSize - 1; i++) {
        const y0 = Number(yData[i]);
        const y1 = Number(yData[i + 1]);
        const dx_i = xValues[i + 1]! - xValues[i]!;
        f16[0] += 0.5 * (y0 + y1) * dx_i;
      }
      return Number(f16[0]!);
    }
    if (yDtype === 'float32') {
      const f32 = new Float32Array(1);
      f32[0] = 0;
      for (let i = 0; i < axisSize - 1; i++) {
        const y0 = Number(yData[i]);
        const y1 = Number(yData[i + 1]);
        const dx_i = xValues[i + 1]! - xValues[i]!;
        f32[0] += 0.5 * (y0 + y1) * dx_i;
      }
      return f32[0]!;
    }
    let integral = 0;
    for (let i = 0; i < axisSize - 1; i++) {
      const y0 = Number(yData[i]);
      const y1 = Number(yData[i + 1]);
      const dx_i = xValues[i + 1]! - xValues[i]!;
      integral += 0.5 * (y0 + y1) * dx_i;
    }
    return integral;
  }

  // For N-D arrays
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const yDtypeND = y.dtype;
  const useF16ND = yDtypeND === 'float16' && hasFloat16;
  const useF32ND = yDtypeND === 'float32';
  const outDtype = useF16ND
    ? ('float16' as const)
    : useF32ND
      ? ('float32' as const)
      : ('float64' as const);
  const resultStorage = ArrayStorage.empty(outShape, outDtype);
  const resultData = resultStorage.data;

  // Compute strides for the input array
  const yStrides: number[] = new Array(ndim);
  let stride = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    yStrides[i] = stride;
    stride *= yShape[i]!;
  }

  // Compute strides for the output array
  const outStrides: number[] = new Array(outShape.length);
  stride = 1;
  for (let i = outShape.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= outShape[i]!;
  }

  const yData = y.data;

  // Iterate over all output elements
  for (let outIdx = 0; outIdx < outSize; outIdx++) {
    // Compute coordinates in output array
    const outCoords: number[] = [];
    let remaining = outIdx;
    for (let d = 0; d < outShape.length; d++) {
      const coord = Math.floor(remaining / outStrides[d]!);
      remaining %= outStrides[d]!;
      outCoords.push(coord);
    }

    // Map to input coordinates (insert axis dimension)
    const inCoordsBase: number[] = [];
    let outD = 0;
    for (let d = 0; d < ndim; d++) {
      if (d === axis) {
        inCoordsBase.push(0); // Placeholder for axis
      } else {
        inCoordsBase.push(outCoords[outD]!);
        outD++;
      }
    }

    // Compute integral along axis
    if (useF16ND) {
      const f16 = new Float16Array(1);
      f16[0] = 0;
      for (let i = 0; i < axisSize - 1; i++) {
        inCoordsBase[axis] = i;
        let idx0 = 0;
        for (let d = 0; d < ndim; d++) {
          idx0 += inCoordsBase[d]! * yStrides[d]!;
        }
        inCoordsBase[axis] = i + 1;
        let idx1 = 0;
        for (let d = 0; d < ndim; d++) {
          idx1 += inCoordsBase[d]! * yStrides[d]!;
        }
        f16[0] +=
          0.5 * (Number(yData[idx0]) + Number(yData[idx1])) * (xValues[i + 1]! - xValues[i]!);
      }
      resultData[outIdx] = f16[0]!;
    } else if (useF32ND) {
      const f32 = new Float32Array(1);
      f32[0] = 0;
      for (let i = 0; i < axisSize - 1; i++) {
        inCoordsBase[axis] = i;
        let idx0 = 0;
        for (let d = 0; d < ndim; d++) {
          idx0 += inCoordsBase[d]! * yStrides[d]!;
        }
        inCoordsBase[axis] = i + 1;
        let idx1 = 0;
        for (let d = 0; d < ndim; d++) {
          idx1 += inCoordsBase[d]! * yStrides[d]!;
        }
        f32[0] +=
          0.5 * (Number(yData[idx0]) + Number(yData[idx1])) * (xValues[i + 1]! - xValues[i]!);
      }
      resultData[outIdx] = f32[0]!;
    } else {
      let integral = 0;
      for (let i = 0; i < axisSize - 1; i++) {
        inCoordsBase[axis] = i;
        let idx0 = 0;
        for (let d = 0; d < ndim; d++) {
          idx0 += inCoordsBase[d]! * yStrides[d]!;
        }
        inCoordsBase[axis] = i + 1;
        let idx1 = 0;
        for (let d = 0; d < ndim; d++) {
          idx1 += inCoordsBase[d]! * yStrides[d]!;
        }
        const y0 = Number(yData[idx0]);
        const y1 = Number(yData[idx1]);
        const dx_i = xValues[i + 1]! - xValues[i]!;
        integral += 0.5 * (y0 + y1) * dx_i;
      }
      resultData[outIdx] = integral;
    }
  }

  if (outShape.length === 0) {
    resultStorage.dispose();
    return Number(resultData[0]!);
  }

  return resultStorage;
}
