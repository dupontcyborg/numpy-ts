/**
 * Statistics operations (histogram, bincount, correlate, cov, etc.)
 *
 * Pure functions for statistical analysis.
 * @module ops/statistics
 */

import { ArrayStorage } from '../core/storage';

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
  minlength: number = 0
): ArrayStorage {
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
    const resultData = new Float64Array(outputSize);

    for (let i = 0; i < size; i++) {
      const idx = Number(xData[i]);
      resultData[idx]! += Number(weightsData[i]);
    }

    return ArrayStorage.fromData(resultData, [outputSize], 'float64');
  } else {
    // Unweighted bincount
    const resultData = new Float64Array(outputSize);

    for (let i = 0; i < size; i++) {
      const idx = Number(xData[i]);
      resultData[idx]!++;
    }

    return ArrayStorage.fromData(resultData, [outputSize], 'float64');
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
  right: boolean = false
): ArrayStorage {
  const xData = x.data;
  const binsData = bins.data;
  const xSize = x.size;
  const numBins = bins.size;

  const resultData = new Float64Array(xSize);

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

  return ArrayStorage.fromData(resultData, [...x.shape], 'float64');
}

// Binary search helpers
function lowerBound(arr: any, size: number, val: number): number {
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

function upperBound(arr: any, size: number, val: number): number {
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

/**
 * Compute the histogram of a set of data.
 *
 * @param a - Input data (flattened if not 1D)
 * @param bins - Number of bins (default: 10) or array of bin edges
 * @param range - Lower and upper range of bins. If not provided, uses [a.min(), a.max()]
 * @param density - If true, return probability density function (default: false)
 * @param weights - Optional weights for each data point
 * @returns Tuple of [hist, bin_edges]
 */
export function histogram(
  a: ArrayStorage,
  bins: number | ArrayStorage = 10,
  range?: [number, number],
  density: boolean = false,
  weights?: ArrayStorage
): { hist: ArrayStorage; bin_edges: ArrayStorage } {
  const aData = a.data;
  const aSize = a.size;

  // Determine bin edges
  let binEdges: number[];

  if (typeof bins === 'number') {
    // Compute bin edges from range
    let minVal: number, maxVal: number;

    if (range) {
      [minVal, maxVal] = range;
    } else {
      minVal = Infinity;
      maxVal = -Infinity;
      for (let i = 0; i < aSize; i++) {
        const val = Number(aData[i]);
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
      // Handle case where all values are the same
      if (minVal === maxVal) {
        minVal = minVal - 0.5;
        maxVal = maxVal + 0.5;
      }
    }

    binEdges = [];
    const step = (maxVal - minVal) / bins;
    for (let i = 0; i <= bins; i++) {
      binEdges.push(minVal + i * step);
    }
  } else {
    // bins is an ArrayStorage of bin edges
    const binsData = bins.data;
    binEdges = [];
    for (let i = 0; i < bins.size; i++) {
      binEdges.push(Number(binsData[i]));
    }
  }

  const numBins = binEdges.length - 1;
  const hist = new Float64Array(numBins);

  // Compute histogram
  const weightsData = weights?.data;

  for (let i = 0; i < aSize; i++) {
    const val = Number(aData[i]);
    const w = weightsData ? Number(weightsData[i]) : 1;

    // Find bin index using binary search
    let binIdx = upperBound(binEdges, binEdges.length, val) - 1;

    // Handle edge cases
    if (binIdx < 0) {
      // Value is below the first bin edge
      continue;
    }
    if (binIdx >= numBins) {
      // Value is at or above the last bin edge
      // Include values exactly at the last edge in the last bin
      if (val === binEdges[numBins]) {
        binIdx = numBins - 1;
      } else {
        continue;
      }
    }

    hist[binIdx]! += w;
  }

  // Apply density normalization if requested
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

  return {
    hist: ArrayStorage.fromData(hist, [numBins], 'float64'),
    bin_edges: ArrayStorage.fromData(new Float64Array(binEdges), [binEdges.length], 'float64'),
  };
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
  weights?: ArrayStorage
): { hist: ArrayStorage; x_edges: ArrayStorage; y_edges: ArrayStorage } {
  const xData = x.data;
  const yData = y.data;
  const size = x.size;

  if (y.size !== size) {
    throw new Error('x and y must have the same length');
  }

  // Determine bin edges for x and y
  let xEdges: number[];
  let yEdges: number[];

  // Parse bins parameter
  let nxBins: number | ArrayStorage;
  let nyBins: number | ArrayStorage;

  if (typeof bins === 'number') {
    nxBins = bins;
    nyBins = bins;
  } else if (Array.isArray(bins) && bins.length === 2) {
    if (typeof bins[0] === 'number') {
      nxBins = bins[0] as number;
      nyBins = bins[1] as number;
    } else {
      nxBins = bins[0] as ArrayStorage;
      nyBins = bins[1] as ArrayStorage;
    }
  } else {
    nxBins = 10;
    nyBins = 10;
  }

  // Compute x edges
  if (typeof nxBins === 'number') {
    let xMin: number, xMax: number;
    if (range) {
      [xMin, xMax] = range[0];
    } else {
      xMin = Infinity;
      xMax = -Infinity;
      for (let i = 0; i < size; i++) {
        const val = Number(xData[i]);
        if (val < xMin) xMin = val;
        if (val > xMax) xMax = val;
      }
      if (xMin === xMax) {
        xMin -= 0.5;
        xMax += 0.5;
      }
    }
    xEdges = [];
    const step = (xMax - xMin) / nxBins;
    for (let i = 0; i <= nxBins; i++) {
      xEdges.push(xMin + i * step);
    }
  } else {
    const edgesData = nxBins.data;
    xEdges = [];
    for (let i = 0; i < nxBins.size; i++) {
      xEdges.push(Number(edgesData[i]));
    }
  }

  // Compute y edges
  if (typeof nyBins === 'number') {
    let yMin: number, yMax: number;
    if (range) {
      [yMin, yMax] = range[1];
    } else {
      yMin = Infinity;
      yMax = -Infinity;
      for (let i = 0; i < size; i++) {
        const val = Number(yData[i]);
        if (val < yMin) yMin = val;
        if (val > yMax) yMax = val;
      }
      if (yMin === yMax) {
        yMin -= 0.5;
        yMax += 0.5;
      }
    }
    yEdges = [];
    const step = (yMax - yMin) / nyBins;
    for (let i = 0; i <= nyBins; i++) {
      yEdges.push(yMin + i * step);
    }
  } else {
    const edgesData = nyBins.data;
    yEdges = [];
    for (let i = 0; i < nyBins.size; i++) {
      yEdges.push(Number(edgesData[i]));
    }
  }

  const nx = xEdges.length - 1;
  const ny = yEdges.length - 1;
  const hist = new Float64Array(nx * ny);

  // Compute histogram
  const weightsData = weights?.data;

  for (let i = 0; i < size; i++) {
    const xVal = Number(xData[i]);
    const yVal = Number(yData[i]);
    const w = weightsData ? Number(weightsData[i]) : 1;

    // Find bin indices
    let xIdx = upperBound(xEdges, xEdges.length, xVal) - 1;
    let yIdx = upperBound(yEdges, yEdges.length, yVal) - 1;

    // Handle edge cases
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

  // Apply density normalization if requested
  if (density) {
    let totalWeight = 0;
    for (let i = 0; i < hist.length; i++) {
      totalWeight += hist[i]!;
    }

    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const xWidth = xEdges[i + 1]! - xEdges[i]!;
        const yWidth = yEdges[j + 1]! - yEdges[j]!;
        const area = xWidth * yWidth;
        hist[i * ny + j]! = hist[i * ny + j]! / (totalWeight * area);
      }
    }
  }

  return {
    hist: ArrayStorage.fromData(hist, [nx, ny], 'float64'),
    x_edges: ArrayStorage.fromData(new Float64Array(xEdges), [xEdges.length], 'float64'),
    y_edges: ArrayStorage.fromData(new Float64Array(yEdges), [yEdges.length], 'float64'),
  };
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
  weights?: ArrayStorage
): { hist: ArrayStorage; edges: ArrayStorage[] } {
  const shape = sample.shape;
  const data = sample.data;

  // Determine number of samples and dimensions
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

  // Determine bin counts for each dimension
  let binCounts: number[];
  if (typeof bins === 'number') {
    binCounts = new Array(D).fill(bins);
  } else {
    binCounts = bins;
    if (binCounts.length !== D) {
      throw new Error('bins array length must match number of dimensions');
    }
  }

  // Compute edges for each dimension
  const allEdges: number[][] = [];

  for (let d = 0; d < D; d++) {
    let minVal: number, maxVal: number;

    if (range && range[d]) {
      [minVal, maxVal] = range[d]!;
    } else {
      minVal = Infinity;
      maxVal = -Infinity;
      for (let i = 0; i < N; i++) {
        const val = D === 1 ? Number(data[i]) : Number(data[i * D + d]);
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
      if (minVal === maxVal) {
        minVal -= 0.5;
        maxVal += 0.5;
      }
    }

    const numBins = binCounts[d]!;
    const edges: number[] = [];
    const step = (maxVal - minVal) / numBins;
    for (let i = 0; i <= numBins; i++) {
      edges.push(minVal + i * step);
    }
    allEdges.push(edges);
  }

  // Compute histogram shape
  const histShape = binCounts.slice();
  const histSize = histShape.reduce((a, b) => a * b, 1);
  const hist = new Float64Array(histSize);

  // Compute strides for indexing into histogram
  const strides: number[] = new Array(D);
  strides[D - 1] = 1;
  for (let d = D - 2; d >= 0; d--) {
    strides[d] = strides[d + 1]! * binCounts[d + 1]!;
  }

  // Compute histogram
  const weightsData = weights?.data;

  for (let i = 0; i < N; i++) {
    const w = weightsData ? Number(weightsData[i]) : 1;
    let histIdx = 0;
    let outOfBounds = false;

    for (let d = 0; d < D; d++) {
      const val = D === 1 ? Number(data[i]) : Number(data[i * D + d]);
      const edges = allEdges[d]!;
      const numBins = binCounts[d]!;

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

  // Apply density normalization if requested
  if (density) {
    let totalWeight = 0;
    for (let i = 0; i < histSize; i++) {
      totalWeight += hist[i]!;
    }

    // Compute volume of each bin
    const binVolumes = new Float64Array(histSize);
    for (let i = 0; i < histSize; i++) {
      let volume = 1;
      let idx = i;
      for (let d = 0; d < D; d++) {
        const binIdx = Math.floor(idx / strides[d]!) % binCounts[d]!;
        const edges = allEdges[d]!;
        volume *= edges[binIdx + 1]! - edges[binIdx]!;
      }
      binVolumes[i] = volume;
    }

    for (let i = 0; i < histSize; i++) {
      hist[i]! = hist[i]! / (totalWeight * binVolumes[i]!);
    }
  }

  // Convert edges to ArrayStorage
  const edgeStorages = allEdges.map((edges) =>
    ArrayStorage.fromData(new Float64Array(edges), [edges.length], 'float64')
  );

  return {
    hist: ArrayStorage.fromData(hist, histShape, 'float64'),
    edges: edgeStorages,
  };
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
  mode: 'full' | 'same' | 'valid' = 'full'
): ArrayStorage {
  const aData = a.data;
  const vData = v.data;
  const aLen = a.size;
  const vLen = v.size;

  // Compute full correlation first
  const fullLen = aLen + vLen - 1;
  const fullResult = new Float64Array(fullLen);

  // Cross-correlation: c[k] = sum_n(a[n] * conj(v[n-k]))
  // For real arrays, this is: c[k] = sum_n(a[n] * v[n-k])
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

  // Return based on mode
  if (mode === 'full') {
    return ArrayStorage.fromData(fullResult, [fullLen], 'float64');
  } else if (mode === 'same') {
    // Return central part with same length as a
    const start = Math.floor((fullLen - aLen) / 2);
    const result = new Float64Array(aLen);
    for (let i = 0; i < aLen; i++) {
      result[i] = fullResult[start + i]!;
    }
    return ArrayStorage.fromData(result, [aLen], 'float64');
  } else {
    // mode === 'valid'
    const validLen = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
    const start = Math.min(aLen, vLen) - 1;
    const result = new Float64Array(validLen);
    for (let i = 0; i < validLen; i++) {
      result[i] = fullResult[start + i]!;
    }
    return ArrayStorage.fromData(result, [validLen], 'float64');
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
  mode: 'full' | 'same' | 'valid' = 'full'
): ArrayStorage {
  // Convolution is correlation with v reversed
  const vData = v.data;
  const vLen = v.size;

  // Create reversed v
  const vReversed = new Float64Array(vLen);
  for (let i = 0; i < vLen; i++) {
    vReversed[i] = Number(vData[vLen - 1 - i]);
  }

  const vReversedStorage = ArrayStorage.fromData(vReversed, [vLen], 'float64');

  return correlate(a, vReversedStorage, mode);
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
  ddof?: number
): ArrayStorage {
  const mShape = m.shape;
  const mData = m.data;

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

      if (y.size !== n) {
        throw new Error('m and y must have same length');
      }

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

  // Compute means for each variable
  const means = new Float64Array(numVars);
  for (let i = 0; i < numVars; i++) {
    let sum = 0;
    for (let j = 0; j < numObs; j++) {
      const idx = rowvar ? i * numObs + j : j * numVars + i;
      sum += Number(mData[idx]);
    }
    means[i] = sum / numObs;
  }

  // Compute covariance matrix
  const covMatrix = new Float64Array(numVars * numVars);
  const divisor = numObs - effectiveDdof;

  if (divisor <= 0) {
    covMatrix.fill(NaN);
    return ArrayStorage.fromData(covMatrix, [numVars, numVars], 'float64');
  }

  for (let i = 0; i < numVars; i++) {
    for (let j = i; j < numVars; j++) {
      let sum = 0;
      for (let k = 0; k < numObs; k++) {
        const idxI = rowvar ? i * numObs + k : k * numVars + i;
        const idxJ = rowvar ? j * numObs + k : k * numVars + j;
        const di = Number(mData[idxI]) - means[i]!;
        const dj = Number(mData[idxJ]) - means[j]!;
        sum += di * dj;
      }
      const covVal = sum / divisor;
      covMatrix[i * numVars + j] = covVal;
      covMatrix[j * numVars + i] = covVal; // Symmetric
    }
  }

  return ArrayStorage.fromData(covMatrix, [numVars, numVars], 'float64');
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
  // Handle 1D array without y - return scalar 1.0 (0-d array like NumPy)
  if (x.shape.length === 1 && y === undefined) {
    return ArrayStorage.fromData(new Float64Array([1]), [], 'float64');
  }

  // Compute covariance matrix
  const covMatrix = cov(x, y, rowvar, false);

  const covData = covMatrix.data;
  const shape = covMatrix.shape;
  const n = shape[0]!;

  // Compute correlation coefficients from covariance matrix
  // corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
  const corrData = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const covIJ = Number(covData[i * n + j]);
      const varI = Number(covData[i * n + i]);
      const varJ = Number(covData[j * n + j]);

      if (varI <= 0 || varJ <= 0) {
        corrData[i * n + j] = NaN;
      } else {
        corrData[i * n + j] = covIJ / Math.sqrt(varI * varJ);
      }
    }
  }

  return ArrayStorage.fromData(corrData, [n, n], 'float64');
}
