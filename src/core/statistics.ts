/**
 * Statistics Functions - Tree-shakeable standalone functions
 *
 * This module provides statistical operations that can be
 * imported independently for optimal tree-shaking.
 */

import * as statisticsOps from '../ops/statistics';
import { NDArrayCore, toStorage, fromStorage, ArrayStorage } from './types';

type BinStrategyString = 'auto' | 'fd' | 'doane' | 'scott' | 'stone' | 'rice' | 'sturges' | 'sqrt';

/** Count occurrences of values */
export function bincount(x: NDArrayCore, weights?: NDArrayCore, minlength?: number): NDArrayCore {
  const weightsStorage = weights ? toStorage(weights) : undefined;
  return fromStorage(statisticsOps.bincount(toStorage(x), weightsStorage, minlength));
}

/** Digitize values into bins */
export function digitize(x: NDArrayCore, bins: NDArrayCore, right?: boolean): NDArrayCore {
  return fromStorage(statisticsOps.digitize(toStorage(x), toStorage(bins), right));
}

/** Compute histogram */
export function histogram(
  a: NDArrayCore,
  bins?: number | NDArrayCore | BinStrategyString,
  range?: [number, number],
  density?: boolean,
  weights?: NDArrayCore
): { hist: NDArrayCore; bin_edges: NDArrayCore } {
  const binsArg = bins instanceof NDArrayCore ? toStorage(bins) : bins;
  const weightsArg = weights ? toStorage(weights) : undefined;
  const result = statisticsOps.histogram(
    toStorage(a),
    binsArg as number | ArrayStorage | undefined,
    range,
    density,
    weightsArg
  );
  return {
    hist: fromStorage(result.hist),
    bin_edges: fromStorage(result.bin_edges),
  };
}

/** Compute 2D histogram */
export function histogram2d(
  x: NDArrayCore,
  y: NDArrayCore,
  bins?: number | [number, number] | [NDArrayCore, NDArrayCore],
  range?: [[number, number], [number, number]],
  density?: boolean,
  weights?: NDArrayCore
): { H: NDArrayCore; x_edges: NDArrayCore; y_edges: NDArrayCore } {
  let binsArg: number | [number, number] | [ArrayStorage, ArrayStorage] | undefined;
  if (Array.isArray(bins) && bins.length === 2) {
    const b0 = bins[0] instanceof NDArrayCore ? toStorage(bins[0]) : bins[0];
    const b1 = bins[1] instanceof NDArrayCore ? toStorage(bins[1]) : bins[1];
    binsArg = [b0 as number | ArrayStorage, b1 as number | ArrayStorage] as
      | [number, number]
      | [ArrayStorage, ArrayStorage];
  } else {
    binsArg = bins as number | undefined;
  }
  const weightsArg = weights ? toStorage(weights) : undefined;
  const result = statisticsOps.histogram2d(
    toStorage(x),
    toStorage(y),
    binsArg,
    range,
    density,
    weightsArg
  );
  return {
    H: fromStorage(result.hist),
    x_edges: fromStorage(result.x_edges),
    y_edges: fromStorage(result.y_edges),
  };
}

/** Compute N-dimensional histogram */
export function histogramdd(
  sample: NDArrayCore,
  bins?: number | number[],
  range?: [number, number][],
  density?: boolean,
  weights?: NDArrayCore
): { hist: NDArrayCore; edges: NDArrayCore[] } {
  const weightsArg = weights ? toStorage(weights) : undefined;
  const result = statisticsOps.histogramdd(toStorage(sample), bins, range, density, weightsArg);
  return {
    hist: fromStorage(result.hist),
    edges: result.edges.map((e) => fromStorage(e)),
  };
}

/** Cross-correlation */
export function correlate(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full'
): NDArrayCore {
  return fromStorage(statisticsOps.correlate(toStorage(a), toStorage(v), mode));
}

/** Convolution */
export function convolve(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full'
): NDArrayCore {
  return fromStorage(statisticsOps.convolve(toStorage(a), toStorage(v), mode));
}

/** Covariance matrix */
export function cov(
  m: NDArrayCore,
  y?: NDArrayCore,
  rowvar?: boolean,
  bias?: boolean,
  ddof?: number
): NDArrayCore {
  return fromStorage(
    statisticsOps.cov(toStorage(m), y ? toStorage(y) : undefined, rowvar, bias, ddof)
  );
}

/** Correlation coefficients */
export function corrcoef(x: NDArrayCore, y?: NDArrayCore, rowvar?: boolean): NDArrayCore {
  return fromStorage(statisticsOps.corrcoef(toStorage(x), y ? toStorage(y) : undefined, rowvar));
}

/** Compute histogram bin edges */
export function histogram_bin_edges(
  a: NDArrayCore,
  bins?: number | BinStrategyString,
  range?: [number, number],
  weights?: NDArrayCore
): NDArrayCore {
  return fromStorage(
    statisticsOps.histogram_bin_edges(
      toStorage(a),
      bins,
      range,
      weights ? toStorage(weights) : undefined
    )
  );
}

/** Integrate using trapezoidal rule */
export function trapezoid(
  y: NDArrayCore,
  x?: NDArrayCore,
  dx?: number,
  axis?: number
): NDArrayCore | number {
  const result = statisticsOps.trapezoid(toStorage(y), x ? toStorage(x) : undefined, dx, axis);
  if (typeof result === 'number') return result;
  return fromStorage(result);
}
