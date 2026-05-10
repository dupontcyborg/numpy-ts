/**
 * Statistics Functions - Tree-shakeable standalone functions
 *
 * This module provides statistical operations that can be
 * imported independently for optimal tree-shaking.
 */

import { Complex } from '../common/complex';
import * as statisticsOps from '../common/ops/statistics';
import { asarray } from './creation';
import {
  type ArrayLike,
  type ArrayStorage,
  fromStorage,
  type NDArrayCore,
  toStorage,
} from './types';

type BinStrategyString = 'auto' | 'fd' | 'doane' | 'scott' | 'stone' | 'rice' | 'sturges' | 'sqrt';

/** Count occurrences of values */
export function bincount(x: ArrayLike, weights?: ArrayLike, minlength?: number): NDArrayCore {
  const weightsStorage = weights !== undefined ? toStorage(asarray(weights)) : undefined;
  return fromStorage(statisticsOps.bincount(toStorage(asarray(x)), weightsStorage, minlength));
}

/** Digitize values into bins */
export function digitize(x: ArrayLike, bins: ArrayLike, right?: boolean): NDArrayCore {
  return fromStorage(
    statisticsOps.digitize(toStorage(asarray(x)), toStorage(asarray(bins)), right),
  );
}

/** Compute histogram */
export function histogram(
  a: ArrayLike,
  bins?: number | ArrayLike | BinStrategyString,
  range?: [number, number],
  density?: boolean,
  weights?: ArrayLike,
): [NDArrayCore, NDArrayCore] {
  const binsArg =
    typeof bins === 'number' || typeof bins === 'string' || bins === undefined
      ? bins
      : toStorage(asarray(bins));
  const weightsArg = weights !== undefined ? toStorage(asarray(weights)) : undefined;
  const result = statisticsOps.histogram(
    toStorage(asarray(a)),
    binsArg as number | ArrayStorage | undefined,
    range,
    density,
    weightsArg,
  );
  const hist = result.hist;
  return [fromStorage(hist), fromStorage(result.bin_edges)];
}

/** Compute 2D histogram */
export function histogram2d(
  x: ArrayLike,
  y: ArrayLike,
  bins?: number | [number, number] | [ArrayLike, ArrayLike],
  range?: [[number, number], [number, number]],
  density?: boolean,
  weights?: ArrayLike,
): [NDArrayCore, NDArrayCore, NDArrayCore] {
  let binsArg: number | [number, number] | [ArrayStorage, ArrayStorage] | undefined;
  if (Array.isArray(bins) && bins.length === 2) {
    const b0raw = bins[0];
    const b1raw = bins[1];
    const b0 = typeof b0raw === 'number' ? b0raw : toStorage(asarray(b0raw));
    const b1 = typeof b1raw === 'number' ? b1raw : toStorage(asarray(b1raw));
    binsArg = [b0 as number | ArrayStorage, b1 as number | ArrayStorage] as
      | [number, number]
      | [ArrayStorage, ArrayStorage];
  } else {
    binsArg = bins as number | undefined;
  }
  const weightsArg = weights !== undefined ? toStorage(asarray(weights)) : undefined;
  const result = statisticsOps.histogram2d(
    toStorage(asarray(x)),
    toStorage(asarray(y)),
    binsArg,
    range,
    density,
    weightsArg,
  );
  return [fromStorage(result.hist), fromStorage(result.x_edges), fromStorage(result.y_edges)];
}

/** Compute N-dimensional histogram */
export function histogramdd(
  sample: ArrayLike,
  bins?: number | number[],
  range?: [number, number][],
  density?: boolean,
  weights?: ArrayLike,
): [NDArrayCore, NDArrayCore[]] {
  const weightsArg = weights !== undefined ? toStorage(asarray(weights)) : undefined;
  const result = statisticsOps.histogramdd(
    toStorage(asarray(sample)),
    bins,
    range,
    density,
    weightsArg,
  );
  return [fromStorage(result.hist), result.edges.map((e) => fromStorage(e))];
}

/** Cross-correlation */
export function correlate(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full',
): NDArrayCore {
  return fromStorage(statisticsOps.correlate(toStorage(a), toStorage(v), mode));
}

/** Convolution */
export function convolve(
  a: NDArrayCore,
  v: NDArrayCore,
  mode?: 'valid' | 'same' | 'full',
): NDArrayCore {
  return fromStorage(statisticsOps.convolve(toStorage(a), toStorage(v), mode));
}

/** Covariance matrix */
export function cov(
  m: NDArrayCore,
  y?: NDArrayCore,
  rowvar?: boolean,
  bias?: boolean,
  ddof?: number,
): NDArrayCore {
  return fromStorage(
    statisticsOps.cov(toStorage(m), y ? toStorage(y) : undefined, rowvar, bias, ddof),
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
  weights?: NDArrayCore,
): NDArrayCore {
  return fromStorage(
    statisticsOps.histogram_bin_edges(
      toStorage(a),
      bins,
      range,
      weights ? toStorage(weights) : undefined,
    ),
  );
}

/** Integrate using trapezoidal rule */
export function trapezoid(
  y: NDArrayCore,
  x?: NDArrayCore,
  dx?: number,
  axis?: number,
): NDArrayCore | number | Complex {
  const result = statisticsOps.trapezoid(toStorage(y), x ? toStorage(x) : undefined, dx, axis);
  if (typeof result === 'number') return result;
  if (result instanceof Complex) return result;
  return fromStorage(result);
}
