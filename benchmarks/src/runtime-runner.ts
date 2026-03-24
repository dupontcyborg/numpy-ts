/**
 * Standalone benchmark runner for multi-runtime support.
 *
 * This script is spawned as a subprocess under Node, Deno, or Bun.
 * It reads benchmark specs + config from stdin (JSON), executes benchmarks,
 * prints progress to stderr, and outputs BenchmarkTiming[] as JSON to stdout.
 *
 * Uses globalThis.performance.now() so it works across all runtimes without
 * needing to import perf_hooks.
 */

import * as np from '../../src/core';
import { serializeNpy, parseNpy, serializeNpzSync, parseNpzSync } from '../../src/io';
import { wasmConfig } from '../../src/common/wasm/config';
import type { BenchmarkCase, BenchmarkTiming, BenchmarkSetup } from './types';

// Benchmark configuration - set from stdin input
let MIN_SAMPLE_TIME_MS = 100;
let TARGET_SAMPLES = 5;

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

function std(arr: number[]): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function setupArrays(setup: BenchmarkSetup, operation?: string): Record<string, any> {
  const arrays: Record<string, any> = {};

  for (const [key, spec] of Object.entries(setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = spec;

    if (
      key === 'n' ||
      key === 'axis' ||
      key === 'new_shape' ||
      key === 'shape' ||
      key === 'fill_value' ||
      key === 'target_shape' ||
      key === 'dims' ||
      key === 'kth'
    ) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape' || key === 'target_shape' || key === 'dims') {
        arrays[key] = shape;
      }
      continue;
    }

    if (key === 'subscripts') {
      arrays[key] = spec.value;
      continue;
    }

    if (key === 'indices') {
      arrays[key] = shape;
      continue;
    }

    if (value !== undefined) {
      arrays[key] = np.full(shape, value, dtype);
    } else if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'complex' || fill === 'complex_small') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const complexValues = [];
      for (let i = 0; i < size; i++) {
        const v = fill === 'complex_small' ? (i % 10) + 1 : i + 1;
        complexValues.push(new np.Complex(v, v));
      }
      const flat = np.array(complexValues);
      const reshaped = np.reshape(flat, shape);
      // Cast to target dtype (e.g. complex64) if specified
      arrays[key] = dtype === 'complex64' ? np.asarray(reshaped, 'complex64') : reshaped;
    } else if (fill === 'random') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = np.reshape(flat, shape);
    } else if (fill === 'shuffled') {
      // Deterministic Fisher-Yates shuffle of arange using LCG (seed=42)
      // Create arange with target dtype first (wraps for narrow types), then shuffle
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const wrapped = np.arange(0, size, 1, dtype);
      const vals = (np.tolist(np.flatten(wrapped)) as number[]).flat() as number[];
      let seed = 42;
      for (let i = size - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        const j = seed % (i + 1);
        [vals[i], vals[j]] = [vals[j]!, vals[i]!];
      }
      arrays[key] = np.reshape(np.array(vals, dtype), shape);
    } else if (fill === 'arange') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = np.reshape(flat, shape);
    } else if (fill === 'invertible') {
      const n = shape[0];
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      const arangeMatrix = np.reshape(flat, shape);
      const identity = np.eye(n, undefined, undefined, dtype);
      arrays[key] = np.add(arangeMatrix, np.multiply(identity, n * n));
    }
  }

  if (operation === 'parseNpy' && arrays['a']) {
    arrays['_npyBytes'] = serializeNpy(arrays['a']);
  } else if (operation === 'serializeNpzSync' && arrays['a']) {
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzArrays'] = npzArrays;
  } else if (operation === 'parseNpzSync' && arrays['a']) {
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzBytes'] = serializeNpzSync(npzArrays);
  }

  return arrays;
}

function executeOperation(operation: string, arrays: Record<string, any>): any {
  // Array creation
  if (operation === 'zeros') return np.zeros(arrays['shape']);
  if (operation === 'ones') return np.ones(arrays['shape']);
  if (operation === 'empty') return np.empty(arrays['shape']);
  if (operation === 'full') return np.full(arrays['shape'], arrays['fill_value']);
  if (operation === 'arange') return np.arange(0, arrays['n'], 1);
  if (operation === 'linspace') return np.linspace(0, 100, arrays['n']);
  if (operation === 'logspace') return np.logspace(0, 3, arrays['n']);
  if (operation === 'geomspace') return np.geomspace(1, 1000, arrays['n']);
  if (operation === 'eye') return np.eye(arrays['n']);
  if (operation === 'identity') return np.identity(arrays['n']);
  if (operation === 'copy') return np.copy(arrays['a']);
  if (operation === 'zeros_like') return np.zeros_like(arrays['a']);
  if (operation === 'ones_like') return np.ones_like(arrays['a']);
  if (operation === 'empty_like') return np.empty_like(arrays['a']);
  if (operation === 'full_like') return np.full_like(arrays['a'], 7);

  // Arithmetic
  if (operation === 'add') return np.add(arrays['a'], arrays['b']);
  if (operation === 'subtract') return np.subtract(arrays['a'], arrays['b']);
  if (operation === 'multiply') return np.multiply(arrays['a'], arrays['b']);
  if (operation === 'divide') return np.divide(arrays['a'], arrays['b']);
  if (operation === 'mod') return np.mod(arrays['a'], arrays['b']);
  if (operation === 'floor_divide') return np.floor_divide(arrays['a'], arrays['b']);
  if (operation === 'reciprocal') return np.reciprocal(arrays['a']);
  if (operation === 'positive') return np.positive(arrays['a']);
  if (operation === 'cbrt') return np.cbrt(arrays['a']);
  if (operation === 'fabs') return np.fabs(arrays['a']);
  if (operation === 'divmod') return np.divmod(arrays['a'], arrays['b']);
  if (operation === 'gcd') {
    const scalar = arrays['b'].size === 1 ? arrays['b'].get([0]) : arrays['b'];
    return np.gcd(arrays['a'], scalar);
  }
  if (operation === 'lcm') {
    const scalar = arrays['b'].size === 1 ? arrays['b'].get([0]) : arrays['b'];
    return np.lcm(arrays['a'], scalar);
  }
  if (operation === 'float_power') {
    const scalar = arrays['b'].size === 1 ? arrays['b'].get([0]) : arrays['b'];
    return np.float_power(arrays['a'], scalar);
  }
  if (operation === 'square') return np.square(arrays['a']);
  if (operation === 'remainder') return np.remainder(arrays['a'], arrays['b']);
  if (operation === 'heaviside') return np.heaviside(arrays['a'], arrays['b']);
  if (operation === 'fmod') return np.fmod(arrays['a'], arrays['b']);
  if (operation === 'frexp') return np.frexp(arrays['a']);
  if (operation === 'ldexp') return np.ldexp(arrays['a'], arrays['b']);
  if (operation === 'modf') return np.modf(arrays['a']);

  // Math
  if (operation === 'sqrt') return np.sqrt(arrays['a']);
  if (operation === 'power') return np.power(arrays['a'], arrays['b']);
  if (operation === 'absolute') return np.absolute(arrays['a']);
  if (operation === 'negative') return np.negative(arrays['a']);
  if (operation === 'sign') return np.sign(arrays['a']);

  // Trig
  if (operation === 'sin') return np.sin(arrays['a']);
  if (operation === 'cos') return np.cos(arrays['a']);
  if (operation === 'tan') return np.tan(arrays['a']);
  if (operation === 'arcsin') return np.arcsin(arrays['a']);
  if (operation === 'arccos') return np.arccos(arrays['a']);
  if (operation === 'arctan') return np.arctan(arrays['a']);
  if (operation === 'arctan2') return np.arctan2(arrays['a'], arrays['b']);
  if (operation === 'hypot') return np.hypot(arrays['a'], arrays['b']);

  // Hyperbolic
  if (operation === 'sinh') return np.sinh(arrays['a']);
  if (operation === 'cosh') return np.cosh(arrays['a']);
  if (operation === 'tanh') return np.tanh(arrays['a']);
  if (operation === 'arcsinh') return np.arcsinh(arrays['a']);
  if (operation === 'arccosh') return np.arccosh(arrays['a']);
  if (operation === 'arctanh') return np.arctanh(arrays['a']);

  // Exponential
  if (operation === 'exp') return np.exp(arrays['a']);
  if (operation === 'exp2') return np.exp2(arrays['a']);
  if (operation === 'expm1') return np.expm1(arrays['a']);
  if (operation === 'log') return np.log(arrays['a']);
  if (operation === 'log2') return np.log2(arrays['a']);
  if (operation === 'log10') return np.log10(arrays['a']);
  if (operation === 'log1p') return np.log1p(arrays['a']);
  if (operation === 'logaddexp') return np.logaddexp(arrays['a'], arrays['b']);
  if (operation === 'logaddexp2') return np.logaddexp2(arrays['a'], arrays['b']);

  // Gradient
  if (operation === 'diff') return np.diff(arrays['a']);
  if (operation === 'gradient') return np.gradient(arrays['a']);
  if (operation === 'cross') return np.cross(arrays['a'], arrays['b']);

  // Linear algebra
  if (operation === 'dot') return np.dot(arrays['a'], arrays['b']);
  if (operation === 'inner') return np.inner(arrays['a'], arrays['b']);
  if (operation === 'outer') return np.outer(arrays['a'], arrays['b']);
  if (operation === 'tensordot') return np.tensordot(arrays['a'], arrays['b'], arrays['axes'] ?? 2);
  if (operation === 'matmul') return np.matmul(arrays['a'], arrays['b']);
  if (operation === 'trace') return np.trace(arrays['a']);
  if (operation === 'transpose') return np.transpose(arrays['a']);
  if (operation === 'diagonal') return np.diagonal(arrays['a']);
  if (operation === 'kron') return np.kron(arrays['a'], arrays['b']);
  if (operation === 'einsum') return np.einsum(arrays['subscripts'], arrays['a'], arrays['b']);
  if (operation === 'deg2rad') return np.deg2rad(arrays['a']);
  if (operation === 'rad2deg') return np.rad2deg(arrays['a']);

  // numpy.linalg
  if (operation === 'linalg_det') return np.linalg.det(arrays['a']);
  if (operation === 'linalg_inv') return np.linalg.inv(arrays['a']);
  if (operation === 'linalg_solve') return np.linalg.solve(arrays['a'], arrays['b']);
  if (operation === 'linalg_qr') return np.linalg.qr(arrays['a']);
  if (operation === 'linalg_cholesky') {
    const a = arrays['a'];
    const n = a.shape[0];
    const aT = np.transpose(a);
    const aTa = np.matmul(aT, a);
    const posdef = np.add(aTa, np.multiply(np.eye(n), n));
    return np.linalg.cholesky(posdef);
  }
  if (operation === 'linalg_svd') return np.linalg.svd(arrays['a']);
  if (operation === 'linalg_eig') return np.linalg.eig(arrays['a']);
  if (operation === 'linalg_eigh') {
    const a = arrays['a'];
    const aT = np.transpose(a);
    const sym = np.multiply(np.add(a, aT), 0.5);
    return np.linalg.eigh(sym);
  }
  if (operation === 'linalg_norm') return np.linalg.norm(arrays['a']);
  if (operation === 'linalg_matrix_rank') return np.linalg.matrix_rank(arrays['a']);
  if (operation === 'linalg_pinv') return np.linalg.pinv(arrays['a']);
  if (operation === 'linalg_cond') return np.linalg.cond(arrays['a']);
  if (operation === 'linalg_matrix_power') return np.linalg.matrix_power(arrays['a'], 3);
  if (operation === 'linalg_lstsq') return np.linalg.lstsq(arrays['a'], arrays['b']);
  if (operation === 'linalg_cross') return np.linalg.cross(arrays['a'], arrays['b']);
  if (operation === 'linalg_slogdet') return np.linalg.slogdet(arrays['a']);
  if (operation === 'linalg_svdvals') return np.linalg.svdvals(arrays['a']);
  if (operation === 'linalg_multi_dot')
    return np.linalg.multi_dot([arrays['a'], arrays['b'], arrays['c']]);
  if (operation === 'vdot') return np.vdot(arrays['a'], arrays['b']);
  if (operation === 'vecdot') return np.vecdot(arrays['a'], arrays['b']);
  if (operation === 'matrix_transpose') return np.matrix_transpose(arrays['a']);
  if (operation === 'matvec') return np.matvec(arrays['a'], arrays['b']);
  if (operation === 'vecmat') return np.vecmat(arrays['a'], arrays['b']);

  // Bitwise
  if (operation === 'bitwise_and') return np.bitwise_and(arrays['a'], arrays['b']);
  if (operation === 'bitwise_or') return np.bitwise_or(arrays['a'], arrays['b']);
  if (operation === 'bitwise_xor') return np.bitwise_xor(arrays['a'], arrays['b']);
  if (operation === 'bitwise_not') return np.bitwise_not(arrays['a']);
  if (operation === 'invert') return np.invert(arrays['a']);
  if (operation === 'left_shift') return np.left_shift(arrays['a'], arrays['b']);
  if (operation === 'right_shift') return np.right_shift(arrays['a'], arrays['b']);
  if (operation === 'packbits') return np.packbits(arrays['a']);
  if (operation === 'unpackbits') return np.unpackbits(arrays['a']);
  if (operation === 'bitwise_count') return np.bitwise_count(arrays['a']);

  // Reductions
  if (operation === 'sum') return np.sum(arrays['a'], arrays['axis']);
  if (operation === 'mean') return np.mean(arrays['a'], arrays['axis']);
  if (operation === 'max') return np.max(arrays['a'], arrays['axis']);
  if (operation === 'min') return np.min(arrays['a'], arrays['axis']);
  if (operation === 'prod') return np.prod(arrays['a'], arrays['axis']);
  if (operation === 'argmin') return np.argmin(arrays['a'], arrays['axis']);
  if (operation === 'argmax') return np.argmax(arrays['a'], arrays['axis']);
  if (operation === 'var') return np.var_(arrays['a'], arrays['axis']);
  if (operation === 'std') return np.std(arrays['a'], arrays['axis']);
  if (operation === 'all') return np.all(arrays['a'], arrays['axis']);
  if (operation === 'any') return np.any(arrays['a'], arrays['axis']);
  if (operation === 'cumsum') return np.cumsum(arrays['a']);
  if (operation === 'cumprod') return np.cumprod(arrays['a']);
  if (operation === 'ptp') return np.ptp(arrays['a']);
  if (operation === 'median') return np.median(arrays['a']);
  if (operation === 'percentile') return np.percentile(arrays['a'], 50);
  if (operation === 'quantile') return np.quantile(arrays['a'], 0.5);
  if (operation === 'average') return np.average(arrays['a']);
  if (operation === 'nansum') return np.nansum(arrays['a']);
  if (operation === 'nanmean') return np.nanmean(arrays['a']);
  if (operation === 'nanmin') return np.nanmin(arrays['a']);
  if (operation === 'nanmax') return np.nanmax(arrays['a']);
  if (operation === 'nanquantile') return np.nanquantile(arrays['a'], 0.5);
  if (operation === 'nanpercentile') return np.nanpercentile(arrays['a'], 50);

  // Array creation - extra
  if (operation === 'asarray_chkfinite') return np.asarray_chkfinite(arrays['a']);
  if (operation === 'require') return np.require(arrays['a'], undefined, 'C');

  // Reshape
  if (operation === 'reshape') return np.reshape(arrays['a'], arrays['new_shape']);
  if (operation === 'flatten') return np.flatten(arrays['a']);
  if (operation === 'ravel') return np.ravel(arrays['a']);
  if (operation === 'squeeze') return np.squeeze(arrays['a']);

  // Slicing
  if (operation === 'slice') return arrays['a'].slice('0:100', '0:100');

  // Array manipulation
  if (operation === 'swapaxes') return np.swapaxes(arrays['a'], 0, 1);
  if (operation === 'concatenate') return np.concatenate([arrays['a'], arrays['b']], 0);
  if (operation === 'stack') return np.stack([arrays['a'], arrays['b']], 0);
  if (operation === 'vstack') return np.vstack([arrays['a'], arrays['b']]);
  if (operation === 'hstack') return np.hstack([arrays['a'], arrays['b']]);
  if (operation === 'tile') return np.tile(arrays['a'], [2, 2]);
  if (operation === 'repeat') return np.repeat(arrays['a'], 2);
  if (operation === 'broadcast_to') return np.broadcast_to(arrays['a'], arrays['target_shape']);
  if (operation === 'take') return np.take(arrays['a'], arrays['indices']);
  if (operation === 'flip') return np.flip(arrays['a']);
  if (operation === 'rot90') return np.rot90(arrays['a']);
  if (operation === 'roll') return np.roll(arrays['a'], 10);
  if (operation === 'pad') return np.pad(arrays['a'], 2);
  if (operation === 'concat') return np.concat([arrays['a'], arrays['b']], 0);
  if (operation === 'unstack') return np.unstack(arrays['a'], 0);
  if (operation === 'block') return np.block([arrays['a'], arrays['b']]);
  if (operation === 'item') return np.item(arrays['a'], 0);
  if (operation === 'tolist') return np.tolist(arrays['a']);

  // Creation functions
  if (operation === 'diag') return np.diag(arrays['a']);
  if (operation === 'tri') return np.tri(arrays['shape'][0], arrays['shape'][1]);
  if (operation === 'tril') return np.tril(arrays['a']);
  if (operation === 'triu') return np.triu(arrays['a']);

  // Indexing
  if (operation === 'take_along_axis') return np.take_along_axis(arrays['a'], arrays['b'], 0);
  if (operation === 'compress') return np.compress(arrays['b'], arrays['a'], 0);
  if (operation === 'diag_indices') return np.diag_indices(arrays['n']);
  if (operation === 'tril_indices') return np.tril_indices(arrays['n']);
  if (operation === 'triu_indices') return np.triu_indices(arrays['n']);
  if (operation === 'indices') return np.indices(arrays['shape']);
  if (operation === 'ravel_multi_index')
    return np.ravel_multi_index([arrays['a'], arrays['b']], arrays['dims']);
  if (operation === 'unravel_index') return np.unravel_index(arrays['a'], arrays['dims']);

  // IO
  if (operation === 'serializeNpy') return serializeNpy(arrays['a']);
  if (operation === 'parseNpy') return parseNpy(arrays['_npyBytes']);
  if (operation === 'serializeNpzSync') return serializeNpzSync(arrays['_npzArrays']);
  if (operation === 'parseNpzSync') return parseNpzSync(arrays['_npzBytes']);

  // Sorting
  if (operation === 'sort') return np.sort(arrays['a']);
  if (operation === 'argsort') return np.argsort(arrays['a']);
  if (operation === 'partition') return np.partition(arrays['a'], arrays['kth'] || 0);
  if (operation === 'argpartition') return np.argpartition(arrays['a'], arrays['kth'] || 0);
  if (operation === 'lexsort') return np.lexsort([arrays['a'], arrays['b']]);
  if (operation === 'sort_complex') return np.sort_complex(arrays['a']);

  // Searching
  if (operation === 'nonzero') return np.nonzero(arrays['a']);
  if (operation === 'argwhere') return np.argwhere(arrays['a']);
  if (operation === 'flatnonzero') return np.flatnonzero(arrays['a']);
  if (operation === 'where') return np.where(arrays['a'], arrays['b'], arrays['c']);
  if (operation === 'searchsorted') return np.searchsorted(arrays['a'], arrays['b']);
  if (operation === 'extract') return np.extract(arrays['condition'], arrays['a']);
  if (operation === 'count_nonzero') return np.count_nonzero(arrays['a']);

  // Statistics
  if (operation === 'bincount') return np.bincount(arrays['a']);
  if (operation === 'digitize') return np.digitize(arrays['a'], arrays['b']);
  if (operation === 'histogram') return np.histogram(arrays['a'], 10);
  if (operation === 'histogram2d') return np.histogram2d(arrays['a'], arrays['b'], 10);
  if (operation === 'correlate') return np.correlate(arrays['a'], arrays['b'], 'full');
  if (operation === 'convolve') return np.convolve(arrays['a'], arrays['b'], 'full');
  if (operation === 'cov') return np.cov(arrays['a']);
  if (operation === 'corrcoef') return np.corrcoef(arrays['a']);
  if (operation === 'histogram_bin_edges') return np.histogram_bin_edges(arrays['a'], 10);
  if (operation === 'trapezoid') return np.trapezoid(arrays['a']);

  // Set operations
  if (operation === 'trim_zeros') return np.trim_zeros(arrays['a']);
  if (operation === 'unique_values') return np.unique_values(arrays['a']);
  if (operation === 'unique_counts') return np.unique_counts(arrays['a']);

  // Logic
  if (operation === 'logical_and')
    return arrays['b'] ? np.logical_and(arrays['a'], arrays['b']) : np.logical_and(arrays['a'], arrays['scalar']);
  if (operation === 'logical_or')
    return arrays['b'] ? np.logical_or(arrays['a'], arrays['b']) : np.logical_or(arrays['a'], arrays['scalar']);
  if (operation === 'logical_not') return np.logical_not(arrays['a']);
  if (operation === 'logical_xor')
    return arrays['b'] ? np.logical_xor(arrays['a'], arrays['b']) : np.logical_xor(arrays['a'], arrays['scalar']);
  if (operation === 'isfinite') return np.isfinite(arrays['a']);
  if (operation === 'isinf') return np.isinf(arrays['a']);
  if (operation === 'isnan') return np.isnan(arrays['a']);
  if (operation === 'isneginf') return np.isneginf(arrays['a']);
  if (operation === 'isposinf') return np.isposinf(arrays['a']);
  if (operation === 'isreal') return np.isreal(arrays['a']);
  if (operation === 'signbit') return np.signbit(arrays['a']);
  if (operation === 'copysign')
    return arrays['b'] ? np.copysign(arrays['a'], arrays['b']) : np.copysign(arrays['a'], arrays['scalar']);

  // Random
  if (operation === 'random_random') return np.random.random(arrays['shape']);
  if (operation === 'random_rand') return np.random.rand(...arrays['shape']);
  if (operation === 'random_randn') return np.random.randn(...arrays['shape']);
  if (operation === 'random_randint') return np.random.randint(0, 100, arrays['shape']);
  if (operation === 'random_uniform') return np.random.uniform(0, 1, arrays['shape']);
  if (operation === 'random_normal') return np.random.normal(0, 1, arrays['shape']);
  if (operation === 'random_standard_normal') return np.random.standard_normal(arrays['shape']);
  if (operation === 'random_exponential') return np.random.exponential(1, arrays['shape']);
  if (operation === 'random_poisson') return np.random.poisson(5, arrays['shape']);
  if (operation === 'random_binomial') return np.random.binomial(10, 0.5, arrays['shape']);
  if (operation === 'random_choice') return np.random.choice(arrays['n'], 100);
  if (operation === 'random_permutation') return np.random.permutation(arrays['n']);
  if (operation === 'random_gamma') return np.random.gamma(2, 1, arrays['shape']);
  if (operation === 'random_beta') return np.random.beta(2, 5, arrays['shape']);
  if (operation === 'random_chisquare') return np.random.chisquare(5, arrays['shape']);
  if (operation === 'random_laplace') return np.random.laplace(0, 1, arrays['shape']);
  if (operation === 'random_geometric') return np.random.geometric(0.5, arrays['shape']);
  if (operation === 'random_dirichlet') return np.random.dirichlet([1, 2, 3], arrays['shape'][0]);

  // Complex
  if (operation === 'complex_zeros') return np.zeros(arrays['shape'], 'complex128');
  if (operation === 'complex_ones') return np.ones(arrays['shape'], 'complex128');
  if (operation === 'complex_add') return np.add(arrays['a'], arrays['b']);
  if (operation === 'complex_multiply') return np.multiply(arrays['a'], arrays['b']);
  if (operation === 'complex_divide') return np.divide(arrays['a'], arrays['b']);
  if (operation === 'complex_real') return np.real(arrays['a']);
  if (operation === 'complex_imag') return np.imag(arrays['a']);
  if (operation === 'complex_conj') return np.conj(arrays['a']);
  if (operation === 'complex_angle') return np.angle(arrays['a']);
  if (operation === 'complex_abs') return np.abs(arrays['a']);
  if (operation === 'complex_sqrt') return np.sqrt(arrays['a']);
  if (operation === 'complex_sum') return np.sum(arrays['a']);
  if (operation === 'complex_mean') return np.mean(arrays['a']);
  if (operation === 'complex_prod') return np.prod(arrays['a']);

  // Other Math
  if (operation === 'clip') return np.clip(arrays['a'], 10, 100);
  if (operation === 'maximum') return np.maximum(arrays['a'], arrays['b']);
  if (operation === 'minimum') return np.minimum(arrays['a'], arrays['b']);
  if (operation === 'fmax') return np.fmax(arrays['a'], arrays['b']);
  if (operation === 'fmin') return np.fmin(arrays['a'], arrays['b']);
  if (operation === 'nan_to_num') return np.nan_to_num(arrays['a']);
  if (operation === 'interp') return np.interp(arrays['x'], arrays['xp'], arrays['fp']);
  if (operation === 'unwrap') return np.unwrap(arrays['a']);
  if (operation === 'sinc') return np.sinc(arrays['a']);
  if (operation === 'i0') return np.i0(arrays['a']);

  // Polynomial
  if (operation === 'poly') return np.poly(arrays['a']);
  if (operation === 'polyadd') return np.polyadd(arrays['a'], arrays['b']);
  if (operation === 'polyder') return np.polyder(arrays['a']);
  if (operation === 'polydiv') return np.polydiv(arrays['a'], arrays['b']);
  if (operation === 'polyfit') return np.polyfit(arrays['a'], arrays['b'], 2);
  if (operation === 'polyint') return np.polyint(arrays['a']);
  if (operation === 'polymul') return np.polymul(arrays['a'], arrays['b']);
  if (operation === 'polysub') return np.polysub(arrays['a'], arrays['b']);
  if (operation === 'polyval') return np.polyval(arrays['a'], arrays['b']);
  if (operation === 'roots') return np.roots(arrays['a']);

  // Type checking
  if (operation === 'can_cast') return np.can_cast('int32', 'float64');
  if (operation === 'result_type') return np.result_type('int32', 'float64');
  if (operation === 'min_scalar_type') return np.min_scalar_type(1000);
  if (operation === 'issubdtype') return np.issubdtype('int32', 'integer');

  // FFT
  if (operation === 'fft') return np.fft.fft(arrays['a']);
  if (operation === 'ifft') return np.fft.ifft(arrays['a']);
  if (operation === 'fft2') return np.fft.fft2(arrays['a']);
  if (operation === 'ifft2') return np.fft.ifft2(arrays['a']);
  if (operation === 'fftn') return np.fft.fftn(arrays['a']);
  if (operation === 'ifftn') return np.fft.ifftn(arrays['a']);
  if (operation === 'rfft') return np.fft.rfft(arrays['a']);
  if (operation === 'irfft') return np.fft.irfft(arrays['a']);
  if (operation === 'rfft2') return np.fft.rfft2(arrays['a']);
  if (operation === 'irfft2') return np.fft.irfft2(arrays['a']);
  if (operation === 'rfftn') return np.fft.rfftn(arrays['a']);
  if (operation === 'irfftn') return np.fft.irfftn(arrays['a']);
  if (operation === 'hfft') return np.fft.hfft(arrays['a']);
  if (operation === 'ihfft') return np.fft.ihfft(arrays['a']);
  if (operation === 'fftfreq') return np.fft.fftfreq(arrays['n']);
  if (operation === 'rfftfreq') return np.fft.rfftfreq(arrays['n']);
  if (operation === 'fftshift') return np.fft.fftshift(arrays['a']);
  if (operation === 'ifftshift') return np.fft.ifftshift(arrays['a']);

  throw new Error(`Unknown operation: ${operation}`);
}

function calibrateOpsPerSample(
  operation: string,
  arrays: Record<string, any>,
  targetTimeMs: number = MIN_SAMPLE_TIME_MS
): number {
  let opsPerSample = 1;
  let calibrationRuns = 0;
  const maxCalibrationRuns = 10;

  while (calibrationRuns < maxCalibrationRuns) {
    const start = globalThis.performance.now();
    for (let i = 0; i < opsPerSample; i++) {
      const result = executeOperation(operation, arrays);
      void result;
    }
    const elapsed = globalThis.performance.now() - start;

    if (elapsed >= targetTimeMs) break;

    if (elapsed < targetTimeMs / 10) {
      opsPerSample *= 10;
    } else if (elapsed < targetTimeMs / 2) {
      opsPerSample *= 2;
    } else {
      const targetOps = Math.ceil((opsPerSample * targetTimeMs) / elapsed);
      opsPerSample = Math.max(targetOps, opsPerSample + 1);
      break;
    }
    calibrationRuns++;
  }

  return Math.min(opsPerSample, 100000);
}

function runBenchmark(spec: BenchmarkCase): BenchmarkTiming {
  const { name, operation, setup, warmup } = spec;
  const arrays = setupArrays(setup, operation);

  // Warmup — also used to detect WASM kernel usage
  wasmConfig.wasmCallCount = 0;
  for (let i = 0; i < warmup; i++) {
    executeOperation(operation, arrays);
  }
  const wasmUsed = wasmConfig.wasmCallCount > 0;

  // Calibrate
  const opsPerSample = calibrateOpsPerSample(operation, arrays);

  // Benchmark
  const sampleTimes: number[] = [];
  let totalOps = 0;

  for (let sample = 0; sample < TARGET_SAMPLES; sample++) {
    const start = globalThis.performance.now();
    for (let i = 0; i < opsPerSample; i++) {
      const result = executeOperation(operation, arrays);
      void result;
    }
    const elapsed = globalThis.performance.now() - start;
    sampleTimes.push(elapsed / opsPerSample);
    totalOps += opsPerSample;
  }

  const meanMs = mean(sampleTimes);
  return {
    name,
    mean_ms: meanMs,
    median_ms: median(sampleTimes),
    min_ms: Math.min(...sampleTimes),
    max_ms: Math.max(...sampleTimes),
    std_ms: std(sampleTimes),
    ops_per_sec: 1000 / meanMs,
    total_ops: totalOps,
    total_samples: TARGET_SAMPLES,
    wasmUsed,
  };
}

// --- Main: read stdin, run benchmarks, write stdout ---

async function main() {
  // Read all of stdin
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk));
  }
  const input = JSON.parse(Buffer.concat(chunks).toString('utf-8'));
  const specs: BenchmarkCase[] = input.specs;
  const config = input.config;

  MIN_SAMPLE_TIME_MS = config.minSampleTimeMs;
  TARGET_SAMPLES = config.targetSamples;
  if (config.noWasm) {
    wasmConfig.thresholdMultiplier = Infinity;
  }

  // Detect runtime name and version
  let runtimeName = 'node';
  let runtimeVersion = typeof process !== 'undefined' ? process.version : 'unknown';

  // Deno sets Deno global, Bun sets Bun global
  if (typeof (globalThis as any).Deno !== 'undefined') {
    runtimeName = 'deno';
    runtimeVersion = (globalThis as any).Deno.version.deno;
  } else if (typeof (globalThis as any).Bun !== 'undefined') {
    runtimeName = 'bun';
    runtimeVersion = (globalThis as any).Bun.version;
  }

  console.error(`${runtimeName} ${runtimeVersion}`);
  console.error(
    `Running ${specs.length} benchmarks with auto-calibration...`
  );
  console.error(
    `Target: ${MIN_SAMPLE_TIME_MS}ms per sample, ${TARGET_SAMPLES} samples per benchmark\n`
  );

  const results: BenchmarkTiming[] = [];

  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    try {
      const result = runBenchmark(spec);
      results.push(result);

      console.error(
        `  [${i + 1}/${specs.length}] ${spec.name.padEnd(40)} ${result.mean_ms.toFixed(3).padStart(8)}ms  ${Math.round(result.ops_per_sec).toLocaleString().padStart(12)} ops/sec`
      );
    } catch (err) {
      // Push a placeholder result so indices stay aligned with Python results
      results.push({
        name: spec.name,
        mean_ms: 0,
        median_ms: 0,
        min_ms: 0,
        max_ms: 0,
        std_ms: 0,
        ops_per_sec: 0,
        total_ops: 0,
        total_samples: 0,
        wasmUsed: false,
      });
      console.error(
        `  [${i + 1}/${specs.length}] ${spec.name.padEnd(40)}   FAILED: ${err instanceof Error ? err.message : err}`
      );
    }
  }

  // Output results as JSON to stdout
  const output = JSON.stringify(results);
  process.stdout.write(output);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
