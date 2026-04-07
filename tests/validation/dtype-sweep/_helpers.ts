/**
 * Shared helpers for dtype-sweep tests.
 */
export { ALL_DTYPES } from './_dtype-matrix';
export { runNumPy, arraysClose, checkNumPyAvailable } from '../numpy-oracle';

export function npDtype(d: string) {
  return d === 'int64' ? 'np.int64' : d === 'uint64' ? 'np.uint64' : `np.${d}`;
}

export const isInt = (d: string) => d.startsWith('int') || d.startsWith('uint');
export const isComplex = (d: string) => d.startsWith('complex');
export const isBool = (d: string) => d === 'bool';
