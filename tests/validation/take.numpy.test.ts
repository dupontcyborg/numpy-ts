/**
 * NumPy validation tests for np.take — verifies ndarray index support
 * and that output shape preserves the indices' shape.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { arange, array, take } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: take', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('accepts a plain number[] (legacy)', () => {
    const a = arange(12).reshape(3, 4);
    const result = take(a, [0, 2], 1);
    const np = runNumPy(`
a = np.arange(12).reshape(3, 4)
result = np.take(a, [0, 2], axis=1)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts a 1D ndarray of indices', () => {
    const a = arange(12).reshape(3, 4);
    const idx = array([0, 2, 1]);
    const result = take(a, idx, 1);
    const np = runNumPy(`
a = np.arange(12).reshape(3, 4)
idx = np.array([0, 2, 1])
result = np.take(a, idx, axis=1)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('preserves 2D index shape in the output', () => {
    const a = arange(20);
    const idx = array([
      [0, 1, 2],
      [3, 4, 5],
    ]);
    const result = take(a, idx);
    const np = runNumPy(`
a = np.arange(20)
idx = np.array([[0, 1, 2], [3, 4, 5]])
result = np.take(a, idx)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('preserves 2D index shape along an axis', () => {
    const a = arange(24).reshape(2, 4, 3);
    const idx = array([
      [0, 2],
      [1, 3],
    ]);
    const result = take(a, idx, 1);
    const np = runNumPy(`
a = np.arange(24).reshape(2, 4, 3)
idx = np.array([[0, 2], [1, 3]])
result = np.take(a, idx, axis=1)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });

  it('accepts a scalar index', () => {
    const a = arange(12).reshape(3, 4);
    const result = take(a, 2, 1);
    const np = runNumPy(`
a = np.arange(12).reshape(3, 4)
result = np.take(a, 2, axis=1)
`);
    expect(Array.from(result.shape)).toEqual(np.shape);
    expect(arraysClose(result.toArray() as number[], np.value, 1e-10)).toBe(true);
  });
});
