/**
 * NumPy validation tests for np.meshgrid — verifies indexing/sparse/copy kwargs.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { array, meshgrid } from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

describe('NumPy Validation: meshgrid', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it("default 'xy' indexing matches NumPy", () => {
    const x = array([1, 2, 3]);
    const y = array([10, 20]);
    const [X, Y] = meshgrid(x, y);
    const npX = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
result, _ = np.meshgrid(x, y)
`);
    const npY = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
_, result = np.meshgrid(x, y)
`);
    expect(Array.from(X!.shape)).toEqual(npX.shape);
    expect(arraysClose(X!.toArray() as number[], npX.value, 1e-10)).toBe(true);
    expect(arraysClose(Y!.toArray() as number[], npY.value, 1e-10)).toBe(true);
  });

  it("'ij' indexing matches NumPy", () => {
    const x = array([1, 2, 3]);
    const y = array([10, 20]);
    const [X, Y] = meshgrid(x, y, { indexing: 'ij' });
    const npX = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
result, _ = np.meshgrid(x, y, indexing='ij')
`);
    const npY = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
_, result = np.meshgrid(x, y, indexing='ij')
`);
    expect(Array.from(X!.shape)).toEqual(npX.shape);
    expect(arraysClose(X!.toArray() as number[], npX.value, 1e-10)).toBe(true);
    expect(arraysClose(Y!.toArray() as number[], npY.value, 1e-10)).toBe(true);
  });

  it('sparse=true matches NumPy', () => {
    const x = array([1, 2, 3]);
    const y = array([10, 20]);
    const [X, Y] = meshgrid(x, y, { sparse: true });
    const npX = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
result, _ = np.meshgrid(x, y, sparse=True)
`);
    const npY = runNumPy(`
x = np.array([1, 2, 3]); y = np.array([10, 20])
_, result = np.meshgrid(x, y, sparse=True)
`);
    expect(Array.from(X!.shape)).toEqual(npX.shape);
    expect(Array.from(Y!.shape)).toEqual(npY.shape);
    expect(arraysClose(X!.toArray() as number[], npX.value, 1e-10)).toBe(true);
    expect(arraysClose(Y!.toArray() as number[], npY.value, 1e-10)).toBe(true);
  });

  it("3D 'ij' indexing matches NumPy", () => {
    const x = array([1, 2]);
    const y = array([10, 20, 30]);
    const z = array([100, 200]);
    const [X, Y, Z] = meshgrid(x, y, z, { indexing: 'ij' });
    const npX = runNumPy(`
x = np.array([1, 2]); y = np.array([10, 20, 30]); z = np.array([100, 200])
result, _, _ = np.meshgrid(x, y, z, indexing='ij')
`);
    const npY = runNumPy(`
x = np.array([1, 2]); y = np.array([10, 20, 30]); z = np.array([100, 200])
_, result, _ = np.meshgrid(x, y, z, indexing='ij')
`);
    const npZ = runNumPy(`
x = np.array([1, 2]); y = np.array([10, 20, 30]); z = np.array([100, 200])
_, _, result = np.meshgrid(x, y, z, indexing='ij')
`);
    expect(Array.from(X!.shape)).toEqual(npX.shape);
    expect(arraysClose(X!.toArray() as number[], npX.value, 1e-10)).toBe(true);
    expect(arraysClose(Y!.toArray() as number[], npY.value, 1e-10)).toBe(true);
    expect(arraysClose(Z!.toArray() as number[], npZ.value, 1e-10)).toBe(true);
  });
});
