/**
 * Python NumPy validation tests for vindex (advanced indexing mixed with slices)
 *
 * These tests mirror the "mixing advanced and slice indices" cases from
 * tests/unit/vindex.test.ts and validate against real NumPy output.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { arange, vindex, array } from '../../src';
import { runNumPyBatch, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

describe('NumPy Validation: vindex (mixed advanced + slice)', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n' +
          '   This test suite requires Python with NumPy installed.\n\n' +
          '   Setup options:\n' +
          '   1. Using system Python: pip install numpy\n' +
          '   2. Using conda: conda install numpy\n' +
          '   3. Set custom Python: NUMPY_PYTHON="conda run -n myenv python" npm test\n\n' +
          `   Current Python command: ${process.env.NUMPY_PYTHON || 'python3'}\n`
      );
    }

    const info = getPythonInfo();
    console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
  });

  it('contiguous array indices are processed in place', () => {
    const a = arange(30).reshape(5, 2, 3);
    const idx = array([0, 1, 0]);
    const result = vindex(a, ':', idx, ':');

    const np = runNumPyBatch({
      values: `
a = np.arange(30).reshape(5, 2, 3)
idx = np.array([0, 1, 0])
result = a[:, idx, :]
`,
      strides: `
a = np.arange(30).reshape(5, 2, 3)
idx = np.array([0, 1, 0])
r = a[:, idx, :]
result = np.array([s // r.itemsize for s in r.strides])
`,
    });

    const npValues = np.get('values')!;
    const npStrides = np.get('strides')!;

    expect(result.shape).toEqual(npValues.shape);
    expect([...result.strides]).toEqual(npStrides.value);
    expect(arraysClose(result.toArray(), npValues.value)).toBe(true);
    expect(result.storage.offset).toBe(0);
  });

  it('discontiguous array indices are moved to front', () => {
    const a = arange(210).reshape(7, 5, 2, 3);
    const idx1 = array([0]);
    const idx2 = array([0]);
    const result = vindex(a, ':', idx1, ':', idx2);

    const np = runNumPyBatch({
      values: `
a = np.arange(210).reshape(7, 5, 2, 3)
idx1 = np.array([0])
idx2 = np.array([0])
result = a[:, idx1, :, idx2]
`,
      strides: `
a = np.arange(210).reshape(7, 5, 2, 3)
idx1 = np.array([0])
idx2 = np.array([0])
r = a[:, idx1, :, idx2]
result = np.array([s // r.itemsize for s in r.strides])
`,
    });

    const npValues = np.get('values')!;
    const npStrides = np.get('strides')!;

    expect(result.shape).toEqual(npValues.shape);
    expect([...result.strides]).toEqual(npStrides.value);
    expect(arraysClose(result.toArray(), npValues.value)).toBe(true);
    expect(result.storage.offset).toBe(0);
  });
});
