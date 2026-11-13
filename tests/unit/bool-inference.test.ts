/**
 * Dtype inference from plain JavaScript data: all-boolean data is bool, and mixed data
 * keeps the library defaults (number literals infer float64 on purpose to avoid bigint).
 */

import { describe, expect, it } from 'vitest';
import { array, asarray, Complex } from '../../src';

describe('array() dtype inference with booleans', () => {
  it('infers bool for all-boolean data at any depth', () => {
    expect(array([true, false]).dtype).toBe('bool');
    expect(array([[true], [false]]).dtype).toBe('bool');
    expect(array([[[true, false]]]).dtype).toBe('bool');
    expect(array(true).dtype).toBe('bool');
    expect(array(false).shape).toEqual([]);
    expect(asarray([true, false]).dtype).toBe('bool');
    expect(asarray(true).dtype).toBe('bool');
  });

  it('keeps float64 for empty data', () => {
    expect(array([]).dtype).toBe('float64');
    expect(array([[], []]).dtype).toBe('float64');
  });

  it('keeps float64 when booleans are mixed with numbers', () => {
    expect(array([true, 1]).dtype).toBe('float64');
    expect(array([true, 2.5]).dtype).toBe('float64');
    expect(array([[true], [1]]).dtype).toBe('float64');
    expect(Array.from(array([true, 2.5]).toArray() as number[])).toEqual([1, 2.5]);
  });

  it('keeps the bigint and complex inference when booleans are mixed in', () => {
    expect(array([true, 1n]).dtype).toBe('int64');
    expect(array([true, new Complex(1, 2)]).dtype).toBe('complex128');
  });

  it('leaves explicit dtypes and typed arrays alone', () => {
    expect(array([true, false], 'float64').dtype).toBe('float64');
    expect(array([true, false], 'uint8').dtype).toBe('uint8');
    expect(array(new Uint8Array([1, 0])).dtype).toBe('uint8');
    expect(array([1, 0]).dtype).toBe('float64');
  });
});
