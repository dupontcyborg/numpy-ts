/**
 * Tests for standalone function exports
 *
 * These tests verify that the standalone functions exported from numpy-ts
 * work correctly. The underlying implementations are tested more thoroughly
 * in the method-based tests (arithmetic.test.ts, reduction.test.ts, etc.).
 */

import { describe, it, expect } from 'vitest';
import {
  // Basic arithmetic
  add,
  subtract,
  multiply,
  divide,
  // Reductions
  sum,
  mean,
  prod,
  std,
  variance,
  argmin,
  argmax,
  all,
  any,
  // Array creation
  array,
  zeros,
} from '../../src';

describe('Standalone Functions', () => {
  describe('Basic Arithmetic', () => {
    it('add() adds arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = add(a, b);
      expect(Array.from(result.data)).toEqual([5, 7, 9]);
    });

    it('add() adds scalar to array', () => {
      const a = array([1, 2, 3]);
      const result = add(a, 10);
      expect(Array.from(result.data)).toEqual([11, 12, 13]);
    });

    it('subtract() subtracts arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([1, 2, 3]);
      const result = subtract(a, b);
      expect(Array.from(result.data)).toEqual([9, 18, 27]);
    });

    it('multiply() multiplies arrays element-wise', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      const result = multiply(a, b);
      expect(Array.from(result.data)).toEqual([4, 10, 18]);
    });

    it('divide() divides arrays element-wise', () => {
      const a = array([10, 20, 30]);
      const b = array([2, 4, 5]);
      const result = divide(a, b);
      expect(Array.from(result.data)).toEqual([5, 5, 6]);
    });
  });

  describe('Reductions', () => {
    it('sum() returns sum of all elements', () => {
      const a = array([1, 2, 3, 4]);
      expect(sum(a)).toBe(10);
    });

    it('sum() with axis returns array', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const result = sum(a, 0);
      expect(Array.from(result.data)).toEqual([4, 6]);
    });

    it('mean() returns mean of all elements', () => {
      const a = array([1, 2, 3, 4]);
      expect(mean(a)).toBe(2.5);
    });

    it('prod() returns product of all elements', () => {
      const a = array([1, 2, 3, 4]);
      expect(prod(a)).toBe(24);
    });

    it('std() returns standard deviation', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = std(a);
      expect(result).toBeCloseTo(Math.sqrt(2), 5);
    });

    it('variance() returns variance', () => {
      const a = array([1, 2, 3, 4, 5]);
      const result = variance(a);
      expect(result).toBeCloseTo(2, 5);
    });

    it('argmin() returns index of minimum', () => {
      const a = array([3, 1, 4, 1, 5]);
      expect(argmin(a)).toBe(1);
    });

    it('argmax() returns index of maximum', () => {
      const a = array([3, 1, 4, 1, 5]);
      expect(argmax(a)).toBe(4);
    });

    it('all() returns true if all elements are truthy', () => {
      expect(all(array([1, 2, 3]))).toBe(true);
      expect(all(array([1, 0, 3]))).toBe(false);
    });

    it('any() returns true if any element is truthy', () => {
      expect(any(array([0, 0, 1]))).toBe(true);
      expect(any(array([0, 0, 0]))).toBe(false);
    });
  });

  describe('Method Chaining Still Works', () => {
    it('array results support method chaining', () => {
      const a = array([1, 2, 3, 4]);
      // Standalone function returns NDArray which has methods
      const result = add(a, 10).multiply(2).reshape([2, 2]);
      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([22, 24, 26, 28]);
    });

    it('zeros/ones support method chaining', () => {
      const a = zeros([2, 3]).add(5);
      expect(a.shape).toEqual([2, 3]);
      expect(Array.from(a.data)).toEqual([5, 5, 5, 5, 5, 5]);
    });
  });
});
