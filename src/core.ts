/**
 * numpy-ts/core - Tree-shakeable entry point
 *
 * This module provides standalone functions that return NDArrayCore.
 * Use this entry point when bundle size is critical and you don't need method chaining.
 *
 * All functions in this module are designed for optimal tree-shaking.
 * Import only what you need to get minimal bundle sizes.
 *
 * @example
 * ```typescript
 * import { array, add, reshape } from 'numpy-ts/core';
 *
 * const a = array([1, 2, 3, 4]);
 * const b = add(a, 10);              // Standalone function
 * const c = reshape(b, [2, 2]);      // Standalone function
 * console.log(c.shape);              // [2, 2]
 * ```
 *
 * For method chaining, use the main entry point instead:
 * ```typescript
 * import { array } from 'numpy-ts';
 *
 * const c = array([1, 2, 3, 4]).add(10).reshape([2, 2]);
 * ```
 *
 * @module numpy-ts/core
 */

export * from './core/index';
