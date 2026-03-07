/**
 * WASM Core Module — Tree-shakeable standalone functions with WASM acceleration.
 *
 * Re-exports everything from the JS core, overriding specific functions
 * with WASM-accelerated wrappers. Users import from 'numpy-ts/wasm/core'
 * instead of 'numpy-ts/core' to get WASM acceleration.
 *
 * Functions without WASM kernels pass through from core as-is.
 */

// Re-export everything from JS core
export * from '../core';

// Override with WASM-accelerated versions
export { matmul } from './kernels/matmul';
