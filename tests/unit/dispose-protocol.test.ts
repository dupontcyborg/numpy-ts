/**
 * Symbol.dispose protocol tests
 *
 * Verifies that NDArrayCore and ArrayStorage always expose [Symbol.dispose],
 * regardless of whether the runtime provides Symbol.dispose natively (Node 22+)
 * or the library polyfills it via Symbol.for("Symbol.dispose") (older runtimes,
 * Safari). The polyfill key matches what TypeScript, esbuild, Babel, and SWC
 * emit in their downlevel `using` helpers.
 */

import { describe, expect, it } from 'vitest';
import { NDArrayCore } from '../../src/common/ndarray-core';
import { ArrayStorage } from '../../src/common/storage';

// After importing storage.ts, Symbol.dispose is guaranteed to be defined
// (either natively or via the polyfill).
describe('Symbol.dispose polyfill', () => {
  it('Symbol.dispose is defined', () => {
    expect(Symbol.dispose).toBeDefined();
    expect(typeof Symbol.dispose).toBe('symbol');
  });

  it('on polyfilled runtimes, equals Symbol.for("Symbol.dispose")', () => {
    // On runtimes without native Symbol.dispose, the polyfill sets it to
    // Symbol.for("Symbol.dispose"). On native runtimes (Node 22+), it's a
    // distinct built-in symbol and the polyfill is a no-op.
    // Both paths are correct — transpilers use:
    //   Symbol.dispose || Symbol.for("Symbol.dispose")
    // which picks the native symbol when available.
    const isNative = Symbol.dispose !== Symbol.for('Symbol.dispose');
    if (!isNative) {
      expect(Symbol.dispose).toBe(Symbol.for('Symbol.dispose'));
    }
  });
});

describe('ArrayStorage[Symbol.dispose]', () => {
  it('is a function on the prototype', () => {
    expect(typeof ArrayStorage.prototype[Symbol.dispose]).toBe('function');
  });

  it('is non-optional (always present on instances)', () => {
    const s = new ArrayStorage(new Float64Array([1, 2, 3]), [3], 'float64');
    expect(s[Symbol.dispose]).toBeInstanceOf(Function);
  });

  it('calls dispose() and releases resources', () => {
    const s = new ArrayStorage(new Float64Array([1, 2, 3]), [3], 'float64');

    // Should not throw regardless of whether the storage is WASM-backed
    expect(() => s[Symbol.dispose]()).not.toThrow();
  });

  it('double [Symbol.dispose] is safe (mirrors dispose())', () => {
    const s = new ArrayStorage(new Float64Array([1, 2, 3]), [3], 'float64');
    s[Symbol.dispose]();
    expect(() => s[Symbol.dispose]()).not.toThrow();
  });
});

describe('NDArrayCore[Symbol.dispose]', () => {
  function makeArray(): NDArrayCore {
    const storage = new ArrayStorage(new Float64Array([1, 2, 3, 4]), [2, 2], 'float64');
    return new NDArrayCore(storage);
  }

  it('is a function on the prototype', () => {
    expect(typeof NDArrayCore.prototype[Symbol.dispose]).toBe('function');
  });

  it('is non-optional (always present on instances)', () => {
    const arr = makeArray();
    expect(arr[Symbol.dispose]).toBeInstanceOf(Function);
  });

  it('calls dispose() and frees the underlying storage', () => {
    const arr = makeArray();
    arr[Symbol.dispose]();
    // After dispose, the storage's WASM region is released
    expect(arr.storage.isWasmBacked).toBe(false);
  });

  it('double [Symbol.dispose] is safe', () => {
    const arr = makeArray();
    arr[Symbol.dispose]();
    expect(() => arr[Symbol.dispose]()).not.toThrow();
  });
});

describe('transpiler interop', () => {
  // Transpilers emit: Symbol.dispose || Symbol.for("Symbol.dispose")
  // On native runtimes, Symbol.dispose is used directly.
  // On polyfilled runtimes, Symbol.for("Symbol.dispose") is used.
  const transpilerKey = Symbol.dispose || Symbol.for('Symbol.dispose');

  it('NDArrayCore is discoverable via transpiler key', () => {
    const arr = new NDArrayCore(new ArrayStorage(new Float64Array([1]), [1], 'float64'));

    const method = (arr as any)[transpilerKey];
    expect(typeof method).toBe('function');

    method.call(arr);
    expect(arr.storage.isWasmBacked).toBe(false);
  });

  it('ArrayStorage is discoverable via transpiler key', () => {
    const s = new ArrayStorage(new Float64Array([1, 2]), [2], 'float64');

    const method = (s as any)[transpilerKey];
    expect(typeof method).toBe('function');

    method.call(s);
    expect(s.isWasmBacked).toBe(false);
  });
});
