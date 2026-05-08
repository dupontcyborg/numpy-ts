/**
 * Tree-shaking test fixture: Symbol.dispose protocol through transpiled `using`
 *
 * When bundled with a downlevel target (e.g. ES2020), the `using` keyword is
 * transpiled to a helper that looks up [Symbol.dispose] or
 * [Symbol.for("Symbol.dispose")]. This fixture verifies that:
 *   1. The polyfill side-effect survives bundling (not tree-shaken away)
 *   2. The transpiled helper finds the dispose method
 *   3. Execution completes without "not disposable" errors
 *
 * Exit code 0 = PASS, non-zero = FAIL.
 */
import { zeros } from 'numpy-ts/core';

function testUsing() {
  // Transpiled `using` — the bundler's downlevel helper must find [Symbol.dispose]
  using arr = zeros([2, 3]);
  if (arr.shape[0] !== 2 || arr.shape[1] !== 3) {
    throw new Error('FAIL: unexpected shape');
  }
}

function testLoop() {
  // Transpiled `using` inside a loop — each iteration should dispose
  for (let i = 0; i < 3; i++) {
    using temp = zeros([4]);
    if (temp.size !== 4) throw new Error('FAIL: unexpected size');
  }
}

function testExplicitKey() {
  // Verify the Symbol.for key matches (this is what transpilers use as fallback)
  const key = Symbol.dispose || Symbol.for('Symbol.dispose');
  const arr = zeros([1]);
  if (typeof (arr as any)[key] !== 'function') {
    throw new Error('FAIL: dispose method not found via transpiler key');
  }
  (arr as any)[key]();
}

testUsing();
testLoop();
testExplicitKey();

console.log('PASS: dispose protocol works');
