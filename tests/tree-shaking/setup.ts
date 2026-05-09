/**
 * Setup file for tree-shaking tests.
 * Runs as part of the test project (respects groupOrder), NOT at vitest init.
 * Rebuilds with production (ReleaseFast) WASM so bundle sizes reflect real output.
 *
 * Three workers start in parallel (one per bundler). The first to acquire the
 * lock file builds; the other two wait for the done sentinel.
 * global-setup.ts clears both sentinel files before any tests run.
 */
import { execSync } from 'node:child_process';
import { closeSync, existsSync, openSync, rmSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeAll } from 'vitest';

const root = resolve(__dirname, '../..');
const lockFile = resolve(root, '.tree-shaking-build.lock');
const doneFile = resolve(root, '.tree-shaking-build.done');

beforeAll(() => {
  let isBuilder = false;
  try {
    // O_EXCL: atomic exclusive create — fails if another worker already created it
    const fd = openSync(lockFile, 'wx');
    closeSync(fd);
    isBuilder = true;
  } catch {
    // Another worker won the lock and is building
  }

  if (isBuilder) {
    try {
      console.log('[tree-shaking] Running production build (ReleaseFast)...');
      execSync('npm run build', { cwd: root, stdio: 'inherit' });
      console.log('[tree-shaking] Production build complete.');
      writeFileSync(doneFile, '');
    } catch (e) {
      rmSync(lockFile, { force: true });
      throw e;
    }
  } else {
    console.log('[tree-shaking] Waiting for production build...');
    const start = Date.now();
    while (!existsSync(doneFile)) {
      if (Date.now() - start > 295_000) {
        throw new Error('[tree-shaking] Timed out waiting for production build');
      }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 500);
    }
    console.log('[tree-shaking] Build ready.');
  }
}, 300000);
