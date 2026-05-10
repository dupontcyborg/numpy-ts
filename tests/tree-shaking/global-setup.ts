/**
 * Global setup for tree-shaking tests.
 * Only responsibility: clear stale sentinel files from a previous run so the
 * per-worker setup (setup.ts) always performs a fresh build.
 */
import { rmSync } from 'node:fs';
import { resolve } from 'node:path';

export function setup() {
  const root = resolve(__dirname, '../..');
  rmSync(resolve(root, '.tree-shaking-build.lock'), { force: true });
  rmSync(resolve(root, '.tree-shaking-build.done'), { force: true });
}
