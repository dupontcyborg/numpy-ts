/**
 * Rebuilds the project with production (ReleaseFast) WASM before tree-shaking
 * tests run, so bundle sizes reflect real output.
 */
import { execSync } from 'child_process';
import { resolve } from 'path';

export function setup() {
  const root = resolve(__dirname, '../..');
  console.log('[tree-shaking] Running production build (ReleaseFast)...');
  execSync('npm run build', { cwd: root, stdio: 'inherit' });
  console.log('[tree-shaking] Production build complete.');
}
