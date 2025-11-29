#!/usr/bin/env node
import { build } from 'esbuild';
import { readFileSync } from 'node:fs';

// Read version from package.json
const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
const VERSION = packageJson.version;

async function buildAll() {
  // Main bundle for Node.js (CJS) - core functionality only
  console.log('Building Node.js CJS bundle...');
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    outfile: 'dist/numpy-ts.node.cjs',
    sourcemap: true,
    minify: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });
  console.log('✓ Node.js CJS build complete');

  // Node.js bundle with file IO (for import from 'numpy-ts/node')
  console.log('Building Node.js bundle with file IO...');
  await build({
    entryPoints: ['src/node.ts'],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    outfile: 'dist/numpy-ts.node-io.cjs',
    sourcemap: true,
    minify: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });
  console.log('✓ Node.js IO build complete');

  // Node.js ESM bundle with file IO
  console.log('Building Node.js ESM bundle with file IO...');
  await build({
    entryPoints: ['src/node.ts'],
    bundle: true,
    platform: 'node',
    format: 'esm',
    outfile: 'dist/numpy-ts.node-io.mjs',
    sourcemap: true,
    minify: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });
  console.log('✓ Node.js ESM IO build complete');

  // Browser bundle (IIFE)
  console.log('Building browser bundle...');
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'browser',
    format: 'iife',
    globalName: 'np',
    outfile: 'dist/numpy-ts.browser.js',
    sourcemap: true,
    minify: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });
  console.log('✓ Browser build complete');

  // ESM bundle (for modern browsers and bundlers)
  console.log('Building ESM bundle...');
  const result = await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    outfile: 'dist/numpy-ts.esm.js',
    sourcemap: true,
    minify: true,
    metafile: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });

  // Write metafile for analysis
  await import('node:fs/promises').then((fs) =>
    fs.writeFile('dist/meta.json', JSON.stringify(result.metafile, null, 2))
  );
  console.log('✓ ESM build complete');

  console.log('\n✓ All builds completed successfully!');
}

buildAll().catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
