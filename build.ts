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
    sourcemap: false, // Browser bundles don't need source maps in npm package
    minify: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });
  console.log('✓ Browser build complete');

  // Tree-shakeable ESM build (preserves module structure for optimal bundler tree-shaking)
  // This transpiles each source file individually without bundling, preserving imports
  console.log('Building tree-shakeable ESM modules...');
  const glob = await import('node:fs').then((fs) => {
    const files: string[] = [];
    const walkDir = (dir: string) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = `${dir}/${entry.name}`;
        if (entry.isDirectory()) {
          walkDir(fullPath);
        } else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.d.ts')) {
          files.push(fullPath);
        }
      }
    };
    walkDir('src');
    return files;
  });

  const esmResult = await build({
    entryPoints: glob,
    bundle: false, // Don't bundle - preserve module structure
    platform: 'neutral',
    format: 'esm',
    outdir: 'dist/esm',
    outbase: 'src',
    sourcemap: false,
    minify: true,
    metafile: true,
    define: {
      __VERSION_PLACEHOLDER__: JSON.stringify(VERSION),
    },
  });

  // Write metafile for analysis
  await import('node:fs/promises').then((fs) =>
    fs.writeFile('dist/meta.json', JSON.stringify(esmResult.metafile, null, 2))
  );
  console.log('✓ Tree-shakeable ESM build complete');

  console.log('\n✓ All builds completed successfully!');
}

buildAll().catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
