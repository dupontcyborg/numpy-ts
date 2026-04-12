import { build } from 'esbuild';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { execSync } from 'node:child_process';

/**
 * Post-process ESM output to add .js extensions to relative import/export
 * specifiers, so Node's ESM loader can resolve them.
 *
 * Handles: './foo' → './foo.js', './dir' → './dir/index.js'
 */
async function fixEsmExtensions(outdir: string, srcDir: string) {
  const fsp = await import('node:fs/promises');
  const path = await import('node:path');

  async function walk(dir: string): Promise<string[]> {
    const entries = await fsp.readdir(dir, { withFileTypes: true });
    const files: string[] = [];
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) files.push(...(await walk(full)));
      else if (entry.name.endsWith('.js')) files.push(full);
    }
    return files;
  }

  const files = await walk(outdir);

  // Match: from"./relative/path" or from "./relative/path"
  const importRe = /(from\s*["'])(\.\.?\/[^"']+)(["'])/g;

  for (const file of files) {
    let content = await fsp.readFile(file, 'utf-8');
    let changed = false;

    content = content.replace(importRe, (match, prefix, specifier, suffix) => {
      if (specifier.endsWith('.js') || specifier.endsWith('.mjs')) return match;

      // Resolve against the source dir to check what the specifier points to
      const relFromOut = path.relative(outdir, path.dirname(file));
      const srcFile = path.resolve(srcDir, relFromOut, specifier);

      // Directory with index.ts → append /index.js
      if (existsSync(srcFile) && statSync(srcFile).isDirectory()) {
        if (existsSync(path.join(srcFile, 'index.ts'))) {
          changed = true;
          return `${prefix}${specifier}/index.js${suffix}`;
        }
      }

      // File with .ts extension → append .js
      if (existsSync(srcFile + '.ts')) {
        changed = true;
        return `${prefix}${specifier}.js${suffix}`;
      }

      return match;
    });

    if (changed) await fsp.writeFile(file, content);
  }
}

// Read version from package.json
const packageJson = JSON.parse(readFileSync('./package.json', 'utf-8'));
const VERSION = packageJson.version;

function buildWasm() {
  try {
    execSync('zig version', { stdio: 'ignore' });
    console.log('Building WASM kernels...');
    execSync('npx tsx scripts/build-wasm.ts', { stdio: 'inherit' });
    console.log('✓ WASM build complete\n');
  } catch {
    console.warn('⚠ Zig not found — using checked-in WASM binaries\n');
  }
}

const skipWasm = process.argv.includes('--skip-wasm');

async function buildAll() {
  // Build WASM kernels (optional — skipped if Zig not installed or --skip-wasm)
  if (!skipWasm) buildWasm();

  // Browser bundle (ESM — single file for CDN / <script type="module">)
  console.log('Building browser bundle...');
  await build({
    entryPoints: ['src/index.ts'],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    outfile: 'dist/numpy-ts.browser.js',
    sourcemap: false,
    minify: true,
    external: ['node:fs', 'node:fs/promises', 'node:module'],
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

  // Add .js extensions to relative imports so Node's ESM loader works
  await fixEsmExtensions('dist/esm', 'src');
  console.log('✓ Tree-shakeable ESM build complete');

  console.log('\n✓ All builds completed successfully!');
}

buildAll().catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
