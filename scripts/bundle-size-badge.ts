/**
 * Measures gzipped bundle sizes and rewrites the two "size" badges in README.md:
 *   - size (full):   the whole API bundled together (`export *` from the package)
 *   - size (single): a minimal tree-shaken core import (`array` + `add`)
 *
 * Mirrors scripts/compare-api-coverage.py: run manually after a build,
 * then commit the README change. Requires `pnpm run build` first
 * (reads the bundled output from dist/esm).
 *
 * Usage: tsx scripts/bundle-size-badge.ts
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { build } from 'esbuild';

const ROOT = join(fileURLToPath(import.meta.url), '..', '..');
const ESM_DIR = join(ROOT, 'dist/esm');
const README = join(ROOT, 'README.md');

const targets = [
  {
    label: 'full',
    desc: 'Full API (export *)',
    // Pull in the entire public surface, as a bundler would when nothing tree-shakes.
    sample: `export * from './index.js';\n`,
    badgeAlt: 'full bundle gzipped size',
  },
  {
    label: 'single',
    desc: 'Core import { array, add }',
    // A representative minimal import: array construction + one arithmetic op.
    sample: `import { array, add } from './core.js';\nconsole.log(array, add);\n`,
    badgeAlt: 'core import gzipped size',
  },
];

if (!existsSync(join(ESM_DIR, 'index.js')) || !existsSync(join(ESM_DIR, 'core.js'))) {
  console.error(`Missing built ESM in ${ESM_DIR}. Run \`pnpm run build\` first.`);
  process.exit(1);
}

// Bundle the same way a real consumer's bundler would (minified ESM).
async function gzipKb(sample: string): Promise<{ raw: number; gzip: number; kb: number }> {
  const result = await build({
    stdin: { contents: sample, resolveDir: ESM_DIR, sourcefile: 'sample.js', loader: 'js' },
    bundle: true,
    minify: true,
    format: 'esm',
    target: 'esnext',
    legalComments: 'none',
    external: ['node:*'],
    write: false,
  });
  const out = result.outputFiles[0]!.contents;
  const gzip = gzipSync(out).length;
  return { raw: out.length, gzip, kb: Math.round(gzip / 1024) };
}

let content = readFileSync(README, 'utf-8');

for (const t of targets) {
  const { raw, gzip, kb } = await gzipKb(t.sample);
  const value = `${kb}%20kB`;

  // ![<alt>](https://img.shields.io/badge/size%20(<label>)-<value>-blue)
  const badgeRe = new RegExp(
    `(!\\[${t.badgeAlt}\\]\\(https://img\\.shields\\.io/badge/size%20\\(${t.label}\\)-)[^-]*(-blue\\))`,
  );

  if (!badgeRe.test(content)) {
    console.error(`Could not find the "size (${t.label})" badge in README.md.`);
    process.exit(1);
  }

  content = content.replace(badgeRe, `$1${value}$2`);
  console.log(`${t.desc}: ${raw} B min, ${gzip} B gzip → size (${t.label}) = ${kb} kB`);
}

writeFileSync(README, content);
