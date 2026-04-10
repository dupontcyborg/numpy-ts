/**
 * Detect dead (unused) Zig functions across three categories:
 *
 *  1. Private `fn`  — not called anywhere else in the same .zig file
 *  2. `pub fn`      — not used in any .zig file (including its own)
 *  3. `export fn`   — re-exported through *.wasm.ts (auto-gen) but the
 *                     TS wrapper function is never imported/used elsewhere
 *
 * Usage: npx tsx scripts/zig-dead-code.ts [--verbose] [--list]
 */

import { readFileSync, readdirSync, existsSync } from 'fs';
import { join, basename } from 'path';

// --- Config ---
const ZIG_DIR = 'src/common/wasm/zig';
const BINS_DIR = 'src/common/wasm/bins';
const WASM_DIR = 'src/common/wasm';
const TS_ROOTS = ['src', 'tests', 'benchmarks'];
// Utility .zig files that only export pub symbols for other .zig files
const SKIP_FILES = new Set(['simd.zig', 'alloc.zig']);

const verbose = process.argv.includes('--verbose');
const list = process.argv.includes('--list');

// --- Helpers ---
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;

// --- Collect all .zig sources ---
const zigFiles = readdirSync(ZIG_DIR)
  .filter((f) => f.endsWith('.zig'))
  .sort();

const zigSources = new Map<string, string>();
for (const file of zigFiles) {
  zigSources.set(file, readFileSync(join(ZIG_DIR, file), 'utf-8'));
}

// --- Collect all TypeScript sources for export fn usage checking ---
function collectTsFiles(roots: string[]): string[] {
  const result: string[] = [];
  function walk(dir: string) {
    if (!existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory() && entry.name !== 'node_modules') {
        walk(full);
      } else if (entry.isFile() && entry.name.endsWith('.ts') && !entry.name.endsWith('.wasm.ts')) {
        result.push(full);
      }
    }
  }
  roots.forEach(walk);
  return result;
}

const allTsFiles = collectTsFiles(TS_ROOTS);
const tsSourceCache = new Map<string, string>();
function getTsSource(path: string): string {
  let src = tsSourceCache.get(path);
  if (src === undefined) {
    src = readFileSync(path, 'utf-8');
    tsSourceCache.set(path, src);
  }
  return src;
}

// --- 1. Private fn: unused in same file ---

interface DeadFn {
  file: string;
  name: string;
  kind: 'fn' | 'pub fn' | 'export fn';
  reason: string;
}

const deadFns: DeadFn[] = [];

for (const [file, src] of zigSources) {
  if (SKIP_FILES.has(file)) continue;

  // Extract private fn declarations (not pub, not export)
  // Match lines like: `fn foo(` or `inline fn foo(`
  const privateFns = [...src.matchAll(/^(?:inline )?fn (\w+)\s*\(/gm)].map((m) => m[1]);

  for (const fn of privateFns) {
    // Count occurrences in the file — declaration itself is 1, so need ≥2
    const re = new RegExp(`\\b${fn}\\b`, 'g');
    const matches = src.match(re);
    if (!matches || matches.length < 2) {
      deadFns.push({ file, name: fn, kind: 'fn', reason: 'unused in same file' });
    }
  }
}

// --- 2. pub fn: unused across all .zig files ---

for (const [file, src] of zigSources) {
  if (SKIP_FILES.has(file)) continue;

  const pubFns = [...src.matchAll(/^pub (?:inline )?fn (\w+)\s*\(/gm)].map((m) => m[1]);

  for (const fn of pubFns) {
    let used = false;
    for (const [otherFile, otherSrc] of zigSources) {
      // In the defining file: need ≥2 occurrences (decl + usage)
      // In other files: need ≥1 occurrence
      const re = new RegExp(`\\b${fn}\\b`, 'g');
      const matches = otherSrc.match(re);
      if (otherFile === file) {
        if (matches && matches.length >= 2) {
          used = true;
          break;
        }
      } else {
        if (matches && matches.length >= 1) {
          used = true;
          break;
        }
      }
    }
    if (!used) {
      deadFns.push({ file, name: fn, kind: 'pub fn', reason: 'unused across all .zig files' });
    }
  }
}

// --- 3. export fn: TS wrapper function unused outside *.wasm.ts ---

// For each .zig file with export fns, check if the corresponding TS wrapper
// re-exports are actually imported/used somewhere in the codebase.

// Build a map: wasm module base name -> set of export fn names
const exportFnsByModule = new Map<string, string[]>();
for (const [file, src] of zigSources) {
  if (SKIP_FILES.has(file)) continue;
  const exportFns = [...src.matchAll(/^export fn (\w+)/gm)].map((m) => m[1]);
  if (exportFns.length > 0) {
    const mod = file.replace('.zig', '');
    exportFnsByModule.set(mod, exportFns);
  }
}

// Build index: for each zig module, find which wrapper .ts files import from
// its .wasm.ts binding. Wrappers may bundle multiple zig modules (e.g. dot.ts
// imports from both dot_float.wasm and dot_int.wasm).
const wrapperTsFiles = readdirSync(WASM_DIR)
  .filter((f) => f.endsWith('.ts') && !f.endsWith('.wasm.ts'))
  .map((f) => join(WASM_DIR, f));

// Cache wrapper sources
const wrapperSources = new Map<string, string>();
for (const wp of wrapperTsFiles) {
  wrapperSources.set(wp, readFileSync(wp, 'utf-8'));
}

// Map: zig module name -> list of wrapper paths that import from its .wasm.ts
function findWrappers(mod: string): string[] {
  const results: string[] = [];
  // Match imports from ./bins/<mod>.wasm or ./bins/<mod>-relaxed.wasm etc.
  const importPattern = new RegExp(`from\\s*['"]\\.\\/bins\\/${mod}(?:-[^'"]*)?[.]wasm['"]`);
  for (const [wp, src] of wrapperSources) {
    if (importPattern.test(src)) {
      results.push(wp);
    }
  }
  return results;
}

for (const [mod, exportFns] of exportFnsByModule) {
  const wrappers = findWrappers(mod);
  if (wrappers.length === 0) {
    for (const fn of exportFns) {
      deadFns.push({
        file: `${mod}.zig`,
        name: fn,
        kind: 'export fn',
        reason: 'no TS wrapper imports this module',
      });
    }
    continue;
  }

  // Collect all wrapper sources for this module
  const combinedWrapperSrc = wrappers.map((wp) => wrapperSources.get(wp)!);
  const wrapperNames = wrappers.map((wp) => basename(wp));

  for (const fn of exportFns) {
    // Check if fn is referenced in any wrapper body (outside import lines).
    // Handles named imports (direct use), namespace imports accessed via
    // helper functions (e.g. float().dot_f64(...)), and any other pattern.
    const fnRe = new RegExp(`\\b${fn}\\b`);
    let usedInWrapper = false;

    for (const src of combinedWrapperSrc) {
      const withoutImports = src.replace(/^import\s+.*$/gm, '');
      if (fnRe.test(withoutImports)) {
        usedInWrapper = true;
        break;
      }
    }

    if (!usedInWrapper) {
      deadFns.push({
        file: `${mod}.zig`,
        name: fn,
        kind: 'export fn',
        reason: `not used in wrapper${wrapperNames.length === 1 ? ` ${wrapperNames[0]}` : 's'}`,
      });
    }
  }
}

// Now check wrapper-level: are the wrapper's exported functions used anywhere?
// This catches cases where the wrapper uses the fn internally but the wrapper
// export itself is dead.
interface DeadExport {
  wrapperFile: string;
  exportName: string;
}

const deadExports: DeadExport[] = [];
const checkedWrappers = new Set<string>();

for (const [mod] of exportFnsByModule) {
  const wrappers = findWrappers(mod);
  for (const wrapperPath of wrappers) {
    if (checkedWrappers.has(wrapperPath)) continue;
    checkedWrappers.add(wrapperPath);

    const wrapperSrc = wrapperSources.get(wrapperPath)!;

    // Find all exported function names from the wrapper
    const wrapperExportNames = [...wrapperSrc.matchAll(/^export function (\w+)/gm)].map(
      (m) => m[1]
    );

    for (const expFn of wrapperExportNames) {
      const importRe = new RegExp(`\\b${expFn}\\b`);
      let found = false;

      for (const tsFile of allTsFiles) {
        if (tsFile === wrapperPath) continue;
        const tsSrc = getTsSource(tsFile);
        if (importRe.test(tsSrc)) {
          found = true;
          break;
        }
      }

      if (!found) {
        deadExports.push({
          wrapperFile: basename(wrapperPath),
          exportName: expFn,
        });
      }
    }
  }
}

// --- Output ---
const fileCol = 30;
const nameCol = 35;
const kindCol = 12;

console.log();
console.log(bold(' Zig dead code analysis'));

// Group by kind
const privateDead = deadFns.filter((d) => d.kind === 'fn');
const pubDead = deadFns.filter((d) => d.kind === 'pub fn');
const exportDead = deadFns.filter((d) => d.kind === 'export fn');

const sep = dim('-'.repeat(fileCol + nameCol + kindCol + 30));

function printSection(title: string, items: DeadFn[]) {
  console.log();
  console.log(bold(`  ${title}`));
  if (items.length === 0) {
    console.log(green('  No dead code found'));
    return;
  }
  console.log(sep);
  console.log('  ' + dim('File'.padEnd(fileCol)) + dim('Function'.padEnd(nameCol)) + dim('Reason'));
  console.log(sep);
  for (const d of items) {
    console.log('  ' + d.file.padEnd(fileCol) + yellow(d.name.padEnd(nameCol)) + dim(d.reason));
  }
}

printSection('Private fn — unused in same file', privateDead);
printSection('pub fn — unused across all .zig files', pubDead);
printSection('export fn — unused in TypeScript layer', exportDead);

if (deadExports.length > 0) {
  console.log();
  console.log(bold('  Wrapper exports — not imported anywhere in codebase'));
  console.log(sep);
  console.log('  ' + dim('Wrapper'.padEnd(fileCol)) + dim('Export'.padEnd(nameCol)));
  console.log(sep);
  for (const d of deadExports) {
    console.log('  ' + d.wrapperFile.padEnd(fileCol) + yellow(d.exportName));
  }
}

const totalDead = privateDead.length + pubDead.length + exportDead.length + deadExports.length;

console.log();
console.log(sep);
console.log(
  bold('  Summary: ') +
    (totalDead === 0
      ? green('No dead code detected')
      : red(`${totalDead} dead code issue${totalDead > 1 ? 's' : ''} found`))
);

if (list && totalDead > 0) {
  console.log();
  console.log(bold('  All dead code:'));
  for (const d of deadFns) {
    console.log(`    ${dim(d.file + ':')} ${d.kind} ${d.name}`);
  }
  for (const d of deadExports) {
    console.log(`    ${dim(d.wrapperFile + ':')} export ${d.exportName}`);
  }
}

console.log();

if (totalDead > 0) {
  process.exit(1);
}
