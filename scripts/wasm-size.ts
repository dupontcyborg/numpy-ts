/**
 * Reports WASM binary sizes (raw and gzipped) per kernel and total.
 * Usage: tsx scripts/wasm-size.ts
 */

import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { gzipSync } from 'zlib';

const __filename = fileURLToPath(import.meta.url);
const BINS_DIR = join(__filename, '..', '..', 'src/common/wasm/bins');

interface Entry {
  name: string;
  raw: number;
  gzip: number;
  exports: number;
}

const files = readdirSync(BINS_DIR)
  .filter((f) => f.endsWith('.wasm.ts'))
  .sort();

const entries: Entry[] = [];

for (const file of files) {
  const src = readFileSync(join(BINS_DIR, file), 'utf-8');

  // Extract base64 WASM blob size (variable is named B64)
  const b64Match = src.match(/const B64\s*=\s*\n?\s*'([A-Za-z0-9+/=\s]+)'/s);
  if (!b64Match) continue;

  const b64 = b64Match[1]!.replace(/\s/g, '');
  const wasmBytes = Buffer.from(b64, 'base64');
  const gzipped = gzipSync(wasmBytes);

  // Count exports (each is `export function name(...)`)
  const exportCount = (src.match(/^export function /gm) || []).length;

  entries.push({
    name: file.replace('.wasm.ts', ''),
    raw: wasmBytes.length,
    gzip: gzipped.length,
    exports: exportCount,
  });
}

// Sort by raw size descending
entries.sort((a, b) => b.raw - a.raw);

const totalRaw = entries.reduce((s, e) => s + e.raw, 0);
const totalGzip = entries.reduce((s, e) => s + e.gzip, 0);
const totalExports = entries.reduce((s, e) => s + e.exports, 0);

function fmt(n: number): string {
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

function pad(s: string, w: number, right = false): string {
  return right ? s.padStart(w) : s.padEnd(w);
}

const nameW = Math.max(8, ...entries.map((e) => e.name.length));
const rawW = 10;
const gzW = 10;
const expW = 8;

console.log();
console.log(
  `${pad('Kernel', nameW)}  ${pad('Raw', rawW, true)}  ${pad('Gzipped', gzW, true)}  ${pad('Exports', expW, true)}`,
);
console.log(`${'─'.repeat(nameW)}  ${'─'.repeat(rawW)}  ${'─'.repeat(gzW)}  ${'─'.repeat(expW)}`);

for (const e of entries) {
  console.log(
    `${pad(e.name, nameW)}  ${pad(fmt(e.raw), rawW, true)}  ${pad(fmt(e.gzip), gzW, true)}  ${pad(String(e.exports), expW, true)}`,
  );
}

console.log(`${'─'.repeat(nameW)}  ${'─'.repeat(rawW)}  ${'─'.repeat(gzW)}  ${'─'.repeat(expW)}`);
console.log(
  `${pad('TOTAL', nameW)}  ${pad(fmt(totalRaw), rawW, true)}  ${pad(fmt(totalGzip), gzW, true)}  ${pad(String(totalExports), expW, true)}`,
);
console.log();
console.log(`${entries.length} kernels, ${totalExports} exports`);
console.log(`Compression ratio: ${((1 - totalGzip / totalRaw) * 100).toFixed(1)}%`);
console.log();
