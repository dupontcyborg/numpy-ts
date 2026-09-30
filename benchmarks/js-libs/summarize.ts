/**
 * Aggregate every results/js-libs/latest-<dtype>.json into one cross-dtype
 * summary (console + SUMMARY.md). Run after the full sweep:
 *   npx tsx benchmarks/js-libs/summarize.ts
 */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, '../results/js-libs');

interface LibSummary {
  name: string; version: string; coverage: number; geomeanOpsPerSec: number; geomeanRatio: number;
}
interface FileShape {
  dtype: string;
  summary: { totalSpecs: number; reference: { geomeanOpsPerSec: number; nativeDtypes: number }; libs: LibSummary[] };
  results: Array<{ name: string; category: string; operation: string; ratios: Record<string, number>; timings: Record<string, { ops_per_sec: number }> }>;
}

const ORDER = ['float64', 'float32', 'float16', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'complex64', 'complex128'];

const files = readdirSync(dir).filter((f) => /^latest-.+\.json$/.test(f) && !f.includes('full') && !f.includes('runtimes') && !f.includes('pyodide') && !f.includes('small') && !f.includes('large'));
const data: FileShape[] = files
  .map((f) => JSON.parse(readFileSync(resolve(dir, f), 'utf-8')) as FileShape)
  .filter((d) => d.dtype && d.summary)
  .sort((a, b) => ORDER.indexOf(a.dtype) - ORDER.indexOf(b.dtype));

const fmt = (x: number) => (!Number.isFinite(x) ? '·' : x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(0)}K` : x.toFixed(0));
const out: string[] = [];
const log = (s = '') => { out.push(s); console.log(s); };

log('# numpy-ts vs JS numerical libraries — full sweep summary\n');
log(`Dtypes covered: ${data.map((d) => d.dtype).join(', ')}`);
log('Headline = geomean ops/sec (higher = faster); ratio = lib / numpy-ts.\n');

// All library names seen.
const allLibs = [...new Set(data.flatMap((d) => d.summary.libs.map((l) => l.name)))];

// Per-dtype headline table.
log('## Per-dtype geomean ops/sec\n');
log('| dtype | specs | numpy-ts | ' + allLibs.join(' | ') + ' |');
log('|-' + '|-'.repeat(allLibs.length + 2) + '|');
for (const d of data) {
  const ref = fmt(d.summary.reference.geomeanOpsPerSec);
  const cells = allLibs.map((n) => {
    const l = d.summary.libs.find((x) => x.name === n);
    return l ? fmt(l.geomeanOpsPerSec) : '·';
  });
  log(`| ${d.dtype} | ${d.summary.totalSpecs} | ${ref} | ${cells.join(' | ')} |`);
}

// Coverage (specs run per lib per dtype).
log('\n## Coverage (specs run / total)\n');
log('| dtype | ' + allLibs.join(' | ') + ' |');
log('|-' + '|-'.repeat(allLibs.length + 1) + '|');
for (const d of data) {
  const cells = allLibs.map((n) => {
    const l = d.summary.libs.find((x) => x.name === n);
    return l ? `${l.coverage}/${d.summary.totalSpecs}` : '·';
  });
  log(`| ${d.dtype} | ${cells.join(' | ')} |`);
}

// vs numpy-ts (geomean ratio; <1 = faster than numpy-ts).
log('\n## Geomean ratio vs numpy-ts (×; <1 = faster than numpy-ts)\n');
log('| dtype | ' + allLibs.join(' | ') + ' |');
log('|-' + '|-'.repeat(allLibs.length + 1) + '|');
for (const d of data) {
  const cells = allLibs.map((n) => {
    const l = d.summary.libs.find((x) => x.name === n);
    return l && Number.isFinite(l.geomeanRatio) ? l.geomeanRatio.toFixed(2) : '·';
  });
  log(`| ${d.dtype} | ${cells.join(' | ')} |`);
}

// Where does a competitor beat numpy-ts? (per-spec ratio < 1)
log('\n## Specs where a competitor beats numpy-ts (ratio < 1)\n');
const beats: Record<string, { count: number; ex: Array<[string, string, number]> }> = {};
let totalCmp = 0, totalBeat = 0;
for (const d of data) {
  for (const r of d.results) {
    for (const [lib, ratio] of Object.entries(r.ratios)) {
      totalCmp++;
      if (ratio < 1) {
        totalBeat++;
        const b = (beats[lib] ??= { count: 0, ex: [] });
        b.count++;
        b.ex.push([d.dtype, r.name, ratio]);
      }
    }
  }
}
log(`numpy-ts is fastest on ${(((totalCmp - totalBeat) / totalCmp) * 100).toFixed(1)}% of all ${totalCmp} head-to-heads.\n`);
for (const [lib, b] of Object.entries(beats).sort((a, c) => c[1].count - a[1].count)) {
  const ex = b.ex.sort((a, c) => a[2] - c[2]).slice(0, 4).map(([dt, n, r]) => `${n} [${dt}] ${(1 / r).toFixed(1)}×`).join('; ');
  log(`- **${lib}**: ${b.count} specs — e.g. ${ex}`);
}

// numpy-ts vs numpy(pyodide) head-to-head (the gold-standard check).
log('\n## numpy-ts vs NumPy (Pyodide) — same-input, same-timer\n');
log('| dtype | numpy-ts ops/sec | numpy(pyodide) ops/sec | numpy-ts vs NumPy |');
log('|-|-|-|-|');
for (const d of data) {
  const py = d.summary.libs.find((l) => l.name === 'numpy(pyodide)');
  if (!py) continue;
  // ratio = NumPy.mean / numpy-ts.mean; >1 ⇒ numpy-ts faster by that factor.
  const r = py.geomeanRatio;
  const verdict = r >= 1 ? `${r.toFixed(2)}× faster` : `${(1 / r).toFixed(2)}× slower`;
  log(`| ${d.dtype} | ${fmt(d.summary.reference.geomeanOpsPerSec)} | ${fmt(py.geomeanOpsPerSec)} | ${verdict} |`);
}

writeFileSync(resolve(dir, 'SUMMARY.md'), out.join('\n') + '\n');
console.log(`\nwrote ${resolve(dir, 'SUMMARY.md')}`);
