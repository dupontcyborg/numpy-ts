/**
 * Generate PNG plots for the JS-library sweep from results/js-libs/latest-*.json.
 *   npx tsx benchmarks/js-libs/plots.ts
 * Renders to benchmarks/results/plots/js-libs/:
 *   - dtype-<dtype>.png        per-dtype geomean ops/sec by library (linear scale)
 *   - <dtype>-by-category.png  per-category grouped ops/sec (float64/float32/int32)
 *   - coverage.png             specs run per library per dtype
 * (complex dtypes + the numpy-ts-vs-NumPy view are covered separately.)
 * Uses chartjs-node-canvas from the repo-root node_modules.
 */

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(here, '../results/js-libs');
const outDir = resolve(here, '../results/plots/js-libs');
mkdirSync(outDir, { recursive: true });

const ORDER = ['float64', 'float32', 'float16', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'complex64', 'complex128'];

interface FileShape {
  dtype: string;
  summary: { totalSpecs: number; reference: { geomeanOpsPerSec: number }; libs: Array<{ name: string; coverage: number; geomeanOpsPerSec: number }> };
  results: Array<{ category: string; ratios: Record<string, number>; timings: Record<string, { ops_per_sec: number }> }>;
}

const data: FileShape[] = readdirSync(dir)
  .filter((f) => /^latest-.+\.json$/.test(f) && !/full|runtimes|pyodide|small|large/.test(f))
  .map((f) => JSON.parse(readFileSync(resolve(dir, f), 'utf-8')) as FileShape)
  // complex dtypes are numpy-ts-vs-NumPy only, covered by the separate NumPy comparison.
  .filter((d) => d.dtype && d.summary && !d.dtype.startsWith('complex'))
  .sort((a, b) => ORDER.indexOf(a.dtype) - ORDER.indexOf(b.dtype));

// Stable color per library.
const COLORS: Record<string, string> = {
  'numpy-ts': '#35A8B1',
  'numpy(pyodide)': '#4C72B0',
  'jax-js': '#DD8452',
  tfjs: '#C44E52',
  mathjs: '#8172B3',
  numeric: '#937860',
  'ml-matrix': '#DA8BC3',
  numjs: '#8C8C8C',
  stdlib: '#CCB974',
};
const color = (n: string) => COLORS[n] ?? '#999999';

const canvas = (w: number, h: number) => new ChartJSNodeCanvas({ width: w, height: h, backgroundColour: 'white' });

function geomean(v: number[]): number {
  const f = v.filter((x) => Number.isFinite(x) && x > 0);
  return f.length ? Math.exp(f.reduce((s, x) => s + Math.log(x), 0) / f.length) : Number.NaN;
}

async function render(c: ChartJSNodeCanvas, config: any, file: string) {
  const buf = await c.renderToBuffer(config as any);
  writeFileSync(resolve(outDir, file), buf);
  console.log('wrote', file);
}

async function perDtypeBars(d: FileShape) {
  const parts = [
    { name: 'numpy-ts', ops: d.summary.reference.geomeanOpsPerSec },
    ...d.summary.libs.map((l) => ({ name: l.name, ops: l.geomeanOpsPerSec })),
  ].filter((p) => Number.isFinite(p.ops) && p.ops > 0).sort((a, b) => b.ops - a.ops);

  await render(canvas(900, 360), {
    type: 'bar',
    data: {
      labels: parts.map((p) => p.name),
      datasets: [{
        label: 'geomean ops/sec',
        data: parts.map((p) => Math.round(p.ops)),
        backgroundColor: parts.map((p) => color(p.name)),
      }],
    },
    options: {
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        title: { display: true, text: `${d.dtype} — geomean ops/sec by library (${d.summary.totalSpecs} specs, default size)`, font: { size: 16 } },
      },
      scales: { x: { type: 'linear', title: { display: true, text: 'ops/sec' } } },
    },
  }, `dtype-${d.dtype}.png`);
}

async function byCategory(d: FileShape) {
  const cats = [...new Set(d.results.map((r) => r.category))].sort();
  const libs = ['numpy-ts', ...d.summary.libs.map((l) => l.name)];
  const datasets = libs.map((lib) => ({
    label: lib,
    backgroundColor: color(lib),
    data: cats.map((cat) => {
      const rows = d.results.filter((r) => r.category === cat);
      const g = geomean(rows.map((r) => r.timings[lib]?.ops_per_sec).filter((x): x is number => x != null));
      return Number.isFinite(g) ? Math.round(g) : null;
    }),
  }));
  await render(canvas(1100, 520), {
    type: 'bar',
    data: { labels: cats, datasets },
    options: {
      plugins: { title: { display: true, text: `${d.dtype} — geomean ops/sec by category`, font: { size: 16 } }, legend: { position: 'top' } },
      scales: { y: { type: 'linear', title: { display: true, text: 'ops/sec' } } },
    },
  }, `${d.dtype}-by-category.png`);
}

async function coverage() {
  const libs = [...new Set(data.flatMap((d) => d.summary.libs.map((l) => l.name)))];
  const datasets = libs.map((lib) => ({
    label: lib,
    backgroundColor: color(lib),
    data: data.map((d) => d.summary.libs.find((l) => l.name === lib)?.coverage ?? 0),
  }));
  await render(canvas(1200, 520), {
    type: 'bar',
    data: { labels: data.map((d) => d.dtype), datasets },
    options: {
      plugins: { title: { display: true, text: 'Coverage — specs run per library per dtype', font: { size: 16 } }, legend: { position: 'top' } },
      scales: { y: { title: { display: true, text: 'specs run' } } },
    },
  }, 'coverage.png');
}

async function main() {
  for (const d of data) await perDtypeBars(d);
  for (const dt of ['float64', 'float32', 'int32']) {
    const d = data.find((x) => x.dtype === dt);
    if (d) await byCategory(d);
  }
  await coverage();
  console.log(`\nAll plots in ${outDir}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
