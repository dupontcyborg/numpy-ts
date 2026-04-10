/**
 * JS vs WASM head-to-head chart generator.
 *
 * Reads js-only-full.json (--no-wasm run) and latest-full.json (WASM run),
 * joins on benchmark name, and generates:
 *   - plots/js-vs-wasm-h2h.png   — JS vs WASM vs NumPy ops/sec by category (log scale)
 *   - plots/js-vs-wasm-speedup.png — WASM/JS speedup ratio by category
 *   - plots/js-vs-wasm.html       — full HTML report with per-benchmark tables
 */

import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import * as fs from 'fs';
import * as path from 'path';
import type { BenchmarkComparison } from './types';

const RESULTS_DIR = path.resolve(__dirname, '../results');
const PLOTS_DIR = path.resolve(RESULTS_DIR, 'plots');

interface H2HEntry {
  name: string;
  category: string;
  numpy_ops: number;
  js_ops: number;
  wasm_ops: number;
  speedup: number; // wasm / js
  numpy_ratio_js: number; // js / numpy (how much slower JS is vs NumPy)
  numpy_ratio_wasm: number; // wasm / numpy
}

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1]! + s[m]!) / 2 : s[m]!;
}

function geoMean(arr: number[]): number {
  return Math.exp(arr.reduce((s, x) => s + Math.log(x), 0) / arr.length);
}

function formatOps(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return v.toFixed(0);
}

function formatRatio(r: number): string {
  if (r < 1.05) return '1.0x';
  return r.toFixed(1) + 'x';
}

function load(): H2HEntry[] {
  const jsReport = JSON.parse(
    fs.readFileSync(path.join(RESULTS_DIR, 'js-only-full.json'), 'utf-8')
  ) as { results: BenchmarkComparison[] };
  const wasmReport = JSON.parse(
    fs.readFileSync(path.join(RESULTS_DIR, 'latest-full.json'), 'utf-8')
  ) as { results: BenchmarkComparison[] };

  const jsMap = new Map(jsReport.results.map((r) => [r.name, r]));

  const entries: H2HEntry[] = [];
  for (const wasmEntry of wasmReport.results) {
    if (!wasmEntry.wasmUsed) continue;
    const jsEntry = jsMap.get(wasmEntry.name);
    if (!jsEntry) continue;

    const js_ops = jsEntry.numpyjs.ops_per_sec;
    const wasm_ops = wasmEntry.numpyjs.ops_per_sec;
    const numpy_ops = wasmEntry.numpy.ops_per_sec;
    if (!isFinite(js_ops) || js_ops <= 0 || !isFinite(wasm_ops) || wasm_ops <= 0) continue;

    entries.push({
      name: wasmEntry.name,
      category: wasmEntry.category,
      numpy_ops,
      js_ops,
      wasm_ops,
      speedup: wasm_ops / js_ops,
      numpy_ratio_js: js_ops > 0 && numpy_ops > 0 ? js_ops / numpy_ops : 0,
      numpy_ratio_wasm: wasm_ops > 0 && numpy_ops > 0 ? wasm_ops / numpy_ops : 0,
    });
  }
  return entries;
}

async function generateH2HChart(entries: H2HEntry[]): Promise<void> {
  // Group by category
  const catMap = new Map<string, { numpy: number[]; js: number[]; wasm: number[] }>();
  for (const e of entries) {
    if (!catMap.has(e.category)) catMap.set(e.category, { numpy: [], js: [], wasm: [] });
    const c = catMap.get(e.category)!;
    c.numpy.push(e.numpy_ops);
    c.js.push(e.js_ops);
    c.wasm.push(e.wasm_ops);
  }

  const categories = Array.from(catMap.keys());
  const numpyMedians = categories.map((c) => median(catMap.get(c)!.numpy));
  const jsMedians = categories.map((c) => median(catMap.get(c)!.js));
  const wasmMedians = categories.map((c) => median(catMap.get(c)!.wasm));

  const overallJs = median(entries.map((e) => e.js_ops));
  const overallWasm = median(entries.map((e) => e.wasm_ops));
  const overallNumpy = median(entries.map((e) => e.numpy_ops));
  const overallSpeedup = geoMean(entries.map((e) => e.speedup));

  const canvas = new ChartJSNodeCanvas({ width: 1200, height: 600, backgroundColour: 'white' });

  const config = {
    type: 'bar' as const,
    data: {
      labels: categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [
        {
          label: 'Python NumPy',
          data: numpyMedians,
          backgroundColor: 'rgba(255, 99, 132, 0.75)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
        },
        {
          label: 'numpy-ts (JS)',
          data: jsMedians,
          backgroundColor: 'rgba(255, 159, 64, 0.75)',
          borderColor: 'rgba(255, 159, 64, 1)',
          borderWidth: 2,
        },
        {
          label: 'numpy-ts (WASM)',
          data: wasmMedians,
          backgroundColor: 'rgba(75, 192, 192, 0.75)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: false,
      plugins: {
        title: {
          display: true,
          text: [
            'numpy-ts: JS vs WASM — Head to Head (WASM-accelerated ops only)',
            `Overall WASM speedup: ${overallSpeedup.toFixed(1)}x | WASM: ${formatOps(overallWasm)} ops/sec | JS: ${formatOps(overallJs)} ops/sec | NumPy: ${formatOps(overallNumpy)} ops/sec (median)`,
          ],
          font: { size: 18, weight: 'bold' as const },
          padding: { top: 10, bottom: 20 },
        },
        subtitle: {
          display: true,
          text: `${entries.length} WASM-accelerated benchmarks | Generated: ${new Date().toLocaleDateString()}`,
          font: { size: 12 },
          padding: { bottom: 10 },
        },
        legend: { display: true, position: 'top' as const, labels: { font: { size: 14 } } },
      },
      scales: {
        y: {
          type: 'logarithmic' as const,
          title: {
            display: true,
            text: 'Median ops/sec (log₁₀ scale, higher is better)',
            font: { size: 14, weight: 'bold' as const },
          },
          ticks: {
            font: { size: 11 },
            callback: function (value: any) {
              const v = Number(value);
              const log = Math.log10(v);
              return Math.abs(log - Math.round(log)) < 0.01 ? formatOps(v) : '';
            },
          },
          grid: {
            color: function (context: any) {
              const v = context.tick?.value;
              if (v == null) return 'rgba(0,0,0,0.1)';
              const log = Math.log10(v);
              return Math.abs(log - Math.round(log)) < 0.01 ? 'rgba(0,0,0,0.1)' : 'transparent';
            },
          },
        },
        x: {
          title: { display: true, text: 'Category', font: { size: 14, weight: 'bold' as const } },
          ticks: { font: { size: 12 } },
        },
      },
    },
    plugins: [
      {
        id: 'bg',
        beforeDraw: (chart: any) => {
          const ctx = chart.ctx;
          ctx.save();
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, chart.width, chart.height);
          ctx.restore();
        },
      },
    ],
  };

  const buf = await canvas.renderToBuffer(config as any);
  const outPath = path.join(PLOTS_DIR, 'js-vs-wasm-h2h.png');
  fs.writeFileSync(outPath, buf);
  console.log('H2H chart saved to:', outPath);
}

async function generateSpeedupChart(entries: H2HEntry[]): Promise<void> {
  // Group speedups by category
  const catMap = new Map<string, number[]>();
  for (const e of entries) {
    if (!catMap.has(e.category)) catMap.set(e.category, []);
    catMap.get(e.category)!.push(e.speedup);
  }

  const categories = Array.from(catMap.keys()).sort(
    (a, b) => geoMean(catMap.get(b)!) - geoMean(catMap.get(a)!)
  );
  const speedups = categories.map((c) => geoMean(catMap.get(c)!));
  const overall = geoMean(entries.map((e) => e.speedup));

  const canvas = new ChartJSNodeCanvas({ width: 1200, height: 600, backgroundColour: 'white' });

  const config = {
    type: 'bar' as const,
    data: {
      labels: categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [
        {
          label: 'WASM speedup over JS (geo mean)',
          data: speedups,
          backgroundColor: speedups.map((s) =>
            s >= 20
              ? 'rgba(46, 213, 115, 0.8)'
              : s >= 5
                ? 'rgba(75, 192, 192, 0.8)'
                : 'rgba(255, 195, 18, 0.8)'
          ),
          borderColor: speedups.map((s) =>
            s >= 20
              ? 'rgba(46, 213, 115, 1)'
              : s >= 5
                ? 'rgba(75, 192, 192, 1)'
                : 'rgba(255, 195, 18, 1)'
          ),
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: false,
      plugins: {
        title: {
          display: true,
          text: [
            'numpy-ts WASM Speedup over Pure JS — by Category',
            `Overall geo mean: ${overall.toFixed(1)}x faster with WASM (${entries.length} benchmarks)`,
          ],
          font: { size: 18, weight: 'bold' as const },
          padding: { top: 10, bottom: 20 },
        },
        subtitle: {
          display: true,
          text:
            'Green ≥20x · Teal ≥5x · Yellow <5x | Generated: ' + new Date().toLocaleDateString(),
          font: { size: 12 },
          padding: { bottom: 10 },
        },
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Speedup ratio (higher is better)',
            font: { size: 14, weight: 'bold' as const },
          },
          ticks: {
            font: { size: 12 },
            callback: (v: any) => v + 'x',
          },
        },
        x: {
          title: { display: true, text: 'Category', font: { size: 14, weight: 'bold' as const } },
          ticks: { font: { size: 12 } },
        },
      },
    },
    plugins: [
      {
        id: 'bg',
        beforeDraw: (chart: any) => {
          const ctx = chart.ctx;
          ctx.save();
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, chart.width, chart.height);
          ctx.restore();
        },
      },
      {
        id: 'valueLabels',
        afterDatasetsDraw: (chart: any) => {
          const { ctx, data } = chart;
          ctx.save();
          ctx.font = 'bold 13px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillStyle = '#333';
          chart.data.datasets[0].data.forEach((value: number, i: number) => {
            const meta = chart.getDatasetMeta(0);
            const bar = meta.data[i];
            ctx.fillText(value.toFixed(1) + 'x', bar.x, bar.y - 4);
          });
          ctx.restore();
        },
      },
    ],
  };

  const buf = await canvas.renderToBuffer(config as any);
  const outPath = path.join(PLOTS_DIR, 'js-vs-wasm-speedup.png');
  fs.writeFileSync(outPath, buf);
  console.log('Speedup chart saved to:', outPath);
}

function generateHTML(entries: H2HEntry[]): void {
  const catMap = new Map<string, H2HEntry[]>();
  for (const e of entries) {
    if (!catMap.has(e.category)) catMap.set(e.category, []);
    catMap.get(e.category)!.push(e);
  }

  const overallSpeedup = geoMean(entries.map((e) => e.speedup));
  const overallMedianSpeedup = median(entries.map((e) => e.speedup));
  const best = [...entries].sort((a, b) => b.speedup - a.speedup)[0]!;
  const worst = [...entries].sort((a, b) => a.speedup - b.speedup)[0]!;

  const catSummary = Array.from(catMap.entries()).map(([cat, es]) => ({
    cat,
    count: es.length,
    geo: geoMean(es.map((e) => e.speedup)),
    med: median(es.map((e) => e.speedup)),
  }));

  const catSpeedupData = catSummary.map((c) => c.geo);
  const catLabels = catSummary.map((c) => c.cat.charAt(0).toUpperCase() + c.cat.slice(1));

  let tables = '';
  for (const [cat, es] of catMap) {
    const sorted = [...es].sort((a, b) => b.speedup - a.speedup);
    tables += `
    <div class="category-section">
      <h2>${cat.toUpperCase()} <span class="cat-summary">${es.length} ops · geo mean ${geoMean(es.map((e) => e.speedup)).toFixed(1)}x speedup</span></h2>
      <table>
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>NumPy (ops/sec)</th>
            <th>JS (ops/sec)</th>
            <th>WASM (ops/sec)</th>
            <th>WASM Speedup</th>
            <th>JS vs NumPy</th>
            <th>WASM vs NumPy</th>
          </tr>
        </thead>
        <tbody>
          ${sorted
            .map((e) => {
              const speedClass =
                e.speedup >= 20 ? 'great' : e.speedup >= 5 ? 'good' : e.speedup >= 2 ? 'ok' : 'meh';
              const jsRatioClass =
                e.numpy_ratio_js >= 0.5 ? 'good' : e.numpy_ratio_js >= 0.2 ? 'ok' : 'bad';
              const wasmRatioClass =
                e.numpy_ratio_wasm >= 0.5 ? 'good' : e.numpy_ratio_wasm >= 0.2 ? 'ok' : 'bad';
              return `
          <tr>
            <td>${e.name}</td>
            <td>${formatOps(e.numpy_ops)}</td>
            <td>${formatOps(e.js_ops)}</td>
            <td>${formatOps(e.wasm_ops)}</td>
            <td><span class="speedup ${speedClass}">${e.speedup.toFixed(1)}x</span></td>
            <td><span class="ratio ${jsRatioClass}">${formatRatio(1 / e.numpy_ratio_js)}</span></td>
            <td><span class="ratio ${wasmRatioClass}">${formatRatio(1 / e.numpy_ratio_wasm)}</span></td>
          </tr>`;
            })
            .join('')}
        </tbody>
      </table>
    </div>`;
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>numpy-ts: JS vs WASM Head-to-Head</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; margin-bottom: 6px; font-size: 2.2em; }
    .subtitle { color: #7f8c8d; margin-bottom: 24px; font-size: 1.1em; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 36px; }
    .card { padding: 18px; border-radius: 8px; text-align: center; color: white; }
    .card.blue { background: linear-gradient(135deg, #667eea, #764ba2); }
    .card.green { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .card.orange { background: linear-gradient(135deg, #f7971e, #ffd200); color: #333; }
    .card.red { background: linear-gradient(135deg, #ee0979, #ff6a00); }
    .card h3 { font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; opacity: 0.9; }
    .card .val { font-size: 2.2em; font-weight: bold; }
    .card .sub { font-size: 0.8em; opacity: 0.8; margin-top: 2px; }
    .chart-container { margin: 32px 0; padding: 20px; background: white; border-radius: 8px; border: 1px solid #e0e0e0; }
    .chart-container h2 { margin-bottom: 16px; color: #2c3e50; }
    .category-section { margin: 36px 0; }
    .category-section h2 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; margin-bottom: 16px; font-size: 1.3em; }
    .cat-summary { font-size: 0.75em; font-weight: normal; color: #7f8c8d; margin-left: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
    th { background: #34495e; color: white; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
    tr:hover { background: #f8f9fa; }
    .speedup { font-weight: bold; padding: 2px 8px; border-radius: 4px; }
    .speedup.great { background: #d4f5e2; color: #065f46; }
    .speedup.good  { background: #d1fae5; color: #166534; }
    .speedup.ok    { background: #fef3c7; color: #92400e; }
    .speedup.meh   { background: #fee2e2; color: #991b1b; }
    .ratio { font-weight: 600; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
    .ratio.good { background: #d4edda; color: #155724; }
    .ratio.ok   { background: #fff3cd; color: #856404; }
    .ratio.bad  { background: #f8d7da; color: #721c24; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; text-align: center; color: #7f8c8d; font-size: 0.9em; }
  </style>
</head>
<body>
<div class="container">
  <h1>numpy-ts: JS vs WASM Head-to-Head</h1>
  <p class="subtitle">WASM-accelerated operations only — comparing pure JS fallback vs WASM/SIMD kernels</p>

  <div class="summary-grid">
    <div class="card blue"><h3>Geo Mean Speedup</h3><div class="val">${overallSpeedup.toFixed(1)}x</div><div class="sub">WASM over JS</div></div>
    <div class="card blue"><h3>Median Speedup</h3><div class="val">${overallMedianSpeedup.toFixed(1)}x</div><div class="sub">WASM over JS</div></div>
    <div class="card green"><h3>Best Speedup</h3><div class="val">${best.speedup.toFixed(0)}x</div><div class="sub">${best.name}</div></div>
    <div class="card orange"><h3>Worst Speedup</h3><div class="val">${worst.speedup.toFixed(1)}x</div><div class="sub">${worst.name}</div></div>
    <div class="card blue"><h3>Benchmarks</h3><div class="val">${entries.length}</div><div class="sub">WASM-accelerated ops</div></div>
  </div>

  <div class="chart-container">
    <h2>WASM Speedup over JS by Category</h2>
    <canvas id="speedupChart"></canvas>
  </div>

  <div class="chart-container">
    <h2>JS vs WASM vs NumPy — Median ops/sec by Category (log scale)</h2>
    <canvas id="h2hChart"></canvas>
  </div>

  ${tables}

  <div class="footer">
    <p>Generated by numpy-ts benchmark suite · ${new Date().toLocaleString()}</p>
    <p>JS = pure JavaScript fallback (--no-wasm) · WASM = SIMD/WASM kernel · NumPy = Python baseline</p>
  </div>
</div>

<script>
  const catLabels = ${JSON.stringify(catLabels)};
  const catSpeedups = ${JSON.stringify(catSpeedupData)};

  new Chart(document.getElementById('speedupChart'), {
    type: 'bar',
    data: {
      labels: catLabels,
      datasets: [{
        label: 'WASM speedup (geo mean)',
        data: catSpeedups,
        backgroundColor: catSpeedups.map(s => s >= 20 ? 'rgba(46,213,115,0.8)' : s >= 5 ? 'rgba(75,192,192,0.8)' : 'rgba(255,195,18,0.8)'),
        borderColor: catSpeedups.map(s => s >= 20 ? 'rgba(46,213,115,1)' : s >= 5 ? 'rgba(75,192,192,1)' : 'rgba(255,195,18,1)'),
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + 'x faster' } }
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: 'Speedup ratio' }, ticks: { callback: v => v + 'x' } }
      }
    }
  });

  const catMapH2H = ${JSON.stringify(
    Array.from(catMap.entries()).map(([cat, es]) => ({
      cat: cat.charAt(0).toUpperCase() + cat.slice(1),
      numpy: median(es.map((e) => e.numpy_ops)),
      js: median(es.map((e) => e.js_ops)),
      wasm: median(es.map((e) => e.wasm_ops)),
    }))
  )};

  new Chart(document.getElementById('h2hChart'), {
    type: 'bar',
    data: {
      labels: catMapH2H.map(c => c.cat),
      datasets: [
        { label: 'Python NumPy', data: catMapH2H.map(c => c.numpy), backgroundColor: 'rgba(255,99,132,0.75)', borderColor: 'rgba(255,99,132,1)', borderWidth: 2 },
        { label: 'numpy-ts JS',  data: catMapH2H.map(c => c.js),    backgroundColor: 'rgba(255,159,64,0.75)',  borderColor: 'rgba(255,159,64,1)',  borderWidth: 2 },
        { label: 'numpy-ts WASM',data: catMapH2H.map(c => c.wasm),  backgroundColor: 'rgba(75,192,192,0.75)',  borderColor: 'rgba(75,192,192,1)',  borderWidth: 2 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true, position: 'top' } },
      scales: {
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Median ops/sec (log scale)' },
          ticks: { callback: v => { const l = Math.log10(Number(v)); return Math.abs(l - Math.round(l)) < 0.01 ? (Number(v) >= 1e6 ? (Number(v)/1e6).toFixed(0)+'M' : Number(v) >= 1e3 ? (Number(v)/1e3).toFixed(0)+'K' : String(v)) : ''; } }
        }
      }
    }
  });
</script>
</body>
</html>`;

  const outPath = path.join(PLOTS_DIR, 'js-vs-wasm.html');
  fs.writeFileSync(outPath, html);
  console.log('HTML report saved to:', outPath);
}

function saveJSON(entries: H2HEntry[]): void {
  const catMap = new Map<string, H2HEntry[]>();
  for (const e of entries) {
    if (!catMap.has(e.category)) catMap.set(e.category, []);
    catMap.get(e.category)!.push(e);
  }

  const categorySummaries = Array.from(catMap.entries()).map(([cat, es]) => ({
    category: cat,
    count: es.length,
    geo_mean_speedup: geoMean(es.map((e) => e.speedup)),
    median_speedup: median(es.map((e) => e.speedup)),
    median_js_ops: median(es.map((e) => e.js_ops)),
    median_wasm_ops: median(es.map((e) => e.wasm_ops)),
    median_numpy_ops: median(es.map((e) => e.numpy_ops)),
  }));

  const output = {
    timestamp: new Date().toISOString(),
    summary: {
      total_benchmarks: entries.length,
      geo_mean_speedup: geoMean(entries.map((e) => e.speedup)),
      median_speedup: median(entries.map((e) => e.speedup)),
      best: entries.reduce((a, b) => (b.speedup > a.speedup ? b : a)),
      worst: entries.reduce((a, b) => (b.speedup < a.speedup ? b : a)),
    },
    categories: categorySummaries,
    benchmarks: entries,
  };

  const outPath = path.join(RESULTS_DIR, 'js-wasm-speedup.json');
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
  console.log('JSON saved to:', outPath);
}

async function main() {
  if (!fs.existsSync(PLOTS_DIR)) fs.mkdirSync(PLOTS_DIR, { recursive: true });

  const entries = load();
  console.log(`Loaded ${entries.length} WASM-accelerated entries`);

  const overall = geoMean(entries.map((e) => e.speedup));
  const med = median(entries.map((e) => e.speedup));
  console.log(`Overall: geo mean ${overall.toFixed(2)}x · median ${med.toFixed(2)}x`);

  await Promise.all([generateH2HChart(entries), generateSpeedupChart(entries)]);
  generateHTML(entries);
  saveJSON(entries);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
