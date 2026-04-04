/**
 * Benchmark visualization - HTML report generation
 */

import * as fs from 'fs';
import * as path from 'path';
import type {
  BenchmarkReport,
  BenchmarkComparison,
  MultiRuntimeReport,
  RuntimeComparison,
} from './types';
import {
  groupByCategory,
  getCategorySummaries,
  getDtypeSummaries,
  getMultiRuntimeDtypeSummaries,
  formatDuration,
  formatRatio,
  formatOpsPerSec,
  groupMultiRuntimeByCategory,
} from './analysis';

const DTYPE_COLORS: Record<string, { bg: string; text: string }> = {
  float64: { bg: '#dbeafe', text: '#1e40af' },
  float32: { bg: '#e0f2fe', text: '#0369a1' },
  float16: { bg: '#e0f7fa', text: '#00695c' },
  complex128: { bg: '#ede9fe', text: '#6d28d9' },
  complex64: { bg: '#f3e8ff', text: '#7e22ce' },
  int64: { bg: '#fef3c7', text: '#92400e' },
  int32: { bg: '#fff7ed', text: '#9a3412' },
  int16: { bg: '#ffedd5', text: '#9a3412' },
  int8: { bg: '#fee2e2', text: '#991b1b' },
  uint64: { bg: '#d1fae5', text: '#065f46' },
  uint32: { bg: '#dcfce7', text: '#166534' },
  uint16: { bg: '#f0fdf4', text: '#166534' },
  uint8: { bg: '#ecfdf5', text: '#065f46' },
  bool: { bg: '#f1f5f9', text: '#475569' },
};

const DTYPE_RE =
  /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;

function formatBenchmarkName(name: string, wasmUsed?: boolean): string {
  const m = name.match(DTYPE_RE);
  const dtype = m ? m[1]! : 'float64';
  const baseName = m ? name.slice(0, -m[0].length) : name;
  const colors = DTYPE_COLORS[dtype] ?? DTYPE_COLORS['float64']!;
  const opacity = m ? '1' : '0.55'; // implicit float64 is dimmer
  const dtypeBadge = `<span style="display:inline-block;font-size:0.75em;font-weight:600;padding:1px 6px;border-radius:3px;background:${colors.bg};color:${colors.text};opacity:${opacity};vertical-align:middle">${dtype}</span>`;
  const wasmBadge = wasmUsed
    ? ` <span style="display:inline-block;font-size:0.75em;font-weight:700;padding:1px 6px;border-radius:3px;background:#e0e7ff;color:#3730a3;vertical-align:middle" title="Accelerated with a WASM/SIMD kernel">WASM</span>`
    : '';
  return `${baseName} ${dtypeBadge}${wasmBadge}`;
}

export function generateHTMLReport(report: BenchmarkReport, outputPath: string): void {
  const html = createHTML(report, outputPath);
  fs.writeFileSync(outputPath, html, 'utf-8');
}

function createHTML(report: BenchmarkReport, outputPath: string = ''): string {
  const { timestamp, environment, results, summary } = report;
  const groups = groupByCategory(results);
  const categorySummaries = getCategorySummaries(results);
  const dtypeSummaries = getDtypeSummaries(results);

  // Prepare data for charts
  const categories = Array.from(groups.keys());
  const categoryAvgSlowdowns = categories.map((cat) => categorySummaries.get(cat)!.geo_mean);

  // All benchmark names and ratios for detailed chart
  const benchmarkNames = results.map((r) => r.name);
  const benchmarkRatios = results.map((r) => r.ratio);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NumPy vs numpy-ts Benchmark Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      background: #f5f5f5;
      padding: 20px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    h1 {
      color: #2c3e50;
      margin-bottom: 10px;
      font-size: 2.5em;
    }

    .subtitle {
      color: #7f8c8d;
      margin-bottom: 30px;
      font-size: 1.1em;
    }

    .meta {
      background: #ecf0f1;
      padding: 15px;
      border-radius: 5px;
      margin-bottom: 30px;
      font-family: monospace;
      font-size: 0.9em;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }

    .summary-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }

    .summary-card.best {
      background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .summary-card.worst {
      background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }

    .summary-card h3 {
      font-size: 0.9em;
      opacity: 0.9;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .summary-card .value {
      font-size: 2.5em;
      font-weight: bold;
    }

    .chart-container {
      margin: 40px 0;
      padding: 20px;
      background: white;
      border-radius: 8px;
      border: 1px solid #e0e0e0;
    }

    .chart-container h2 {
      margin-bottom: 20px;
      color: #2c3e50;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }

    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e0e0e0;
    }

    th {
      background: #34495e;
      color: white;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.85em;
      letter-spacing: 0.5px;
    }

    tr:hover {
      background: #f8f9fa;
    }

    .ratio {
      font-weight: bold;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .ratio.good {
      background: #d4edda;
      color: #155724;
    }

    .ratio.ok {
      background: #fff3cd;
      color: #856404;
    }

    .ratio.bad {
      background: #f8d7da;
      color: #721c24;
    }

    .category-section {
      margin: 40px 0;
    }

    .category-section h2 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 2px solid #e0e0e0;
      text-align: center;
      color: #7f8c8d;
      font-size: 0.9em;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚀 NumPy vs numpy-ts Benchmark Results</h1>
    <p class="subtitle">Performance comparison of numpy-ts against ${environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy'}</p>

    <div class="meta">
      <div><strong>Timestamp:</strong> ${new Date(timestamp).toLocaleString()}</div>
      ${environment.machine ? `<div><strong>Machine:</strong> ${environment.machine}</div>` : ''}
      <div><strong>Node:</strong> ${environment.node_version}</div>
      ${environment.python_version ? `<div><strong>Python:</strong> ${environment.python_version}</div>` : ''}
      ${environment.numpy_version ? `<div><strong>NumPy:</strong> ${environment.numpy_version}</div>` : ''}
      <div><strong>numpy-ts:</strong> ${environment.numpyjs_version}</div>
      <div><strong>Total Benchmarks:</strong> ${summary.total_benchmarks}</div>
    </div>

    <div class="summary-grid">
      <div class="summary-card">
        <h3>Overall Slowdown</h3>
        <div class="value">${formatRatio(summary.geo_mean)}</div>
      </div>
      <div class="summary-card">
        <h3>Median Slowdown</h3>
        <div class="value">${formatRatio(summary.median_slowdown)}</div>
      </div>
      <div class="summary-card best">
        <h3>Best Case</h3>
        <div class="value">${formatRatio(summary.best_case)}</div>
      </div>
      <div class="summary-card worst">
        <h3>Worst Case</h3>
        <div class="value">${formatRatio(summary.worst_case)}</div>
      </div>
    </div>

    ${generateDtypeBreakdown(dtypeSummaries)}

    <div class="chart-container">
      <h2>📊 Overall Slowdown by Category</h2>
      <canvas id="categoryChart"></canvas>
    </div>

    <div class="chart-container">
      <h2>📈 Detailed Results (All Benchmarks)</h2>
      <canvas id="detailedChart"></canvas>
    </div>

    ${generateCategoryTables(groups)}

    <div class="footer">
      <p>Generated by numpy-ts Benchmark Suite</p>
      <p>Lower ratios are better (closer to NumPy performance)</p>
    </div>
  </div>

  <script>
    // Category chart
    new Chart(document.getElementById('categoryChart'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(categories)},
        datasets: [{
          label: 'Overall Slowdown (x times slower than NumPy)',
          data: ${JSON.stringify(categoryAvgSlowdowns)},
          backgroundColor: 'rgba(102, 126, 234, 0.8)',
          borderColor: 'rgba(102, 126, 234, 1)',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Slowdown Ratio'
            }
          }
        }
      }
    });

    // Detailed chart
    new Chart(document.getElementById('detailedChart'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(benchmarkNames)},
        datasets: [{
          label: 'Slowdown Ratio (x times slower)',
          data: ${JSON.stringify(benchmarkRatios)},
          backgroundColor: ${JSON.stringify(
            benchmarkRatios.map((r: number) =>
              r < 2
                ? 'rgba(46, 213, 115, 0.8)'
                : r < 5
                  ? 'rgba(255, 195, 18, 0.8)'
                  : 'rgba(235, 77, 75, 0.8)'
            )
          )},
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        indexAxis: 'y',
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Slowdown Ratio'
            }
          }
        }
      }
    });

    ${generateDtypeChartScript(dtypeSummaries)}
  </script>
</body>
</html>`;
}

// --- Multi-runtime HTML report ---

export function generateMultiRuntimeHTMLReport(
  report: MultiRuntimeReport,
  outputPath: string
): void {
  const html = createMultiRuntimeHTML(report);
  fs.writeFileSync(outputPath, html, 'utf-8');
}

function createMultiRuntimeHTML(report: MultiRuntimeReport): string {
  const { timestamp, environment, results, summaries } = report;
  const groups = groupMultiRuntimeByCategory(results);
  const runtimeNames = Object.keys(environment.runtimes);

  // Colors per runtime
  const runtimeColors: Record<string, string> = {
    node: 'rgba(54, 162, 235, 0.8)',
    deno: 'rgba(75, 192, 192, 0.8)',
    bun: 'rgba(255, 159, 64, 0.8)',
  };

  // Compute per-category avg slowdown for each runtime
  const categories = Array.from(groups.keys());
  const datasets = runtimeNames.map((rt) => {
    const data = categories.map((cat) => {
      const items = groups.get(cat)!;
      const ratios = items
        .filter((item) => item.runtimes[rt])
        .map((item) => item.runtimes[rt]!.ratio);
      return ratios.length > 0 ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
    });
    return {
      label: rt,
      data,
      backgroundColor: runtimeColors[rt] || 'rgba(153, 102, 255, 0.8)',
    };
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NumPy vs numpy-ts Multi-Runtime Benchmark Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }
    .subtitle { color: #7f8c8d; margin-bottom: 30px; font-size: 1.1em; }
    .meta { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 30px; font-family: monospace; font-size: 0.9em; }
    .runtime-cards { display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap; }
    .runtime-card { background: #34495e; color: white; padding: 12px 20px; border-radius: 8px; flex: 1; min-width: 200px; }
    .runtime-card h3 { font-size: 1.1em; margin-bottom: 5px; text-transform: capitalize; }
    .runtime-card .version { opacity: 0.8; font-size: 0.9em; }
    .runtime-card.node { border-left: 4px solid #36a2eb; }
    .runtime-card.deno { border-left: 4px solid #4bc0c0; }
    .runtime-card.bun { border-left: 4px solid #ff9f40; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }
    .summary-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
    .summary-card h3 { font-size: 0.85em; opacity: 0.9; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .summary-card .runtime-label { font-size: 0.8em; opacity: 0.7; margin-bottom: 8px; text-transform: capitalize; }
    .summary-card .value { font-size: 2em; font-weight: bold; }
    .chart-container { margin: 40px 0; padding: 20px; background: white; border-radius: 8px; border: 1px solid #e0e0e0; }
    .chart-container h2 { margin-bottom: 20px; color: #2c3e50; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; font-size: 0.9em; }
    th { background: #34495e; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; }
    tr:hover { background: #f8f9fa; }
    .ratio { font-weight: bold; padding: 3px 6px; border-radius: 4px; }
    .ratio.good { background: #d4edda; color: #155724; }
    .ratio.ok { background: #fff3cd; color: #856404; }
    .ratio.bad { background: #f8d7da; color: #721c24; }
    .ratio.best { outline: 2px solid #28a745; }
    .category-section { margin: 40px 0; }
    .category-section h2 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; text-align: center; color: #7f8c8d; font-size: 0.9em; }
  </style>
</head>
<body>
  <div class="container">
    <h1>NumPy vs numpy-ts Multi-Runtime Benchmarks</h1>
    <p class="subtitle">Performance comparison across Node.js, Deno, and Bun (baseline: ${environment.baseline === 'pyodide' ? 'Pyodide NumPy/WASM' : 'Python NumPy'})</p>

    <div class="meta">
      <div><strong>Timestamp:</strong> ${new Date(timestamp).toLocaleString()}</div>
      ${environment.machine ? `<div><strong>Machine:</strong> ${environment.machine}</div>` : ''}
      ${environment.python_version ? `<div><strong>Python:</strong> ${environment.python_version}</div>` : ''}
      ${environment.numpy_version ? `<div><strong>NumPy:</strong> ${environment.numpy_version}</div>` : ''}
      <div><strong>numpy-ts:</strong> ${environment.numpyjs_version}</div>
      <div><strong>Total Benchmarks:</strong> ${results.length}</div>
    </div>

    <div class="runtime-cards">
      ${runtimeNames
        .map(
          (rt) => `
        <div class="runtime-card ${rt}">
          <h3>${rt}</h3>
          <div class="version">v${environment.runtimes[rt]}</div>
          ${summaries[rt] ? `<div class="version">Overall: ${formatRatio(summaries[rt]!.geo_mean)} | Median: ${formatRatio(summaries[rt]!.median_slowdown)}</div>` : ''}
        </div>
      `
        )
        .join('')}
    </div>

    <div class="summary-grid">
      ${runtimeNames
        .map((rt) => {
          const s = summaries[rt];
          if (!s) return '';
          return `
        <div class="summary-card">
          <h3>Overall Slowdown</h3>
          <div class="runtime-label">${rt}</div>
          <div class="value">${formatRatio(s.geo_mean)}</div>
        </div>`;
        })
        .join('')}
    </div>

    ${runtimeNames
      .map((rt) => {
        const dtypeSums = getMultiRuntimeDtypeSummaries(results, rt);
        if (dtypeSums.size <= 1) return '';
        return generateDtypeBreakdownMultiRuntime(dtypeSums, rt);
      })
      .join('')}

    <div class="chart-container">
      <h2>Overall Slowdown by Category (Grouped)</h2>
      <canvas id="categoryChart"></canvas>
    </div>

    ${generateMultiRuntimeCategoryTables(groups, runtimeNames)}

    <div class="footer">
      <p>Generated by numpy-ts Benchmark Suite</p>
      <p>Lower ratios are better (closer to NumPy performance)</p>
    </div>
  </div>

  <script>
    new Chart(document.getElementById('categoryChart'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)))},
        datasets: ${JSON.stringify(
          datasets.map((ds) => ({
            label: ds.label.charAt(0).toUpperCase() + ds.label.slice(1),
            data: ds.data,
            backgroundColor: ds.backgroundColor,
            borderWidth: 1,
          }))
        )}
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Slowdown Ratio (lower is better)' }
          }
        }
      }
    });

    ${runtimeNames
      .map((rt) => {
        const dtypeSums = getMultiRuntimeDtypeSummaries(results, rt);
        return generateDtypeChartScript(dtypeSums, `dtypeChart_${rt}`);
      })
      .join('\n')}
  </script>
</body>
</html>`;
}

function generateMultiRuntimeCategoryTables(
  groups: Map<string, RuntimeComparison[]>,
  runtimeNames: string[]
): string {
  let html = '';

  for (const [category, items] of groups) {
    html += `
    <div class="category-section">
      <h2>${category.toUpperCase()}</h2>
      <table>
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>NumPy (ms)</th>
            ${runtimeNames.map((rt) => `<th>${rt} (ms)</th><th>${rt} ratio</th>`).join('')}
          </tr>
        </thead>
        <tbody>`;

    for (const item of items) {
      // Find best (lowest) ratio
      let bestRatio = Infinity;
      let bestRuntime = '';
      for (const rt of runtimeNames) {
        const entry = item.runtimes[rt];
        if (entry && entry.ratio < bestRatio) {
          bestRatio = entry.ratio;
          bestRuntime = rt;
        }
      }

      html += `
          <tr>
            <td>${formatBenchmarkName(item.name, item.wasmUsed)}</td>
            <td>${formatDuration(item.numpy.mean_ms)}</td>`;

      for (const rt of runtimeNames) {
        const entry = item.runtimes[rt];
        if (entry) {
          const ratioClass = entry.ratio < 2 ? 'good' : entry.ratio < 5 ? 'ok' : 'bad';
          const bestClass = rt === bestRuntime && runtimeNames.length > 1 ? ' best' : '';
          html += `
            <td>${formatDuration(entry.timing.mean_ms)}</td>
            <td><span class="ratio ${ratioClass}${bestClass}">${formatRatio(entry.ratio)}</span></td>`;
        } else {
          html += `<td>-</td><td>-</td>`;
        }
      }

      html += `</tr>`;
    }

    html += `
        </tbody>
      </table>
    </div>`;
  }

  return html;
}

function generateDtypeBreakdown(
  dtypeSummaries: Map<string, { geo_mean: number; median_slowdown: number; count: number }>
): string {
  if (dtypeSummaries.size <= 1) return ''; // Only float64 — nothing interesting to show

  let html = `
    <div class="chart-container">
      <h2>Slowdown by DType</h2>
      <div style="display:flex;gap:30px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:1;min-width:400px">
          <canvas id="dtypeChart"></canvas>
        </div>
        <div style="flex:0 0 auto">
          <table style="margin:0">
            <thead>
              <tr>
                <th>DType</th>
                <th>Count</th>
                <th>Avg Slowdown</th>
                <th>Median Slowdown</th>
              </tr>
            </thead>
            <tbody>`;

  for (const [dtype, data] of dtypeSummaries) {
    const colors = DTYPE_COLORS[dtype] ?? DTYPE_COLORS['float64']!;
    const ratioClass = data.geo_mean < 2 ? 'good' : data.geo_mean < 5 ? 'ok' : 'bad';
    html += `
              <tr>
                <td><span style="display:inline-block;font-size:0.85em;font-weight:600;padding:2px 8px;border-radius:3px;background:${colors.bg};color:${colors.text}">${dtype}</span></td>
                <td>${data.count}</td>
                <td><span class="ratio ${ratioClass}">${formatRatio(data.geo_mean)}</span></td>
                <td>${formatRatio(data.median_slowdown)}</td>
              </tr>`;
  }

  html += `
            </tbody>
          </table>
        </div>
      </div>
    </div>`;

  return html;
}

function generateDtypeBreakdownMultiRuntime(
  dtypeSummaries: Map<string, { geo_mean: number; median_slowdown: number; count: number }>,
  runtimeName: string
): string {
  if (dtypeSummaries.size <= 1) return '';

  const canvasId = `dtypeChart_${runtimeName}`;
  let html = `
    <div class="chart-container">
      <h2>Slowdown by DType (${runtimeName})</h2>
      <div style="display:flex;gap:30px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:1;min-width:400px">
          <canvas id="${canvasId}"></canvas>
        </div>
        <div style="flex:0 0 auto">
          <table style="margin:0">
            <thead>
              <tr>
                <th>DType</th>
                <th>Count</th>
                <th>Avg Slowdown</th>
                <th>Median Slowdown</th>
              </tr>
            </thead>
            <tbody>`;

  for (const [dtype, data] of dtypeSummaries) {
    const colors = DTYPE_COLORS[dtype] ?? DTYPE_COLORS['float64']!;
    const ratioClass = data.geo_mean < 2 ? 'good' : data.geo_mean < 5 ? 'ok' : 'bad';
    html += `
              <tr>
                <td><span style="display:inline-block;font-size:0.85em;font-weight:600;padding:2px 8px;border-radius:3px;background:${colors.bg};color:${colors.text}">${dtype}</span></td>
                <td>${data.count}</td>
                <td><span class="ratio ${ratioClass}">${formatRatio(data.geo_mean)}</span></td>
                <td>${formatRatio(data.median_slowdown)}</td>
              </tr>`;
  }

  html += `
            </tbody>
          </table>
        </div>
      </div>
    </div>`;

  return html;
}

function generateDtypeChartScript(
  dtypeSummaries: Map<string, { geo_mean: number; median_slowdown: number; count: number }>,
  canvasId: string = 'dtypeChart'
): string {
  if (dtypeSummaries.size <= 1) return '';

  const labels = Array.from(dtypeSummaries.keys());
  const avgData = labels.map((d) => dtypeSummaries.get(d)!.geo_mean);
  const medianData = labels.map((d) => dtypeSummaries.get(d)!.median_slowdown);
  const bgColors = labels.map((d) => {
    const c = DTYPE_COLORS[d] ?? DTYPE_COLORS['float64']!;
    return c.bg;
  });
  const borderColors = labels.map((d) => {
    const c = DTYPE_COLORS[d] ?? DTYPE_COLORS['float64']!;
    return c.text;
  });

  return `
    new Chart(document.getElementById('${canvasId}'), {
      type: 'bar',
      data: {
        labels: ${JSON.stringify(labels)},
        datasets: [
          {
            label: 'Avg Slowdown',
            data: ${JSON.stringify(avgData)},
            backgroundColor: ${JSON.stringify(bgColors)},
            borderColor: ${JSON.stringify(borderColors)},
            borderWidth: 2
          },
          {
            label: 'Median Slowdown',
            data: ${JSON.stringify(medianData)},
            backgroundColor: ${JSON.stringify(bgColors.map((c) => c + '80'))},
            borderColor: ${JSON.stringify(borderColors)},
            borderWidth: 1,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Slowdown Ratio (lower is better)' }
          }
        }
      }
    });`;
}

function generateCategoryTables(groups: Map<string, BenchmarkComparison[]>): string {
  let html = '';

  for (const [category, items] of groups) {
    html += `
    <div class="category-section">
      <h2>${category.toUpperCase()}</h2>
      <table>
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>NumPy (ops/sec)</th>
            <th>numpy-ts (ops/sec)</th>
            <th>Time (ms)</th>
            <th>Ratio</th>
          </tr>
        </thead>
        <tbody>`;

    for (const item of items) {
      const ratioClass = item.ratio < 2 ? 'good' : item.ratio < 5 ? 'ok' : 'bad';
      html += `
          <tr>
            <td>${formatBenchmarkName(item.name, item.wasmUsed)}</td>
            <td>${formatOpsPerSec(item.numpy.ops_per_sec)}</td>
            <td>${formatOpsPerSec(item.numpyjs.ops_per_sec)}</td>
            <td>${formatDuration(item.numpyjs.mean_ms)}</td>
            <td><span class="ratio ${ratioClass}">${formatRatio(item.ratio)}</span></td>
          </tr>`;
    }

    html += `
        </tbody>
      </table>
    </div>`;
  }

  return html;
}
