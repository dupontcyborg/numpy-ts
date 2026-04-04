/**
 * PNG chart generation using Chart.js
 */

import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import * as fs from 'fs';
import * as path from 'path';
import type { BenchmarkReport, MultiRuntimeReport } from './types';
import { getCategorySummaries, groupMultiRuntimeByCategory } from './analysis';

/**
 * Format a ratio for display in titles. If <1.0, flip to "Nx faster".
 */
function formatTitleRatio(ratio: number): string {
  if (ratio <= 0 || !isFinite(ratio)) return 'N/A';
  if (ratio < 1.0) {
    return `${(1 / ratio).toFixed(1)}x faster`;
  }
  if (ratio < 1.05) return '1.0x (parity)';
  return `${ratio.toFixed(1)}x slower`;
}

/**
 * Infer array size qualifier from the output path filename.
 * e.g. "latest-full-large.png" → " — Large Arrays"
 */
function getSizeQualifier(outputPath: string): string {
  const basename = path.basename(outputPath).toLowerCase();
  if (basename.includes('-small')) return ' — Small Arrays';
  if (basename.includes('-large')) return ' — Large Arrays';
  return '';
}

export async function generatePNGChart(report: BenchmarkReport, outputPath: string): Promise<void> {
  const { results, summary } = report;
  const categorySummaries = getCategorySummaries(results);

  // Prepare data
  const categories = Array.from(categorySummaries.keys());
  const geoSlowdowns = categories.map((cat) => categorySummaries.get(cat)!.geo_mean);

  // Chart dimensions
  const width = 1200;
  const height = 600;

  // Create canvas
  const chartJSNodeCanvas = new ChartJSNodeCanvas({
    width,
    height,
    backgroundColour: 'white',
  });

  // Define chart configuration
  const configuration = {
    type: 'bar' as const,
    data: {
      labels: categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [
        {
          label: 'Overall Slowdown (x times slower than NumPy)',
          data: geoSlowdowns,
          backgroundColor: [
            'rgba(75, 192, 192, 0.8)', // creation - teal
            'rgba(255, 159, 64, 0.8)', // arithmetic - orange
            'rgba(153, 102, 255, 0.8)', // linalg - purple
            'rgba(255, 99, 132, 0.8)', // reductions - red
            'rgba(54, 162, 235, 0.8)', // reshape - blue
          ],
          borderColor: [
            'rgba(75, 192, 192, 1)',
            'rgba(255, 159, 64, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
          ],
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
            `numpy-ts vs ${report.environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy'} Performance${getSizeQualifier(outputPath)}`,
            `Overall: ${formatTitleRatio(summary.geo_mean)} | Best: ${formatTitleRatio(summary.best_case)} | Worst: ${formatTitleRatio(summary.worst_case)}`,
          ],
          font: {
            size: 18,
            weight: 'bold',
          },
          padding: {
            top: 10,
            bottom: 20,
          },
        },
        legend: {
          display: true,
          position: 'top' as const,
          labels: {
            font: {
              size: 14,
            },
          },
        },
        subtitle: {
          display: true,
          text: `Total Benchmarks: ${summary.total_benchmarks} | Generated: ${new Date(report.timestamp).toLocaleDateString()}`,
          font: {
            size: 12,
          },
          padding: {
            bottom: 10,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Slowdown Ratio (lower is better)',
            font: {
              size: 14,
              weight: 'bold',
            },
          },
          ticks: {
            font: {
              size: 12,
            },
            callback: function (value: any) {
              return value + 'x';
            },
          },
        },
        x: {
          title: {
            display: true,
            text: 'Operation Category',
            font: {
              size: 14,
              weight: 'bold',
            },
          },
          ticks: {
            font: {
              size: 12,
            },
          },
        },
      },
    },
    plugins: [
      {
        id: 'customBackground',
        beforeDraw: (chart: any) => {
          const ctx = chart.canvas.getContext('2d');
          ctx.save();
          ctx.globalCompositeOperation = 'destination-over';
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, chart.width, chart.height);
          ctx.restore();
        },
      },
    ],
  };

  // Render chart to buffer
  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration as any);

  // Write to file
  fs.writeFileSync(outputPath, imageBuffer);
}

/**
 * Head-to-head grouped bar chart: NumPy vs numpy-ts median ops/sec per category.
 * Taller bars = faster. Makes it easy to see where we win vs lose.
 */
export async function generateH2HChart(report: BenchmarkReport, outputPath: string): Promise<void> {
  const { results } = report;

  // Group by category, compute median ops/sec for both numpy and numpyjs
  const catMap = new Map<string, { numpy: number[]; numpyjs: number[] }>();
  for (const r of results) {
    if (!catMap.has(r.category)) catMap.set(r.category, { numpy: [], numpyjs: [] });
    const entry = catMap.get(r.category)!;
    entry.numpy.push(r.numpy.ops_per_sec);
    entry.numpyjs.push(r.numpyjs.ops_per_sec);
  }

  const categories = Array.from(catMap.keys());
  const median = (arr: number[]) => {
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[m - 1]! + s[m]!) / 2 : s[m]!;
  };

  const numpyMedians = categories.map((c) => median(catMap.get(c)!.numpy));
  const numpytsMedians = categories.map((c) => median(catMap.get(c)!.numpyjs));

  // Overall median ops/sec across all benchmarks
  const allNumpyOps = results.map((r) => r.numpy.ops_per_sec);
  const allNumpytsOps = results.map((r) => r.numpyjs.ops_per_sec);
  const overallNumpy = median(allNumpyOps);
  const overallNumpyts = median(allNumpytsOps);

  // Format large numbers
  const formatOps = (v: number) => {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
    return v.toFixed(0);
  };

  const baseline = report.environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy';

  const width = 1200;
  const height = 600;
  const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height, backgroundColour: 'white' });

  const configuration = {
    type: 'bar' as const,
    data: {
      labels: categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets: [
        {
          label: baseline,
          data: numpyMedians,
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
        },
        {
          label: 'numpy-ts',
          data: numpytsMedians,
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
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
            `numpy-ts vs ${baseline} — Head to Head${getSizeQualifier(outputPath)}`,
            `numpy-ts: ${formatOps(overallNumpyts)} ops/sec | ${baseline}: ${formatOps(overallNumpy)} ops/sec (median across all benchmarks)`,
          ],
          font: { size: 18, weight: 'bold' },
          padding: { top: 10, bottom: 20 },
        },
        subtitle: {
          display: true,
          text: `Total Benchmarks: ${results.length} | Generated: ${new Date(report.timestamp).toLocaleDateString()}`,
          font: { size: 12 },
          padding: { bottom: 10 },
        },
        legend: {
          display: true,
          position: 'top' as const,
          labels: { font: { size: 14 } },
        },
      },
      scales: {
        y: {
          type: 'logarithmic' as const,
          title: {
            display: true,
            text: 'Median ops/sec (log₁₀ scale)',
            font: { size: 14, weight: 'bold' },
          },
          ticks: {
            font: { size: 11 },
            // Show only powers of 10: 1, 10, 100, 1K, 10K, 100K, 1M, 10M
            callback: function (value: any) {
              const v = Number(value);
              const log = Math.log10(v);
              if (Math.abs(log - Math.round(log)) < 0.01) return formatOps(v);
              return '';
            },
          },
          grid: {
            // Only draw gridlines at powers of 10 (where labels are shown)
            color: function (context: any) {
              const v = context.tick?.value;
              if (v == null) return 'rgba(0,0,0,0.1)';
              const log = Math.log10(v);
              if (Math.abs(log - Math.round(log)) < 0.01) return 'rgba(0,0,0,0.1)';
              return 'transparent';
            },
          },
        },
        x: {
          title: {
            display: true,
            text: 'Operation Category',
            font: { size: 14, weight: 'bold' },
          },
          ticks: { font: { size: 12 } },
        },
      },
    },
    plugins: [
      {
        id: 'chartAreaBorder',
        afterDraw: (chart: any) => {
          const { ctx, chartArea: { left, top, right, bottom } } = chart;
          ctx.save();
          ctx.strokeStyle = 'rgba(0,0,0,0.1)';
          ctx.lineWidth = 1;
          ctx.strokeRect(left, top, right - left, bottom - top);
          ctx.restore();
        },
      },
      {
        id: 'customBackground',
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

  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration as any);
  fs.writeFileSync(outputPath, imageBuffer);
}

export async function generateComparisonPNG(
  report: BenchmarkReport,
  outputPath: string
): Promise<void> {
  const { results } = report;

  // Get top 10 slowest operations
  const sorted = [...results].sort((a, b) => b.ratio - a.ratio).slice(0, 10);

  const width = 1200;
  const height = 700;

  const chartJSNodeCanvas = new ChartJSNodeCanvas({
    width,
    height,
    backgroundColour: 'white',
  });

  const configuration = {
    type: 'bar' as const,
    data: {
      labels: sorted.map((r) => r.name),
      datasets: [
        {
          label: 'NumPy (ops/sec)',
          data: sorted.map((r) => r.numpy.ops_per_sec),
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 2,
        },
        {
          label: 'numpy-ts (ops/sec)',
          data: sorted.map((r) => r.numpyjs.ops_per_sec),
          backgroundColor: 'rgba(255, 99, 132, 0.8)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: false,
      indexAxis: 'y' as const,
      plugins: {
        title: {
          display: true,
          text: 'Top 10 Slowest Operations - Throughput Comparison',
          font: {
            size: 18,
            weight: 'bold',
          },
          padding: {
            top: 10,
            bottom: 20,
          },
        },
        legend: {
          display: true,
          position: 'top' as const,
          labels: {
            font: {
              size: 14,
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Operations per Second (higher is better)',
            font: {
              size: 14,
              weight: 'bold',
            },
          },
          ticks: {
            font: {
              size: 12,
            },
            callback: function (value: any) {
              // Format large numbers with K/M suffix
              if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
              if (value >= 1000) return (value / 1000).toFixed(0) + 'K';
              return value.toString();
            },
          },
        },
        y: {
          ticks: {
            font: {
              size: 10,
            },
          },
        },
      },
    },
    plugins: [
      {
        id: 'customBackground',
        beforeDraw: (chart: any) => {
          const ctx = chart.canvas.getContext('2d');
          ctx.save();
          ctx.globalCompositeOperation = 'destination-over';
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, chart.width, chart.height);
          ctx.restore();
        },
      },
    ],
  };

  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration as any);
  fs.writeFileSync(outputPath, imageBuffer);
}

// --- Multi-runtime grouped bar chart ---

const RUNTIME_COLORS: Record<string, { bg: string; border: string }> = {
  node: { bg: 'rgba(54, 162, 235, 0.8)', border: 'rgba(54, 162, 235, 1)' },
  deno: { bg: 'rgba(75, 192, 192, 0.8)', border: 'rgba(75, 192, 192, 1)' },
  bun: { bg: 'rgba(255, 159, 64, 0.8)', border: 'rgba(255, 159, 64, 1)' },
};

export async function generateMultiRuntimePNGChart(
  report: MultiRuntimeReport,
  outputPath: string
): Promise<void> {
  const { results, summaries, environment } = report;
  const groups = groupMultiRuntimeByCategory(results);
  const runtimeNames = Object.keys(environment.runtimes);

  const categories = Array.from(groups.keys());

  // Build one dataset per runtime
  const datasets = runtimeNames.map((rt) => {
    const data = categories.map((cat) => {
      const items = groups.get(cat)!;
      const ratios = items
        .filter((item) => item.runtimes[rt])
        .map((item) => item.runtimes[rt]!.ratio);
      return ratios.length > 0 ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
    });
    const colors = RUNTIME_COLORS[rt] || {
      bg: 'rgba(153, 102, 255, 0.8)',
      border: 'rgba(153, 102, 255, 1)',
    };
    return {
      label: `${rt.charAt(0).toUpperCase() + rt.slice(1)} (overall slowdown)`,
      data,
      backgroundColor: colors.bg,
      borderColor: colors.border,
      borderWidth: 2,
    };
  });

  // Build subtitle parts
  const subtitleParts = runtimeNames
    .filter((rt) => summaries[rt])
    .map((rt) => `${rt}: ${formatTitleRatio(summaries[rt]!.geo_mean)}`);

  const width = 1200;
  const height = 600;

  const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height, backgroundColour: 'white' });

  const configuration = {
    type: 'bar' as const,
    data: {
      labels: categories.map((c) => c.charAt(0).toUpperCase() + c.slice(1)),
      datasets,
    },
    options: {
      responsive: false,
      plugins: {
        title: {
          display: true,
          text: [
            `numpy-ts vs ${report.environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy'} — Multi-Runtime Performance${getSizeQualifier(outputPath)}`,
            subtitleParts.join(' | '),
          ],
          font: { size: 18, weight: 'bold' },
          padding: { top: 10, bottom: 20 },
        },
        legend: {
          display: true,
          position: 'top' as const,
          labels: { font: { size: 14 } },
        },
        subtitle: {
          display: true,
          text: `Total Benchmarks: ${results.length} | Generated: ${new Date(report.timestamp).toLocaleDateString()}`,
          font: { size: 12 },
          padding: { bottom: 10 },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Slowdown Ratio (lower is better)',
            font: { size: 14, weight: 'bold' },
          },
          ticks: {
            font: { size: 12 },
            callback: function (value: any) {
              return value + 'x';
            },
          },
        },
        x: {
          title: {
            display: true,
            text: 'Operation Category',
            font: { size: 14, weight: 'bold' },
          },
          ticks: { font: { size: 12 } },
        },
      },
    },
    plugins: [
      {
        id: 'customBackground',
        beforeDraw: (chart: any) => {
          const ctx = chart.canvas.getContext('2d');
          ctx.save();
          ctx.globalCompositeOperation = 'destination-over';
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, chart.width, chart.height);
          ctx.restore();
        },
      },
    ],
  };

  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration as any);
  fs.writeFileSync(outputPath, imageBuffer);
}
