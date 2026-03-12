/**
 * PNG chart generation using Chart.js
 */

import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import * as fs from 'fs';
import type { BenchmarkReport, MultiRuntimeReport } from './types';
import { getCategorySummaries, groupMultiRuntimeByCategory } from './analysis';

export async function generatePNGChart(report: BenchmarkReport, outputPath: string): Promise<void> {
  const { results, summary } = report;
  const categorySummaries = getCategorySummaries(results);

  // Prepare data
  const categories = Array.from(categorySummaries.keys());
  const avgSlowdowns = categories.map((cat) => categorySummaries.get(cat)!.avg_slowdown);

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
          label: 'Average Slowdown (x times slower than NumPy)',
          data: avgSlowdowns,
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
            `numpy-ts vs ${report.environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy'} Performance`,
            `Overall: ${summary.avg_slowdown.toFixed(1)}x slower (avg) | Best: ${summary.best_case.toFixed(1)}x | Worst: ${summary.worst_case.toFixed(1)}x`,
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
    const colors = RUNTIME_COLORS[rt] || { bg: 'rgba(153, 102, 255, 0.8)', border: 'rgba(153, 102, 255, 1)' };
    return {
      label: `${rt.charAt(0).toUpperCase() + rt.slice(1)} (avg slowdown)`,
      data,
      backgroundColor: colors.bg,
      borderColor: colors.border,
      borderWidth: 2,
    };
  });

  // Build subtitle parts
  const subtitleParts = runtimeNames
    .filter((rt) => summaries[rt])
    .map((rt) => `${rt}: ${summaries[rt]!.avg_slowdown.toFixed(1)}x avg`);

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
            `numpy-ts vs ${report.environment.baseline === 'pyodide' ? 'Pyodide NumPy (WASM)' : 'Python NumPy'} — Multi-Runtime Performance`,
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
