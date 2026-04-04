/**
 * Regenerate HTML and PNG charts from existing benchmark JSON files.
 * Useful when chart formatting changes without re-running benchmarks.
 *
 * Usage: npx tsx benchmarks/scripts/regenerate-charts.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { generateHTMLReport, generateMultiRuntimeHTMLReport } from '../src/visualization';
import { generatePNGChart, generateH2HChart, generateMultiRuntimePNGChart } from '../src/chart-generator';
import { calculateSummary } from '../src/analysis';
import type { BenchmarkReport, MultiRuntimeReport } from '../src/types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const resultsDir = path.join(__dirname, '..', 'results');
const plotsDir = path.join(resultsDir, 'plots');

if (!fs.existsSync(plotsDir)) fs.mkdirSync(plotsDir, { recursive: true });

const jsonFiles = fs
  .readdirSync(resultsDir)
  .filter((f) => f.startsWith('latest') && f.endsWith('.json'));

console.log(`Found ${jsonFiles.length} benchmark files to regenerate:\n`);

async function main() {
  for (const jsonFile of jsonFiles) {
    const jsonPath = path.join(resultsDir, jsonFile);
    const baseName = jsonFile.replace('.json', '');
    const htmlPath = path.join(plotsDir, `${baseName}.html`);
    const pngPath = path.join(plotsDir, `${baseName}.png`);

    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

    // Detect if this is a multi-runtime report
    const isMultiRuntime = 'runtimes' in (data.environment ?? {});

    console.log(`  ${jsonFile} → ${isMultiRuntime ? 'multi-runtime' : 'single-runtime'}`);

    try {
      if (isMultiRuntime) {
        const report = data as MultiRuntimeReport;
        generateMultiRuntimeHTMLReport(report, htmlPath);
        await generateMultiRuntimePNGChart(report, pngPath);
      } else {
        const report = data as BenchmarkReport;
        // Recalculate summary (picks up fixes like geo_mean epsilon clamping)
        report.summary = calculateSummary(report.results);
        generateHTMLReport(report, htmlPath);
        await generatePNGChart(report, pngPath);
        const h2hPath = path.join(plotsDir, `${baseName}-h2h.png`);
        await generateH2HChart(report, h2hPath);
      }
      console.log(`    ✓ ${baseName}.html + ${baseName}.png`);
    } catch (err) {
      console.log(`    ✗ Error: ${(err as Error).message}`);
    }
  }

  console.log('\nDone.');
}

main();
