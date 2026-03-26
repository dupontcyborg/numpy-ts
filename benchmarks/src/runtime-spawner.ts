/**
 * Multi-runtime benchmark spawner
 *
 * Detects available JS runtimes and spawns the standalone runtime-runner
 * under each one, collecting results via stdout.
 */

import { spawn, execFile } from 'child_process';
import { resolve } from 'path';
import type { BenchmarkCase, BenchmarkTiming, RuntimeName, RuntimeInfo } from './types';

const RUNNER_PATH = resolve(__dirname, '../dist/runtime-runner.mjs');

/**
 * Check if a runtime is available on PATH by running `<runtime> --version`
 */
function checkRuntime(name: RuntimeName): Promise<RuntimeInfo | null> {
  return new Promise((resolve) => {
    const cmd = name;
    execFile(cmd, ['--version'], { timeout: 5000 }, (err, stdout, stderr) => {
      if (err) {
        resolve(null);
        return;
      }
      // Extract version string from output
      const output = (stdout || stderr || '').trim();
      let version = output;
      // Node outputs "v22.0.0", Deno outputs "deno 2.1.0", Bun outputs "1.2.0"
      const match = output.match(/v?([\d.]+)/);
      if (match) {
        version = match[1]!;
      }
      resolve({ name, version });
    });
  });
}

/**
 * Detect which JS runtimes are available on the system
 */
export async function detectRuntimes(): Promise<RuntimeInfo[]> {
  const checks = await Promise.all([
    checkRuntime('node'),
    checkRuntime('deno'),
    checkRuntime('bun'),
  ]);
  return checks.filter((r): r is RuntimeInfo => r !== null);
}

/**
 * Build the spawn command for a given runtime
 */
function getRuntimeArgs(runtime: RuntimeName): { cmd: string; args: string[] } {
  switch (runtime) {
    case 'node':
      return { cmd: 'node', args: [RUNNER_PATH] };
    case 'deno':
      return { cmd: 'deno', args: ['run', '--allow-read', '--allow-env', RUNNER_PATH] };
    case 'bun':
      return { cmd: 'bun', args: ['run', RUNNER_PATH] };
  }
}

/**
 * Spawn the benchmark runner under a specific JS runtime
 */
export async function spawnRuntimeBenchmark(
  runtime: RuntimeName,
  specs: BenchmarkCase[],
  minSampleTimeMs: number,
  targetSamples: number,
  noWasm: boolean = false
): Promise<{ results: BenchmarkTiming[]; version: string }> {
  const { cmd, args } = getRuntimeArgs(runtime);

  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let version = 'unknown';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;

      // First line of stderr contains runtime name + version
      const versionMatch = text.match(/^(\w+)\s+(v?[\d.]+)/m);
      if (versionMatch) {
        version = versionMatch[2]!;
        if (version.startsWith('v')) version = version.slice(1);
      }

      // Forward progress to parent stderr
      process.stderr.write(text);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`${runtime} runner exited with code ${code}\n${stderr}`));
        return;
      }

      try {
        const results = JSON.parse(stdout) as BenchmarkTiming[];
        resolve({ results, version });
      } catch (err) {
        reject(
          new Error(
            `Failed to parse ${runtime} output: ${err}\nstdout: ${stdout.substring(0, 500)}`
          )
        );
      }
    });

    child.on('error', (err) => {
      reject(new Error(`Failed to spawn ${runtime}: ${err.message}`));
    });

    // Send specs + config via stdin
    child.stdin.write(
      JSON.stringify({
        specs,
        config: {
          minSampleTimeMs,
          targetSamples,
          noWasm,
        },
      })
    );
    child.stdin.end();
  });
}
