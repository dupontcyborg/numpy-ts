/**
 * WASM vs WebGPU matmul benchmark
 *
 * f32: WASM (Rust) vs WebGPU (tiled WGSL shader)
 * f64: WASM (Zig micro-kernel) only — WebGPU has no native f64
 *
 * Run:
 *   deno run --allow-read --unstable-webgpu harness/bench-matmul-gpu.ts
 */

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST_DIR = resolve(__dirname, '..', 'dist');

// ─── WASM Loading ────────────────────────────────────────────────────────────

const PAGE_SIZE = 65536;

interface WasmModule {
  exports: Record<string, any>;
  memory: WebAssembly.Memory;
}

async function loadWasm(file: string): Promise<WasmModule> {
  const buf = readFileSync(resolve(DIST_DIR, file));
  const mod = await WebAssembly.compile(buf);
  const inst = await WebAssembly.instantiate(mod, {});
  const memory = inst.exports.memory as WebAssembly.Memory;
  return { exports: inst.exports as Record<string, any>, memory };
}

function ensureMemory(memory: WebAssembly.Memory, bytesNeeded: number) {
  const current = memory.buffer.byteLength;
  if (bytesNeeded > current) {
    memory.grow(Math.ceil((bytesNeeded - current) / PAGE_SIZE));
  }
}

// ─── WebGPU Setup ────────────────────────────────────────────────────────────

const WGSL_MATMUL = /* wgsl */ `
struct Dims { M: u32, N: u32, K: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TS: u32 = 16u;
var<workgroup> sA: array<array<f32, 16>, 16>;
var<workgroup> sB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;

  var acc: f32 = 0.0;
  let tiles = (K + TS - 1u) / TS;

  for (var t: u32 = 0u; t < tiles; t++) {
    let aCol = t * TS + lc;
    let bRow = t * TS + lr;
    sA[lr][lc] = select(0.0, A[row * K + aCol], row < M && aCol < K);
    sB[lr][lc] = select(0.0, B[bRow * N + col], bRow < K && col < N);
    workgroupBarrier();

    for (var k: u32 = 0u; k < TS; k++) {
      acc += sA[lr][k] * sB[k][lc];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
`;

interface GpuCtx {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
}

async function initGpu(): Promise<GpuCtx | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: WGSL_MATMUL }),
      entryPoint: 'main',
    },
  });
  return { device, pipeline };
}

// ─── Config ──────────────────────────────────────────────────────────────────

interface Shape { label: string; M: number; N: number; K: number }

const SHAPES: Shape[] = [
  { label: '64×64×64',       M: 64,   N: 64,   K: 64 },
  { label: '128×128×128',    M: 128,  N: 128,  K: 128 },
  { label: '256×256×256',    M: 256,  N: 256,  K: 256 },
  { label: '512×512×512',    M: 512,  N: 512,  K: 512 },
  { label: '1024×1024×1024', M: 1024, N: 1024, K: 1024 },
];

const WARMUP = 3;
const MIN_TIME_MS = 500;

function flops(M: number, N: number, K: number) { return 2 * M * N * K; }

function fillRandom32(arr: Float32Array) {
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random() * 2 - 1;
}
function fillRandom64(arr: Float64Array) {
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random() * 2 - 1;
}

// ─── WASM Bench ──────────────────────────────────────────────────────────────

function benchWasm<T extends Float32Array | Float64Array>(
  fn: Function,
  memory: WebAssembly.Memory,
  jsA: T, jsB: T,
  M: number, N: number, K: number,
  bytesPerElem: number,
): { ms: number; iters: number } {
  const aElems = M * K, bElems = K * N, cElems = M * N;
  ensureMemory(memory, (aElems + bElems + cElems) * bytesPerElem);
  const aPtr = 0;
  const bPtr = aElems * bytesPerElem;
  const cPtr = (aElems + bElems) * bytesPerElem;

  const Ctor = bytesPerElem === 8 ? Float64Array : Float32Array;
  new Ctor(memory.buffer, aPtr, aElems).set(jsA as any);
  new Ctor(memory.buffer, bPtr, bElems).set(jsB as any);

  for (let w = 0; w < WARMUP; w++) fn(aPtr, bPtr, cPtr, M, N, K);

  let totalMs = 0, iters = 0;
  while (totalMs < MIN_TIME_MS) {
    const t0 = performance.now();
    fn(aPtr, bPtr, cPtr, M, N, K);
    totalMs += performance.now() - t0;
    iters++;
  }
  return { ms: totalMs / iters, iters };
}

// ─── GPU Bench ───────────────────────────────────────────────────────────────

async function benchGpu(
  ctx: GpuCtx,
  jsA: Float32Array, jsB: Float32Array,
  M: number, N: number, K: number,
): Promise<{ ms: number; iters: number }> {
  const { device, pipeline } = ctx;
  const cBytes = M * N * 4;

  const bufA = device.createBuffer({ size: jsA.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufB = device.createBuffer({ size: jsB.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufC = device.createBuffer({ size: cBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bufU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(bufA, 0, jsA);
  device.queue.writeBuffer(bufB, 0, jsB);
  device.queue.writeBuffer(bufU, 0, new Uint32Array([M, N, K]));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufU } },
      { binding: 1, resource: { buffer: bufA } },
      { binding: 2, resource: { buffer: bufB } },
      { binding: 3, resource: { buffer: bufC } },
    ],
  });

  const wgX = Math.ceil(N / 16);
  const wgY = Math.ceil(M / 16);

  function submit() {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // Warmup
  for (let w = 0; w < WARMUP; w++) {
    submit();
    await device.queue.onSubmittedWorkDone();
  }

  // Timed
  let totalMs = 0, iters = 0;
  while (totalMs < MIN_TIME_MS) {
    const t0 = performance.now();
    submit();
    await device.queue.onSubmittedWorkDone();
    totalMs += performance.now() - t0;
    iters++;
  }

  bufA.destroy(); bufB.destroy(); bufC.destroy(); bufU.destroy();
  return { ms: totalMs / iters, iters };
}

// ─── Main ────────────────────────────────────────────────────────────────────

function gfStr(gf: number) { return gf.toFixed(2).padStart(10); }

async function main() {
  const zigMod = await loadWasm('matmul_zig.wasm');
  const rustMod = await loadWasm('matmul_rust.wasm');
  const gpu = await initGpu();

  if (!gpu) {
    console.error('WebGPU not available. Run with:');
    console.error('  deno run --allow-read --unstable-webgpu harness/bench-matmul-gpu.ts');
    process.exit(1);
  }

  console.log('WASM vs WebGPU Matmul Benchmark');
  console.log('═'.repeat(80));
  console.log('GPU timing: submit → onSubmittedWorkDone (compute only, no readback)');
  console.log(`Warmup: ${WARMUP} iters, min measurement time: ${MIN_TIME_MS}ms`);
  console.log();

  // ── f32: Rust WASM vs WebGPU ─────────────────────────────────────────────

  console.log('▸ f32  (Rust WASM vs WebGPU)');
  const h32 = [
    'Shape'.padEnd(20),
    'WASM ms'.padStart(10),
    'GPU ms'.padStart(10),
    'WASM GF/s'.padStart(10),
    'GPU GF/s'.padStart(10),
    'GPU/WASM'.padStart(10),
  ];
  console.log(h32.join(' │ '));
  console.log('─'.repeat(h32.join(' │ ').length));

  for (const { label, M, N, K } of SHAPES) {
    const a32 = new Float32Array(M * K);
    const b32 = new Float32Array(K * N);
    fillRandom32(a32);
    fillRandom32(b32);

    const wasm = benchWasm(rustMod.exports.matmul_f32, rustMod.memory, a32, b32, M, N, K, 4);
    const gpuR = await benchGpu(gpu, a32, b32, M, N, K);

    const gf = flops(M, N, K) / 1e9;
    const wasmGF = gf / (wasm.ms / 1000);
    const gpuGF = gf / (gpuR.ms / 1000);
    const speedup = wasm.ms / gpuR.ms;

    console.log([
      label.padEnd(20),
      wasm.ms.toFixed(2).padStart(10),
      gpuR.ms.toFixed(2).padStart(10),
      gfStr(wasmGF),
      gfStr(gpuGF),
      `${speedup.toFixed(2)}x`.padStart(10),
    ].join(' │ '));
  }

  console.log();

  // ── f64: Zig WASM only (no GPU f64) ──────────────────────────────────────

  console.log('▸ f64  (Zig WASM micro-kernel — WebGPU has no native f64)');
  const h64 = [
    'Shape'.padEnd(20),
    'WASM ms'.padStart(10),
    'WASM GF/s'.padStart(10),
  ];
  console.log(h64.join(' │ '));
  console.log('─'.repeat(h64.join(' │ ').length));

  for (const { label, M, N, K } of SHAPES) {
    const a64 = new Float64Array(M * K);
    const b64 = new Float64Array(K * N);
    fillRandom64(a64);
    fillRandom64(b64);

    const wasm = benchWasm(zigMod.exports.matmul_f64_micro, zigMod.memory, a64, b64, M, N, K, 8);
    const gf = flops(M, N, K) / 1e9;
    const wasmGF = gf / (wasm.ms / 1000);

    console.log([
      label.padEnd(20),
      wasm.ms.toFixed(2).padStart(10),
      gfStr(wasmGF),
    ].join(' │ '));
  }

  console.log();
  console.log('GPU/WASM: >1x means GPU is faster');

  gpu.device.destroy();
}

main().catch(console.error);
