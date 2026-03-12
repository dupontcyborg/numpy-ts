/**
 * WASM Runtime — shared memory management for all WASM kernels.
 *
 * All kernel modules share a single WebAssembly.Memory instance.
 * Uses a bump allocator that resets on every kernel call (no fragmentation).
 * Memory grows monotonically (high-water-mark pattern).
 */

import type { TypedArray } from '../dtype';

// Shared memory instance — grows as needed, never shrinks
let memory: WebAssembly.Memory | null = null;

// Bump allocator offset (reset to heapBase before each kernel call)
let offset = 0;
let heapBase = 0;

/**
 * Get the shared WebAssembly.Memory instance.
 * All WASM kernel modules import this same memory.
 * Starts with 17 pages (~1.1MB) — enough for debug (ReleaseSmall/Safe) WASM builds
 * which have larger binaries and declare higher minimum memory.
 */
export function getSharedMemory(): WebAssembly.Memory {
  if (!memory) {
    memory = new WebAssembly.Memory({ initial: 17 });
  }
  return memory;
}

/**
 * Ensure the shared memory has at least `bytes` of usable space
 * (above heapBase). Grows if necessary.
 */
export function ensureMemory(bytes: number): void {
  const mem = getSharedMemory();
  const needed = heapBase + bytes;
  const current = mem.buffer.byteLength;
  if (needed > current) {
    const pagesNeeded = Math.ceil((needed - current) / 65536);
    mem.grow(pagesNeeded);
  }
}

/**
 * Reset the bump allocator. Call before each kernel invocation.
 */
export function resetAllocator(base: number = heapBase): void {
  offset = base;
}

/**
 * Set the heap base offset. Called once when the first WASM instance
 * is initialized (reads __heap_base from the WASM exports).
 */
export function setHeapBase(base: number): void {
  if (base > heapBase) {
    heapBase = base;
  }
}

/**
 * Bump-allocate `bytes` from WASM memory. Returns the byte offset.
 * Always 8-byte aligned for TypedArray compatibility.
 */
export function alloc(bytes: number): number {
  const aligned = (offset + 7) & ~7;
  offset = aligned + bytes;
  return aligned;
}

/**
 * Copy a JS TypedArray into WASM memory. Returns the byte offset.
 */
export function copyIn(src: TypedArray): number {
  const ptr = alloc(src.byteLength);
  const mem = getSharedMemory();
  new Uint8Array(mem.buffer, ptr, src.byteLength).set(
    new Uint8Array(src.buffer, src.byteOffset, src.byteLength)
  );
  return ptr;
}

/**
 * Copy data from WASM memory into a new JS TypedArray.
 */
export function copyOut<T extends TypedArray>(
  ptr: number,
  length: number,
  Ctor: new (buffer: ArrayBuffer, byteOffset: number, length: number) => T
): T {
  const mem = getSharedMemory();
  const result = new Ctor(
    new ArrayBuffer(length * (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT),
    0,
    length
  );
  new Uint8Array(result.buffer, 0, result.byteLength).set(
    new Uint8Array(mem.buffer, ptr, result.byteLength)
  );
  return result;
}
