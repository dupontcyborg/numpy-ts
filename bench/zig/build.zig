const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const threads = b.option(bool, "threads", "Enable multi-threaded build") orelse false;

    const target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const exe = b.addExecutable(.{
        .name = if (threads) "numpy_bench_threads" else "numpy_bench",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // WASM-specific settings
    exe.entry = .disabled;
    exe.rdynamic = true;

    b.installArtifact(exe);
}
