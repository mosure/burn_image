# Performance

Performance work targets interactive image generation and editing without weakening artifact,
numeric, or GPU correctness.

## Goals

- Keep ordinary 1024 px requests near or below 30 seconds on capable desktop GPUs.
- Keep every public variant's resident-Q4 plan within a 16 GB device budget.
- Keep selected weights resident for fast subsequent requests.
- Avoid blocking the Bevy update loop during artifact I/O or inference.
- Avoid full-bundle Wasm allocations and repeated network transfers.

These are engineering targets, not universal latency guarantees. Reports must identify hardware,
browser/driver, resolution, profile, cache state, and whether kernel tuning occurred.

## Runtime policy

Ordinary execution prioritizes resident packed-Q4 weights for Turbo, Edit Turbo, and Edit Turbo
1.5K. Packed linear, embedding, and convolution kernels accumulate in F32 without widening whole
model stages. This reduces both VRAM residency and upload traffic compared with dense F32
materialization. The runtime unloads modules that are not used by a newly selected variant before
loading replacement weights. It does not cycle model weights out after each successful request
merely to minimize a peak number.

Autotuning is opt-in. Interactive defaults use static kernels so the first request does not absorb a
large tuning pause.

## Browser loading

- only the selected model closure is planned;
- VRAM feasibility is checked before large transfers;
- physical parts are fetched or read from Cache Storage concurrently within explicit bounds;
- each part is verified before its logical object is reconstructed;
- logical objects are uploaded sequentially so Wasm linear memory remains bounded;
- aggregate progress reports bytes, objects, parts, ranges, throughput, and ETA;
- cached-stage application is shown as active work rather than a completed transfer bar.

The physical part size is 20 MiB, while browser reads and GPU uploads use bounds chosen by the
runtime. Part size is a CDN/cache contract, not a claim that the entire part must remain as an extra
Wasm copy.

## Measuring a request

Measure at least:

1. manifest/layout authentication;
2. network or persistent-cache reads;
3. logical reconstruction and digest verification;
4. GPU upload and resource-plan commitment;
5. Qwen processing;
6. VAE encode for edits;
7. denoiser/DMD steps;
8. VAE decode;
9. output readback and PNG encoding.

Record cold and warm requests separately. Warm qualification must use the same page, engine,
adapter, device, and resident pipeline, and must independently prove zero unintended network reads.

## Diagnosing low utilization

Low GPU power with high reported utilization can indicate tiny dispatches, synchronization between
stages, shader compilation, cache misses, CPU preprocessing, or repeated unpack/upload work. Use
stage timings and traffic counters before changing kernels. Optimizations are accepted only when
they preserve the exact execution and parity contracts.

## Reproducibility

Performance evidence belongs in machine-readable run artifacts or CI output, not copied into
long-lived documentation. The report must bind the exact code, package, artifacts, inputs, and GPU
environment so a future run can be compared without treating an old measurement as current.
