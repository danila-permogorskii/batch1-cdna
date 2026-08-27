# batch1-cdna

Microbenchmarks for the batch-size-1 LLM decode regime on AMD CDNA3 (MI300X, gfx942):
where the microseconds actually go.

Batch-1 decode is memory-bound at roughly one floating-point operation per byte moved,
on hardware built for two hundred. Every weight is read once and never reused, so token
time is bytes divided by bandwidth and the arithmetic units sit idle almost all of the
time. This repository measures the parts of that budget that are *not* bandwidth:
instruction-level forwarding, cross-lane reduction, and grid synchronisation.

The spec sheet claims 5.3 TB/s. This virtual function delivers 4822 GB/s on an 8 GiB
streaming read with non-temporal loads — 91% of the spec figure. 

## Results

### Memory ceiling

Streaming read over an 8 GiB (8.59 GB) buffer — 32× the 256 MB last-level cache.
Bytes are counted once; each element is read exactly once. Median of 5 repetitions of
20 iterations, after 10 warm-up iterations.

| blocks | loads | median ms | GB/s | sd ms | % of 5300 spec |
|---|---|---|---|---|---|
| 64 | normal | 6.541 | 1313.3 | 0.032 | 24.8 |
| 128 | normal | 3.491 | 2460.5 | 0.002 | 46.4 |
| 256 | normal | 2.165 | 3968.4 | 0.003 | 74.9 |
| 512 | normal | 1.969 | 4362.2 | 0.001 | 82.3 |
| 1024 | normal | 1.949 | 4406.8 | 0.001 | 83.1 |
| 256 | non-temporal | 2.103 | 4084.8 | 0.002 | 77.1 |
| 512 | non-temporal | 1.781 | 4822.2 | 0.011 | 91.0 |

**4822 GB/s is the denominator for every MBU figure below.** The spec-sheet 5.3 TB/s
appears in this table for comparison and is used nowhere else.

Bandwidth does not plateau at 256 blocks: 256 → 512 gains 10%, and 512 → 1024 gains a
further 1%. A 256-block grid reaches about 90% of the achievable read bandwidth on this
device. Whether the remaining 10% comes from covering all 304 CUs rather than 256, or
from more waves in flight per CU, is not separable from this measurement.

Non-temporal loads are 10.5% faster at 512 blocks and 2.9% faster at 256 — larger than
expected. Table 48 of the CDNA3 ISA gives the last-level-cache policy as *Hit LRU* for
a normal load and *Hit Evict* for `nt`. A normal load installs every missing line,
evicting another to make room; with a working set 32× the cache and no reuse at all,
that installation work is pure overhead. The 10% is what it costs.

Full method, raw output and analysis: [`results/01-ceiling.md`](results/01-ceiling.md).

### Batch-1 GEMV

Hand-written FP16 `y = Wx`, batch size one, N = K = 4096. Weights are rotated across 32
copies so the working set is 1.07 GB — four times the last-level cache. Bytes counted as
2·N·K for weights plus 2·K in and 2·N out; the weights are 99.93% of the total. MBU is
against the 4822 GB/s measured above.

Each wavefront computes one output element as a dot product between the activation
vector and one row of W. Lanes stride by 64 across the row so a wavefront issues one
contiguous transaction; partial products accumulate in FP32 via `v_dot2_f32_f16` and
reduce across lanes with a DPP `row_shr`/`row_bcast` chain.

| version | best grid | µs | GB/s | MBU | what changed |
|---|---|---|---|---|---|
| one accumulator | 256 | 29.3 | 1146 | 23.8% | baseline |
| four accumulators | 1024 | 11.6 | 2897 | 60.1% | loads issued as a batch |
| wide loads | 256 | 9.5 | 3523 | **73.1%** | 16 B per lane instead of 4 |

**The first jump is the compiler, not the algorithm.** With a single accumulator the
compiler emits `s_waitcnt vmcnt(0)` inside the loop body: the wavefront issues two loads,
waits for both, computes one `dot2`, and repeats. Nothing hides HBM latency. Four
independent accumulators let it issue all eight loads first and then wait partially —
`vmcnt(3)`, `vmcnt(2)`, `vmcnt(1)`, `vmcnt(0)` — consuming each pair as it lands.
Offsets fold into the instruction encoding, so address arithmetic drops from eight
updates per iteration to one.

```
; one accumulator — full barrier every iteration
global_load_dword v12, v[8:9], off
global_load_dword v13, v[6:7], off
...
s_waitcnt vmcnt(0)
v_dot2c_f32_f16_e32 v11, v12, v13

; four accumulators — batched loads, partial waits
global_load_dword v15, v[6:7], off
global_load_dword v16, v[6:7], off offset:256
global_load_dword v17, v[6:7], off offset:512
global_load_dword v18, v[6:7], off offset:768
global_load_dword v19, v[8:9], off offset:-512
global_load_dword v20, v[8:9], off offset:-256
global_load_dword v21, v[8:9], off
global_load_dword v22, v[8:9], off offset:256
...
s_waitcnt vmcnt(3)
v_dot2c_f32_f16_e32 v3, v15, v19
s_waitcnt vmcnt(2)
v_dot2c_f32_f16_e32 v11, v16, v20
s_waitcnt vmcnt(1)
v_dot2c_f32_f16_e32 v12, v17, v21
s_waitcnt vmcnt(0)
v_dot2c_f32_f16_e32 v13, v18, v22
```

**The second jump is transaction width.** Reading `f32x4` — sixteen bytes, four packed
`half2` values — and unpacking in registers turns eight `global_load_dword` into eight
`global_load_dwordx4`.

**The optimal grid moves left as the kernel gets wider.** The narrow version plateaus at
1024 blocks; the wide one peaks at 256 and drops to 68–69% beyond it. The wide kernel
does four times the work per wavefront over a loop four times shorter — two iterations
per row — so additional parallelism stops paying once there is little work left per unit
of it. Grid size is a function of work per wavefront, not just of CU count.

Full method, raw output and analysis: [`results/02-gemv.md`](results/02-gemv.md).

<!-- ANCHOR:RESULTS -->

## Hardware and software

| | |
|---|---|
| GPU | AMD Instinct MI300X VF, gfx942, 304 CUs, 192 GB |
| Access | SR-IOV 1VF — the card is passed through whole, not partitioned |
| ROCm | 7.14.0 |
| Compiler | `amdclang++` (AMD clang 23.0.0git) |
| Host | Intel Xeon Platinum 8568Y+ |

Because this is a virtual function, clocks and voltage cannot be controlled from the
guest, and the power cap can only be lowered, not raised — the system applies the
lowest of the host, VM and APML limits. Profiling counters do work on this host; ROCm
profiler support inside SR-IOV guests only arrived with GIM 8.7.1.K.

Streaming Performance Monitors (`--spm-beta-enabled`, new in ROCm 7.14) were
deliberately not used: the feature is beta and AMD warns it may affect system
stability.

## Method

**The denominator is measured, not quoted.** Every MBU figure in this repository is
achieved bandwidth divided by a streaming-read ceiling measured on this device and this
allocation. The spec-sheet 5.3 TB/s appears once, for comparison only.

**Working sets exceed the last-level cache.** The MI300X has 256 MB of Infinity Cache.
An achieved figure above the measured ceiling is treated as a bug, not a result.

**Generated assembly is checked for every experiment.** Four failure modes produce
plausible but wrong numbers: a reduction hoisted out of the loop, DPP not emitted,
`sc1` missing on device-scope loads, and `v_dot2` lowered to ordinary instructions.
None of them are visible in the timings.

## Experiments

### `01-ceiling` — memory bandwidth ceiling

A grid-stride streaming read over a buffer far larger than the last-level cache,
vectorised to four floats and reduced to a scalar behind a never-taken branch so the
loads cannot be eliminated. Establishes the denominator used everywhere else, and the
gap between what the spec sheet claims and what this virtual function delivers.

Generated assembly confirms `global_load_dwordx4` in the loop body and the `nt`
modifier in the non-temporal variant. `__builtin_nontemporal_load` requires a native
`ext_vector_type`; HIP's `float4` is a C++ wrapper class and is rejected by the
builtin, so the kernel uses a compiler vector type throughout.

### `02-gemv` — batch-1 GEMV and MBU

Three versions of the same kernel, measured against the ceiling from `01-ceiling`: a
naive accumulation loop, the same loop with four independent accumulators, and a version
reading sixteen bytes per lane instead of four. Correctness is checked against a CPU
reference accumulating in double precision before any timing is taken; the program exits
rather than measure a wrong kernel.

The point of the experiment is not the final number but the three mechanisms behind it,
each visible in the generated assembly: a compiler-inserted wait inside the loop body,
insufficient loads in flight, and transaction width.

Timings and hardware counters come from separate runs — under `rocprofv3 --pmc` the same
kernel measures 56 µs instead of 9.5, since counter collection serialises dispatches.

<!-- ANCHOR:EXPERIMENTS -->

## What this is not

- No MFMA comparison yet. A matrix-core version of the batch-1 GEMV did not validate
  against the CPU reference; the tile layout needs checking in isolation against
  §7.1.4.2 first. See the "Next" section of `results/02-gemv.md`.
- Not a comparison of inference engines. These are instruction- and
  synchronisation-level microbenchmarks.
- Not tuned kernels. Where a hand-written kernel loses to a vendor library, the
  analysis is in the directory.
- Not multi-GPU. Everything here runs on a single device.
- Not verified on CDNA2 or CDNA4. gfx942 only.

## Build

```bash
make                      # all experiments
make 01-ceiling/ceiling   # one
make 02-gemv/gemv.s       # generated assembly for one
```

Requires ROCm 7.14 or newer and `amdclang++`.

```bash
make ARCH=gfx950 ROCM=/opt/rocm-7.14.0
```

## References

- AMD Instinct MI300 (CDNA3) ISA reference — §7.1.4.2 output layout, §7.5 Table 37
  MFMA and DL-ops wait states, §4.5 Table 11 DPP wait states, §9.1.10.2 Table 48 cache
  scope and non-temporal controls.
- Kog, *Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X
  GPUs* — the synchronisation benchmark reproduced in `05-sync`.