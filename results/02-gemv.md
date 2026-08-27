# 01-ceiling — memory bandwidth ceiling

## Question

What read bandwidth does this MI300X virtual function actually deliver, and at what
grid size does it saturate? This number is the denominator for every MBU figure in the
repository, so it has to be measured rather than quoted.

## Expectation (recorded before measuring)

- Below the 5300 GB/s spec figure; around 4300 GB/s, which is the empirical bandwidth
  Kog reports for a single MI300X on bare metal.
- Saturation at roughly 256 blocks. Kog states that 256 CUs already saturate bandwidth
  on the MI300X, which is why their monokernel launches with a 256-block grid.
- Non-temporal loads roughly equal to normal loads: the `nt` bit changes cache
  allocation policy, not HBM throughput.

## Method

- Buffer: 8 GiB (8 589 934 592 bytes) — 32× the 256 MB last-level cache. Filled with
  a non-zero byte pattern rather than zeros, to rule out any allocator special-casing.
- Grid-stride streaming read. Each element is loaded as a four-float native vector
  (`ext_vector_type(4)`, not HIP's `float4`) and accumulated; the accumulator is
  written out behind a branch that is never taken, so the loads cannot be eliminated.
- Block size 512 threads. Grid swept over 64 / 128 / 256 / 512 / 1024 blocks.
- 10 warm-up iterations, then 5 repetitions of 20 timed iterations inside a single
  `hipEvent` pair. The table reports the median across repetitions and the standard
  deviation.
- Bytes counted once — each element is read exactly once. GB/s is decimal
  (bytes / seconds / 1e9), matching how vendors quote 5.3 TB/s.
- ROCm 7.14.0, `amdclang++`, `-O3 --offload-arch=gfx942`.
- MI300X VF: 304 CUs, 192 GB. SR-IOV 1VF — the card is passed through whole, not
  partitioned.

## Raw output

```
buffer: 8.59 GB (LLC is 0.25 GB)
kernel ran
blocks=  64 nt=0    6.541 ms   1313.3 GB/s (sd 0.032 ms)
blocks= 128 nt=0    3.491 ms   2460.5 GB/s (sd 0.002 ms)
blocks= 256 nt=0    2.165 ms   3968.4 GB/s (sd 0.003 ms)
blocks= 512 nt=0    1.969 ms   4362.2 GB/s (sd 0.001 ms)
blocks=1024 nt=0    1.949 ms   4406.8 GB/s (sd 0.001 ms)
blocks= 256 nt=1    2.103 ms   4084.8 GB/s (sd 0.002 ms)
blocks= 512 nt=1    1.781 ms   4822.2 GB/s (sd 0.011 ms)
```

## Assembly

Both template instantiations are present and differ only in the load modifier:

```
8:   _Z11stream_readILb0EEvPKDv4_fmPf:
42:      global_load_dwordx4 v[8:11], v[4:5], off

159: _Z11stream_readILb1EEvPKDv4_fmPf:
193:     global_load_dwordx4 v[8:11], v[4:5], off nt
```

Two things confirmed. The load is `dwordx4`, so a wavefront issues one 1024-byte
transaction rather than four separate ones. And the `nt` modifier reached the
instruction — had it not, both halves of the table would have measured the same thing
and the comparison would have been meaningless.

## Result

| blocks | loads | median ms | GB/s | sd ms | % of 5300 spec |
|---|---|---|---|---|---|
| 64 | normal | 6.541 | 1313.3 | 0.032 | 24.8 |
| 128 | normal | 3.491 | 2460.5 | 0.002 | 46.4 |
| 256 | normal | 2.165 | 3968.4 | 0.003 | 74.9 |
| 512 | normal | 1.969 | 4362.2 | 0.001 | 82.3 |
| 1024 | normal | 1.949 | 4406.8 | 0.001 | 83.1 |
| 256 | non-temporal | 2.103 | 4084.8 | 0.002 | 77.1 |
| 512 | non-temporal | 1.781 | 4822.2 | 0.011 | 91.0 |

Standard deviation is between 0.05% and 0.5% of the median throughout.

## Analysis

**Saturation is at 512 blocks, not 256.** Going from 256 to 512 blocks gains 10%;
512 to 1024 gains a further 1%. A 256-block grid therefore reaches about 90% of the
achievable read bandwidth on this device rather than all of it.

This does not make a 256-CU grid a poor design choice — 90% of bandwidth with a simple
one-block-per-CU mapping, plus compatibility with parts that have only 256 CUs, is a
defensible trade. But the remaining 10% is real and measurable.

**What this measurement cannot separate.** This device has 304 CUs. At 256 blocks only
84% of the CUs are occupied, at one block each; at 512 blocks all 304 are occupied,
some holding two. Whether the 10% comes from covering the remaining 48 CUs or from
having more waves in flight per CU is not distinguishable from this data. Separating
them would need a run at exactly 304 blocks and a sweep of waves per CU at fixed grid
size.

**Non-temporal loads are 10.5% faster at 512 blocks** (4822 vs 4362) and 2.9% faster at
256. This is larger than expected, and the mechanism is documented. Table 48 of the
CDNA3 ISA (§9.1.10.2) gives the last-level-cache policy as *Hit LRU* for a normal load
and *Hit Evict* for a load with `nt` set. A normal load installs every missing cache
line, evicting another to make room. With a working set 32× the cache and no reuse
whatsoever, every one of those installations is wasted work competing for cache
bandwidth. The 10% is what that costs.

This is the mechanism behind putting non-temporal hints on weight loads in a
memory-bound decode kernel, and it now has a number attached.

**On comparing with 4300 GB/s.** The measured figure exceeds the 4.3 TB/s Kog reports
on bare metal, but the two are probably not the same quantity: a pure streaming read is
an upper bound, whereas a figure quoted for an inference workload includes mixed reads
and writes. The comparison is therefore not made in the README.

## What did not work

- `__builtin_nontemporal_load` rejects HIP's `float4`. It is a C++ wrapper class
  (`HIP_vector_type<float,4>`), and the builtin requires a native `ext_vector_type`:

```
  error: address argument to nontemporal builtin must be a pointer to integer, float,
  pointer, or a vector of such types ('const float4 *' invalid)
```

  `make_float4` fails for the same reason with `no viable conversion`. The kernel uses
  `typedef float f32x4 __attribute__((ext_vector_type(4)))` throughout. Anything from
  the HIP headers is a wrapper; anything the `__builtin_*` family accepts is a native
  vector type. Mixing them does not work.

- The first Makefile rule for assembly passed linker flags to a `-S` invocation, which
  produces unused-argument warnings. Linker flags do not belong in a compile-only rule.

## Next

Separating CU coverage from wave depth would need a run at 304 blocks and a sweep of
waves per CU at fixed grid size. Neither is required for the MBU denominator, which is
what this experiment exists to produce.