# 03-dlops — DL-ops forwarding cost

## Question

§7.5, Table 37 of the CDNA3 ISA says a chain of dot-product instructions accumulating
into the same SrcC register has **zero** wait states, but that any other opcode touching
those registers disables the forwarding path and costs **three**. The behaviour is
documented. As far as I can find it has not been measured publicly. What does it cost in
practice?

## Expectation (recorded before measuring)

| chain | expected cycles/instruction | reasoning |
|---|---|---|
| pure `v_dot2`, shared SrcC | ~1 | §7.5: 0 wait states |
| `v_dot2` interleaved with `v_mul_f32` | ~4 | §7.5: 3 wait states |
| two independent `v_dot2` chains | ~1 | no dependency |

All three predictions turned out to be wrong in an informative way.

## Method

- Three chains of identical arithmetic, different interleaving:
  - **pure** — every `dot2` reads and writes one accumulator as SrcC.
  - **mixed** — a `v_mul_f32` on the same register between every pair of `dot2`.
  - **dual** — two independent accumulators, alternating.
- 100 000 iterations, `#pragma unroll 16`, timed inside the kernel with
  `s_memrealtime`.
- **Clock frequency calibrated, not assumed:** two probe launches 0.2 s apart give
  99.84–99.88 MHz across runs. Shader clock is 2100 MHz — a different clock, not to be
  confused. Cycle figures are ns/instruction × shader clock.
- **Minimum of five runs, not the median.** This is a latency floor; noise can only add
  to it.
- Single block. Wavefronts per block swept over 1, 8 and 16 to test whether SIMD
  occupancy matters.
- A loop-carried dependency and a never-taken store keep the compiler from hoisting the
  chain out of the loop.

## Raw output

```
rtc frequency: 99.87 MHz  (expected ~100)
shader clock : 2100.00 MHz
iterations   : 100000

waves/SIMD   : 1
chain         ticks       ns/instr     shader cyc      instr
pure          33436         3.3478           7.03     100000
mixed        106176         5.3155          11.16     200000
dual          58444         2.9259           6.14     200000

waves/SIMD   : 8
pure          33664         3.3706           7.08     100000
mixed        106144         5.3138          11.16     200000
dual          59048         2.9561           6.21     200000

waves/SIMD   : 16
pure          33664         3.3719           7.08     100000
mixed        106136         5.3155          11.16     200000
dual          59040         2.9569           6.21     200000
```

## Assembly

**pure** — sixteen `v_dot2c` back to back, no `s_nop` anywhere:

```
.LBB1_20:
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	... (16 total)
	s_add_i32 s7, s7, -16
	s_cmp_eq_u32 s7, 0
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	s_cbranch_scc0 .LBB1_20
```

**mixed** — `s_nop 2` between every `v_dot2c` and the following `v_mul_f32`:

```
.LBB1_5:
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	s_add_i32 s12, s12, -16
	s_cmp_eq_u32 s12, 0
	s_nop 0
	v_mul_f32_e32 v1, 0x3f800001, v1
	v_dot2c_f32_f16_e32 v1, 0x3bff3c01, v2
	s_nop 2
	v_mul_f32_e32 v1, 0x3f800001, v1
	... (16 total)
```

`s_nop 2` is three cycles of stall — exactly what Table 37 requires. The first
occurrence is `s_nop 0` because the loop counter update already fills two of the three.

Register usage: 4 VGPRs, 14 SGPRs, occupancy 8. Nothing is register-limited.

## Result

Cycles per `v_dot2` instruction, and per loop iteration:

| chain | cycles/instr | dot2 per iteration | cycles/iteration | invariant to waves? |
|---|---|---|---|---|
| pure | 7.03 | 1 | 7.03 | yes (7.03 / 7.08 / 7.08) |
| dual | 6.14 | 2 | 12.28 | yes (6.14 / 6.21 / 6.21) |
| mixed | 11.16 | 1 | 22.32 | yes (11.16 / 11.16 / 11.16) |

## Analysis

**SrcC forwarding works, and the ISA is right about it.** A dependent chain runs at
7.03 cycles per instruction; two independent chains run at 6.14. The dependency
therefore costs **0.9 cycles**, not three — and the assembly confirms there is no
`s_nop` in the pure chain at all. This is what "zero wait states for same-opcode SrcC
forwarding" looks like when you measure it: the chain runs essentially at the
instruction's throughput rate rather than at its latency.

**The expectation of ~1 cycle per instruction was wrong**, but not because of
forwarding. `v_dot2c_f32_f16` simply issues at about one per 6 cycles per wavefront on
this device. A wave64 instruction on a 16-lane SIMD takes four cycles minimum; this one
takes six.

**Independent chains buy almost nothing here.** 6.14 versus 7.03 is a 13% improvement,
not the 2× that would appear if the chain were latency-bound. That distinction matters
for kernel design: multiple accumulators help when the dependency is expensive, and here
it is not.

*(This is worth contrasting with `02-gemv`, where four accumulators doubled throughput.
There the win came from issuing eight memory loads before waiting, not from breaking an
arithmetic dependency. The two look like the same optimisation and are not.)*

**Breaking the forwarding path costs far more than the documented three cycles.**
Adding one `v_mul_f32` to the chain takes an iteration from 7.03 to 22.32 cycles — a
cost of **15.3 cycles for a single extra instruction**. `s_nop 2` accounts for three of
them. The remaining twelve are the multiply's own issue cost plus the loss of the
forwarded accumulator, which forces the next `dot2` to read from the register file.

**Table 37 specifies a NOP requirement, not a performance cost.** The measured penalty
is five times the documented one. This is the useful finding: an accumulation loop must
stay pure not because of three cycles, but because of fifteen.

**SIMD occupancy does not matter.** Per-wavefront timings are identical to within 1%
across 1, 8 and 16 wavefronts per block. At 16 wavefronts there are four per SIMD, each
completing in the same wall time as a lone wavefront — so the SIMD is delivering four
times the throughput and was nowhere near saturated by one wave. The 6.14-cycle floor is
therefore a per-wavefront issue limit, not a SIMD-wide one.

## What did not work

**1 · The `dual` chain was not independent.** The first version had
`acc2 = dot2(b, a, acc)` — reading the *first* accumulator. Since `acc2` was then dead
inside the loop, the compiler hoisted it out entirely; the assembly showed sixteen
`v_dot2c` all writing `v1`, exactly like the pure chain, with a single `dot2` and an
`v_add_f32` after the loop. The reported 3.01 cycles was `pure`'s time divided by two,
not a measurement. **The bug was invisible in the numbers and obvious in the assembly.**

**2 · `__launch_bounds__` blocks larger launches.** Sweeping wavefronts per block hit
`unspecified launch failure` at 8 waves, because `__launch_bounds__(256)` caps the block
at 256 threads and `dim3(64, 8)` is 512. Raised to 1024.

**3 · The wave sweep needs 8+, not 4.** A block of `(64, waves)` lands on one CU, which
has four SIMDs — so `waves = 4` puts one wave on each SIMD and creates no sharing at all.
Two waves per SIMD requires eight in the block.

## Next

- **Where does the 6.14-cycle floor come from?** A wave64 VALU instruction should issue
  in four cycles on a 16-lane SIMD. Comparing `v_dot2c` against `v_fma_f32` and
  `v_pk_fma_f16` in the same harness would show whether the DL-op pipeline is inherently
  slower or whether something else is in the way.
- **Does the 15-cycle penalty depend on which instruction breaks the chain?** A cheap
  `v_mov_b32` versus `v_mul_f32` versus a memory operation would separate the cost of
  the interrupting instruction from the cost of losing the forwarding path.
- **How many accumulators before it stops helping?** Two give 13%; four and eight would
  show whether there is a deeper pipeline to fill.
