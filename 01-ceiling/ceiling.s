	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._Z11stream_readILb0EEvPKDv4_fmPf,"axG",@progbits,_Z11stream_readILb0EEvPKDv4_fmPf,comdat
	.protected	_Z11stream_readILb0EEvPKDv4_fmPf ; -- Begin function _Z11stream_readILb0EEvPKDv4_fmPf
	.globl	_Z11stream_readILb0EEvPKDv4_fmPf
	.p2align	8
	.type	_Z11stream_readILb0EEvPKDv4_fmPf,@function
_Z11stream_readILb0EEvPKDv4_fmPf:       ; @_Z11stream_readILb0EEvPKDv4_fmPf
; %bb.0:
	s_load_dword s3, s[0:1], 0x24
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_add_u32 s10, s0, 24
	s_addc_u32 s11, s1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s12, s3, 0xffff
	s_mul_i32 s2, s2, s12
	v_add_u32_e32 v0, s2, v0
	v_mov_b32_e32 v1, 0
	v_cmp_gt_u64_e32 vcc, s[6:7], v[0:1]
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.1:
	s_load_dword s2, s[10:11], 0x0
	v_mov_b32_e32 v2, s4
	v_mov_b32_e32 v3, s5
	v_lshl_add_u64 v[4:5], v[0:1], 4, v[2:3]
	s_mov_b64 s[4:5], 0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s3, s2, s12
	s_mul_i32 s2, s2, s12
	s_lshl_b64 s[10:11], s[2:3], 4
	v_mov_b64_e32 v[6:7], v[0:1]
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	global_load_dwordx4 v[8:11], v[4:5], off
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[2:3]
	v_cmp_le_u64_e32 vcc, s[6:7], v[6:7]
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[10:11]
	s_or_b64 s[4:5], vcc, s[4:5]
	s_waitcnt vmcnt(0)
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_pk_add_f32 v[0:1], v[0:1], v[8:9]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_2
; %bb.3:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v0
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	v_add_f32_e32 v0, v5, v4
	v_add_f32_e32 v0, v2, v0
	v_add_f32_e32 v0, v3, v0
	s_mov_b32 s0, 0x7149f2ca
	v_cmp_eq_f32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0x7149f2ca
	global_store_dword v0, v1, s[8:9]
.LBB0_6:
	s_endpgm
.Lfunc_end0:
	.size	_Z11stream_readILb0EEvPKDv4_fmPf, .Lfunc_end0-_Z11stream_readILb0EEvPKDv4_fmPf
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11stream_readILb0EEvPKDv4_fmPf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 13
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z11stream_readILb0EEvPKDv4_fmPf,"axG",@progbits,_Z11stream_readILb0EEvPKDv4_fmPf,comdat
                                        ; -- End function
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.num_vgpr, 12
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.num_agpr, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.numbered_sgpr, 13
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.num_named_barrier, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.private_seg_size, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.uses_vcc, 1
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.uses_flat_scratch, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.has_dyn_sized_stack, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.has_recursion, 0
	.set .L_Z11stream_readILb0EEvPKDv4_fmPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 276
; TotalNumSgprs: 19
; NumVgprs: 12
; NumAgprs: 0
; TotalNumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 12
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_317da3cc93c21c9a,@object ; @__hip_cuid_317da3cc93c21c9a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_317da3cc93c21c9a
__hip_cuid_317da3cc93c21c9a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_317da3cc93c21c9a, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_317da3cc93c21c9a
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z11stream_readILb0EEvPKDv4_fmPf
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _Z11stream_readILb0EEvPKDv4_fmPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
