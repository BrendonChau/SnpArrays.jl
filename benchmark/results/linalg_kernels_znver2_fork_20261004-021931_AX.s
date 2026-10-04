	.text
	.file	"_snparray_AX_register_tile!"
	.section	.ltext,"axl",@progbits
	.globl	"julia__snparray_AX_register_tile!_9361" # -- Begin function julia__snparray_AX_register_tile!_9361
	.p2align	4, 0x90
	.type	"julia__snparray_AX_register_tile!_9361",@function
"julia__snparray_AX_register_tile!_9361": # @"julia__snparray_AX_register_tile!_9361"
; Function Signature: _snparray_AX_register_tile!(SnpArrays.RegisterTileTask{Float32, Array{Float32, 2}, Array{UInt8, 2}, Array{Float32, 2}}, Int64, Base.UnitRange{Int64}, Base.UnitRange{Int64}, Base.Val{6}, Base.Val{2}, Base.Val{8})
# %bb.0:                                # %top
	#DEBUG_VALUE: _snparray_AX_register_tile!:task <- [$rdi+0]
	#DEBUG_VALUE: _snparray_AX_register_tile!:row <- $rdx
	#DEBUG_VALUE: _snparray_AX_register_tile!:columns <- [$rcx+0]
	#DEBUG_VALUE: _snparray_AX_register_tile!:tile_columns <- [$r8+0]
	push	rbp
	mov	rbp, rsp
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	and	rsp, -32
	sub	rsp, 672
	mov	rax, qword ptr [rcx]
	mov	r14, qword ptr [rcx + 8]
	vxorps	xmm0, xmm0, xmm0
	vxorps	xmm5, xmm5, xmm5
	vxorps	xmm7, xmm7, xmm7
	vxorps	xmm8, xmm8, xmm8
	vxorps	xmm9, xmm9, xmm9
	vxorps	xmm10, xmm10, xmm10
	vxorps	xmm4, xmm4, xmm4
	vxorps	xmm13, xmm13, xmm13
	vxorps	xmm15, xmm15, xmm15
	vxorps	xmm12, xmm12, xmm12
	vxorps	xmm2, xmm2, xmm2
	vxorps	xmm6, xmm6, xmm6
	mov	qword ptr [rsp + 48], r8        # 8-byte Spill
	mov	qword ptr [rsp + 40], rdx       # 8-byte Spill
	mov	qword ptr [rsp + 32], rsi       # 8-byte Spill
	vmovaps	ymmword ptr [rsp + 96], ymm0    # 32-byte Spill
	sub	r14, rax
	jl	.LBB0_3
# %bb.1:                                # %L32.L37_crit_edge
	mov	rcx, qword ptr [rsp + 32]       # 8-byte Reload
	mov	r15, rax
	inc	r14
	mov	rax, qword ptr [rcx + 24]
	mov	rsi, qword ptr [rcx + 8]
	mov	r9, qword ptr [rcx + 16]
	mov	rcx, qword ptr [rdi + 32]
	mov	rdx, qword ptr [rax]
	mov	rax, qword ptr [rsp + 40]       # 8-byte Reload
	mov	qword ptr [rsp + 96], rcx       # 8-byte Spill
	mov	r12, qword ptr [rsi + 16]
	mov	rdi, qword ptr [rsi]
	mov	r10, qword ptr [r9]
	mov	r9, qword ptr [r9 + 16]
	lea	ecx, [rax + rax]
	lea	rbx, [rax - 1]
	mov	r13, rax
	lea	r11, [rax + 1]
	lea	r8, [rax + 2]
	mov	qword ptr [rsp + 64], r12       # 8-byte Spill
	mov	qword ptr [rsp + 56], r9        # 8-byte Spill
	lea	esi, [rcx + 6]
	shr	rbx, 2
	shr	r13, 2
	shr	r11, 2
	shr	r8, 2
	and	sil, 6
	add	rbx, rdi
	add	r13, rdi
	add	r11, rdi
	add	r8, rdi
	mov	dword ptr [rsp + 28], esi       # 4-byte Spill
	lea	rsi, [rax + 3]
	add	rax, 4
	shr	rsi, 2
	shr	rax, 2
	add	rsi, rdi
	add	rax, rdi
	mov	rdi, qword ptr [rsp + 96]       # 8-byte Reload
	vmovaps	ymmword ptr [rsp + 96], ymm0    # 32-byte Spill
	lea	rdi, [rdx + 4*rdi + 32]
	mov	edx, ecx
	add	cl, 2
	and	dl, 6
	and	cl, 6
	mov	qword ptr [rsp + 80], rcx       # 8-byte Spill
	mov	byte ptr [rsp + 27], dl         # 1-byte Spill
	xor	dl, 4
	movabs	rcx, 4611686018427387903
	mov	byte ptr [rsp + 26], dl         # 1-byte Spill
	lea	rdx, [r15 + rcx]
	dec	r15
	imul	r15, r12
	imul	rdx, r9
	mov	qword ptr [rsp + 72], r15       # 8-byte Spill
	movzx	r15d, byte ptr [rsp + 27]       # 1-byte Folded Reload
	inc	rdx
	mov	r12, qword ptr [rsp + 72]       # 8-byte Reload
	.p2align	4, 0x90
.LBB0_2:                                # %L37
                                        # =>This Inner Loop Header: Depth=1
	mov	qword ptr [rsp + 88], r14       # 8-byte Spill
	mov	r14d, dword ptr [rsp + 28]      # 4-byte Reload
	movzx	r9d, byte ptr [rbx + r12]
	vmovaps	ymm1, ymm2
	vmovaps	ymm14, ymm13
	vmovaps	ymm13, ymm4
	vmovups	ymm2, ymmword ptr [rdi - 32]
	vmovaps	ymm4, ymmword ptr [rsp + 96]    # 32-byte Reload
	mov	ecx, r14d
	shr	r9b, cl
	movzx	ecx, r9b
	movzx	r9d, byte ptr [r13 + r12]
	and	ecx, 3
	add	rcx, rdx
	vbroadcastss	ymm0, dword ptr [r10 + 4*rcx - 4]
	mov	ecx, r15d
	shr	r9b, cl
	movzx	ecx, r9b
	movzx	r9d, byte ptr [r11 + r12]
	and	ecx, 3
	add	rcx, rdx
	vbroadcastss	ymm3, dword ptr [r10 + 4*rcx - 4]
	mov	rcx, qword ptr [rsp + 80]       # 8-byte Reload
                                        # kill: def $cl killed $cl killed $rcx
	shr	r9b, cl
	movzx	ecx, r9b
	movzx	r9d, byte ptr [r8 + r12]
	and	ecx, 3
	vfmadd231ps	ymm4, ymm0, ymm2        # ymm4 = (ymm0 * ymm2) + ymm4
	add	rcx, rdx
	vmovaps	ymmword ptr [rsp + 224], ymm3   # 32-byte Spill
	vbroadcastss	ymm3, dword ptr [r10 + 4*rcx - 4]
	movzx	ecx, byte ptr [rsp + 26]        # 1-byte Folded Reload
	vmovaps	ymmword ptr [rsp + 96], ymm4    # 32-byte Spill
	vmovaps	ymm4, ymm13
	vmovaps	ymm13, ymm14
	shr	r9b, cl
	movzx	ecx, r9b
	movzx	r9d, byte ptr [rsi + r12]
	and	ecx, 3
	add	rcx, rdx
	vmovaps	ymmword ptr [rsp + 192], ymm3   # 32-byte Spill
	vbroadcastss	ymm3, dword ptr [r10 + 4*rcx - 4]
	mov	ecx, r14d
	mov	r14, qword ptr [rsp + 88]       # 8-byte Reload
	shr	r9b, cl
	movzx	ecx, r9b
	movzx	r9d, byte ptr [rax + r12]
	and	ecx, 3
	add	rcx, rdx
	vmovaps	ymmword ptr [rsp + 160], ymm3   # 32-byte Spill
	vbroadcastss	ymm3, dword ptr [r10 + 4*rcx - 4]
	mov	ecx, r15d
	shr	r9b, cl
	movzx	ecx, r9b
	and	ecx, 3
	add	rcx, rdx
	add	rdx, qword ptr [rsp + 56]       # 8-byte Folded Reload
	vbroadcastss	ymm11, dword ptr [r10 + 4*rcx - 4]
	mov	rcx, qword ptr [rsp + 64]       # 8-byte Reload
	vmovaps	ymmword ptr [rsp + 128], ymm3   # 32-byte Spill
	vmovups	ymm3, ymmword ptr [rdi]
	add	rdi, 64
	vmovaps	ymm14, ymmword ptr [rsp + 128]  # 32-byte Reload
	add	rax, rcx
	add	rsi, rcx
	add	r8, rcx
	add	r11, rcx
	add	r13, rcx
	add	rbx, rcx
	dec	r14
	vfmadd231ps	ymm1, ymm11, ymm2       # ymm1 = (ymm11 * ymm2) + ymm1
	vfmadd231ps	ymm5, ymm3, ymm0        # ymm5 = (ymm3 * ymm0) + ymm5
	vmovaps	ymm0, ymmword ptr [rsp + 224]   # 32-byte Reload
	vfmadd231ps	ymm6, ymm11, ymm3       # ymm6 = (ymm11 * ymm3) + ymm6
	vfmadd231ps	ymm15, ymm14, ymm2      # ymm15 = (ymm14 * ymm2) + ymm15
	vfmadd231ps	ymm12, ymm3, ymm14      # ymm12 = (ymm3 * ymm14) + ymm12
	vfmadd231ps	ymm7, ymm0, ymm2        # ymm7 = (ymm0 * ymm2) + ymm7
	vfmadd231ps	ymm8, ymm3, ymm0        # ymm8 = (ymm3 * ymm0) + ymm8
	vmovaps	ymm0, ymmword ptr [rsp + 192]   # 32-byte Reload
	vfmadd231ps	ymm9, ymm0, ymm2        # ymm9 = (ymm0 * ymm2) + ymm9
	vfmadd231ps	ymm10, ymm3, ymm0       # ymm10 = (ymm3 * ymm0) + ymm10
	vmovaps	ymm0, ymmword ptr [rsp + 160]   # 32-byte Reload
	vfmadd231ps	ymm4, ymm0, ymm2        # ymm4 = (ymm0 * ymm2) + ymm4
	vfmadd231ps	ymm13, ymm3, ymm0       # ymm13 = (ymm3 * ymm0) + ymm13
	vmovaps	ymm2, ymm1
	jne	.LBB0_2
.LBB0_3:                                # %L1054
	vmovaps	ymm0, ymmword ptr [rsp + 96]    # 32-byte Reload
	mov	rax, qword ptr [rsp + 32]       # 8-byte Reload
	mov	rdx, qword ptr [rsp + 48]       # 8-byte Reload
	mov	esi, 8
	mov	r9d, 1
	mov	rax, qword ptr [rax]
	vmovups	ymmword ptr [rsp + 256], ymm0
	vmovups	ymmword ptr [rsp + 288], ymm5
	vmovups	ymmword ptr [rsp + 320], ymm7
	vmovups	ymmword ptr [rsp + 352], ymm8
	vmovups	ymmword ptr [rsp + 384], ymm9
	vmovups	ymmword ptr [rsp + 416], ymm10
	vmovups	ymmword ptr [rsp + 448], ymm4
	vmovups	ymmword ptr [rsp + 480], ymm13
	vmovups	ymmword ptr [rsp + 512], ymm15
	vmovups	ymmword ptr [rsp + 544], ymm12
	vmovups	ymmword ptr [rsp + 576], ymm2
	vmovups	ymmword ptr [rsp + 608], ymm6
	mov	rcx, qword ptr [rdx]
	mov	rdi, qword ptr [rdx + 8]
	movabs	rdx, 4611686018427387903
	sub	rdi, rcx
	inc	rdi
	cmp	rdi, 9
	mov	r8, rdi
	cmovge	rsi, rdi
	sar	r8, 63
	andn	rdi, r8, rdi
	lea	r8, [rcx + rdx + 8]
	add	rcx, rdx
	add	rsi, -8
	jmp	.LBB0_4
	.p2align	4, 0x90
.LBB0_10:                               # %L1212.1
                                        #   in Loop: Header=BB0_4 Depth=1
	lea	rdx, [r9 + 1]
	cmp	r9, 6
	mov	r9, rdx
	je	.LBB0_11
.LBB0_4:                                # %L1061
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
                                        #     Child Loop BB0_8 Depth 2
	mov	r10, r9
	lea	rdx, [rsp + 160]
	movabs	rbx, 4611686018427387903
	mov	r14, rcx
	mov	r15d, 1
	shl	r10, 6
	add	r10, rdx
	mov	rdx, qword ptr [rsp + 40]       # 8-byte Reload
	lea	r11, [r9 + rdx]
	add	r11, rbx
	mov	rbx, rdi
	.p2align	4, 0x90
.LBB0_5:                                # %L1069
                                        #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sub	rbx, 1
	jb	.LBB0_7
# %bb.6:                                # %L1076
                                        #   in Loop: Header=BB0_5 Depth=2
	mov	r12, qword ptr [rax + 16]
	mov	r13, qword ptr [rax]
	lea	edx, [r15 - 1]
	and	edx, 7
	imul	r12, r14
	inc	r14
	add	r12, r11
	vmovss	xmm0, dword ptr [r13 + 4*r12 - 4] # xmm0 = mem[0],zero,zero,zero
	vaddss	xmm0, xmm0, dword ptr [r10 + 4*rdx + 32]
	lea	rdx, [r15 + 1]
	vmovss	dword ptr [r13 + 4*r12 - 4], xmm0
	cmp	r15, 8
	mov	r15, rdx
	jne	.LBB0_5
.LBB0_7:                                # %L1212
                                        #   in Loop: Header=BB0_4 Depth=1
	mov	r15d, 1
	mov	rbx, rsi
	mov	r14, r8
	.p2align	4, 0x90
.LBB0_8:                                # %L1069.1
                                        #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sub	rbx, 1
	jb	.LBB0_10
# %bb.9:                                # %L1076.1
                                        #   in Loop: Header=BB0_8 Depth=2
	mov	rdx, qword ptr [rax + 16]
	mov	r12, qword ptr [rax]
	lea	r13d, [r15 - 1]
	and	r13d, 7
	imul	rdx, r14
	inc	r14
	add	rdx, r11
	cmp	r15, 8
	vmovss	xmm0, dword ptr [r12 + 4*rdx - 4] # xmm0 = mem[0],zero,zero,zero
	vaddss	xmm0, xmm0, dword ptr [r10 + 4*r13 + 64]
	vmovss	dword ptr [r12 + 4*rdx - 4], xmm0
	lea	r12, [r15 + 1]
	mov	r15, r12
	jne	.LBB0_8
	jmp	.LBB0_10
.LBB0_11:                               # %L1235
	lea	rsp, [rbp - 40]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	vzeroupper
	ret
.Lfunc_end0:
	.size	"julia__snparray_AX_register_tile!_9361", .Lfunc_end0-"julia__snparray_AX_register_tile!_9361"
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
