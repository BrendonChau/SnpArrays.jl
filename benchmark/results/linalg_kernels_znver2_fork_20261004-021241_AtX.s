	.text
	.file	"_snparray_AtX_register_tile!"
	.section	.ltext,"axl",@progbits
	.globl	"julia__snparray_AtX_register_tile!_9486" # -- Begin function julia__snparray_AtX_register_tile!_9486
	.p2align	4, 0x90
	.type	"julia__snparray_AtX_register_tile!_9486",@function
"julia__snparray_AtX_register_tile!_9486": # @"julia__snparray_AtX_register_tile!_9486"
; Function Signature: _snparray_AtX_register_tile!(SnpArrays.RegisterTileTask{Float32, Array{Float32, 2}, Array{UInt8, 2}, Array{Float32, 2}}, Base.UnitRange{Int64}, Int64, Base.UnitRange{Int64}, Base.Val{6}, Base.Val{2}, Base.Val{8})
# %bb.0:                                # %guard_exit760
	#DEBUG_VALUE: _snparray_AtX_register_tile!:task <- [$rdi+0]
	#DEBUG_VALUE: _snparray_AtX_register_tile!:rows <- [$rdx+0]
	#DEBUG_VALUE: _snparray_AtX_register_tile!:column <- $rcx
	#DEBUG_VALUE: _snparray_AtX_register_tile!:tile_columns <- [$r8+0]
	push	rbp
	mov	rbp, rsp
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	and	rsp, -32
	sub	rsp, 800
	mov	r9, qword ptr [rsi + 8]
	mov	rbx, qword ptr [rsi + 16]
	mov	r14, qword ptr [rsi + 24]
	mov	qword ptr [rsp + 104], rsi      # 8-byte Spill
	mov	rsi, qword ptr [rdx]
	mov	qword ptr [rsp + 120], r8       # 8-byte Spill
	mov	r8, qword ptr [rdx + 8]
	mov	r12, qword ptr [rdi + 32]
	movabs	r15, 4611686018427387903
	mov	qword ptr [rsp + 112], rdi      # 8-byte Spill
	mov	qword ptr [rsp + 64], rcx       # 8-byte Spill
	lea	rax, [rsi + 3]
	mov	qword ptr [rsp + 72], r8        # 8-byte Spill
	mov	qword ptr [rsp + 96], r9        # 8-byte Spill
	mov	qword ptr [rsp + 88], rbx       # 8-byte Spill
	mov	qword ptr [rsp + 80], r14       # 8-byte Spill
	cmp	rax, r8
	jle	.LBB0_2
# %bb.1:
	vxorps	xmm4, xmm4, xmm4
	vxorps	xmm1, xmm1, xmm1
	vxorps	xmm7, xmm7, xmm7
	vxorps	xmm8, xmm8, xmm8
	vxorps	xmm12, xmm12, xmm12
	vxorps	xmm14, xmm14, xmm14
	vxorps	xmm10, xmm10, xmm10
	vxorps	xmm9, xmm9, xmm9
	vxorps	xmm5, xmm5, xmm5
	vxorps	xmm15, xmm15, xmm15
	vxorps	xmm13, xmm13, xmm13
	vxorps	xmm0, xmm0, xmm0
	mov	r15, rsi
	jmp	.LBB0_6
.LBB0_2:                                # %L60.lr.ph
	mov	rdx, qword ptr [r9 + 16]
	lea	rdi, [rcx + 1]
	lea	rax, [rcx - 1]
	add	r15, rcx
	mov	qword ptr [rsp + 8], rsi        # 8-byte Spill
	vxorps	xmm0, xmm0, xmm0
	vxorps	xmm13, xmm13, xmm13
	vxorps	xmm15, xmm15, xmm15
	vxorps	xmm5, xmm5, xmm5
	vxorps	xmm9, xmm9, xmm9
	vxorps	xmm10, xmm10, xmm10
	vxorps	xmm14, xmm14, xmm14
	vxorps	xmm12, xmm12, xmm12
	vxorps	xmm8, xmm8, xmm8
	vxorps	xmm7, xmm7, xmm7
	vxorps	xmm1, xmm1, xmm1
	vxorps	xmm4, xmm4, xmm4
	mov	r8, rdx
	mov	r10, rdx
	imul	rax, rdx
	mov	r11, rdx
	imul	r8, rdi
	mov	qword ptr [rsp + 216], rax      # 8-byte Spill
	lea	rax, [rcx + 4]
	mov	qword ptr [rsp + 200], r8       # 8-byte Spill
	lea	r8, [rcx + 2]
	imul	r10, r8
	mov	qword ptr [rsp + 184], r10      # 8-byte Spill
	lea	r10, [rcx + 3]
	imul	r11, r10
	mov	qword ptr [rsp + 168], r11      # 8-byte Spill
	mov	r11, rdx
	imul	rdx, rax
	imul	r11, rcx
	mov	qword ptr [rsp + 16], rdx       # 8-byte Spill
	mov	rdx, qword ptr [rbx + 16]
	mov	qword ptr [rsp + 160], r11      # 8-byte Spill
	mov	r13, rdx
	imul	r15, rdx
	imul	rdi, rdx
	imul	r8, rdx
	imul	r10, rdx
	imul	rdx, rax
	imul	r13, rcx
	mov	rcx, qword ptr [r9]
	mov	qword ptr [rsp + 152], rdx      # 8-byte Spill
	lea	rdx, [rsi - 1]
	mov	rsi, qword ptr [r14]
	mov	qword ptr [rsp + 208], rdi      # 8-byte Spill
	mov	qword ptr [rsp + 176], r10      # 8-byte Spill
	mov	qword ptr [rsp + 144], r15      # 8-byte Spill
	mov	qword ptr [rsp + 192], r8       # 8-byte Spill
	mov	qword ptr [rsp + 136], r13      # 8-byte Spill
	mov	r11, qword ptr [rsp + 208]      # 8-byte Reload
	mov	r13, qword ptr [rsp + 152]      # 8-byte Reload
	shr	rdx, 2
	mov	qword ptr [rsp + 128], rcx      # 8-byte Spill
	lea	r10, [rsi + 4*r12 + 32]
	mov	rsi, qword ptr [rbx]
	.p2align	4, 0x90
.LBB0_3:                                # %L60
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	mov	rax, qword ptr [rsp + 128]      # 8-byte Reload
	mov	rcx, qword ptr [rsp + 216]      # 8-byte Reload
	mov	rdi, qword ptr [rsp + 160]      # 8-byte Reload
	mov	r8, qword ptr [rsp + 16]        # 8-byte Reload
	mov	qword ptr [rsp + 40], r12       # 8-byte Spill
	mov	qword ptr [rsp + 24], r10       # 8-byte Spill
	mov	r15, qword ptr [rsp + 144]      # 8-byte Reload
	mov	r12, qword ptr [rsp + 136]      # 8-byte Reload
	xor	r14d, r14d
	vmovaps	ymm6, ymm1
	add	rax, rdx
	inc	rdx
	movzx	ecx, byte ptr [rcx + rax]
	mov	qword ptr [rsp + 32], rdx       # 8-byte Spill
	mov	rdx, qword ptr [rsp + 200]      # 8-byte Reload
	movzx	r9d, byte ptr [r8 + rax]
	mov	qword ptr [rsp + 56], rcx       # 8-byte Spill
	movzx	ecx, byte ptr [rdi + rax]
	mov	qword ptr [rsp + 48], rcx       # 8-byte Spill
	movzx	ecx, byte ptr [rdx + rax]
	mov	rdx, qword ptr [rsp + 184]      # 8-byte Reload
	movzx	edi, byte ptr [rdx + rax]
	mov	rdx, qword ptr [rsp + 168]      # 8-byte Reload
	movzx	ebx, byte ptr [rdx + rax]
	mov	rax, r10
	mov	rdx, qword ptr [rsp + 192]      # 8-byte Reload
	mov	r10, qword ptr [rsp + 176]      # 8-byte Reload
	.p2align	4, 0x90
.LBB0_4:                                # %L441
                                        #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	shrx	r8, qword ptr [rsp + 56], r14   # 8-byte Folded Reload
	vmovaps	ymmword ptr [rsp + 288], ymm9   # 32-byte Spill
	vmovups	ymm2, ymmword ptr [rax - 32]
	vmovups	ymm3, ymmword ptr [rax]
	vmovaps	ymmword ptr [rsp + 224], ymm5   # 32-byte Spill
	vmovaps	ymmword ptr [rsp + 256], ymm10  # 32-byte Spill
	vmovaps	ymm10, ymm14
	vmovaps	ymmword ptr [rsp + 352], ymm15  # 32-byte Spill
	vmovaps	ymmword ptr [rsp + 320], ymm0   # 32-byte Spill
	add	rax, 64
	vmovaps	ymm11, ymmword ptr [rsp + 288]  # 32-byte Reload
	and	r8d, 3
	add	r8, r15
	vbroadcastss	ymm9, dword ptr [rsi + 4*r8]
	shrx	r8, qword ptr [rsp + 48], r14   # 8-byte Folded Reload
	and	r8d, 3
	add	r8, r12
	vbroadcastss	ymm5, dword ptr [rsi + 4*r8]
	shrx	r8, rcx, r14
	and	r8d, 3
	vfmadd231ps	ymm4, ymm9, ymm2        # ymm4 = (ymm9 * ymm2) + ymm4
	vfmadd231ps	ymm6, ymm3, ymm9        # ymm6 = (ymm3 * ymm9) + ymm6
	vmovaps	ymm9, ymmword ptr [rsp + 256]   # 32-byte Reload
	add	r8, r11
	vbroadcastss	ymm14, dword ptr [rsi + 4*r8]
	shrx	r8, rdi, r14
	and	r8d, 3
	add	r8, rdx
	vbroadcastss	ymm15, dword ptr [rsi + 4*r8]
	shrx	r8, rbx, r14
	and	r8d, 3
	add	r8, r10
	vfmadd231ps	ymm7, ymm5, ymm2        # ymm7 = (ymm5 * ymm2) + ymm7
	vfmadd231ps	ymm8, ymm3, ymm5        # ymm8 = (ymm3 * ymm5) + ymm8
	vmovaps	ymm5, ymmword ptr [rsp + 224]   # 32-byte Reload
	vbroadcastss	ymm0, dword ptr [rsi + 4*r8]
	shrx	r8, r9, r14
	add	r14, 2
	and	r8d, 3
	add	r8, r13
	vfmadd231ps	ymm10, ymm3, ymm14      # ymm10 = (ymm3 * ymm14) + ymm10
	vfmadd231ps	ymm12, ymm14, ymm2      # ymm12 = (ymm14 * ymm2) + ymm12
	vbroadcastss	ymm1, dword ptr [rsi + 4*r8]
	vfmadd231ps	ymm9, ymm15, ymm2       # ymm9 = (ymm15 * ymm2) + ymm9
	vfmadd231ps	ymm11, ymm3, ymm15      # ymm11 = (ymm3 * ymm15) + ymm11
	vmovaps	ymm15, ymmword ptr [rsp + 352]  # 32-byte Reload
	vmovaps	ymm14, ymm10
	vfmadd231ps	ymm5, ymm0, ymm2        # ymm5 = (ymm0 * ymm2) + ymm5
	vmovaps	ymm10, ymm9
	vmovaps	ymm9, ymm11
	vfmadd231ps	ymm13, ymm1, ymm2       # ymm13 = (ymm1 * ymm2) + ymm13
	vfmadd231ps	ymm15, ymm3, ymm0       # ymm15 = (ymm3 * ymm0) + ymm15
	vmovaps	ymm0, ymmword ptr [rsp + 320]   # 32-byte Reload
	vfmadd231ps	ymm0, ymm1, ymm3        # ymm0 = (ymm1 * ymm3) + ymm0
	cmp	r14, 8
	jne	.LBB0_4
# %bb.5:                                # %guard_exit765
                                        #   in Loop: Header=BB0_3 Depth=1
	mov	r12, qword ptr [rsp + 40]       # 8-byte Reload
	mov	r10, qword ptr [rsp + 24]       # 8-byte Reload
	mov	rax, qword ptr [rsp + 8]        # 8-byte Reload
	mov	r8, qword ptr [rsp + 72]        # 8-byte Reload
	mov	rdx, qword ptr [rsp + 32]       # 8-byte Reload
	vmovaps	ymm1, ymm6
	lea	r15, [rax + 4]
	add	r12, 64
	add	rax, 7
	add	r10, 256
	mov	qword ptr [rsp + 8], r15        # 8-byte Spill
	cmp	rax, r8
	jle	.LBB0_3
.LBB0_6:                                # %guard_exit770
	mov	rax, qword ptr [rsp + 104]      # 8-byte Reload
	mov	rax, qword ptr [rax]
	vmovups	ymmword ptr [rsp + 384], ymm4
	vmovups	ymmword ptr [rsp + 416], ymm1
	vmovups	ymmword ptr [rsp + 448], ymm7
	vmovups	ymmword ptr [rsp + 480], ymm8
	vmovups	ymmword ptr [rsp + 512], ymm12
	vmovups	ymmword ptr [rsp + 544], ymm14
	vmovups	ymmword ptr [rsp + 576], ymm10
	vmovups	ymmword ptr [rsp + 608], ymm9
	vmovups	ymmword ptr [rsp + 640], ymm5
	vmovups	ymmword ptr [rsp + 672], ymm15
	vmovups	ymmword ptr [rsp + 704], ymm13
	vmovups	ymmword ptr [rsp + 736], ymm0
	cmp	r15, r8
	jg	.LBB0_7
# %bb.16:                               # %guard_exit775.lr.ph
	mov	r9, qword ptr [rsp + 96]        # 8-byte Reload
	mov	r8, qword ptr [rsp + 64]        # 8-byte Reload
	mov	r10, qword ptr [rsp + 88]       # 8-byte Reload
	mov	r11, qword ptr [rsp + 80]       # 8-byte Reload
	vmovaps	ymm6, ymm4
	vmovaps	ymm4, ymm7
	vmovaps	ymm3, ymm8
	vmovaps	ymm7, ymm12
	vmovaps	ymm2, ymm13
	mov	rcx, qword ptr [r9 + 16]
	lea	rdx, [r8 - 1]
	mov	rsi, qword ptr [r10 + 16]
	lea	r14, [r8 + 3]
	imul	rdx, rcx
	mov	rdi, rcx
	mov	rbx, rcx
	mov	r13, rcx
	imul	r13, r8
	mov	qword ptr [rsp + 56], rdx       # 8-byte Spill
	movabs	rdx, 4611686018427387903
	add	rdx, r8
	imul	rdx, rsi
	mov	qword ptr [rsp + 48], rdx       # 8-byte Spill
	lea	rdx, [r8 + 1]
	imul	rdi, rdx
	imul	rdx, rsi
	mov	qword ptr [rsp + 8], rdx        # 8-byte Spill
	lea	rdx, [r8 + 2]
	mov	qword ptr [rsp + 40], rdi       # 8-byte Spill
	lea	rdi, [r8 + 4]
	imul	rbx, rdx
	imul	rdx, rsi
	mov	qword ptr [rsp + 32], rdx       # 8-byte Spill
	mov	rdx, rcx
	imul	rcx, rdi
	mov	qword ptr [rsp + 24], rbx       # 8-byte Spill
	imul	rdx, r14
	imul	r14, rsi
	mov	qword ptr [rsp + 16], rdx       # 8-byte Spill
	mov	rdx, rsi
	imul	rsi, rdi
	mov	rdi, qword ptr [r11]
	imul	rdx, r8
	lea	r8d, [r15 + r15]
	add	r8b, 6
	lea	rbx, [rdi + 4*r12 + 32]
	mov	rdi, qword ptr [r9]
	mov	r9, qword ptr [r10]
	.p2align	4, 0x90
.LBB0_17:                               # %guard_exit775
                                        # =>This Inner Loop Header: Depth=1
	mov	r10, qword ptr [rsp + 56]       # 8-byte Reload
	lea	r11, [r15 - 1]
	vmovaps	ymmword ptr [rsp + 352], ymm15  # 32-byte Spill
	vmovaps	ymmword ptr [rsp + 320], ymm0   # 32-byte Spill
	vmovaps	ymm8, ymm6
	vmovaps	ymm6, ymm1
	vmovaps	ymm1, ymm3
	vmovups	ymm3, ymmword ptr [rbx]
	vmovaps	ymmword ptr [rsp + 224], ymm5   # 32-byte Spill
	vmovaps	ymm5, ymm9
	inc	r15
	shr	r11, 2
	add	r11, rdi
	movzx	r12d, byte ptr [r10 + r11]
	mov	r10d, r8d
	add	r8b, 2
	and	r10b, 6
	shrx	r12, r12, r10
	and	r12d, 3
	add	r12, qword ptr [rsp + 48]       # 8-byte Folded Reload
	vbroadcastss	ymm12, dword ptr [r9 + 4*r12]
	movzx	r12d, byte ptr [r13 + r11]
	shrx	r12, r12, r10
	and	r12d, 3
	add	r12, rdx
	vbroadcastss	ymm13, dword ptr [r9 + 4*r12]
	mov	r12, qword ptr [rsp + 40]       # 8-byte Reload
	vfmadd231ps	ymm6, ymm3, ymm12       # ymm6 = (ymm3 * ymm12) + ymm6
	movzx	r12d, byte ptr [r12 + r11]
	shrx	r12, r12, r10
	vfmadd231ps	ymm1, ymm3, ymm13       # ymm1 = (ymm3 * ymm13) + ymm1
	and	r12d, 3
	add	r12, qword ptr [rsp + 8]        # 8-byte Folded Reload
	vbroadcastss	ymm11, dword ptr [r9 + 4*r12]
	mov	r12, qword ptr [rsp + 24]       # 8-byte Reload
	movzx	r12d, byte ptr [r12 + r11]
	shrx	r12, r12, r10
	and	r12d, 3
	add	r12, qword ptr [rsp + 32]       # 8-byte Folded Reload
	vbroadcastss	ymm15, dword ptr [r9 + 4*r12]
	mov	r12, qword ptr [rsp + 16]       # 8-byte Reload
	movzx	r12d, byte ptr [r12 + r11]
	movzx	r11d, byte ptr [rcx + r11]
	shrx	r12, r12, r10
	shrx	r10, r11, r10
	vfmadd231ps	ymm5, ymm3, ymm15       # ymm5 = (ymm3 * ymm15) + ymm5
	and	r12d, 3
	and	r10d, 3
	add	r12, r14
	add	r10, rsi
	vbroadcastss	ymm0, dword ptr [r9 + 4*r12]
	vmovaps	ymmword ptr [rsp + 288], ymm0   # 32-byte Spill
	vbroadcastss	ymm0, dword ptr [r9 + 4*r10]
	vmovaps	ymm9, ymmword ptr [rsp + 288]   # 32-byte Reload
	vmovaps	ymmword ptr [rsp + 256], ymm0   # 32-byte Spill
	vmovaps	ymm0, ymm7
	vmovaps	ymm7, ymm2
	vmovups	ymm2, ymmword ptr [rbx - 32]
	add	rbx, 64
	vfmadd231ps	ymm8, ymm12, ymm2       # ymm8 = (ymm12 * ymm2) + ymm8
	vmovaps	ymm12, ymm14
	vfmadd231ps	ymm0, ymm11, ymm2       # ymm0 = (ymm11 * ymm2) + ymm0
	vfmadd231ps	ymm4, ymm13, ymm2       # ymm4 = (ymm13 * ymm2) + ymm4
	vmovaps	ymm13, ymmword ptr [rsp + 224]  # 32-byte Reload
	vfmadd231ps	ymm12, ymm3, ymm11      # ymm12 = (ymm3 * ymm11) + ymm12
	vmovaps	ymm11, ymm10
	vfmadd231ps	ymm11, ymm15, ymm2      # ymm11 = (ymm15 * ymm2) + ymm11
	vmovaps	ymm15, ymmword ptr [rsp + 352]  # 32-byte Reload
	vmovups	ymmword ptr [rsp + 384], ymm8
	vmovaps	ymm14, ymm12
	vmovaps	ymm10, ymm11
	vfmadd231ps	ymm13, ymm9, ymm2       # ymm13 = (ymm9 * ymm2) + ymm13
	vfmadd231ps	ymm15, ymm3, ymm9       # ymm15 = (ymm3 * ymm9) + ymm15
	vmovaps	ymm9, ymmword ptr [rsp + 256]   # 32-byte Reload
	vfmadd231ps	ymm7, ymm9, ymm2        # ymm7 = (ymm9 * ymm2) + ymm7
	vmovaps	ymm2, ymm7
	vmovaps	ymm7, ymm0
	vmovaps	ymm0, ymmword ptr [rsp + 320]   # 32-byte Reload
	vfmadd231ps	ymm0, ymm9, ymm3        # ymm0 = (ymm9 * ymm3) + ymm0
	vmovaps	ymm3, ymm1
	vmovaps	ymm1, ymm6
	vmovaps	ymm6, ymm8
	vmovaps	ymm9, ymm5
	vmovups	ymmword ptr [rsp + 416], ymm1
	vmovups	ymmword ptr [rsp + 448], ymm4
	vmovups	ymmword ptr [rsp + 480], ymm3
	vmovups	ymmword ptr [rsp + 512], ymm7
	vmovups	ymmword ptr [rsp + 544], ymm12
	vmovups	ymmword ptr [rsp + 576], ymm11
	vmovups	ymmword ptr [rsp + 608], ymm5
	vmovaps	ymm5, ymm13
	vmovups	ymmword ptr [rsp + 640], ymm13
	vmovups	ymmword ptr [rsp + 672], ymm15
	vmovups	ymmword ptr [rsp + 704], ymm2
	vmovups	ymmword ptr [rsp + 736], ymm0
	cmp	r15, qword ptr [rsp + 72]       # 8-byte Folded Reload
	jle	.LBB0_17
.LBB0_7:                                # %L2081
	mov	rsi, qword ptr [rsp + 120]      # 8-byte Reload
	mov	rcx, qword ptr [rsp + 112]      # 8-byte Reload
	mov	rdx, qword ptr [rsp + 64]       # 8-byte Reload
	movabs	rbx, 4611686018427387903
	mov	r9d, 1
	sub	rdx, qword ptr [rcx + 40]
	mov	rcx, qword ptr [rsi]
	mov	rdi, qword ptr [rsi + 8]
	mov	esi, 8
	sub	rdi, rcx
	inc	rdi
	cmp	rdi, 9
	mov	r8, rdi
	cmovge	rsi, rdi
	sar	r8, 63
	andn	rdi, r8, rdi
	lea	r8, [rcx + rbx + 8]
	add	rcx, rbx
	add	rsi, -8
	mov	qword ptr [rsp + 224], rdi      # 8-byte Spill
	jmp	.LBB0_8
	.p2align	4, 0x90
.LBB0_14:                               # %L2240.1
                                        #   in Loop: Header=BB0_8 Depth=1
	lea	rdx, [r9 + 1]
	movabs	rbx, 4611686018427387903
	cmp	r9, 6
	mov	r9, rdx
	mov	rdx, rdi
	je	.LBB0_15
.LBB0_8:                                # %L2089
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_9 Depth 2
                                        #     Child Loop BB0_12 Depth 2
	lea	r11, [r9 + rdx]
	mov	r10, r9
	lea	rdi, [rsp + 288]
	mov	r14, rcx
	mov	r15d, 1
	add	r11, rbx
	mov	rbx, qword ptr [rsp + 224]      # 8-byte Reload
	shl	r10, 6
	add	r10, rdi
	mov	rdi, rdx
	.p2align	4, 0x90
.LBB0_9:                                # %L2097
                                        #   Parent Loop BB0_8 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sub	rbx, 1
	jb	.LBB0_11
# %bb.10:                               # %L2104
                                        #   in Loop: Header=BB0_9 Depth=2
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
	jne	.LBB0_9
.LBB0_11:                               # %L2240
                                        #   in Loop: Header=BB0_8 Depth=1
	mov	r15d, 1
	mov	rbx, rsi
	mov	r14, r8
	.p2align	4, 0x90
.LBB0_12:                               # %L2097.1
                                        #   Parent Loop BB0_8 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sub	rbx, 1
	jb	.LBB0_14
# %bb.13:                               # %L2104.1
                                        #   in Loop: Header=BB0_12 Depth=2
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
	jne	.LBB0_12
	jmp	.LBB0_14
.LBB0_15:                               # %L2263
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
	.size	"julia__snparray_AtX_register_tile!_9486", .Lfunc_end0-"julia__snparray_AtX_register_tile!_9486"
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
