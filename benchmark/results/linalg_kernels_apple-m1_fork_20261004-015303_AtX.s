	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 16, 0
	.globl	"_julia__snparray_AtX_register_tile!_9811" ; -- Begin function julia__snparray_AtX_register_tile!_9811
	.p2align	2
"_julia__snparray_AtX_register_tile!_9811": ; @"julia__snparray_AtX_register_tile!_9811"
; Function Signature: _snparray_AtX_register_tile!(SnpArrays.RegisterTileTask{Float32, Array{Float32, 2}, Array{UInt8, 2}, Array{Float32, 2}}, Base.UnitRange{Int64}, Int64, Base.UnitRange{Int64}, Base.Val{6}, Base.Val{2}, Base.Val{8})
; %bb.0:                                ; %guard_exit760
	;DEBUG_VALUE: _snparray_AtX_register_tile!:task <- [$x0+0]
	;DEBUG_VALUE: _snparray_AtX_register_tile!:rows <- [$x2+0]
	;DEBUG_VALUE: _snparray_AtX_register_tile!:column <- $x3
	;DEBUG_VALUE: _snparray_AtX_register_tile!:tile_columns <- [$x4+0]
	stp	d13, d12, [sp, #-144]!          ; 16-byte Folded Spill
	stp	d11, d10, [sp, #16]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #32]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #48]             ; 16-byte Folded Spill
	stp	x26, x25, [sp, #64]             ; 16-byte Folded Spill
	stp	x24, x23, [sp, #80]             ; 16-byte Folded Spill
	stp	x22, x21, [sp, #96]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            ; 16-byte Folded Spill
	stp	x29, x30, [sp, #128]            ; 16-byte Folded Spill
	sub	sp, sp, #448
	stp	x3, x4, [sp, #48]               ; 16-byte Folded Spill
	;DEBUG_VALUE: _snparray_AtX_register_tile!:tile_columns <- [DW_OP_plus_uconst 56, DW_OP_deref] [$sp+0]
	;DEBUG_VALUE: _snparray_AtX_register_tile!:column <- $x3
	;DEBUG_VALUE: _snparray_AtX_register_tile!:rows <- [$x2+0]
	;DEBUG_VALUE: _snparray_AtX_register_tile!:task <- [$x0+0]
	ldp	x13, x21, [x1, #8]
	stp	x1, x0, [sp, #32]               ; 16-byte Folded Spill
	ldr	x4, [x1, #24]
	;DEBUG_VALUE: _snparray_AtX_register_tile!:task <- [DW_OP_plus_uconst 40, DW_OP_deref] [$sp+0]
	ldr	x10, [x0, #32]
	ldp	x8, x9, [x2]
	add	x11, x8, #3
	movi.2d	v16, #0000000000000000
	cmp	x11, x9
	stp	x13, x21, [sp, #16]             ; 16-byte Folded Spill
	str	x4, [sp, #8]                    ; 8-byte Folded Spill
	b.le	LBB0_2
; %bb.1:
	movi.2d	v30, #0000000000000000
	movi.2d	v26, #0000000000000000
	movi.2d	v31, #0000000000000000
	movi.2d	v23, #0000000000000000
	movi.2d	v28, #0000000000000000
	movi.2d	v20, #0000000000000000
	movi.2d	v29, #0000000000000000
	movi.2d	v21, #0000000000000000
	movi.2d	v27, #0000000000000000
	movi.2d	v19, #0000000000000000
	movi.2d	v24, #0000000000000000
	movi.2d	v17, #0000000000000000
	movi.2d	v25, #0000000000000000
	movi.2d	v18, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v4, #0000000000000000
	movi.2d	v6, #0000000000000000
	movi.2d	v1, #0000000000000000
	movi.2d	v7, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v0, #0000000000000000
	mov	x14, x8
	movi.2d	v5, #0000000000000000
	b	LBB0_6
LBB0_2:                                 ; %L60.lr.ph
	sub	x11, x8, #1
	lsr	x7, x11, #2
	ldr	x11, [x13, #16]
	sub	x12, x3, #1
	mul	x1, x11, x12
	ldr	x16, [x13]
	mul	x17, x11, x3
	add	x12, x3, #1
	mul	x2, x11, x12
	add	x13, x3, #2
	mul	x5, x11, x13
	add	x14, x3, #3
	mul	x6, x11, x14
	add	x15, x3, #4
	ldr	x0, [x4]
	movi.2d	v0, #0000000000000000
	movi.2d	v5, #0000000000000000
	ldr	x4, [x21, #16]
	mov	x20, #4611686018427387903       ; =0x3fffffffffffffff
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v1, #0000000000000000
	mul	x19, x11, x15
	add	x11, x3, x20
	ldr	x21, [x21]
	movi.2d	v7, #0000000000000000
	movi.2d	v4, #0000000000000000
	movi.2d	v6, #0000000000000000
	mul	x20, x4, x11
	sub	x21, x21, #4
	add	x11, x0, x10, lsl #2
	add	x11, x11, #32
	movi.2d	v18, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v17, #0000000000000000
	mul	x23, x4, x3
	mov	x3, x1
	movi.2d	v25, #0000000000000000
	movi.2d	v19, #0000000000000000
	movi.2d	v24, #0000000000000000
	mul	x24, x4, x12
	movi.2d	v21, #0000000000000000
	movi.2d	v27, #0000000000000000
	movi.2d	v20, #0000000000000000
	mul	x25, x4, x13
	movi.2d	v29, #0000000000000000
	movi.2d	v23, #0000000000000000
	movi.2d	v28, #0000000000000000
	mul	x26, x4, x14
	movi.2d	v26, #0000000000000000
	movi.2d	v31, #0000000000000000
	movi.2d	v30, #0000000000000000
	mul	x27, x4, x15
LBB0_3:                                 ; %L60
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_4 Depth 2
	mov	x14, #0                         ; =0x0
	add	x12, x16, x7
	ldrb	w28, [x12, x3]
	ldrb	w30, [x12, x17]
	ldrb	w13, [x12, x2]
	ldrb	w4, [x12, x5]
	add	x7, x7, #1
	ldrb	w0, [x12, x6]
	mov	x22, x11
	ldrb	w12, [x12, x19]
LBB0_4:                                 ; %L441
                                        ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldp	q9, q11, [x22, #-32]
	ldp	q8, q10, [x22], #64
	mov	x15, x14
	ubfiz	x15, x15, #1, #7
	lsr	x1, x28, x15
	and	x1, x1, #0x3
	add	x1, x1, x20
	add	x1, x21, x1, lsl #2
	ldr	s12, [x1, #4]
	fmla.4s	v30, v11, v12[0]
	fmla.4s	v16, v9, v12[0]
	fmla.4s	v31, v10, v12[0]
	fmla.4s	v26, v8, v12[0]
	lsr	x1, x30, x15
	and	x1, x1, #0x3
	add	x1, x1, x23
	add	x1, x21, x1, lsl #2
	ldr	s12, [x1, #4]
	fmla.4s	v28, v11, v12[0]
	fmla.4s	v23, v9, v12[0]
	fmla.4s	v29, v10, v12[0]
	fmla.4s	v20, v8, v12[0]
	lsr	x1, x13, x15
	and	x1, x1, #0x3
	add	x1, x1, x24
	add	x1, x21, x1, lsl #2
	ldr	s12, [x1, #4]
	fmla.4s	v27, v11, v12[0]
	fmla.4s	v21, v9, v12[0]
	fmla.4s	v24, v10, v12[0]
	fmla.4s	v19, v8, v12[0]
	lsr	x1, x4, x15
	and	x1, x1, #0x3
	add	x1, x1, x25
	add	x1, x21, x1, lsl #2
	ldr	s12, [x1, #4]
	fmla.4s	v25, v11, v12[0]
	fmla.4s	v17, v9, v12[0]
	fmla.4s	v22, v10, v12[0]
	fmla.4s	v18, v8, v12[0]
	lsr	x1, x0, x15
	and	x1, x1, #0x3
	add	x1, x1, x26
	add	x1, x21, x1, lsl #2
	ldr	s12, [x1, #4]
	fmla.4s	v6, v11, v12[0]
	fmla.4s	v4, v9, v12[0]
	fmla.4s	v7, v10, v12[0]
	fmla.4s	v1, v8, v12[0]
	lsr	x15, x12, x15
	and	x15, x15, #0x3
	add	x15, x15, x27
	add	x15, x21, x15, lsl #2
	ldr	s12, [x15, #4]
	fmla.4s	v3, v11, v12[0]
	fmla.4s	v2, v9, v12[0]
	fmla.4s	v5, v10, v12[0]
	fmla.4s	v0, v8, v12[0]
	add	x14, x14, #1
	cmp	x14, #4
	b.ne	LBB0_4
; %bb.5:                                ; %guard_exit765
                                        ;   in Loop: Header=BB0_3 Depth=1
	add	x14, x8, #4
	add	x10, x10, #64
	add	x12, x8, #7
	add	x11, x11, #256
	mov	x8, x14
	cmp	x12, x9
	b.le	LBB0_3
LBB0_6:                                 ; %guard_exit770
	ldr	x8, [sp, #32]                   ; 8-byte Folded Reload
	ldr	x8, [x8]
	stp	q16, q30, [sp, #64]
	stp	q26, q31, [sp, #96]
	stp	q23, q28, [sp, #128]
	stp	q20, q29, [sp, #160]
	stp	q21, q27, [sp, #192]
	stp	q19, q24, [sp, #224]
	stp	q17, q25, [sp, #256]
	stp	q18, q22, [sp, #288]
	stp	q4, q6, [sp, #320]
	stp	q1, q7, [sp, #352]
	stp	q2, q3, [sp, #384]
	stp	q0, q5, [sp, #416]
	cmp	x14, x9
	ldr	x22, [sp, #48]                  ; 8-byte Folded Reload
	b.gt	LBB0_9
; %bb.7:                                ; %guard_exit775.lr.ph
	ldp	x11, x12, [sp, #8]              ; 16-byte Folded Reload
	ldr	x0, [x11]
	ldr	x3, [x12, #16]
	sub	x11, x22, #1
	mul	x13, x3, x11
	ldr	x12, [x12]
	ldr	x16, [sp, #24]                  ; 8-byte Folded Reload
	ldr	x4, [x16, #16]
	mov	x11, #4611686018427387903       ; =0x3fffffffffffffff
	add	x11, x22, x11
	mul	x15, x4, x11
	ldr	x11, [x16]
	sub	x11, x11, #4
	mul	x16, x3, x22
	mul	x17, x4, x22
	add	x2, x22, #1
	mul	x1, x3, x2
	mul	x2, x4, x2
	add	x6, x22, #2
	mul	x5, x3, x6
	mul	x6, x4, x6
	add	x19, x22, #3
	mul	x7, x3, x19
	mul	x19, x4, x19
	add	x21, x22, #4
	mul	x20, x3, x21
	add	x10, x0, x10, lsl #2
	add	x10, x10, #32
	mul	x21, x4, x21
LBB0_8:                                 ; %guard_exit775
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q10, q8, [x10, #-32]
	ldp	q11, q9, [x10], #64
	sub	x0, x14, #1
	add	x0, x12, x0, lsr #2
	ldrb	w3, [x0, x13]
	lsl	w4, w14, #1
	add	w4, w4, #6
	and	x4, x4, #0x6
	lsr	x3, x3, x4
	and	x3, x3, #0x3
	add	x3, x3, x15
	add	x3, x11, x3, lsl #2
	ldr	s12, [x3, #4]
	fmla.4s	v16, v10, v12[0]
	fmla.4s	v30, v8, v12[0]
	fmla.4s	v26, v11, v12[0]
	fmla.4s	v31, v9, v12[0]
	ldrb	w3, [x0, x16]
	lsr	x3, x3, x4
	and	x3, x3, #0x3
	add	x3, x3, x17
	add	x3, x11, x3, lsl #2
	ldr	s12, [x3, #4]
	fmla.4s	v23, v10, v12[0]
	fmla.4s	v28, v8, v12[0]
	fmla.4s	v20, v11, v12[0]
	fmla.4s	v29, v9, v12[0]
	ldrb	w3, [x0, x1]
	lsr	x3, x3, x4
	and	x3, x3, #0x3
	add	x3, x3, x2
	add	x3, x11, x3, lsl #2
	ldr	s12, [x3, #4]
	fmla.4s	v21, v10, v12[0]
	fmla.4s	v27, v8, v12[0]
	fmla.4s	v19, v11, v12[0]
	fmla.4s	v24, v9, v12[0]
	ldrb	w3, [x0, x5]
	lsr	x3, x3, x4
	and	x3, x3, #0x3
	add	x3, x3, x6
	add	x3, x11, x3, lsl #2
	ldr	s12, [x3, #4]
	fmla.4s	v17, v10, v12[0]
	fmla.4s	v25, v8, v12[0]
	fmla.4s	v18, v11, v12[0]
	ldrb	w3, [x0, x7]
	fmla.4s	v22, v9, v12[0]
	lsr	x3, x3, x4
	and	x3, x3, #0x3
	add	x3, x3, x19
	add	x3, x11, x3, lsl #2
	ldr	s12, [x3, #4]
	ldrb	w0, [x0, x20]
	fmla.4s	v4, v10, v12[0]
	fmla.4s	v6, v8, v12[0]
	fmla.4s	v1, v11, v12[0]
	fmla.4s	v7, v9, v12[0]
	lsr	x0, x0, x4
	and	x0, x0, #0x3
	add	x0, x0, x21
	add	x0, x11, x0, lsl #2
	ldr	s12, [x0, #4]
	stp	q16, q30, [sp, #64]
	stp	q26, q31, [sp, #96]
	stp	q23, q28, [sp, #128]
	stp	q20, q29, [sp, #160]
	stp	q21, q27, [sp, #192]
	stp	q19, q24, [sp, #224]
	fmla.4s	v2, v10, v12[0]
	fmla.4s	v3, v8, v12[0]
	fmla.4s	v0, v11, v12[0]
	fmla.4s	v5, v9, v12[0]
	stp	q17, q25, [sp, #256]
	add	x14, x14, #1
	stp	q18, q22, [sp, #288]
	stp	q4, q6, [sp, #320]
	stp	q1, q7, [sp, #352]
	stp	q2, q3, [sp, #384]
	stp	q0, q5, [sp, #416]
	cmp	x14, x9
	b.le	LBB0_8
LBB0_9:                                 ; %L2081
	ldr	x9, [sp, #40]                   ; 8-byte Folded Reload
	ldr	x9, [x9, #40]
	sub	x9, x22, x9
	ldr	x11, [sp, #56]                  ; 8-byte Folded Reload
	ldp	x14, x10, [x11]
	sub	x11, x10, x14
	add	x12, x11, #1
	add	x10, sp, #64
	sub	x10, x10, #96
	cmp	x12, #8
	mov	w13, #8                         ; =0x8
	csinc	x15, x13, x11, le
	bic	x11, x12, x12, asr #63
	mov	x12, #4611686018427387903       ; =0x3fffffffffffffff
	add	x13, x14, x12
	mov	x16, #7                         ; =0x7
	movk	x16, #16384, lsl #48
	add	x14, x14, x16
	sub	x15, x15, #8
	mov	w16, #1                         ; =0x1
	b	LBB0_11
LBB0_10:                                ; %L2240.1
                                        ;   in Loop: Header=BB0_11 Depth=1
	add	x17, x16, #1
	cmp	x16, #6
	mov	x16, x17
	b.eq	LBB0_17
LBB0_11:                                ; %L2089
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_12 Depth 2
                                        ;     Child Loop BB0_15 Depth 2
	add	x17, x16, x9
	add	x17, x17, x12
	mov	x1, x11
	mov	x2, x13
	mov	w3, #1                          ; =0x1
	add	x0, x10, x16, lsl #6
LBB0_12:                                ; %L2097
                                        ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	cbz	x1, LBB0_14
; %bb.13:                               ; %L2104
                                        ;   in Loop: Header=BB0_12 Depth=2
	ldr	x4, [x8, #16]
	madd	x4, x2, x4, x17
	ldr	x5, [x8]
	add	x4, x5, x4, lsl #2
	ldur	s0, [x4, #-4]
	sub	w5, w3, #1
	and	x5, x5, #0x7
	add	x5, x0, x5, lsl #2
	ldr	s1, [x5, #32]
	fadd	s0, s0, s1
	stur	s0, [x4, #-4]
	add	x4, x3, #1
	add	x2, x2, #1
	sub	x1, x1, #1
	cmp	x3, #8
	mov	x3, x4
	b.ne	LBB0_12
LBB0_14:                                ; %L2240
                                        ;   in Loop: Header=BB0_11 Depth=1
	mov	x1, x15
	mov	x2, x14
	mov	w3, #1                          ; =0x1
LBB0_15:                                ; %L2097.1
                                        ;   Parent Loop BB0_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	cbz	x1, LBB0_10
; %bb.16:                               ; %L2104.1
                                        ;   in Loop: Header=BB0_15 Depth=2
	ldr	x4, [x8, #16]
	madd	x4, x2, x4, x17
	ldr	x5, [x8]
	add	x4, x5, x4, lsl #2
	ldur	s0, [x4, #-4]
	sub	w5, w3, #1
	and	x5, x5, #0x7
	add	x5, x0, x5, lsl #2
	ldr	s1, [x5, #64]
	fadd	s0, s0, s1
	stur	s0, [x4, #-4]
	add	x4, x3, #1
	add	x2, x2, #1
	sub	x1, x1, #1
	cmp	x3, #8
	mov	x3, x4
	b.ne	LBB0_15
	b	LBB0_10
LBB0_17:                                ; %L2263
	mov	x0, x8
	add	sp, sp, #448
	ldp	x29, x30, [sp, #128]            ; 16-byte Folded Reload
	ldp	x20, x19, [sp, #112]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             ; 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             ; 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #32]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #16]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp], #144            ; 16-byte Folded Reload
	ret
                                        ; -- End function
.subsections_via_symbols
