	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 16, 0
	.globl	"_julia__snparray_AX_register_tile!_9767" ; -- Begin function julia__snparray_AX_register_tile!_9767
	.p2align	2
"_julia__snparray_AX_register_tile!_9767": ; @"julia__snparray_AX_register_tile!_9767"
; Function Signature: _snparray_AX_register_tile!(SnpArrays.RegisterTileTask{Float32, Array{Float32, 2}, Array{UInt8, 2}, Array{Float32, 2}}, Int64, Base.UnitRange{Int64}, Base.UnitRange{Int64}, Base.Val{6}, Base.Val{2}, Base.Val{8})
; %bb.0:                                ; %top
	;DEBUG_VALUE: _snparray_AX_register_tile!:task <- [$x0+0]
	;DEBUG_VALUE: _snparray_AX_register_tile!:row <- $x2
	;DEBUG_VALUE: _snparray_AX_register_tile!:row <- $x2
	;DEBUG_VALUE: _snparray_AX_register_tile!:columns <- [$x3+0]
	;DEBUG_VALUE: _snparray_AX_register_tile!:columns <- [$x3+0]
	;DEBUG_VALUE: _snparray_AX_register_tile!:tile_columns <- [$x4+0]
	;DEBUG_VALUE: _snparray_AX_register_tile!:tile_columns <- [$x4+0]
	sub	sp, sp, #480
	stp	d13, d12, [sp, #384]            ; 16-byte Folded Spill
	stp	d11, d10, [sp, #400]            ; 16-byte Folded Spill
	stp	d9, d8, [sp, #416]              ; 16-byte Folded Spill
	stp	x28, x27, [sp, #432]            ; 16-byte Folded Spill
	stp	x22, x21, [sp, #448]            ; 16-byte Folded Spill
	stp	x20, x19, [sp, #464]            ; 16-byte Folded Spill
	mov	x8, x0
	;DEBUG_VALUE: _snparray_AX_register_tile!:task <- [$x8+0]
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v4, #0000000000000000
	movi.2d	v5, #0000000000000000
	ldr	x0, [x1]
	movi.2d	v17, #0000000000000000
	movi.2d	v19, #0000000000000000
	ldp	x19, x9, [x3]
	movi.2d	v24, #0000000000000000
	movi.2d	v25, #0000000000000000
	movi.2d	v26, #0000000000000000
	movi.2d	v27, #0000000000000000
	movi.2d	v28, #0000000000000000
	movi.2d	v29, #0000000000000000
	movi.2d	v30, #0000000000000000
	movi.2d	v31, #0000000000000000
	movi.2d	v20, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v21, #0000000000000000
	movi.2d	v23, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v18, #0000000000000000
	movi.2d	v6, #0000000000000000
	movi.2d	v7, #0000000000000000
	subs	x6, x9, x19
	b.lt	LBB0_3
; %bb.1:                                ; %L32.L37_crit_edge
	ldp	x12, x9, [x1, #16]
	ldr	x10, [x1, #8]
	ldr	x20, [x8, #32]
	ldr	x21, [x9]
	sub	x9, x2, #1
	ldr	x8, [x10, #16]
	ldr	x5, [x10]
	add	x9, x5, x9, lsr #2
	lsl	w16, w2, #1
	add	w10, w16, #6
	and	w10, w10, #0x6
	ldr	x11, [x12, #16]
	ldr	x12, [x12]
	sub	x12, x12, #4
	add	x13, x5, x2, lsr #2
	and	w14, w16, #0x6
	add	x15, x2, #1
	add	x15, x5, x15, lsr #2
	add	w16, w16, #2
	and	w16, w16, #0x6
	add	x17, x2, #2
	add	x17, x5, x17, lsr #2
	eor	w1, w14, #0x4
	add	x3, x2, #3
	add	x3, x5, x3, lsr #2
	add	x7, x2, #4
	add	x5, x5, x7, lsr #2
	add	x6, x6, #1
	mov	x7, #4611686018427387903        ; =0x3fffffffffffffff
	add	x7, x19, x7
	mov	x22, #1                         ; =0x1
	madd	x7, x11, x7, x22
	sub	x19, x19, #1
	mul	x19, x8, x19
	add	x20, x21, x20, lsl #2
	add	x20, x20, #32
LBB0_2:                                 ; %L37
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q9, q10, [x20, #-32]
	ldp	q8, q11, [x20], #64
	ldrb	w21, [x9, x19]
	lsr	w21, w21, w10
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s12, [x12, x21, lsl #2]
	fmla.4s	v1, v10, v12[0]
	fmla.4s	v0, v9, v12[0]
	fmla.4s	v3, v11, v12[0]
	fmla.4s	v2, v8, v12[0]
	ldrb	w21, [x13, x19]
	lsr	w21, w21, w14
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s12, [x12, x21, lsl #2]
	fmla.4s	v5, v10, v12[0]
	fmla.4s	v4, v9, v12[0]
	fmla.4s	v19, v11, v12[0]
	fmla.4s	v17, v8, v12[0]
	ldrb	w21, [x15, x19]
	lsr	w21, w21, w16
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s12, [x12, x21, lsl #2]
	fmla.4s	v25, v10, v12[0]
	fmla.4s	v24, v9, v12[0]
	ldrb	w21, [x17, x19]
	fmla.4s	v27, v11, v12[0]
	lsr	w21, w21, w1
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s13, [x12, x21, lsl #2]
	fmla.4s	v26, v8, v12[0]
	fmla.4s	v29, v10, v13[0]
	fmla.4s	v28, v9, v13[0]
	fmla.4s	v31, v11, v13[0]
	ldrb	w21, [x3, x19]
	fmla.4s	v30, v8, v13[0]
	lsr	w21, w21, w10
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s12, [x12, x21, lsl #2]
	fmla.4s	v22, v10, v12[0]
	fmla.4s	v20, v9, v12[0]
	fmla.4s	v23, v11, v12[0]
	fmla.4s	v21, v8, v12[0]
	ldrb	w21, [x5, x19]
	lsr	w21, w21, w14
	and	x21, x21, #0x3
	add	x21, x7, x21
	ldr	s12, [x12, x21, lsl #2]
	add	x7, x7, x11
	fmla.4s	v18, v10, v12[0]
	fmla.4s	v16, v9, v12[0]
	fmla.4s	v7, v11, v12[0]
	add	x5, x5, x8
	add	x3, x3, x8
	add	x17, x17, x8
	add	x15, x15, x8
	fmla.4s	v6, v8, v12[0]
	add	x13, x13, x8
	add	x9, x9, x8
	subs	x6, x6, #1
	b.ne	LBB0_2
LBB0_3:                                 ; %L1054
	stp	q0, q1, [sp]
	stp	q2, q3, [sp, #32]
	stp	q4, q5, [sp, #64]
	stp	q17, q19, [sp, #96]
	stp	q24, q25, [sp, #128]
	stp	q26, q27, [sp, #160]
	stp	q28, q29, [sp, #192]
	stp	q30, q31, [sp, #224]
	ldp	x12, x8, [x4]
	sub	x9, x8, x12
	add	x10, x9, #1
	stp	q20, q22, [sp, #256]
	mov	x8, sp
	sub	x8, x8, #96
	cmp	x10, #8
	mov	w11, #8                         ; =0x8
	stp	q21, q23, [sp, #288]
	csinc	x13, x11, x9, le
	bic	x9, x10, x10, asr #63
	mov	x10, #4611686018427387903       ; =0x3fffffffffffffff
	stp	q16, q18, [sp, #320]
	add	x11, x12, x10
	mov	x14, #7                         ; =0x7
	movk	x14, #16384, lsl #48
	add	x12, x12, x14
	sub	x13, x13, #8
	mov	w14, #1                         ; =0x1
	stp	q6, q7, [sp, #352]
	b	LBB0_5
LBB0_4:                                 ; %L1212.1
                                        ;   in Loop: Header=BB0_5 Depth=1
	add	x15, x14, #1
	cmp	x14, #6
	mov	x14, x15
	b.eq	LBB0_11
LBB0_5:                                 ; %L1061
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
                                        ;     Child Loop BB0_9 Depth 2
	add	x15, x14, x2
	add	x15, x15, x10
	mov	x17, x9
	mov	x1, x11
	mov	w3, #1                          ; =0x1
	add	x16, x8, x14, lsl #6
LBB0_6:                                 ; %L1069
                                        ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	cbz	x17, LBB0_8
; %bb.7:                                ; %L1076
                                        ;   in Loop: Header=BB0_6 Depth=2
	ldr	x4, [x0, #16]
	madd	x4, x1, x4, x15
	ldr	x5, [x0]
	add	x4, x5, x4, lsl #2
	ldur	s0, [x4, #-4]
	sub	w5, w3, #1
	and	x5, x5, #0x7
	add	x5, x16, x5, lsl #2
	ldr	s1, [x5, #32]
	fadd	s0, s0, s1
	stur	s0, [x4, #-4]
	add	x4, x3, #1
	add	x1, x1, #1
	sub	x17, x17, #1
	cmp	x3, #8
	mov	x3, x4
	b.ne	LBB0_6
LBB0_8:                                 ; %L1212
                                        ;   in Loop: Header=BB0_5 Depth=1
	mov	x17, x13
	mov	x1, x12
	mov	w3, #1                          ; =0x1
LBB0_9:                                 ; %L1069.1
                                        ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	cbz	x17, LBB0_4
; %bb.10:                               ; %L1076.1
                                        ;   in Loop: Header=BB0_9 Depth=2
	ldr	x4, [x0, #16]
	madd	x4, x1, x4, x15
	ldr	x5, [x0]
	add	x4, x5, x4, lsl #2
	ldur	s0, [x4, #-4]
	sub	w5, w3, #1
	and	x5, x5, #0x7
	add	x5, x16, x5, lsl #2
	ldr	s1, [x5, #64]
	fadd	s0, s0, s1
	stur	s0, [x4, #-4]
	add	x4, x3, #1
	add	x1, x1, #1
	sub	x17, x17, #1
	cmp	x3, #8
	mov	x3, x4
	b.ne	LBB0_9
	b	LBB0_4
LBB0_11:                                ; %L1235
	ldp	x20, x19, [sp, #464]            ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #448]            ; 16-byte Folded Reload
	ldp	x28, x27, [sp, #432]            ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #416]              ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #400]            ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #384]            ; 16-byte Folded Reload
	add	sp, sp, #480
	ret
                                        ; -- End function
.subsections_via_symbols
