#include "textflag.h"

// func macBlock24(acc, src, w *float32, n, stride int)
// acc[0:24] += sum_{i<n} src[i] * w[i*stride : i*stride+24]
//
// Tuned for in-order Cortex-A53: two input channels per iteration into two
// independent accumulator sets (V0-V5, V6-V11), all loads issued before the
// FMLAs that consume them.
TEXT ·macBlock24(SB), NOSPLIT, $0-40
	MOVD acc+0(FP), R0
	MOVD src+8(FP), R1
	MOVD w+16(FP), R2
	MOVD n+24(FP), R3
	MOVD stride+32(FP), R4
	LSL  $2, R4, R4
	ADD  $64, R0, R5
	VLD1 (R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1 (R5), [V4.S4, V5.S4]
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	LSR  $1, R3, R7
	CBZ  R7, tail

pair:
	VLD1R.P 4(R1), [V16.S4]
	VLD1R.P 4(R1), [V23.S4]
	ADD     $64, R2, R6
	VLD1    (R2), [V17.S4, V18.S4, V19.S4, V20.S4]
	VLD1    (R6), [V21.S4, V22.S4]
	ADD     R4, R2, R2
	ADD     $64, R2, R6
	VLD1    (R2), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1    (R6), [V28.S4, V29.S4]
	ADD     R4, R2, R2
	VFMLA   V16.S4, V17.S4, V0.S4
	VFMLA   V16.S4, V18.S4, V1.S4
	VFMLA   V16.S4, V19.S4, V2.S4
	VFMLA   V16.S4, V20.S4, V3.S4
	VFMLA   V16.S4, V21.S4, V4.S4
	VFMLA   V16.S4, V22.S4, V5.S4
	VFMLA   V23.S4, V24.S4, V6.S4
	VFMLA   V23.S4, V25.S4, V7.S4
	VFMLA   V23.S4, V26.S4, V8.S4
	VFMLA   V23.S4, V27.S4, V9.S4
	VFMLA   V23.S4, V28.S4, V10.S4
	VFMLA   V23.S4, V29.S4, V11.S4
	SUBS    $1, R7, R7
	BNE     pair

tail:
	TBZ     $0, R3, done
	VLD1R.P 4(R1), [V16.S4]
	ADD     $64, R2, R6
	VLD1    (R2), [V17.S4, V18.S4, V19.S4, V20.S4]
	VLD1    (R6), [V21.S4, V22.S4]
	VFMLA   V16.S4, V17.S4, V0.S4
	VFMLA   V16.S4, V18.S4, V1.S4
	VFMLA   V16.S4, V19.S4, V2.S4
	VFMLA   V16.S4, V20.S4, V3.S4
	VFMLA   V16.S4, V21.S4, V4.S4
	VFMLA   V16.S4, V22.S4, V5.S4

done:
	// FADD Vk.4S, Vk.4S, V(k+6).4S for k = 0..5. Raw encodings: the Go 1.24
	// assembler (go.mod minimum) has no VFADD.
	WORD $0x4E26D400
	WORD $0x4E27D421
	WORD $0x4E28D442
	WORD $0x4E29D463
	WORD $0x4E2AD484
	WORD $0x4E2BD4A5
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R0)
	VST1  [V4.S4, V5.S4], (R5)
	RET

// func macBlock24x4(acc *[4]*float32, src *[4]*float32, w *float32, n, stride int)
// Four output pixels share every weight load: for p < 4,
// acc[p][0:24] += sum_{i<n} src[p][i] * w[i*stride : i*stride+24].
// Accumulators V0-V23 (6 per pixel), weights V24-V29, broadcasts V30-V31.
TEXT ·macBlock24x4(SB), NOSPLIT, $0-40
	MOVD acc+0(FP), R0
	MOVD src+8(FP), R1
	MOVD w+16(FP), R2
	MOVD n+24(FP), R3
	MOVD stride+32(FP), R4
	LSL  $2, R4, R4
	MOVD 0(R0), R10
	MOVD 8(R0), R11
	MOVD 16(R0), R12
	MOVD 24(R0), R13
	MOVD 0(R1), R14
	MOVD 8(R1), R15
	MOVD 16(R1), R19
	MOVD 24(R1), R20
	ADD  $64, R10, R6
	VLD1 (R10), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1 (R6), [V4.S4, V5.S4]
	ADD  $64, R11, R6
	VLD1 (R11), [V6.S4, V7.S4, V8.S4, V9.S4]
	VLD1 (R6), [V10.S4, V11.S4]
	ADD  $64, R12, R6
	VLD1 (R12), [V12.S4, V13.S4, V14.S4, V15.S4]
	VLD1 (R6), [V16.S4, V17.S4]
	ADD  $64, R13, R6
	VLD1 (R13), [V18.S4, V19.S4, V20.S4, V21.S4]
	VLD1 (R6), [V22.S4, V23.S4]
	CBZ  R3, store4

loop4:
	VLD1R.P 4(R14), [V30.S4]
	VLD1R.P 4(R15), [V31.S4]
	ADD     $64, R2, R6
	VLD1    (R2), [V24.S4, V25.S4, V26.S4, V27.S4]
	VLD1    (R6), [V28.S4, V29.S4]
	ADD     R4, R2, R2
	VFMLA   V30.S4, V24.S4, V0.S4
	VFMLA   V30.S4, V25.S4, V1.S4
	VFMLA   V30.S4, V26.S4, V2.S4
	VFMLA   V30.S4, V27.S4, V3.S4
	VFMLA   V30.S4, V28.S4, V4.S4
	VFMLA   V30.S4, V29.S4, V5.S4
	VLD1R.P 4(R19), [V30.S4]
	VFMLA   V31.S4, V24.S4, V6.S4
	VFMLA   V31.S4, V25.S4, V7.S4
	VFMLA   V31.S4, V26.S4, V8.S4
	VFMLA   V31.S4, V27.S4, V9.S4
	VFMLA   V31.S4, V28.S4, V10.S4
	VFMLA   V31.S4, V29.S4, V11.S4
	VLD1R.P 4(R20), [V31.S4]
	VFMLA   V30.S4, V24.S4, V12.S4
	VFMLA   V30.S4, V25.S4, V13.S4
	VFMLA   V30.S4, V26.S4, V14.S4
	VFMLA   V30.S4, V27.S4, V15.S4
	VFMLA   V30.S4, V28.S4, V16.S4
	VFMLA   V30.S4, V29.S4, V17.S4
	VFMLA   V31.S4, V24.S4, V18.S4
	VFMLA   V31.S4, V25.S4, V19.S4
	VFMLA   V31.S4, V26.S4, V20.S4
	VFMLA   V31.S4, V27.S4, V21.S4
	VFMLA   V31.S4, V28.S4, V22.S4
	VFMLA   V31.S4, V29.S4, V23.S4
	SUBS    $1, R3, R3
	BNE     loop4

store4:
	ADD  $64, R10, R6
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R10)
	VST1 [V4.S4, V5.S4], (R6)
	ADD  $64, R11, R6
	VST1 [V6.S4, V7.S4, V8.S4, V9.S4], (R11)
	VST1 [V10.S4, V11.S4], (R6)
	ADD  $64, R12, R6
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R12)
	VST1 [V16.S4, V17.S4], (R6)
	ADD  $64, R13, R6
	VST1 [V18.S4, V19.S4, V20.S4, V21.S4], (R13)
	VST1 [V22.S4, V23.S4], (R6)
	RET
