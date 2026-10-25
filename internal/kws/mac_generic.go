//go:build !arm64

package kws

import "unsafe"

func macBlock24(acc, src, w *float32, n, stride int) {
	a := unsafe.Slice(acc, 24)
	s := unsafe.Slice(src, n)
	ws := unsafe.Slice(w, (n-1)*stride+24)
	for i, v := range s {
		r := ws[i*stride : i*stride+24]
		for o := range a {
			a[o] += v * r[o]
		}
	}
}

func macBlock24x4(acc, src *[4]*float32, w *float32, n, stride int) {
	for p := range 4 {
		macBlock24(acc[p], src[p], w, n, stride)
	}
}
