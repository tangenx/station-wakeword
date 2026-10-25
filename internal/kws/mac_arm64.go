package kws

//go:noescape
func macBlock24(acc, src, w *float32, n, stride int)

//go:noescape
func macBlock24x4(acc, src *[4]*float32, w *float32, n, stride int)
