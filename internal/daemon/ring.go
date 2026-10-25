package daemon

// ring keeps the most recent samples of the live stream.
type ring struct {
	buf  []int16
	pos  int // next write index
	full bool
}

func newRing(n int) *ring { return &ring{buf: make([]int16, n)} }

func (r *ring) write(pcm []int16) {
	for len(pcm) > 0 {
		n := copy(r.buf[r.pos:], pcm)
		pcm = pcm[n:]
		r.pos += n
		if r.pos == len(r.buf) {
			r.pos, r.full = 0, true
		}
	}
}

// last returns a copy of the newest n samples (fewer if not yet recorded).
func (r *ring) last(n int) []int16 {
	have := r.pos
	if r.full {
		have = len(r.buf)
	}
	n = min(n, have)
	out := make([]int16, n)
	start := r.pos - n
	if start >= 0 {
		copy(out, r.buf[start:r.pos])
		return out
	}
	k := copy(out, r.buf[len(r.buf)+start:])
	copy(out[k:], r.buf[:r.pos])
	return out
}
