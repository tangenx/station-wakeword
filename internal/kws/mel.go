package kws

import (
	"math"
	"math/cmplx"
)

// Front end of openWakeWord: 16 kHz int16-scale audio, STFT with a 400-point
// periodic Hann window centred in a 512-point frame, hop 160, power -> 32 mel
// bins -> 10*log10 clamped to (max-80 dB) over each 80 ms step, then x/10+2.
const (
	SampleRate = 16000
	MelBins    = 32
	StepSamp   = 1280 // one streaming step = 80 ms = 8 mel frames
	StepFrames = 8
	fftSize    = 512
	hop        = 160
	winLen     = 400
	ctxSamp    = 3 * hop // extra history so each step yields exactly 8 frames
)

type melFront struct {
	filters []float32 // 257 x 32
	window  [fftSize]float64
	tw      [fftSize / 2]complex128
	buf     [fftSize]complex128
	power   [fftSize/2 + 1]float64
}

func newMelFront(filters []float32) *melFront {
	m := &melFront{filters: filters}
	off := (fftSize - winLen) / 2
	for i := range winLen {
		m.window[off+i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/winLen)
	}
	for i := range m.tw {
		m.tw[i] = cmplx.Exp(complex(0, -2*math.Pi*float64(i)/fftSize))
	}
	return m
}

// fft is an in-place iterative radix-2 FFT of m.buf.
func (m *melFront) fft() {
	a := &m.buf
	for i, j := 1, 0; i < fftSize; i++ {
		bit := fftSize >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= fftSize; size <<= 1 {
		half, step := size>>1, fftSize/size
		for start := 0; start < fftSize; start += size {
			for k := range half {
				t := m.tw[k*step] * a[start+k+half]
				a[start+k+half] = a[start+k] - t
				a[start+k] += t
			}
		}
	}
}

// frames computes log-mel frames (each MelBins long) of samples, writing
// them consecutively to dst. Returns the number of frames.
func (m *melFront) frames(samples []float32, dst []float32) int {
	n := 0
	for start := 0; start+fftSize <= len(samples); start += hop {
		for i := range fftSize {
			m.buf[i] = complex(float64(samples[start+i])*m.window[i], 0)
		}
		m.fft()
		for k := range m.power {
			re, im := real(m.buf[k]), imag(m.buf[k])
			m.power[k] = re*re + im*im
		}
		out := dst[n*MelBins : (n+1)*MelBins]
		for b := range MelBins {
			var s float64
			for k, p := range m.power {
				s += p * float64(m.filters[k*MelBins+b])
			}
			out[b] = float32(10 * math.Log10(max(s, 1e-10)))
		}
		n++
	}
	top := float32(math.Inf(-1))
	for _, v := range dst[:n*MelBins] {
		top = max(top, v)
	}
	for i, v := range dst[:n*MelBins] {
		dst[i] = max(v, top-80)/10 + 2
	}
	return n
}
