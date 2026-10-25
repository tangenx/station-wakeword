package kws

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

// Stream is a run of consecutive embeddings (n x EmbDim). Embedding i of a
// stream computed from a fresh Features covers audio ending at sample
// EmbEnd(i).
type Stream []float32

func (s Stream) Len() int { return len(s) / EmbDim }

// Window returns the Ctx embeddings ending at embedding e (inclusive).
func (s Stream) Window(e int) []float32 { return s[(e-Ctx+1)*EmbDim : (e+1)*EmbDim] }

// Windows returns every stride-th full window.
func (s Stream) Windows(stride int) [][]float32 {
	var out [][]float32
	for e := Ctx - 1; e < s.Len(); e += stride {
		out = append(out, s.Window(e))
	}
	return out
}

// firstEmbStep is the 80 ms step that yields a fresh stream's first
// embedding (it needs EmbWindow mel frames).
const firstEmbStep = (EmbWindow + StepFrames - 1) / StepFrames

// EmbEnd is the sample index just past the audio covered by embedding i.
func EmbEnd(i int) int { return (i + firstEmbStep) * StepSamp }

// Embed computes the embedding stream of pcm with f (which it resets).
func Embed(f *Features, pcm []int16) Stream {
	f.Reset()
	out := make(Stream, 0, (len(pcm)/StepSamp)*EmbDim)
	f.Push(pcm, func(e []float32) { out = append(out, e...) })
	return out
}

// FindSpeech locates the word in a take with an energy VAD on 10 ms frames.
// Thresholds are relative to the loudest frame, not only to the noise
// floor, so room reverb and background chatter after the word do not
// stretch it: a core run within 20 dB of the peak (gaps under 150 ms
// bridged) is grown by up to 150 ms of softer frames (onsets, final
// consonants). Returns sample offsets.
func FindSpeech(pcm []int16) (start, end int, ok bool) {
	const fl = SampleRate / 100
	const maxGap, maxGrow = 15, 15 // frames
	n := len(pcm) / fl
	if n < 10 {
		return 0, 0, false
	}
	db := make([]float64, n)
	peak := 0
	for i := range n {
		var e float64
		for _, s := range pcm[i*fl : (i+1)*fl] {
			e += float64(s) * float64(s)
		}
		db[i] = 10 * math.Log10(e/fl+1)
		if db[i] > db[peak] {
			peak = i
		}
	}
	sorted := slices.Clone(db)
	sort.Float64s(sorted)
	floor, top := sorted[n/5], db[peak]
	if top < 45 || top-floor < 15 { // 45 dB ~ RMS 178: near-digital silence
		return 0, 0, false
	}
	core := max(floor+12, top-20)
	grow := func(from, dir int) int {
		last, gap := from, 0
		for i := from + dir; i >= 0 && i < n && gap <= maxGap; i += dir {
			if db[i] >= core {
				last, gap = i, 0
			} else {
				gap++
			}
		}
		soft := max(floor+8, top-32)
		for k := 0; k < maxGrow; k++ {
			i := last + dir
			if i < 0 || i >= n || db[i] < soft {
				break
			}
			last = i
		}
		return last
	}
	return grow(peak, -1) * fl, (grow(peak, 1) + 1) * fl, true
}

// Resample changes speed (and pitch) by factor (>1 = faster) with linear
// interpolation.
func Resample(pcm []int16, factor float64) []int16 {
	n := int(float64(len(pcm)) / factor)
	out := make([]int16, n)
	for i := range out {
		p := float64(i) * factor
		j := int(p)
		if j+1 >= len(pcm) {
			out[i] = pcm[len(pcm)-1]
			continue
		}
		fr := p - float64(j)
		out[i] = int16(float64(pcm[j])*(1-fr) + float64(pcm[j+1])*fr)
	}
	return out
}

func rms(pcm []int16) float64 {
	var e float64
	for _, s := range pcm {
		e += float64(s) * float64(s)
	}
	return math.Sqrt(e/float64(max(len(pcm), 1))) + 1e-9
}

func clip16(v float64) int16 { return int16(max(-32768, min(32767, math.Round(v)))) }

// segment returns n samples of src starting at a random offset (looping).
func segment(rng *rand.Rand, src []int16, n int) []int16 {
	out := make([]int16, n)
	if len(src) == 0 {
		return out
	}
	off := rng.IntN(len(src))
	for i := range out {
		out[i] = src[(off+i)%len(src)]
	}
	return out
}

// Augment builds one positive training signal from a take: random speed and
// gain, placed after preCtx of background (room or the user's own speech)
// and followed by post samples; noise mixed at a random SNR. Returns the
// signal and the sample index where the word ends.
func Augment(rng *rand.Rand, word []int16, backgrounds [][]int16, pre, post int) ([]int16, int) {
	w := Resample(word, 0.88+0.24*rng.Float64())
	gain := math.Pow(10, (-10+16*rng.Float64())/20)
	bg := backgrounds[rng.IntN(len(backgrounds))]
	sig := segment(rng, bg, pre+len(w)+post)
	// Background level relative to the word: 5..30 dB SNR.
	snr := 5 + 25*rng.Float64()
	bgScale := gain * rms(w) / (rms(sig) * math.Pow(10, snr/20))
	out := make([]int16, len(sig))
	for i, s := range sig {
		v := float64(s) * bgScale
		if j := i - pre; j >= 0 && j < len(w) {
			v += float64(w[j]) * gain
		}
		out[i] = clip16(v)
	}
	return out, pre + len(w)
}
