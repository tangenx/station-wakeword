package kws

import (
	"math"
	"os"
	"testing"
)

func golden(t *testing.T) map[string]Tensor {
	t.Helper()
	b, err := os.ReadFile("testdata/golden.bin")
	if err != nil {
		t.Fatal(err)
	}
	g, err := ReadTensors(b)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func maxAbsDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		d = max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d
}

// The Go front end must match the ONNX models it replaces, otherwise the
// pretrained embedding sees out-of-distribution input.
func TestMelStepMatchesONNX(t *testing.T) {
	g := golden(t)
	f, err := NewFeatures()
	if err != nil {
		t.Fatal(err)
	}
	audio := g["audio"].Data
	end := len(audio) / StepSamp * StepSamp // golden mel.step is the last full step
	n := f.mel.frames(audio[end-ctxSamp-StepSamp:end], f.frame)
	want := g["mel.step"].Data
	if n*MelBins != len(want) {
		t.Fatalf("frames: got %d, want %d", n, len(want)/MelBins)
	}
	if d := maxAbsDiff(f.frame[:n*MelBins], want); d > 1e-3 {
		t.Fatalf("mel step max diff %g", d)
	}
}

func TestStreamingEmbeddingsMatchONNX(t *testing.T) {
	g := golden(t)
	f, err := NewFeatures()
	if err != nil {
		t.Fatal(err)
	}
	want := g["emb.stream"]
	audio := g["audio"].Data
	pcm := make([]int16, len(audio))
	for i, v := range audio {
		pcm[i] = int16(v)
	}
	for pass := range 2 { // second pass checks Reset restores a clean stream
		var got [][]float32
		// Odd chunk size: embeddings must not depend on how audio is split.
		for i := 0; i < len(pcm); i += 333 {
			f.Push(pcm[i:min(i+333, len(pcm))], func(e []float32) {
				got = append(got, append([]float32(nil), e...))
			})
		}
		if len(got) != want.Shape[0] {
			t.Fatalf("pass %d: got %d embeddings, want %d", pass, len(got), want.Shape[0])
		}
		for i, e := range got {
			if d := maxAbsDiff(e, want.Data[i*EmbDim:(i+1)*EmbDim]); d > 1e-3 {
				t.Fatalf("pass %d: embedding %d max diff %g", pass, i, d)
			}
		}
		f.Reset()
	}
}

// One 80 ms step of audio: the steady-state cost on the station.
func BenchmarkStep(b *testing.B) {
	f, err := NewFeatures()
	if err != nil {
		b.Fatal(err)
	}
	pcm := make([]int16, StepSamp)
	for i := range pcm {
		pcm[i] = int16((i * 7919) % 2000)
	}
	for range 12 {
		f.Push(pcm, func([]float32) {})
	}
	for b.Loop() {
		f.Push(pcm, func([]float32) {})
	}
}
