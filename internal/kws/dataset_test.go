package kws

import (
	"math"
	"math/rand/v2"
	"testing"
)

// A take as the station records it: low noise, the word, then a long quiet
// tail (room reverb, someone in the next room) well above the noise floor.
// The tail must not be counted as part of the word, or every take in a
// lively room is rejected as too long.
func TestFindSpeechIgnoresReverbTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	pcm := make([]int16, 3*SampleRate)
	for i := range pcm {
		t := float64(i) / SampleRate
		v := rng.NormFloat64() * 30 // noise floor ~30 dB
		switch {
		case t >= 1.0 && t < 1.6: // the word: ~70 dB
			v += 4000 * math.Sin(2*math.Pi*220*t)
		case t >= 1.6 && t < 2.8: // tail: ~25 dB below the word
			v += 220 * math.Sin(2*math.Pi*150*t)
		}
		pcm[i] = int16(v)
	}
	s, e, ok := FindSpeech(pcm)
	if !ok {
		t.Fatal("word not found")
	}
	if got := float64(s) / SampleRate; math.Abs(got-1.0) > 0.05 {
		t.Errorf("start %.2fs, want 1.00s", got)
	}
	if got := float64(e) / SampleRate; got < 1.55 || got > 1.8 {
		t.Errorf("end %.2fs, want ~1.6s (tail must not stretch the word)", got)
	}
}

func TestFindSpeechRejectsSilence(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	pcm := make([]int16, 3*SampleRate)
	for i := range pcm {
		pcm[i] = int16(rng.NormFloat64() * 40)
	}
	if _, _, ok := FindSpeech(pcm); ok {
		t.Fatal("found a word in plain noise")
	}
}

// One window above threshold is a spike, not a word; after a detection the
// gate stays quiet so the command spoken after the word cannot re-trigger.
func TestGate(t *testing.T) {
	var g gate
	fire := func(scores ...float32) (n int) {
		for _, s := range scores {
			if g.step(s, 0.8) {
				n++
			}
		}
		return n
	}
	if n := fire(0.1, 0.95, 0.1, 0.99, 0.2); n != 0 {
		t.Fatalf("isolated spikes fired %d times", n)
	}
	if n := fire(0.9, 0.95); n != 1 {
		t.Fatalf("two consecutive passes fired %d times, want 1", n)
	}
	quiet := make([]float32, refractory)
	for i := range quiet {
		quiet[i] = 1
	}
	if n := fire(quiet...); n != 0 {
		t.Fatalf("fired %d times during the refractory period", n)
	}
	if n := fire(0.9, 0.9); n != 1 {
		t.Fatalf("did not fire again after the refractory period (%d)", n)
	}
}
