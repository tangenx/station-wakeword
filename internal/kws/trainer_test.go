package kws

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestTakeSplitNeverReusesTrainingTake(t *testing.T) {
	for _, tc := range []struct{ n, train, check int }{{3, 2, 1}, {6, 5, 1}, {10, 8, 2}} {
		train, check := splitTakeIndices(tc.n, 42)
		if len(train) != tc.train || len(check) != tc.check {
			t.Fatalf("%d takes: split %v/%v", tc.n, train, check)
		}
		for _, i := range check {
			if slices.Contains(train, i) {
				t.Fatalf("take %d occurs in both train and calibration", i)
			}
		}
	}
}

func TestBackgroundSplitLeavesRawContextGap(t *testing.T) {
	pcm := make([]int16, 8*SampleRate)
	train, check := splitBackground(pcm)
	if len(train) != 67200 || len(check) != 48000 || &check[0] != &pcm[80000] {
		t.Fatalf("split %d/%d samples; want 4.2s train, .8s gap, 3s calibration", len(train), len(check))
	}
	train[len(train)-1] = 7
	if check[0] != 0 {
		t.Fatal("raw calibration audio aliases the training samples")
	}
}

func TestShortBackgroundIsOnlyUsedForTraining(t *testing.T) {
	pcm := make([]int16, SampleRate)
	train, check := splitBackground(pcm)
	if len(train) != SampleRate || len(check) != 0 {
		t.Fatalf("short background should not create unusable calibration: %d/%d", len(train), len(check))
	}
}

func TestAugmentationJittersOnsetWithoutMislabellingPrefix(t *testing.T) {
	word := make([]int16, 8000)
	for i := range word {
		word[i] = 1000
	}
	starts := make(map[int]bool)
	for seed := range 8 {
		pcm, start, end := Augment(rand.New(rand.NewPCG(uint64(seed), 2)), word, [][]int16{make([]int16, SampleRate)}, preCtxSamp, postSamp)
		starts[start] = true
		if start < preCtxSamp || start > preCtxSamp+3*StepSamp || end <= start || end >= len(pcm) {
			t.Fatalf("invalid word offsets %d..%d in %d samples", start, end, len(pcm))
		}
		if pcm[start-1] != 0 || pcm[start] == 0 || pcm[end] != 0 {
			t.Fatal("reported offsets do not bound the inserted word")
		}
	}
	if len(starts) < 2 {
		t.Fatal("word onset is identical in all augmented signals")
	}
}
