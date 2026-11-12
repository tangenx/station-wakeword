package kws

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestMineHardDoesNotRepeatWeightedWindow(t *testing.T) {
	h := newHead(rand.New(rand.NewPCG(1, 2)))
	x := make([]float32, Ctx*EmbDim)
	other := make([]float32, Ctx*EmbDim)
	other[0] = 1
	neg := []Sample{{X: x}, {X: x}, {X: x}, {X: other}}
	got := mineHard(h, neg, 3, 1)
	if len(got) != 2 {
		t.Fatalf("mined %d windows from two independent windows, want 2", len(got))
	}
	if &got[0].X[0] == &got[1].X[0] {
		t.Fatal("weighted copies occupied both hard-negative slots")
	}
}

func smallTrainingSet() (pos, neg []Sample) {
	x := make([]float32, Ctx*EmbDim)
	x[0] = 1
	return []Sample{{X: x, Y: 1}}, []Sample{{X: make([]float32, Ctx*EmbDim)}}
}

func TestTrainReturnsBestValidationEpoch(t *testing.T) {
	pos, neg := smallTrainingSet()
	var checks []int
	var best [][]float32
	lastEpoch := 0
	h, err := Train(pos, neg, TrainConfig{
		Epochs: 10, Batch: 2, Seed: 12, ValidateEvery: 2, Patience: 2,
		OnProgress: func(epoch int, _ float64) { lastEpoch = epoch },
		Validate: func(epoch int, h *Head) float64 {
			checks = append(checks, epoch)
			if epoch == 2 {
				for _, p := range h.params() {
					best = append(best, slices.Clone(p))
				}
			}
			return float64(epoch)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(checks, []int{2, 4, 6}) || lastEpoch != 6 {
		t.Fatalf("validation epochs %v; stopped at %d, want [2 4 6] and 6", checks, lastEpoch)
	}
	for i, p := range h.params() {
		if !slices.Equal(p, best[i]) {
			t.Fatalf("returned parameters %d differ from best epoch 2", i)
		}
	}
}

func TestTrainValidationDefaultsAndFinalEpoch(t *testing.T) {
	pos, neg := smallTrainingSet()
	var checks []int
	_, err := Train(pos, neg, TrainConfig{Epochs: 6, Seed: 3,
		Validate: func(epoch int, _ *Head) float64 {
			checks = append(checks, epoch)
			return float64(-epoch)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(checks, []int{4, 6}) {
		t.Fatalf("validation epochs %v, want [4 6]", checks)
	}
}

func TestTrainDefaultPatienceStopsAfterThreeStaleChecks(t *testing.T) {
	pos, neg := smallTrainingSet()
	var checks []int
	_, err := Train(pos, neg, TrainConfig{Epochs: 24, Seed: 3,
		Validate: func(epoch int, _ *Head) float64 {
			checks = append(checks, epoch)
			return 1
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(checks, []int{4, 8, 12, 16}) {
		t.Fatalf("validation epochs %v, want best at 4 then three stale checks", checks)
	}
}

func TestTrainRejectsNonfiniteValidation(t *testing.T) {
	pos, neg := smallTrainingSet()
	for _, metric := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := Train(pos, neg, TrainConfig{Epochs: 1, Seed: 3,
			Validate: func(int, *Head) float64 { return metric },
		})
		if err == nil || !strings.Contains(err.Error(), "validation") {
			t.Fatalf("metric %v: got error %v, want validation error", metric, err)
		}
	}
}

func TestTrainWithoutValidationRunsAllEpochs(t *testing.T) {
	pos, neg := smallTrainingSet()
	lastEpoch := 0
	h, err := Train(pos, neg, TrainConfig{Epochs: 5, Seed: 9,
		OnProgress: func(epoch int, _ float64) { lastEpoch = epoch },
	})
	if err != nil {
		t.Fatal(err)
	}
	if lastEpoch != 5 {
		t.Fatalf("stopped at epoch %d, want 5", lastEpoch)
	}
	if err := h.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestMineHardDiversifiesRecordingsAndAdjacentFrames(t *testing.T) {
	h := newHead(rand.New(rand.NewPCG(1, 2)))
	for _, p := range h.params() {
		clear(p)
	}
	// A deterministic score that increases with the first input value.
	h.W1[0], h.W2[0], h.W3[0] = 1, 1, 1
	window := func(v float32, source, frame int) Sample {
		x := make([]float32, Ctx*EmbDim)
		x[0] = v
		return Sample{X: x, Source: source, Frame: frame}
	}
	neg := []Sample{
		window(8, 1, 10), window(7, 1, 11), window(6, 1, 12),
		window(5, 1, 30), window(4, 1, 50), window(3, 1, 70),
		window(2, 2, 10), window(1, 3, 10), window(.5, 0, 0),
	}
	got := mineHard(h, neg, 4, 1)
	var sources []int
	for _, s := range got {
		sources = append(sources, s.Source)
	}
	if !slices.Equal(sources, []int{1, 2, 3, 0}) {
		t.Fatalf("mined sources %v, want [1 2 3 0]", sources)
	}
	got = mineHard(h, neg[:6], 4, 1)
	var frames []int
	for _, s := range got {
		frames = append(frames, s.Frame)
	}
	if !slices.Equal(frames, []int{10, 30, 50, 70}) {
		t.Fatalf("mined frames %v, want separated [10 30 50 70]", frames)
	}
}

func TestMineHardIndependentWindowsDoNotShareSourceQuota(t *testing.T) {
	h := newHead(rand.New(rand.NewPCG(1, 2)))
	neg := make([]Sample, 5)
	for i := range neg {
		neg[i].X = make([]float32, Ctx*EmbDim)
		neg[i].X[0] = float32(i)
	}
	if got := mineHard(h, neg, 4, 1); len(got) != 4 {
		t.Fatalf("mined %d independent windows, want 4", len(got))
	}
}
