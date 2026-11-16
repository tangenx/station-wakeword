package kws

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestCalibrationDoesNotLowerSafeThreshold(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{0.999, 0.999}, First: 0, Last: 1}}, [][]float32{{0.975, 0.975}})
	if !c.Feasible || c.Threshold <= 0.975 || c.FalseTriggers != 0 || c.Detected != 1 {
		t.Fatalf("unsafe calibration: %+v", c)
	}
}

func TestPastMistakesConstrainThresholdWithoutInflatingExposure(t *testing.T) {
	h := newHead(rand.New(rand.NewPCG(1, 2)))
	for _, p := range h.params() {
		clear(p)
	}
	h.W1[0], h.W2[0], h.W3[0] = 1, 1, 1
	stream := func(v float32) Stream {
		s := make(Stream, (Ctx+1)*EmbDim)
		for i := range Ctx + 1 {
			s[i*EmbDim] = v
		}
		return s
	}
	c := validateHead(h, []validationTake{{Stream: stream(10), Start: 0, End: 10 * SampleRate}}, []Stream{stream(0)}, []Stream{stream(8)})
	if c.Feasible || c.FalseTriggers != 0 || math.Abs(c.EvalHours-0.16/3600) > 1e-9 {
		t.Fatalf("known mistake was ignored or counted as independent evidence: %+v", c)
	}
}

func TestCalibrationKeepsRecognizableQuietTake(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{0.89, 0.89}, First: 0, Last: 1}}, [][]float32{{0.87, 0.87}})
	if !c.Feasible || c.Detected != 1 || c.Threshold > 0.89 || c.FalseTriggers != 0 {
		t.Fatalf("margin lost the held-out word: %+v", c)
	}
}

func TestCalibrationCannotHideUnseparableNegatives(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{1, 1}, First: 0, Last: 1}}, [][]float32{{1, 1}})
	if c.Feasible || c.FalseTriggers != 1 {
		t.Fatalf("accepted unseparable background: %+v", c)
	}
}

func TestCalibrationCountsEventsAndExposure(t *testing.T) {
	negative := make([]float32, 100)
	negative[5] = 0.99 // isolated spikes do not count as events
	c := calibrateScores([]scoredTake{{Scores: []float32{0.1, 0.9, 0.9}, First: 1, Last: 2}}, [][]float32{negative})
	if !c.Feasible || c.Detected != 1 || c.FalseTriggers != 0 || math.Abs(c.EvalHours-8.0/3600) > 1e-9 {
		t.Fatalf("wrong streaming metrics: %+v", c)
	}
}

func TestPositiveEvaluationHonorsCooldownBeforeWord(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{0.9, 0.9, 0.1, 0.9, 0.9}, First: 3, Last: 4}}, [][]float32{{0.1, 0.1}})
	if c.Detected != 0 {
		t.Fatalf("a pre-word false trigger consumed the cooldown, but recall was counted: %+v", c)
	}
}

func TestCalibrationFindsThresholdAbovePrewordSpike(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{0.5, 0.5, 0.1, 0.9, 0.9}, First: 3, Last: 4}}, [][]float32{{0.1, 0.1}})
	if !c.Feasible || c.Detected != 1 || c.Threshold <= 0.5 || c.Threshold > 0.9 {
		t.Fatalf("failed to suppress pre-word trigger while preserving the real word: %+v", c)
	}
}

func TestCalibrationNeedsRealExposure(t *testing.T) {
	c := calibrateScores([]scoredTake{{Scores: []float32{0.9, 0.9}, First: 0, Last: 1}}, nil)
	if c.Feasible {
		t.Fatal("calibration without measured background must not be feasible")
	}
}
