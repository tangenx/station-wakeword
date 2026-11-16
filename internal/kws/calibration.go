package kws

import (
	"math"
	"slices"
)

// These are score limits, not probabilities of a correct recognition. Keep
// maxThreshold in sync with the detector's supported threshold range.
const minThreshold, maxThreshold float32 = 0.3, 0.99

type scoredTake struct {
	Scores      []float32
	First, Last int // inclusive score indices where the word may trigger
}

type calibration struct {
	Threshold     float32
	Detected      int
	Recall        float64
	FalseTriggers int
	EvalHours     float64
	Feasible      bool
	gap           float64
}

func countTriggers(scores []float32, threshold float32, first, last int) int {
	var g gate
	n := 0
	for i, score := range scores {
		if g.step(score, threshold) && i >= first && i <= last {
			n++
		}
	}
	return n
}

func pairPeak(scores []float32, first, last int) float32 {
	var peak float32
	for i := max(1, first); i <= last && i < len(scores); i++ {
		peak = max(peak, min(scores[i-1], scores[i]))
	}
	return peak
}

// calibrateScores selects a zero-event threshold on the measured background.
// With the small station corpus, allowing even one event would mean several
// false activations/hour. A separate, untouched long test is still needed to
// measure rare-event performance; this is calibration, not such a test.
func calibrateScores(positive []scoredTake, negative [][]float32) calibration {
	c := calibration{Threshold: maxThreshold}
	var negPeak float32
	for _, scores := range negative {
		c.EvalHours += float64(len(scores)) * StepSamp / SampleRate / 3600
		negPeak = max(negPeak, pairPeak(scores, 0, len(scores)-1))
	}
	if len(positive) == 0 || c.EvalHours == 0 {
		return c
	}
	// Two consecutive scores pass at equality, so the safe threshold must
	// be strictly above the largest negative pair. Never lower it by a cap.
	base := max(minThreshold, math.Nextafter32(negPeak, float32(math.Inf(1))))
	c.Feasible = base <= maxThreshold
	if c.Feasible {
		c.Threshold = base
	}
	detected := func(threshold float32) int {
		n := 0
		for _, take := range positive {
			if countTriggers(take.Scores, threshold, take.First, take.Last) > 0 {
				n++
			}
		}
		return n
	}
	c.Detected = detected(c.Threshold)
	if c.Feasible {
		// A higher threshold may prevent a pre-word event from consuming
		// cooldown. Gated recall is not monotonic, so examine every positive
		// pair boundary rather than assuming the lowest safe value is best.
		candidates := []float32{base, maxThreshold}
		for _, take := range positive {
			for i := 1; i < len(take.Scores); i++ {
				v := min(take.Scores[i-1], take.Scores[i])
				for _, t := range []float32{v, math.Nextafter32(v, float32(math.Inf(1)))} {
					if t >= base && t <= maxThreshold {
						candidates = append(candidates, t)
					}
				}
			}
		}
		slices.Sort(candidates)
		for _, t := range slices.Compact(candidates) {
			if n := detected(t); n > c.Detected {
				c.Threshold, c.Detected = t, n
			}
		}
		base = c.Threshold
		// Add up to .03 safety margin only while preserving recall. Raising
		// the threshold can also prevent a pre-word spike consuming cooldown.
		for i := 1; i <= 3; i++ {
			t := min(maxThreshold, base+float32(i)*0.01)
			if n := detected(t); n >= c.Detected {
				c.Threshold, c.Detected = t, n
			}
		}
	}
	for _, scores := range negative {
		c.FalseTriggers += countTriggers(scores, c.Threshold, 0, len(scores)-1)
	}
	c.Recall = float64(c.Detected) / float64(len(positive))
	for _, take := range positive {
		c.gap += float64(pairPeak(take.Scores, take.First, take.Last) - negPeak)
	}
	c.gap /= float64(len(positive))
	return c
}

// Recall is the main selection criterion; separation breaks ties without
// outweighing even one missed real take. Infeasible heads always rank last.
func (c calibration) metric() float64 {
	if !c.Feasible {
		return 3 + float64(c.FalseTriggers)
	}
	return 1 - c.Recall - 0.001*c.gap
}

type validationTake struct {
	Stream     Stream
	Start, End int // sample offsets of the actual word, without trim margins
}

func validateHead(h *Head, positive []validationTake, negative []Stream, regression ...[]Stream) calibration {
	pos := make([]scoredTake, len(positive))
	for i, take := range positive {
		p := &pos[i]
		p.First = -1
		p.Last = -1
		for e := Ctx - 1; e < take.Stream.Len(); e++ {
			end := EmbEnd(e)
			if end >= take.Start && end <= take.End+posSlack {
				if p.First < 0 {
					p.First = len(p.Scores)
				}
				p.Last = len(p.Scores)
			}
			p.Scores = append(p.Scores, h.Score(take.Stream.Window(e)))
		}
	}
	all := append([]Stream(nil), negative...)
	for _, streams := range regression {
		all = append(all, streams...)
	}
	neg := make([][]float32, len(all))
	var evalHours float64
	for i, stream := range all {
		for e := Ctx - 1; e < stream.Len(); e++ {
			neg[i] = append(neg[i], h.Score(stream.Window(e)))
		}
		if i < len(negative) {
			evalHours += float64(len(neg[i])) * StepSamp / SampleRate / 3600
		}
	}
	c := calibrateScores(pos, neg)
	// Known past mistakes also constrain the threshold, but they were used
	// in training and must never inflate independent evaluation exposure.
	c.EvalHours = evalHours
	c.FalseTriggers = 0
	for _, scores := range neg[:len(negative)] {
		c.FalseTriggers += countTriggers(scores, c.Threshold, 0, len(scores)-1)
	}
	return c
}
