package kws

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func syntheticTone(hz, amplitude float64) []int16 {
	pcm := make([]int16, SampleRate)
	for i := SampleRate / 4; i < 3*SampleRate/4; i++ {
		pcm[i] = int16(amplitude * math.Sin(2*math.Pi*hz*float64(i)/SampleRate))
	}
	return pcm
}

func TestSyntheticNormalizationUsesOnlySuppliedRealTrainingWords(t *testing.T) {
	realWord := make([]int16, SampleRate/2)
	for i := range realWord {
		realWord[i] = 700
	}
	quiet, loud := slices.Clone(realWord), slices.Clone(realWord)
	for i := range realWord {
		quiet[i], loud[i] = 100, 20000
	}
	realTrain := [][]int16{loud, realWord, quiet}
	input := syntheticTone(220, 14000)
	before := slices.Clone(input)
	got := prepareSynthetic([][]int16{input}, realTrain)
	if len(got) != 1 {
		t.Fatalf("accepted %d synthetic sources, want 1", len(got))
	}
	if level := rms(got[0]); math.Abs(level-700) > 1 {
		t.Fatalf("normalized synthetic RMS = %.2f, want real train RMS 700", level)
	}
	if !slices.Equal(input, before) {
		t.Fatal("normalization modified original cached audio")
	}
	again := prepareSynthetic([][]int16{input}, realTrain)
	if !slices.Equal(got[0], again[0]) || syntheticDigest(got) != syntheticDigest(again) {
		t.Fatal("same recordings produced nondeterministic normalization or digest")
	}
}

func TestSyntheticRejectsInvalidClipsAndLimitsDistinctSources(t *testing.T) {
	first := syntheticTone(220, 14000)
	clipped := slices.Clone(first)
	clipped[100] = 32767
	input := [][]int16{nil, make([]int16, SampleRate), clipped, first, slices.Clone(first)}
	for i := range 12 {
		input = append(input, syntheticTone(float64(300+20*i), 12000))
	}
	got := prepareSynthetic(input, [][]int16{first})
	if len(got) != 8 {
		t.Fatalf("accepted %d distinct usable sources, want limit 8", len(got))
	}
	seen := make(map[string]bool)
	for _, clip := range got {
		digest := syntheticDigest([][]int16{clip})
		if seen[digest] {
			t.Fatal("duplicate synthetic generation counted as distinct source")
		}
		seen[digest] = true
		for _, v := range clip {
			if v == -32768 || v == 32767 {
				t.Fatal("normalized synthetic source clips at PCM limits")
			}
		}
	}
	if got := prepareSynthetic(input, nil); len(got) != 0 {
		t.Fatal("synthetic normalization accepted an enrollment with no real train words")
	}
}

func TestSyntheticJobsPreserveRealHoldoutAndRandomGenerators(t *testing.T) {
	word := syntheticTone(220, 2000)
	before := []trainingJob{{kind: positiveTrain, pcm: word}, {kind: positiveCheck, pcm: word, start: 1, end: 2}, {kind: negativeCheck, pcm: word}}
	jobs := appendSyntheticJobs(slices.Clone(before), [][]int16{word, word})
	if len(jobs) != 19 {
		t.Fatalf("got %d jobs, want original 3 plus 2 sources x 8 augmentations", len(jobs))
	}
	for i, job := range before {
		if jobs[i].kind != job.kind || jobs[i].start != job.start || jobs[i].end != job.end || &jobs[i].pcm[0] != &job.pcm[0] {
			t.Fatalf("existing training/calibration job %d changed", i)
		}
		// Keeping each real job at its original index also preserves the
		// existing per-job RNG seed used to construct held-out input audio.
	}
	for _, job := range jobs[len(before):] {
		if job.kind != syntheticTrain {
			t.Fatalf("synthetic source entered a real/validation job: kind %d", job.kind)
		}
	}
}

func TestSyntheticWindowsAreCappedAndSpreadAcrossOriginalSources(t *testing.T) {
	groups := [][]Sample{{{Frame: 10}, {Frame: 11}, {Frame: 12}}, {{Frame: 20}, {Frame: 21}}, {{Frame: 30}}}
	got, used := limitSynthetic(groups, 20)
	var frames []int
	for _, sample := range got {
		frames = append(frames, sample.Frame)
	}
	if !slices.Equal(frames, []int{10, 20, 30, 11, 21}) || !slices.Equal(used, []int{0, 1, 2}) {
		t.Fatalf("selected frames %v sources %v, want five diverse windows [10 20 30 11 21]", frames, used)
	}
	if got, used := limitSynthetic(groups, 3); len(got) != 0 || len(used) != 0 {
		t.Fatal("synthetic examples exceeded one quarter of the real-positive count")
	}
	got, used = limitSynthetic(groups, 4)
	if len(got) != 1 || !slices.Equal(used, []int{0}) {
		t.Fatalf("metadata includes unused sources: windows %v sources %v", got, used)
	}
	if syntheticDigest(nil) != "" {
		t.Fatal("empty synthetic set should have no metadata digest")
	}
}

func TestSyntheticCannotReplaceMinimumRealEnrollment(t *testing.T) {
	word := syntheticTone(220, 2000)
	_, err := TrainWord("test", Enrollment{Takes: [][]int16{word, word}, Synthetic: [][]int16{word, word, word, word}}, &Negatives{}, 42, nil)
	if err == nil || !strings.Contains(err.Error(), "only 2 usable takes") {
		t.Fatalf("synthetic clips bypassed the real-take minimum: %v", err)
	}
}

func TestSyntheticTrainWordKeepsRealCalibrationAndUsedMetadata(t *testing.T) {
	word := syntheticTone(220, 2000)
	en := Enrollment{
		Takes: [][]int16{word, word, word}, Ambient: make([]int16, 8*SampleRate),
		Synthetic: [][]int16{syntheticTone(300, 14000)},
	}
	neg := &Negatives{Windows: [][]float32{make([]float32, Ctx*EmbDim)}, Streams: []Stream{make(Stream, 64*EmbDim)}}
	m, err := TrainWord("test", en, neg, 42, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.Takes != 3 || m.EvalTakes != 1 || m.Seed != 42 {
		t.Fatalf("synthetic sources affected real enrollment/calibration: takes %d, eval %d, seed %d", m.Takes, m.EvalTakes, m.Seed)
	}
	if m.SyntheticTakes != 1 || len(m.SyntheticDigest) != 64 {
		t.Fatalf("used synthetic metadata = %d / %q, want one source and SHA-256", m.SyntheticTakes, m.SyntheticDigest)
	}
}
