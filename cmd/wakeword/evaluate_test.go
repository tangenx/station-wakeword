package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

func TestEvaluationMatchesOnlyAnnotatedOccurrences(t *testing.T) {
	words := []evaluationWord{{Start: 1, End: 1.5}, {Start: 4, End: 4.5}, {Start: 7, End: 7.5}}
	hits := []evaluationHit{{Time: 0.5}, {Time: 1.3}, {Time: 1.6}, {Time: 4.9}, {Time: 9}}
	r := matchEvaluation("recording.wav", 10, words, hits)
	if r.Expected != 3 || r.Detected != 2 || r.Missed != 1 || r.FalseTriggers != 3 {
		t.Fatalf("matching: %+v", r)
	}
	if r.Hits[1].WordIndex == nil || *r.Hits[1].WordIndex != 0 || r.Hits[2].WordIndex != nil {
		t.Fatalf("duplicate hit consumed the same annotation: %+v", r.Hits)
	}
	// Each annotation excludes 0.5 seconds plus 0.4 seconds of detection slack.
	if math.Abs(r.NegativeHours*3600-7.3) > 1e-9 {
		t.Fatalf("negative exposure: %.9f seconds", r.NegativeHours*3600)
	}
	if r.Latency == nil || r.Latency.Count != 2 || math.Abs(r.Latency.Mean-0.1) > 1e-9 {
		t.Fatalf("latency relative to word end: %+v", r.Latency)
	}
}

func TestEvaluationMergesSlackAndHandlesZeroExposure(t *testing.T) {
	words := []evaluationWord{{Start: 0, End: 0.6}, {Start: 0.8, End: 1.6}}
	r := matchEvaluation("all-word.wav", 2, words, []evaluationHit{{Time: 0.9}, {Time: 1.8}})
	if r.NegativeHours != 0 || r.FalsePerH != nil || r.Detected != 2 {
		t.Fatalf("overlapping detection slack double-counted or invented background: %+v", r)
	}
	b, err := json.Marshal(r)
	if err != nil || !strings.Contains(string(b), `"false_per_h":null`) {
		t.Fatalf("zero-exposure report: %s, %v", b, err)
	}
}

func TestEvaluationRejectsInvalidAnnotations(t *testing.T) {
	cases := map[string][]evaluationWord{
		"negative": {{Start: -0.1, End: 1}},
		"reversed": {{Start: 1, End: 0.5}},
		"empty":    {{Start: 1, End: 1}},
		"outside":  {{Start: 0.5, End: 3.1}},
		"overlap":  {{Start: 0.5, End: 1.5}, {Start: 1.4, End: 2}},
		"unsorted": {{Start: 2, End: 2.5}, {Start: 0.1, End: 1}},
		"NaN":      {{Start: math.NaN(), End: 1}},
		"infinity": {{Start: 0, End: math.Inf(1)}},
	}
	for name, words := range cases {
		t.Run(name, func(t *testing.T) {
			if err := validateEvaluationWords(words, 3); err == nil {
				t.Fatalf("accepted invalid annotations: %+v", words)
			}
		})
	}
}

func constantEvaluationModel(bias float32) *kws.Model {
	return &kws.Model{Name: "constant", Threshold: 0.5, Head: &kws.Head{
		W1: make([]float32, kws.EmbDim*kws.H1), B1: make([]float32, kws.H1),
		W2: make([]float32, kws.Ctx*kws.H1*kws.H2), B2: make([]float32, kws.H2),
		W3: make([]float32, kws.H2), B3: []float32{bias},
	}}
}

func TestEvaluateManifestReplaysDetectorGateAndCooldown(t *testing.T) {
	dir := t.TempDir()
	if err := wav.WriteFile(filepath.Join(dir, "silence.wav"), make([]int16, 6*kws.SampleRate), kws.SampleRate); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(dir, "evaluation.json")
	if err := os.WriteFile(manifest, []byte(`[{"file":"silence.wav","words":[{"start":2.7,"end":2.9}]}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := evaluateManifest(constantEvaluationModel(10), manifest, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.Expected != 1 || r.Detected != 1 || r.Missed != 0 || r.FalseTriggers != 1 || len(r.Files) != 1 {
		t.Fatalf("real detector replay: %+v", r)
	}
	// Synthetic warmup only initializes features. The real two-window gate
	// starts afresh, then triggers every 3.04 seconds on this broken head.
	hits := r.Files[0].Hits
	if len(hits) != 2 || math.Abs(hits[0].Time-0.16) > 1e-9 || math.Abs(hits[1].Time-3.2) > 1e-9 {
		t.Fatalf("gate timing or warmup offset changed: %+v", hits)
	}
	if math.Abs(r.NegativeHours*3600-5.4) > 1e-9 || r.FalsePerH == nil {
		t.Fatalf("warmup counted as exposure: %+v", r)
	}
}

func TestEvaluateManifestRejectsBadInputs(t *testing.T) {
	dir := t.TempDir()
	if err := wav.WriteFile(filepath.Join(dir, "room.wav"), make([]int16, kws.SampleRate), kws.SampleRate); err != nil {
		t.Fatal(err)
	}
	cases := []string{
		`[]`, `null`, `[{}]`, `[{"file":""}]`, `[{"file":"absent.wav"}]`,
		`[{"file":"room.wav","words":[{"start":0,"end":2}]}]`,
		`[{"file":"room.wav","words":[{"end":0.5}]}]`,
		`[{"file":"room.wav","words":[{"start":null,"end":0.5}]}]`,
		`[{"file":"room.wav","words":[{"start":0}]}]`,
		`[{"file":"room.wav","words":[{"start":0,"end":null}]}]`,
		`[{"file":"room.wav","words":[{"start":0,"end":0.5,"typo":true}]}]`,
		`[{"file":"room.wav"},{"file":"./room.wav"}]`,
		`[{"file":"room.wav","word":[]}]`,
		`[{"file":"room.wav"}] []`,
	}
	for _, content := range cases {
		t.Run(content, func(t *testing.T) {
			manifest := filepath.Join(dir, "evaluation.json")
			if err := os.WriteFile(manifest, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := evaluateManifest(constantEvaluationModel(-10), manifest, 0); err == nil {
				t.Fatalf("accepted invalid manifest: %s", content)
			}
		})
	}
}

func TestEvaluateCountsOnlyCompleteScoredFrames(t *testing.T) {
	for _, tc := range []struct {
		name    string
		samples int
		seconds float64
	}{
		{"shorter than a frame", kws.StepSamp / 2, 0},
		{"partial trailing frame", kws.StepSamp + kws.StepSamp/2, 0.08},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := wav.WriteFile(filepath.Join(dir, "short.wav"), make([]int16, tc.samples), kws.SampleRate); err != nil {
				t.Fatal(err)
			}
			manifest := filepath.Join(dir, "evaluation.json")
			if err := os.WriteFile(manifest, []byte(`[{"file":"short.wav"}]`), 0o644); err != nil {
				t.Fatal(err)
			}
			r, err := evaluateManifest(constantEvaluationModel(-10), manifest, 0)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(r.NegativeHours*3600-tc.seconds) > 1e-9 {
				t.Fatalf("counted unscored tail: %.9f seconds, want %.2f", r.NegativeHours*3600, tc.seconds)
			}
			if math.Abs(r.Files[0].ScoredDuration-tc.seconds) > 1e-9 || math.Abs(r.Files[0].Duration-float64(tc.samples)/kws.SampleRate) > 1e-9 {
				t.Fatalf("full and scored durations not distinguished: %+v", r.Files[0])
			}
			if tc.seconds == 0 && (r.FalsePerH != nil || r.Files[0].FalsePerH != nil) {
				t.Fatalf("invented false/h without a scored frame: %+v", r)
			}
		})
	}
}

func TestEvaluationWarmupCannotSuppressRealFalseTrigger(t *testing.T) {
	hits, err := evaluateAudio(constantEvaluationModel(10), make([]int16, kws.SampleRate), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || math.Abs(hits[0].Time-0.16) > 1e-9 {
		t.Fatalf("synthetic warmup hid an always-positive model: %+v", hits)
	}
}

func TestEvaluateRejectsWordInUnscoredTail(t *testing.T) {
	dir := t.TempDir()
	if err := wav.WriteFile(filepath.Join(dir, "short.wav"), make([]int16, kws.StepSamp+kws.StepSamp/2), kws.SampleRate); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(dir, "evaluation.json")
	if err := os.WriteFile(manifest, []byte(`[{"file":"short.wav","words":[{"start":0,"end":0.1}]}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := evaluateManifest(constantEvaluationModel(-10), manifest, 0); err == nil {
		t.Fatal("accepted word extending beyond the last scored frame at 0.08 seconds")
	}
}

func TestEvaluationAnnotationAllowsExplicitZeroStart(t *testing.T) {
	var word evaluationWord
	if err := json.Unmarshal([]byte(`{"start":0,"end":0.5}`), &word); err != nil || word.Start != 0 || word.End != 0.5 {
		t.Fatalf("explicit zero start rejected: %+v, %v", word, err)
	}
}
