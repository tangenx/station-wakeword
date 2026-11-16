package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

const evaluationSlack = 0.4

type evaluationWord struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

func (w *evaluationWord) UnmarshalJSON(b []byte) error {
	var bounds struct {
		Start *float64 `json:"start"`
		End   *float64 `json:"end"`
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&bounds); err != nil {
		return err
	}
	if bounds.Start == nil || bounds.End == nil {
		return errors.New("each word needs explicit non-null start and end")
	}
	w.Start, w.End = *bounds.Start, *bounds.End
	return nil
}

type evaluationRecording struct {
	File  string           `json:"file"`
	Words []evaluationWord `json:"words"`
}

type evaluationHit struct {
	Time      float64 `json:"time"`
	Score     float32 `json:"score"`
	WordIndex *int    `json:"word_index,omitempty"` // zero-based; absent for a false trigger
}

type evaluationLatency struct {
	Count int     `json:"count"`
	Mean  float64 `json:"mean"`
	Min   float64 `json:"min"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	Max   float64 `json:"max"`
}

type evaluationMetrics struct {
	Expected      int                `json:"expected"`
	Detected      int                `json:"detected"`
	Missed        int                `json:"missed"`
	FalseTriggers int                `json:"false_triggers"`
	NegativeHours float64            `json:"negative_hours"`
	FalsePerH     *float64           `json:"false_per_h"` // null without negative exposure
	Latency       *evaluationLatency `json:"latency_seconds,omitempty"`
}

type evaluationFileResult struct {
	evaluationMetrics
	File           string          `json:"file"`
	Duration       float64         `json:"duration_seconds"`
	ScoredDuration float64         `json:"scored_duration_seconds"`
	Hits           []evaluationHit `json:"hits"`
	MissedWords    []int           `json:"missed_word_indices,omitempty"`
	latencies      []float64
}

type evaluationReport struct {
	evaluationMetrics
	Model              string                 `json:"model"`
	Sensitivity        float64                `json:"sensitivity"`
	EffectiveThreshold float32                `json:"effective_threshold"`
	DetectionSlack     float64                `json:"detection_slack_seconds"`
	Files              []evaluationFileResult `json:"files"`
}

func cmdEvaluate(args []string) error {
	fs := flag.NewFlagSet("evaluate", flag.ContinueOnError)
	modelPath := fs.String("model", "", "trained model JSON")
	manifest := fs.String("manifest", "", "JSON list of independent recordings and annotated word intervals")
	sens := fs.Float64("sensitivity", 0, "threshold offset (negative = more sensitive, as in scan)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *manifest == "" || fs.NArg() != 0 {
		return errors.New("usage: wakeword evaluate -model model.json -manifest evaluation.json [-sensitivity offset]")
	}
	m, err := loadModel(*modelPath)
	if err != nil {
		return err
	}
	report, err := evaluateManifest(m, *manifest, *sens)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func validateEvaluationWords(words []evaluationWord, duration float64) error {
	for i, w := range words {
		if !finite(w.Start) || !finite(w.End) || w.Start < 0 || w.End <= w.Start || w.End > duration {
			return fmt.Errorf("word %d: need 0 <= start < end <= scored recording duration %.3f", i, duration)
		}
		if i > 0 && w.Start < words[i-1].End {
			return fmt.Errorf("word %d: annotations must be chronological and non-overlapping", i)
		}
	}
	return nil
}

func evaluateManifest(m *kws.Model, manifest string, sensitivity float64) (evaluationReport, error) {
	var report evaluationReport
	if !finite(sensitivity) || math.Abs(sensitivity) > math.MaxFloat32 {
		return report, errors.New("sensitivity must be a finite float32 offset")
	}
	if m == nil || m.Head == nil || !finite(float64(m.Threshold)) {
		return report, errors.New("invalid evaluation model")
	}
	if err := m.Head.Validate(); err != nil {
		return report, err
	}
	for _, tensor := range [][]float32{m.Head.W1, m.Head.B1, m.Head.W2, m.Head.B2, m.Head.W3, m.Head.B3} {
		for _, x := range tensor {
			if !finite(float64(x)) {
				return report, errors.New("model contains nonfinite weights")
			}
		}
	}
	f, err := os.Open(manifest)
	if err != nil {
		return report, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var recordings []evaluationRecording
	if err := dec.Decode(&recordings); err != nil {
		return report, fmt.Errorf("manifest: %w", err)
	}
	if err := dec.Decode(new(any)); !errors.Is(err, io.EOF) {
		return report, errors.New("manifest must contain exactly one JSON array")
	}
	if len(recordings) == 0 {
		return report, errors.New("manifest must contain at least one recording")
	}
	base, err := filepath.Abs(filepath.Dir(manifest))
	if err != nil {
		return report, err
	}
	report.Model, report.Sensitivity = m.Name, sensitivity
	report.EffectiveThreshold = min(0.99, max(0.05, m.Threshold+float32(sensitivity)))
	report.DetectionSlack = evaluationSlack
	seen := map[string]bool{}
	var latencies []float64
	for _, recording := range recordings {
		if strings.TrimSpace(recording.File) == "" {
			return report, errors.New("manifest: each recording needs a file path")
		}
		path := recording.File
		if !filepath.IsAbs(path) {
			path = filepath.Join(base, path)
		}
		canonical, err := filepath.EvalSymlinks(path)
		if err != nil {
			return report, fmt.Errorf("%s: %w", recording.File, err)
		}
		if seen[canonical] {
			return report, fmt.Errorf("manifest: duplicate recording %s", recording.File)
		}
		seen[canonical] = true
		// Keep only one decoded WAV in memory; reports retain timestamps, not audio.
		pcm, err := wav.ReadFile(path, kws.SampleRate)
		if err != nil {
			return report, err
		}
		duration := float64(len(pcm)) / kws.SampleRate
		if duration == 0 {
			return report, fmt.Errorf("%s: empty recording", recording.File)
		}
		// A final incomplete 80 ms frame produces no score. Neither count
		// that tail as negative exposure nor accept labels in it.
		scoredSamples := len(pcm) / kws.StepSamp * kws.StepSamp
		scoredDuration := float64(scoredSamples) / kws.SampleRate
		if err := validateEvaluationWords(recording.Words, scoredDuration); err != nil {
			return report, fmt.Errorf("%s: %w", recording.File, err)
		}
		hits, err := evaluateAudio(m, pcm[:scoredSamples], float32(sensitivity))
		if err != nil {
			return report, err
		}
		r := matchEvaluation(recording.File, scoredDuration, recording.Words, hits)
		r.Duration = duration
		report.Expected += r.Expected
		report.Detected += r.Detected
		report.Missed += r.Missed
		report.FalseTriggers += r.FalseTriggers
		report.NegativeHours += r.NegativeHours
		latencies = append(latencies, r.latencies...)
		report.Files = append(report.Files, r)
	}
	report.FalsePerH = falsePerHour(report.FalseTriggers, report.NegativeHours)
	report.Latency = summarizeEvaluationLatency(latencies)
	return report, nil
}

func evaluateAudio(m *kws.Model, pcm []int16, sensitivity float32) ([]evaluationHit, error) {
	d, err := kws.NewDetector([]*kws.Model{m}, []float32{sensitivity})
	if err != nil {
		return nil, err
	}
	// 2.32 seconds is at least 2.3 s and an exact number of 80 ms frames,
	// so every callback below has an exact recording-relative timestamp.
	// The prefix warms only features/history: discarded synthetic activations
	// must not place the real recording inside a cooldown.
	d.Push(make([]int16, 29*kws.StepSamp), func(int, float32) {})
	d.ResetDecisionState()
	hits := make([]evaluationHit, 0)
	for pos := 0; pos < len(pcm); {
		n := min(kws.StepSamp, len(pcm)-pos)
		d.Push(pcm[pos:pos+n], func(_ int, score float32) {
			hits = append(hits, evaluationHit{Time: float64(pos+n) / kws.SampleRate, Score: score})
		})
		pos += n
	}
	return hits, nil
}

// Match chronological hits to the earliest unmatched annotation whose detection
// interval includes the hit. A second activation for an occurrence is false.
func matchEvaluation(file string, duration float64, words []evaluationWord, hits []evaluationHit) evaluationFileResult {
	r := evaluationFileResult{File: file, Duration: duration, ScoredDuration: duration, Hits: slices.Clone(hits)}
	r.Expected = len(words)
	matched := make([]bool, len(words))
	for i := range r.Hits {
		hit := &r.Hits[i]
		hit.WordIndex = nil
		for j, word := range words {
			if !matched[j] && hit.Time >= word.Start && hit.Time <= min(duration, word.End+evaluationSlack) {
				matched[j] = true
				hit.WordIndex = new(int)
				*hit.WordIndex = j
				r.Detected++
				r.latencies = append(r.latencies, hit.Time-word.End)
				break
			}
		}
		if hit.WordIndex == nil {
			r.FalseTriggers++
		}
	}
	var excluded, coveredUntil float64
	for i, word := range words {
		if !matched[i] {
			r.MissedWords = append(r.MissedWords, i)
		}
		end := min(duration, word.End+evaluationSlack)
		excluded += max(0, end-max(word.Start, coveredUntil))
		coveredUntil = max(coveredUntil, end)
	}
	r.Missed = r.Expected - r.Detected
	r.NegativeHours = max(0, duration-excluded) / 3600
	r.FalsePerH = falsePerHour(r.FalseTriggers, r.NegativeHours)
	r.Latency = summarizeEvaluationLatency(r.latencies)
	return r
}

func falsePerHour(events int, hours float64) *float64 {
	if hours <= 0 {
		return nil
	}
	rate := float64(events) / hours
	return &rate
}

// Latency is relative to the annotated word end; negative values are early
// detections. Quantiles use nearest rank, including for small sample counts.
func summarizeEvaluationLatency(latencies []float64) *evaluationLatency {
	if len(latencies) == 0 {
		return nil
	}
	sorted := slices.Clone(latencies)
	slices.Sort(sorted)
	var total float64
	for _, v := range sorted {
		total += v
	}
	n := len(sorted)
	return &evaluationLatency{Count: n, Mean: total / float64(n), Min: sorted[0], Max: sorted[n-1],
		P50: sorted[int(math.Ceil(0.5*float64(n)))-1], P95: sorted[int(math.Ceil(0.95*float64(n)))-1]}
}
