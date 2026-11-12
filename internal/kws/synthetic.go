package kws

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"slices"
)

const syntheticSources, syntheticAugments = 8, 8

// prepareSynthetic keeps generated voices subordinate to real enrollment.
// Its reference level comes only from trimmed real TRAIN words; held-out
// recordings must never determine synthetic preprocessing or normalization.
func prepareSynthetic(input, realTrain [][]int16) [][]int16 {
	levels := make([]float64, 0, len(realTrain))
	for _, pcm := range realTrain {
		if len(pcm) > 0 {
			levels = append(levels, rms(pcm))
		}
	}
	if len(levels) == 0 {
		return nil
	}
	slices.Sort(levels)
	level := levels[len(levels)/2]
	if len(levels)%2 == 0 {
		level = (level + levels[len(levels)/2-1]) / 2
	}
	var clips [][]int16
	seen := make(map[string]bool)
	for _, pcm := range input {
		if len(clips) >= syntheticSources {
			break
		}
		clipped := slices.Contains(pcm, int16(-32768)) || slices.Contains(pcm, int16(32767))
		start, end, ok := FindSpeech(pcm)
		if clipped || !ok || end-start < SampleRate/5 || end-start > 2*SampleRate {
			continue
		}
		word := pcm[max(0, start-marginSamp):min(len(pcm), end+marginSamp)]
		var peak float64
		for _, sample := range word {
			peak = max(peak, math.Abs(float64(sample)))
		}
		scale := min(level/rms(word), 32766/max(peak, 1))
		normalized := make([]int16, len(word))
		for i, sample := range word {
			normalized[i] = clip16(float64(sample) * scale)
		}
		start, end, ok = FindSpeech(normalized)
		if !ok || end-start < SampleRate/5 || end-start > 2*SampleRate {
			continue
		}
		digest := syntheticDigest([][]int16{normalized})
		if !seen[digest] {
			seen[digest] = true
			clips = append(clips, normalized)
		}
	}
	return clips
}

// Include clip boundaries as well as normalized PCM so the digest describes
// exactly the ordered source set used by this model, independent of its cache.
func syntheticDigest(clips [][]int16) string {
	if len(clips) == 0 {
		return ""
	}
	h := sha256.New()
	for _, pcm := range clips {
		data := make([]byte, 4+2*len(pcm))
		binary.LittleEndian.PutUint32(data, uint32(len(pcm)))
		for i, sample := range pcm {
			binary.LittleEndian.PutUint16(data[4+2*i:], uint16(sample))
		}
		h.Write(data)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Append after every real/background job: their indices seed the original
// generators and must remain identical when synthetic augmentation is enabled.
func appendSyntheticJobs(jobs []trainingJob, clips [][]int16) []trainingJob {
	for source, pcm := range clips[:min(len(clips), syntheticSources)] {
		for range syntheticAugments {
			jobs = append(jobs, trainingJob{kind: syntheticTrain, pcm: pcm, syntheticSource: source})
		}
	}
	return jobs
}

// Round-robin original generations before taking further windows from one
// source. At most a fifth of positive examples can then be synthetic: both
// real and synthetic positives retain the same existing BCE weight.
func limitSynthetic(groups [][]Sample, realPos int) ([]Sample, []int) {
	limit := realPos / 4
	var selected []Sample
	var used []int
	for frame := 0; len(selected) < limit; frame++ {
		added := false
		for source, group := range groups {
			if len(selected) >= limit {
				break
			}
			if frame < len(group) {
				selected = append(selected, group[frame])
				if frame == 0 {
					used = append(used, source)
				}
				added = true
			}
		}
		if !added {
			break
		}
	}
	return selected, used
}
