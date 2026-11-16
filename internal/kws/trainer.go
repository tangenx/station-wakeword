package kws

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"time"
)

// Negatives is the bundled general-purpose negative set: independent
// Ctx-long windows (multilingual speech, music, noise) plus continuous
// streams used for false-alarm estimates.
type Negatives struct {
	Windows [][]float32
	Streams []Stream
}

// ReadNegatives parses "NEG1" | u32 nWindows | u32 nStreams | u32 len... |
// float16 data (windows first, then streams), as written by
// tools/build_negatives.py.
func ReadNegatives(b []byte) (*Negatives, error) {
	if len(b) < 12 || string(b[:4]) != "NEG1" {
		return nil, errors.New("kws: bad negatives file")
	}
	nw := int(binary.LittleEndian.Uint32(b[4:]))
	ns := int(binary.LittleEndian.Uint32(b[8:]))
	p := 12
	if len(b) < p+4*ns {
		return nil, errors.New("kws: truncated negatives header")
	}
	lens := make([]int, ns)
	for i := range lens {
		lens[i] = int(binary.LittleEndian.Uint32(b[p:]))
		p += 4
	}
	total := nw * Ctx * EmbDim
	for _, l := range lens {
		total += l * EmbDim
	}
	if len(b) < p+2*total {
		return nil, errors.New("kws: truncated negatives data")
	}
	all := make([]float32, total)
	for i := range all {
		all[i] = f16(binary.LittleEndian.Uint16(b[p:]))
		p += 2
	}
	n := &Negatives{}
	for i := range nw {
		n.Windows = append(n.Windows, all[i*Ctx*EmbDim:(i+1)*Ctx*EmbDim])
	}
	off := nw * Ctx * EmbDim
	for _, l := range lens {
		n.Streams = append(n.Streams, Stream(all[off:off+l*EmbDim]))
		off += l * EmbDim
	}
	return n, nil
}

func f16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x3ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal
		f := float32(frac) / 1024 * float32(math.Pow(2, -14))
		if sign != 0 {
			return -f
		}
		return f
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
}

// Enrollment is what the station recorded for one word.
type Enrollment struct {
	Takes     [][]int16 // one utterance of the word per take (with silence around)
	Ambient   []int16   // room with nobody talking to the station
	Speech    []int16   // the user talking normally (hard negatives)
	False     [][]int16 // audio that wrongly triggered this word earlier
	Confusers [][]int16 // deliberately recorded similar words, without the wake word
	Synthetic [][]int16 // optional generated variants; training only, never real takes
}

// Model is a trained word, as stored on the station.
type Model struct {
	Name              string    `json:"name"`
	Threshold         float32   `json:"threshold"`
	Created           time.Time `json:"created"`
	Takes             int       `json:"takes"`
	Recall            float64   `json:"recall"`      // held-out takes detected at Threshold
	FalsePerH         float64   `json:"false_per_h"` // held-out negatives, triggers per hour
	EvalHours         float64   `json:"eval_hours"`
	Head              *Head     `json:"head"`
	TrainNotes        []string  `json:"notes,omitempty"`
	ValidationVersion int       `json:"validation_version,omitzero"`
	EvalTakes         int       `json:"eval_takes,omitzero"`
	DetectedTakes     int       `json:"detected_takes,omitzero"`
	FalseTriggers     int       `json:"false_triggers,omitzero"`
	BestEpoch         int       `json:"best_epoch,omitzero"`
	Seed              uint64    `json:"seed,omitzero"`
	// Synthetic metadata covers distinct normalized sources actually used
	// by positive training windows; real take/calibration counts exclude them.
	SyntheticTakes  int    `json:"synthetic_takes,omitzero"`
	SyntheticDigest string `json:"synthetic_digest,omitempty"`
}

// ReadModel parses and checks a stored model.
func ReadModel(b []byte) (*Model, error) {
	var m Model
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Head == nil {
		return nil, errors.New("kws: model has no head")
	}
	if err := m.Head.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

type Progress func(stage string, frac float64)

const (
	preCtxSamp = 2300 * SampleRate / 1000 // > 16 embeddings of context before the word end
	postSamp   = 480 * SampleRate / 1000
	posSlack   = 400 * SampleRate / 1000 // windows ending this long after the word count as hits
	marginSamp = 120 * SampleRate / 1000
	// Embeddings (2.9 s) between triggers: the command after the word must
	// not fire again while Alice is already listening.
	refractory = 36
)

const (
	positiveTrain = iota
	positiveCheck
	negativeTrain
	negativeCheck
	pastFalse
	syntheticTrain
)

type trainingJob struct {
	kind, weight    int
	pcm             []int16
	start, end      int
	syntheticSource int
}

// splitTakeIndices holds out whole original recordings, even for the smallest
// supported enrollment. Seeded membership avoids always testing the same place
// in an ordered recording session; returned indices retain recording order.
func splitTakeIndices(n int, seed uint64) (train, check []int) {
	if n < 2 {
		for i := range n {
			train = append(train, i)
		}
		return
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewPCG(seed, 0x74616b6573706c69))
	rng.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	held := make(map[int]bool)
	for _, i := range indices[:max(1, n/4)] {
		held[i] = true
	}
	for i := range n {
		if held[i] {
			check = append(check, i)
		} else {
			train = append(train, i)
		}
	}
	return
}

// Split raw PCM BEFORE augmentation/embedding. A discarded 800 ms gap prevents
// overlapping embedding receptive fields from crossing the split. Short clips
// remain training-only; they cannot provide a full independent detection window.
func splitBackground(pcm []int16) (train, check []int16) {
	hold := max(3*SampleRate, len(pcm)/5)
	cut := len(pcm) - hold - StepSamp*10
	if cut < preCtxSamp {
		return pcm, nil
	}
	return pcm[:cut], pcm[len(pcm)-hold:]
}

// TrainWord augments only training audio, validates on unaugmented held-out
// takes and independent background, and returns the best validation checkpoint.
func TrainWord(name string, en Enrollment, neg *Negatives, seed uint64, progress Progress) (*Model, error) {
	if progress == nil {
		progress = func(string, float64) {}
	}
	if neg == nil {
		return nil, errors.New("kws: no negative dataset")
	}
	type take struct {
		word, raw  []int16
		start, end int
	}
	var takes []take
	var originalTakeIndices []int
	var notes []string
	for i, pcm := range en.Takes {
		start, end, ok := FindSpeech(pcm)
		duration := float64(end-start) / SampleRate
		if !ok || duration < 0.2 || duration > 2 {
			notes = append(notes, fmt.Sprintf("запись %d пропущена: не найдено чёткое слово (%.2f с)", i+1, duration))
			continue
		}
		takes = append(takes, take{pcm[max(0, start-marginSamp):min(len(pcm), end+marginSamp)], pcm, start, end})
		originalTakeIndices = append(originalTakeIndices, i)
	}
	if len(takes) < 3 {
		return nil, fmt.Errorf("kws: only %d usable takes, need at least 3", len(takes))
	}
	var backgrounds, evalPCM [][]int16
	for _, pcm := range [][]int16{en.Ambient, en.Speech} {
		train, check := splitBackground(pcm)
		if len(train) >= SampleRate {
			backgrounds = append(backgrounds, train)
		}
		if len(check) > 0 {
			evalPCM = append(evalPCM, check)
		}
	}
	if len(backgrounds) == 0 {
		return nil, errors.New("kws: no room recording")
	}
	trainTakes, checkTakes := splitTakeIndices(len(takes), seed)
	realTrainWords := make([][]int16, 0, len(trainTakes))
	for _, i := range trainTakes {
		realTrainWords = append(realTrainWords, takes[i].word)
	}
	synthetic := prepareSynthetic(en.Synthetic, realTrainWords)
	var jobs []trainingJob
	for _, i := range trainTakes {
		for range max(12, 320/len(trainTakes)) {
			jobs = append(jobs, trainingJob{kind: positiveTrain, pcm: takes[i].word})
		}
	}
	for _, i := range checkTakes {
		t := takes[i]
		jobs = append(jobs, trainingJob{kind: positiveCheck, pcm: t.raw, start: t.start, end: t.end})
	}
	for _, pcm := range backgrounds {
		jobs = append(jobs, trainingJob{kind: negativeTrain, pcm: pcm, weight: 1})
	}
	for _, pcm := range evalPCM {
		jobs = append(jobs, trainingJob{kind: negativeCheck, pcm: pcm})
	}
	for _, pcm := range en.False {
		jobs = append(jobs, trainingJob{kind: pastFalse, pcm: pcm, weight: 4})
	}
	// A single deliberate confusing clip is valuable training data. With
	// several clips, hold out a whole clip instead of neighboring windows.
	confTrain, confCheck := splitTakeIndices(len(en.Confusers), seed^0x636f6e66757365)
	for _, i := range confTrain {
		jobs = append(jobs, trainingJob{kind: negativeTrain, pcm: en.Confusers[i], weight: 4})
	}
	for _, i := range confCheck {
		jobs = append(jobs, trainingJob{kind: negativeCheck, pcm: en.Confusers[i]})
	}
	jobs = appendSyntheticJobs(jobs, synthetic)
	type signal struct {
		stream     Stream
		start, end int
	}
	signals := make([]signal, len(jobs))
	var done int
	var mu sync.Mutex
	parallel(len(jobs), func(f *Features, i int) {
		j := jobs[i]
		rng := rand.New(rand.NewPCG(seed, uint64(i)+2))
		pcm := j.pcm
		start, end := j.start, j.end
		switch j.kind {
		case positiveTrain, syntheticTrain:
			pcm, start, end = Augment(rng, pcm, backgrounds, preCtxSamp, postSamp)
		case positiveCheck:
			// The held-out take is never resampled, amplified, or mixed. Add
			// independent room context (silence if unavailable) for warm-up.
			var bg []int16
			if len(evalPCM) > 0 {
				bg = evalPCM[0]
			}
			prefix := segment(rng, bg, preCtxSamp)
			pcm = append(prefix, pcm...)
			pcm = append(pcm, make([]int16, postSamp)...)
			start += preCtxSamp
			end += preCtxSamp
		case pastFalse:
			pcm = append(segment(rng, backgrounds[0], preCtxSamp), pcm...)
			start = preCtxSamp
		}
		signals[i] = signal{Embed(f, pcm), start, end}
		mu.Lock()
		done++
		progress("embed", float64(done)/float64(len(jobs)))
		mu.Unlock()
	})

	var pos, negs []Sample
	syntheticPos := make([][]Sample, len(synthetic))
	var validation []validationTake
	var evalStreams []Stream
	var regressionStreams []Stream
	for i, j := range jobs {
		s := signals[i]
		switch j.kind {
		case positiveCheck:
			validation = append(validation, validationTake{s.stream, s.start, s.end})
			continue
		case negativeCheck:
			evalStreams = append(evalStreams, s.stream)
			continue
		}
		if j.kind == pastFalse || (j.kind == negativeTrain && j.weight > 1) {
			regressionStreams = append(regressionStreams, s.stream)
		}
		for e := Ctx - 1; e < s.stream.Len(); e++ {
			end := EmbEnd(e)
			sample := Sample{X: s.stream.Window(e), Source: i + 1, Frame: e}
			if j.kind == positiveTrain || j.kind == syntheticTrain {
				switch {
				case end >= s.end && end <= s.end+posSlack:
					sample.Y = 1
					if j.kind == syntheticTrain {
						syntheticPos[j.syntheticSource] = append(syntheticPos[j.syntheticSource], sample)
					} else {
						pos = append(pos, sample)
					}
				case end <= s.start && e%2 == 0:
					if j.kind == positiveTrain {
						negs = append(negs, sample)
					}
				}
			} else if j.kind != pastFalse || end > s.start {
				for range j.weight {
					negs = append(negs, sample)
				}
			}
		}
	}
	syntheticWindows, usedSynthetic := limitSynthetic(syntheticPos, len(pos))
	pos = append(pos, syntheticWindows...)
	usedSyntheticClips := make([][]int16, 0, len(usedSynthetic))
	for _, source := range usedSynthetic {
		usedSyntheticClips = append(usedSyntheticClips, synthetic[source])
	}
	userNeg := len(negs)
	for _, window := range neg.Windows {
		negs = append(negs, Sample{X: window})
	}
	trainStreams, checkStreams := splitTakeIndices(len(neg.Streams), seed^0x73747265616d73)
	if len(neg.Streams) == 1 {
		trainStreams, checkStreams = nil, []int{0}
	}
	for _, i := range trainStreams {
		stream := neg.Streams[i]
		for e := Ctx - 1; e < stream.Len(); e++ {
			negs = append(negs, Sample{X: stream.Window(e), Source: len(jobs) + i + 1, Frame: e})
		}
	}
	for _, i := range checkStreams {
		evalStreams = append(evalStreams, neg.Streams[i])
	}
	if len(pos) == 0 || len(validation) == 0 {
		return nil, errors.New("kws: no independent positive validation")
	}
	var exposure int
	for _, stream := range evalStreams {
		exposure += max(0, stream.Len()-Ctx+1)
	}
	if exposure < Consecutive {
		return nil, errors.New("kws: no independent background validation")
	}
	weighted := append([]Sample(nil), negs...)
	for range max(1, (len(negs)-userNeg)/max(userNeg, 1)/2) {
		weighted = append(weighted, negs[:userNeg]...)
	}
	bestMetric := math.Inf(1)
	bestEpoch := 0
	head, err := Train(pos, weighted, TrainConfig{
		Seed: seed, Epochs: 40, NegPerEp: max(4*len(pos), 6000), PosWeight: 2,
		Validate: func(epoch int, head *Head) float64 {
			metric := validateHead(head, validation, evalStreams, regressionStreams).metric()
			if metric < bestMetric {
				bestMetric, bestEpoch = metric, epoch
			}
			return metric
		},
		OnProgress: func(epoch int, _ float64) { progress("train", float64(epoch)/40) },
	})
	if err != nil {
		return nil, err
	}
	progress("calibrate", 0)
	c := validateHead(head, validation, evalStreams, regressionStreams)
	if err := trainingQualityError(c, originalTakeIndices, checkTakes, seed); err != nil {
		return nil, err
	}
	if c.Recall < 0.8 {
		notes = append(notes, "Не все проверочные записи распознаны. Добавьте примеры с разных мест комнаты через «Не узнала?».")
	}
	if len(validation) < 3 {
		notes = append(notes, "Проверочных произнесений пока мало. Больше записей с разных мест комнаты сделают проверку надёжнее.")
	}
	m := &Model{
		Name: name, Created: time.Now().UTC(), Takes: len(takes), Head: head, TrainNotes: notes,
		Threshold: c.Threshold, Recall: c.Recall, FalsePerH: float64(c.FalseTriggers) / c.EvalHours,
		EvalHours: c.EvalHours, EvalTakes: len(validation), DetectedTakes: c.Detected,
		FalseTriggers: c.FalseTriggers, BestEpoch: bestEpoch, Seed: seed, ValidationVersion: 1,
		SyntheticTakes: len(usedSyntheticClips), SyntheticDigest: syntheticDigest(usedSyntheticClips),
	}
	progress("calibrate", 1)
	return m, nil
}

// parallel runs fn(i) for i < n over a few workers, each with its own
// Features.
func parallel(n int, fn func(f *Features, i int)) {
	workers := max(1, min(runtime.NumCPU()-1, 4))
	next := make(chan int)
	var wg sync.WaitGroup
	for range workers {
		f, err := NewFeatures()
		if err != nil {
			panic(err) // embedded assets are validated by tests
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(f, i)
			}
		}()
	}
	for i := range n {
		next <- i
	}
	close(next)
	wg.Wait()
}
