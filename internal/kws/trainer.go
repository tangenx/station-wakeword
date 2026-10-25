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
	Takes   [][]int16 // one utterance of the word per take (with silence around)
	Ambient []int16   // room with nobody talking to the station
	Speech  []int16   // the user talking normally (hard negatives)
	False   [][]int16 // audio that wrongly triggered this word earlier
}

// Model is a trained word, as stored on the station.
type Model struct {
	Name       string    `json:"name"`
	Threshold  float32   `json:"threshold"`
	Created    time.Time `json:"created"`
	Takes      int       `json:"takes"`
	Recall     float64   `json:"recall"`      // held-out takes detected at Threshold
	FalsePerH  float64   `json:"false_per_h"` // held-out negatives, triggers per hour
	EvalHours  float64   `json:"eval_hours"`
	Head       *Head     `json:"head"`
	TrainNotes []string  `json:"notes,omitempty"`
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

// TrainWord trims the takes, augments them with the room and the user's own
// speech, trains a head and calibrates its threshold on held-out data.
func TrainWord(name string, en Enrollment, neg *Negatives, seed uint64, progress Progress) (*Model, error) {
	if progress == nil {
		progress = func(string, float64) {}
	}
	var words [][]int16
	var notes []string
	for i, t := range en.Takes {
		s, e, ok := FindSpeech(t)
		d := float64(e-s) / SampleRate
		if !ok || d < 0.2 || d > 2.0 {
			notes = append(notes, fmt.Sprintf("запись %d пропущена: не найдено чёткое слово (%.2f с)", i+1, d))
			continue
		}
		words = append(words, t[max(0, s-marginSamp):min(len(t), e+marginSamp)])
	}
	if len(words) < 3 {
		return nil, fmt.Errorf("kws: only %d usable takes, need at least 3", len(words))
	}
	var backgrounds [][]int16
	for _, b := range [][]int16{en.Ambient, en.Speech} {
		if len(b) >= SampleRate {
			backgrounds = append(backgrounds, b)
		}
	}
	if len(backgrounds) == 0 {
		return nil, errors.New("kws: no room recording")
	}

	// Hold out every 4th take for calibration.
	var trainW, testW [][]int16
	for i, w := range words {
		if len(words) >= 4 && i%4 == 3 {
			testW = append(testW, w)
		} else {
			trainW = append(trainW, w)
		}
	}
	if len(testW) == 0 {
		testW = trainW[len(trainW)-1:]
	}

	augPerTake := max(12, 320/len(trainW))
	type job struct {
		word []int16
		test bool
	}
	var jobs []job
	for _, w := range trainW {
		for range augPerTake {
			jobs = append(jobs, job{w, false})
		}
	}
	for _, w := range testW {
		for range 12 {
			jobs = append(jobs, job{w, true})
		}
	}
	// Each signal has its own generator so results do not depend on worker
	// scheduling, and its audio is dropped once embedded (all signals at
	// once would not fit a Mini's RAM).
	type sig struct {
		end  int
		test bool
	}
	sigs := make([]sig, len(jobs))
	// Stream layout: augmented takes, ambient, speech, false alarms.
	falseAt := len(sigs) + 2
	streams := make([]Stream, falseAt+len(en.False))
	var done int
	var mu sync.Mutex
	parallel(len(streams), func(f *Features, i int) {
		r := rand.New(rand.NewPCG(seed, uint64(i)+2))
		switch {
		case i < len(sigs):
			pcm, end := Augment(r, jobs[i].word, backgrounds, preCtxSamp, postSamp)
			sigs[i] = sig{end, jobs[i].test}
			streams[i] = Embed(f, pcm)
		case i == len(sigs):
			streams[i] = Embed(f, en.Ambient)
		case i == len(sigs)+1:
			streams[i] = Embed(f, en.Speech)
		default:
			// Room audio first so windows ending inside the clip are full.
			streams[i] = Embed(f, append(segment(r, backgrounds[0], preCtxSamp), en.False[i-falseAt]...))
		}
		mu.Lock()
		done++
		progress("embed", float64(done)/float64(len(streams)))
		mu.Unlock()
	})

	var pos, negs []Sample
	var testPos [][][]float32 // per held-out signal: its hit windows
	for i, s := range sigs {
		st := streams[i]
		var hits [][]float32
		for e := Ctx - 1; e < st.Len(); e++ {
			end := EmbEnd(e)
			switch {
			case end >= s.end && end <= s.end+posSlack:
				hits = append(hits, st.Window(e))
			case end <= preCtxSamp && !s.test && e%2 == 0:
				// Ends before the word starts: background or the user's
				// speech only.
				negs = append(negs, Sample{st.Window(e), 0})
			}
		}
		if s.test {
			testPos = append(testPos, hits)
			continue
		}
		for _, h := range hits {
			pos = append(pos, Sample{h, 1})
		}
	}

	// User streams: 80% train, last 20% evaluation. Bundled streams: 75/25.
	var evalStreams []Stream
	split := func(s Stream, frac float64) {
		cut := int(float64(s.Len()) * frac)
		for _, w := range Stream(s[:cut*EmbDim]).Windows(1) {
			negs = append(negs, Sample{w, 0})
		}
		evalStreams = append(evalStreams, s[cut*EmbDim:])
	}
	split(streams[len(sigs)], 0.8)
	split(streams[len(sigs)+1], 0.8)
	for _, st := range streams[falseAt:] {
		for e := Ctx - 1; e < st.Len(); e++ {
			if EmbEnd(e) > preCtxSamp {
				for range 4 { // exact past mistakes: weigh them up
					negs = append(negs, Sample{st.Window(e), 0})
				}
			}
		}
	}
	userNeg := len(negs)
	for _, w := range neg.Windows {
		negs = append(negs, Sample{w, 0})
	}
	for _, s := range neg.Streams {
		split(s, 0.75)
	}
	if len(pos) == 0 {
		return nil, errors.New("kws: no positive windows")
	}

	cfg := TrainConfig{
		Seed:      seed,
		Epochs:    40,
		NegPerEp:  max(4*len(pos), 6000),
		PosWeight: 2,
		OnProgress: func(ep int, _ float64) {
			progress("train", float64(ep)/40)
		},
	}
	// Room and own-speech windows are the false alarms that matter most;
	// make them as likely per epoch as the whole bundled set.
	weighted := append([]Sample(nil), negs...)
	for range max(1, (len(negs)-userNeg)/max(userNeg, 1)/2) {
		weighted = append(weighted, negs[:userNeg]...)
	}
	head, err := Train(pos, weighted, cfg)
	if err != nil {
		return nil, err
	}

	progress("calibrate", 0)
	m := &Model{Name: name, Created: time.Now().UTC(), Takes: len(words), Head: head, TrainNotes: notes}
	// A held-out take counts as found if some pair of consecutive windows
	// both pass, which is what the gate needs.
	maxPos := make([]float32, len(testPos))
	for i, hs := range testPos {
		for j := 1; j < len(hs); j++ {
			maxPos[i] = max(maxPos[i], min(head.Score(hs[j-1]), head.Score(hs[j])))
		}
	}
	scores := make([][]float32, len(evalStreams))
	var embs int
	for i, s := range evalStreams {
		for e := Ctx - 1; e < s.Len(); e++ {
			scores[i] = append(scores[i], head.Score(s.Window(e)))
		}
		embs += s.Len()
	}
	m.EvalHours = float64(embs) * float64(StepSamp) / SampleRate / 3600
	triggers := func(t float32) int {
		n := 0
		for _, sc := range scores {
			var g gate
			for _, v := range sc {
				if g.step(v, t) {
					n++
				}
			}
		}
		return n
	}
	recall := func(t float32) float64 {
		k := 0
		for _, v := range maxPos {
			if v >= t {
				k++
			}
		}
		return float64(k) / float64(len(maxPos))
	}
	// Lowest threshold without false triggers on held-out data, plus margin.
	// Scores of a real word saturate near 1, so the floor costs little recall
	// and keeps a model that saw too few hard negatives from firing on noise.
	thr := float32(0.98)
	for t := float32(0.30); t <= 0.98; t += 0.02 {
		if triggers(t) == 0 {
			thr = t
			break
		}
	}
	m.Threshold = min(0.97, max(0.8, thr+0.05))
	m.Recall = recall(m.Threshold)
	m.FalsePerH = float64(triggers(m.Threshold)) / max(m.EvalHours, 1e-9)
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
