package kws

import (
	_ "embed"
	"fmt"
	"sync"
)

//go:embed assets/oww.bin
var owwBin []byte

var (
	assetsOnce sync.Once
	assets     map[string]Tensor
	assetsErr  error
)

func loadAssets() (map[string]Tensor, error) {
	assetsOnce.Do(func() { assets, assetsErr = ReadTensors(owwBin) })
	return assets, assetsErr
}

// Features turns a 16 kHz mono stream into one 96-dim embedding per 80 ms
// (each covering the last 76 mel frames = 775 ms). Not safe for concurrent
// use; create one per stream.
type Features struct {
	mel   *melFront
	net   *embedNet
	audio []float32 // ctxSamp history + pending samples
	frame []float32 // scratch for one step of mel frames
	nmel  int       // mel frames seen, saturating at EmbWindow
}

func NewFeatures() (*Features, error) {
	t, err := loadAssets()
	if err != nil {
		return nil, err
	}
	filt, ok := t["mel.filters"]
	if !ok || len(filt.Data) != (fftSize/2+1)*MelBins {
		return nil, fmt.Errorf("kws: missing mel filters")
	}
	net, err := newEmbedNet(t)
	if err != nil {
		return nil, err
	}
	return &Features{
		mel:   newMelFront(filt.Data),
		net:   net,
		audio: make([]float32, ctxSamp, ctxSamp+StepSamp),
		frame: make([]float32, (StepFrames+1)*MelBins),
	}, nil
}

// Reset forgets all history (e.g. between independent clips).
func (f *Features) Reset() {
	f.audio = f.audio[:ctxSamp]
	clear(f.audio)
	f.nmel = 0
	f.net.reset()
}

// Push feeds int16-scale samples and calls emit for every new embedding.
// The embedding slice is reused; copy it to keep it.
func (f *Features) Push(samples []int16, emit func(emb []float32)) {
	for _, s := range samples {
		f.audio = append(f.audio, float32(s))
		if len(f.audio) < ctxSamp+StepSamp {
			continue
		}
		n := f.mel.frames(f.audio, f.frame)
		// Slide the history: keep the last ctxSamp samples.
		copy(f.audio, f.audio[len(f.audio)-ctxSamp:])
		f.audio = f.audio[:ctxSamp]

		embs, k := f.net.push(f.frame, n)
		f.nmel = min(f.nmel+n, EmbWindow)
		if f.nmel < EmbWindow || k == 0 {
			continue
		}
		emit(embs[(k-1)*EmbDim : k*EmbDim])
	}
}
