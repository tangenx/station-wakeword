package kws

// Consecutive windows that must pass the threshold for a detection. A word
// passes for several 80 ms steps in a row while one-off spikes on other
// speech rarely do; it costs 80 ms of latency.
const Consecutive = 2

// gate turns per-window scores into detections: Consecutive passes in a
// row, then a refractory period.
type gate struct{ run, quiet int }

func (g *gate) step(score, thr float32) bool {
	if g.quiet > 0 {
		g.quiet--
		g.run = 0
		return false
	}
	if score < thr {
		g.run = 0
		return false
	}
	if g.run++; g.run < Consecutive {
		return false
	}
	g.run, g.quiet = 0, refractory
	return true
}

// Detector scores every new embedding against a set of word models.
type Detector struct {
	feat   *Features
	models []*Model
	sens   []float32 // per-model threshold offset (negative = more sensitive)
	ring   []float32 // last Ctx embeddings, oldest first
	have   int
	gates  []gate
	// Scores holds the latest score of each model (for the UI meter).
	Scores []float32
}

func NewDetector(models []*Model, sensitivity []float32) (*Detector, error) {
	f, err := NewFeatures()
	if err != nil {
		return nil, err
	}
	sens := make([]float32, len(models))
	copy(sens, sensitivity)
	return &Detector{
		feat:   f,
		models: models,
		sens:   sens,
		ring:   make([]float32, Ctx*EmbDim),
		gates:  make([]gate, len(models)),
		Scores: make([]float32, len(models)),
	}, nil
}

// Push feeds audio; hit is called with the model index for each detection.
func (d *Detector) Push(pcm []int16, hit func(i int, score float32)) {
	d.feat.Push(pcm, func(e []float32) {
		copy(d.ring, d.ring[EmbDim:])
		copy(d.ring[(Ctx-1)*EmbDim:], e)
		if d.have < Ctx {
			d.have++
			return
		}
		for i, m := range d.models {
			s := m.Head.Score(d.ring)
			d.Scores[i] = s
			if d.gates[i].step(s, min(0.99, max(0.05, m.Threshold+d.sens[i]))) {
				hit(i, s)
			}
		}
	})
}
