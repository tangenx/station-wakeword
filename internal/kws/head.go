package kws

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sort"
	"sync"
)

// Head is the per-word classifier over the last Ctx embeddings (1.28 s).
// A projection shared across time (96 -> H1) keeps it small enough to train
// on the station from a few dozen takes without memorising them. Both hidden
// widths are 24 so the NEON block kernel runs every layer.
const (
	Ctx = 16
	H1  = 24
	H2  = 24
	in2 = Ctx * H1
)

type Head struct {
	W1 []float32 `json:"w1"` // [EmbDim][H1]
	B1 []float32 `json:"b1"`
	W2 []float32 `json:"w2"` // [in2][H2]
	B2 []float32 `json:"b2"`
	W3 []float32 `json:"w3"` // [H2]
	B3 []float32 `json:"b3"` // [1]
}

func newHead(rng *rand.Rand) *Head {
	h := &Head{
		W1: make([]float32, EmbDim*H1), B1: make([]float32, H1),
		W2: make([]float32, in2*H2), B2: make([]float32, H2),
		W3: make([]float32, H2), B3: make([]float32, 1),
	}
	he := func(w []float32, fanIn int) {
		s := math.Sqrt(2 / float64(fanIn))
		for i := range w {
			w[i] = float32(rng.NormFloat64() * s)
		}
	}
	he(h.W1, EmbDim)
	he(h.W2, in2)
	he(h.W3, H2)
	return h
}

func (h *Head) params() [][]float32 {
	return [][]float32{h.W1, h.B1, h.W2, h.B2, h.W3, h.B3}
}

// Validate checks tensor sizes (a corrupt model must not crash the daemon).
func (h *Head) Validate() error {
	want := []int{EmbDim * H1, H1, in2 * H2, H2, H2, 1}
	for i, p := range h.params() {
		if len(p) != want[i] {
			return fmt.Errorf("kws: head tensor %d has %d values, want %d", i, len(p), want[i])
		}
	}
	return nil
}

// act holds one sample's activations (reused across calls).
type act struct {
	a1, r1 [in2]float32
	a2, r2 [H2]float32
	xt     [EmbDim * Ctx]float32 // x transposed, for the W1 gradient
}

func sigmoid(z float32) float32 { return float32(1 / (1 + math.Exp(-float64(z)))) }

// forward: H1 and H2 are 24 wide so both layers run on macBlock24.
func (h *Head) forward(x []float32, a *act) float32 {
	for t := range Ctx {
		a1 := a.a1[t*H1 : (t+1)*H1]
		copy(a1, h.B1)
		macBlock24(&a1[0], &x[t*EmbDim], &h.W1[0], EmbDim, H1)
	}
	for j, v := range a.a1 {
		a.r1[j] = max(v, 0)
	}
	copy(a.a2[:], h.B2)
	macBlock24(&a.a2[0], &a.r1[0], &h.W2[0], in2, H2)
	z := h.B3[0]
	for k, v := range a.a2 {
		a.r2[k] = max(v, 0)
		z += a.r2[k] * h.W3[k]
	}
	return sigmoid(z)
}

// Score returns the wake-word probability for a window of Ctx embeddings
// (row-major, oldest first).
func (h *Head) Score(x []float32) float32 {
	var a act
	return h.forward(x, &a)
}

// backward accumulates the gradient of weight*BCE(p, y) into g.
func (h *Head) backward(x []float32, a *act, p, y, weight float32, g *Head) {
	dz := weight * (p - y)
	g.B3[0] += dz
	var da2 [H2]float32
	for k := range da2 {
		g.W3[k] += dz * a.r2[k]
		if a.a2[k] > 0 {
			da2[k] = dz * h.W3[k]
		}
	}
	for k := range da2 {
		g.B2[k] += da2[k]
	}
	var da1 [in2]float32
	for j, v := range a.r1 {
		if v == 0 {
			continue // ReLU closed: no gradient to W2 row j or below
		}
		w := h.W2[j*H2 : (j+1)*H2]
		gw := g.W2[j*H2 : (j+1)*H2]
		var s float32
		for k, d := range da2 {
			gw[k] += v * d
			s += w[k] * d
		}
		da1[j] = s
	}
	for t := range Ctx {
		d := da1[t*H1 : (t+1)*H1]
		for j := range d {
			g.B1[j] += d[j]
		}
		for i, v := range x[t*EmbDim : (t+1)*EmbDim] {
			a.xt[i*Ctx+t] = v
		}
	}
	// gW1[i][:] += sum_t x[t][i] * da1[t][:]
	for i := range EmbDim {
		macBlock24(&g.W1[i*H1], &a.xt[i*Ctx], &da1[0], Ctx, H1)
	}
}

// Sample is one training window: Ctx*EmbDim floats and its label.
type Sample struct {
	X []float32
	Y float32
}

type TrainConfig struct {
	Epochs     int
	Batch      int
	LR         float64
	NegPerEp   int     // negatives drawn per epoch (all positives every epoch)
	PosWeight  float32 // loss weight of positives relative to negatives
	Seed       uint64
	OnProgress func(epoch int, loss float64)
}

func (c *TrainConfig) defaults() {
	if c.Epochs == 0 {
		c.Epochs = 40
	}
	if c.Batch == 0 {
		c.Batch = 128
	}
	if c.LR == 0 {
		c.LR = 1e-3
	}
	if c.NegPerEp == 0 {
		c.NegPerEp = 6000
	}
	if c.PosWeight == 0 {
		c.PosWeight = 1
	}
}

// Train fits a new head with Adam on positives and negatives.
func Train(pos, neg []Sample, cfg TrainConfig) (*Head, error) {
	cfg.defaults()
	if len(pos) == 0 || len(neg) == 0 {
		return nil, fmt.Errorf("kws: need positives and negatives (have %d/%d)", len(pos), len(neg))
	}
	rng := rand.New(rand.NewPCG(cfg.Seed, 0x9e3779b97f4a7c15))
	h := newHead(rng)
	params := h.params()
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		m[i] = make([]float32, len(p))
		v[i] = make([]float32, len(p))
	}
	workers := max(1, min(runtime.NumCPU()-1, 4))
	grads := make([]*Head, workers)
	for i := range grads {
		grads[i] = newHead(rng)
	}
	gparams := make([][][]float32, workers)
	for i, g := range grads {
		gparams[i] = g.params()
	}
	const b1, b2, eps = 0.9, 0.999, 1e-8
	step := 0
	var hard []Sample
	for ep := range cfg.Epochs {
		// Hard-negative mining: every few epochs the negatives the head likes
		// most are added to each epoch, so it spends its capacity on
		// near-misses ("Дарвин" for "Джарвис") rather than on silence.
		if ep >= 4 && ep%4 == 0 {
			hard = mineHard(h, neg, max(len(pos), 1000), workers)
		}
		epoch := append([]Sample(nil), pos...)
		epoch = append(epoch, hard...)
		for range min(cfg.NegPerEp, len(neg)) {
			epoch = append(epoch, neg[rng.IntN(len(neg))])
		}
		rng.Shuffle(len(epoch), func(i, j int) { epoch[i], epoch[j] = epoch[j], epoch[i] })
		var lossSum float64
		var lossMu sync.Mutex
		for start := 0; start < len(epoch); start += cfg.Batch {
			batch := epoch[start:min(start+cfg.Batch, len(epoch))]
			var wg sync.WaitGroup
			for w := range workers {
				g := grads[w]
				for _, p := range gparams[w] {
					clear(p)
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					var a act
					var l float64
					for i := w; i < len(batch); i += workers {
						s := batch[i]
						p := h.forward(s.X, &a)
						wt := float32(1)
						if s.Y > 0.5 {
							wt = cfg.PosWeight
						}
						pc := math.Min(math.Max(float64(p), 1e-7), 1-1e-7)
						l -= float64(wt) * (float64(s.Y)*math.Log(pc) + float64(1-s.Y)*math.Log(1-pc))
						h.backward(s.X, &a, p, s.Y, wt, g)
					}
					lossMu.Lock()
					lossSum += l
					lossMu.Unlock()
				}()
			}
			wg.Wait()
			step++
			lr := cfg.LR * math.Sqrt(1-math.Pow(b2, float64(step))) / (1 - math.Pow(b1, float64(step)))
			inv := 1 / float32(len(batch))
			for pi, p := range params {
				for j := range p {
					var gsum float32
					for _, gp := range gparams {
						gsum += gp[pi][j]
					}
					gsum = gsum*inv + 1e-4*p[j] // light L2
					m[pi][j] = b1*m[pi][j] + (1-b1)*gsum
					v[pi][j] = b2*v[pi][j] + (1-b2)*gsum*gsum
					p[j] -= float32(lr * float64(m[pi][j]) / (math.Sqrt(float64(v[pi][j])) + eps))
				}
			}
		}
		if cfg.OnProgress != nil {
			cfg.OnProgress(ep+1, lossSum/float64(len(epoch)))
		}
	}
	return h, nil
}

// mineHard returns the k negatives with the highest current scores.
func mineHard(h *Head, neg []Sample, k, workers int) []Sample {
	scores := make([]float32, len(neg))
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var a act
			for i := w; i < len(neg); i += workers {
				scores[i] = h.forward(neg[i].X, &a)
			}
		}()
	}
	wg.Wait()
	idx := make([]int, len(neg))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
	out := make([]Sample, 0, k)
	for _, i := range idx[:min(k, len(idx))] {
		out = append(out, neg[i])
	}
	return out
}
