package kws

import (
	"fmt"
)

// Google speech_embedding (as shipped by openWakeWord): 76x32 log-mel window
// -> 96-dim embedding. 20 convolutions; time axis unpadded, frequency axis
// "same"-padded; LeakyReLU(0.2) clamped at -0.4 after every conv but the last;
// max-pool after convs 2, 6, 10, 14, 18: 2x2 after 2/10/18, frequency-only
// 1x2 after 6/14. Total time stride is 8 frames, equal to the hop.
const (
	EmbWindow = 76 // mel frames per embedding
	EmbDim    = 96
	EmbHop    = StepFrames
)

type conv struct {
	kh, kw, cin, cout int
	w                 []float32 // [kh][kw][cin][cout]
	b                 []float32
}

// stage is one op of the streaming net. Every op keeps just enough input rows
// (time steps) to produce outputs for rows that arrive later, so each 8-frame
// step costs about 1/8 of a full 76-frame window.
type stage struct {
	conv     *conv // nil: max-pool
	act      bool
	poolTime bool
	w, cin   int // input width (mel axis) and channels
	keep     int // rows carried over between pushes
	buf      []float32
	nbuf     int // rows currently in buf
	out      []float32
}

type embedNet struct {
	stages []stage
}

// pad frames prepended to every stream so 76-frame windows ending at the
// newest frame start on a multiple of the network's time stride (8).
const streamPad = EmbHop - EmbWindow%EmbHop

const maxRowsIn = EmbHop + 2 // new rows per push + conv history

func newEmbedNet(t map[string]Tensor) (*embedNet, error) {
	var convs []conv
	for i := 0; ; i++ {
		wt, ok := t[fmt.Sprintf("emb.conv%d.w", i)]
		if !ok {
			break
		}
		bt := t[fmt.Sprintf("emb.conv%d.b", i)]
		if len(wt.Shape) != 4 {
			return nil, fmt.Errorf("kws: conv%d: bad shape %v", i, wt.Shape)
		}
		co, ci, kh, kw := wt.Shape[0], wt.Shape[1], wt.Shape[2], wt.Shape[3]
		w := make([]float32, len(wt.Data))
		for o := range co {
			for c := range ci {
				for y := range kh {
					for x := range kw {
						w[((y*kw+x)*ci+c)*co+o] = wt.Data[((o*ci+c)*kh+y)*kw+x]
					}
				}
			}
		}
		convs = append(convs, conv{kh, kw, ci, co, w, bt.Data})
	}
	if len(convs) != 20 {
		return nil, fmt.Errorf("kws: expected 20 convs, got %d", len(convs))
	}
	n := &embedNet{}
	w, c := MelBins, 1
	add := func(s stage) {
		s.w, s.cin = w, c
		s.buf = make([]float32, maxRowsIn*w*c)
		outW, outC := w, c
		switch {
		case s.conv != nil:
			s.keep = s.conv.kh - 1
			outC = s.conv.cout
		default:
			outW = w / 2
		}
		s.out = make([]float32, maxRowsIn*outW*outC)
		n.stages = append(n.stages, s)
		w, c = outW, outC
	}
	for i := range convs {
		add(stage{conv: &convs[i], act: i < len(convs)-1})
		switch i {
		case 2, 10, 18:
			add(stage{poolTime: true})
		case 6, 14:
			add(stage{})
		}
	}
	if w != 1 || c != EmbDim {
		return nil, fmt.Errorf("kws: net output %dx%d, want 1x%d", w, c, EmbDim)
	}
	n.reset()
	return n, nil
}

func (n *embedNet) reset() {
	for i := range n.stages {
		n.stages[i].nbuf = 0
	}
	var pad [streamPad * MelBins]float32
	n.push(pad[:], streamPad)
}

// push feeds rows mel frames (row-major, MelBins wide) and returns the
// embeddings completed by them (EmbDim floats each).
func (n *embedNet) push(in []float32, rows int) ([]float32, int) {
	for i := range n.stages {
		s := &n.stages[i]
		row := s.w * s.cin
		copy(s.buf[s.nbuf*row:], in[:rows*row])
		total := s.nbuf + rows
		switch {
		case s.conv != nil:
			rows = max(total-s.conv.kh+1, 0)
			if rows > 0 {
				s.conv.apply(s.buf, total, s.w, s.out, s.act)
			}
		case s.poolTime:
			rows = total / 2
			maxPool(s.buf, 2*rows, s.w, s.cin, true, s.out)
			s.keep = total % 2
		default:
			rows = total
			maxPool(s.buf, total, s.w, s.cin, false, s.out)
		}
		keep := min(s.keep, total)
		copy(s.buf, s.buf[(total-keep)*row:total*row])
		s.nbuf = keep
		in = s.out
		if rows == 0 {
			return nil, 0
		}
	}
	return in[:rows*EmbDim], rows
}

// apply runs one conv over in (h x w x cin) into out; returns output height.
// Loop order keeps one (tap, 24-channel block) weight slice, at most
// 96x24 floats, hot in L1 while it sweeps every output pixel (the deep
// layers' weights overflow the A53's 32 KB L1), and pixels go four at a
// time so each weight load feeds four FMLAs (the A53 is load-bound).
func (c *conv) apply(in []float32, h, w int, out []float32, act bool) int {
	oh := h - c.kh + 1
	pad := (c.kw - 1) / 2
	co, ci := c.cout, c.cin
	out = out[:oh*w*co]
	for p := 0; p < len(out); p += co {
		copy(out[p:p+co], c.b)
	}
	var accs, srcs [4]*float32
	for ky := range c.kh {
		for kx := range c.kw {
			wt := c.w[(ky*c.kw+kx)*ci*co : (ky*c.kw+kx+1)*ci*co]
			x0, x1 := max(pad-kx, 0), min(w, w+pad-kx)
			for b := 0; b < co; b += 24 {
				k := 0
				for y := range oh {
					row := in[(y+ky)*w*ci : (y+ky+1)*w*ci]
					for x := x0; x < x1; x++ {
						accs[k] = &out[(y*w+x)*co+b]
						srcs[k] = &row[(x+kx-pad)*ci]
						if k++; k == 4 {
							macBlock24x4(&accs, &srcs, &wt[b], ci, co)
							k = 0
						}
					}
				}
				for p := range k {
					macBlock24(accs[p], srcs[p], &wt[b], ci, co)
				}
			}
		}
	}
	if act {
		for o, v := range out {
			if v < 0 {
				out[o] = max(v*0.2, -0.4)
			}
		}
	}
	return oh
}

func maxPool(in []float32, h, w, c int, poolTime bool, out []float32) (int, int) {
	sy := 1
	if poolTime {
		sy = 2
	}
	oh, ow := h/sy, w/2
	for y := range oh {
		for x := range ow {
			o := out[(y*ow+x)*c : (y*ow+x+1)*c]
			p00 := in[((sy*y)*w+2*x)*c:]
			p01 := in[((sy*y)*w+2*x+1)*c:]
			if !poolTime {
				for i := range o {
					o[i] = max(p00[i], p01[i])
				}
				continue
			}
			p10 := in[((sy*y+1)*w+2*x)*c:]
			p11 := in[((sy*y+1)*w+2*x+1)*c:]
			for i := range o {
				o[i] = max(p00[i], p01[i], p10[i], p11[i])
			}
		}
	}
	return oh, ow
}
