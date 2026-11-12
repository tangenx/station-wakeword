package kws

import (
	"math/rand/v2"
	"testing"
)

func TestDetectorScoresFirstFullWindow(t *testing.T) {
	head := newHead(rand.New(rand.NewPCG(1, 2)))
	for _, p := range head.params() {
		clear(p)
	}
	head.B3[0] = 3
	d, err := NewDetector([]*Model{{Head: head, Threshold: 0.8}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	hits := 0
	for range 26 { // first embedding at .8s, full window at 2s, pair at 2.08s
		d.Push(make([]int16, StepSamp), func(int, float32) { hits++ })
	}
	if hits != 1 {
		t.Fatalf("got %d hits by 2.08s, want one from the first full window pair", hits)
	}
}
