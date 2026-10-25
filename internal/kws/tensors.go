package kws

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Tensor is a dense little-endian float32 array as written by
// tools/export_oww.py.
type Tensor struct {
	Shape []int
	Data  []float32
}

// ReadTensors parses the "OWW1" flat tensor container.
func ReadTensors(b []byte) (map[string]Tensor, error) {
	if len(b) < 8 || string(b[:4]) != "OWW1" {
		return nil, errors.New("kws: bad tensor file magic")
	}
	n := int(binary.LittleEndian.Uint32(b[4:]))
	p := 8
	out := make(map[string]Tensor, n)
	need := func(k int) error {
		if p+k > len(b) {
			return errors.New("kws: truncated tensor file")
		}
		return nil
	}
	for range n {
		if err := need(2); err != nil {
			return nil, err
		}
		nl := int(binary.LittleEndian.Uint16(b[p:]))
		p += 2
		if err := need(nl + 1); err != nil {
			return nil, err
		}
		name := string(b[p : p+nl])
		p += nl
		nd := int(b[p])
		p++
		if err := need(4 * nd); err != nil {
			return nil, err
		}
		shape := make([]int, nd)
		size := 1
		for d := range shape {
			shape[d] = int(binary.LittleEndian.Uint32(b[p:]))
			size *= shape[d]
			p += 4
		}
		if err := need(4 * size); err != nil {
			return nil, fmt.Errorf("kws: tensor %s: %w", name, err)
		}
		data := make([]float32, size)
		for j := range data {
			data[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[p:]))
			p += 4
		}
		out[name] = Tensor{Shape: shape, Data: data}
	}
	return out, nil
}
