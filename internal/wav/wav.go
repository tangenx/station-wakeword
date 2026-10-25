// Package wav reads and writes 16-bit PCM mono WAV files.
package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// Encode returns a canonical 44-byte-header WAV of mono s16le samples.
func Encode(pcm []int16, rate int) []byte {
	b := make([]byte, 44+2*len(pcm))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*len(pcm)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*len(pcm)))
	for i, s := range pcm {
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(s))
	}
	return b
}

func WriteFile(path string, pcm []int16, rate int) error {
	return os.WriteFile(path, Encode(pcm, rate), 0o644)
}

// Decode parses a PCM s16 WAV; multi-channel input keeps channel 0.
func Decode(b []byte) (pcm []int16, rate int, err error) {
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, errors.New("wav: not a RIFF/WAVE file")
	}
	var channels, bits int
	p := 12
	for p+8 <= len(b) {
		id := string(b[p : p+4])
		size := int(binary.LittleEndian.Uint32(b[p+4:]))
		body := b[p+8:]
		if size > len(body) {
			size = len(body) // tolerate streams with an unfinished header
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("wav: short fmt chunk")
			}
			format := binary.LittleEndian.Uint16(body)
			channels = int(binary.LittleEndian.Uint16(body[2:]))
			rate = int(binary.LittleEndian.Uint32(body[4:]))
			bits = int(binary.LittleEndian.Uint16(body[14:]))
			if (format != 1 && format != 0xfffe) || bits != 16 || channels < 1 {
				return nil, 0, fmt.Errorf("wav: need 16-bit PCM, got format %d, %d bits", format, bits)
			}
		case "data":
			if channels == 0 {
				return nil, 0, errors.New("wav: data before fmt")
			}
			n := size / (2 * channels)
			pcm = make([]int16, n)
			for i := range pcm {
				pcm[i] = int16(binary.LittleEndian.Uint16(body[2*channels*i:]))
			}
			return pcm, rate, nil
		}
		p += 8 + size + size%2
	}
	return nil, 0, io.ErrUnexpectedEOF
}

// ReadFile reads a WAV and requires the given sample rate.
func ReadFile(path string, wantRate int) ([]int16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pcm, rate, err := Decode(b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if rate != wantRate {
		return nil, fmt.Errorf("%s: %d Hz, need %d Hz", path, rate, wantRate)
	}
	return pcm, nil
}
