// Package quasar talks to maind's local IPC services:
// raw TCP on 127.0.0.1, frames of uint32-LE length + an envelope whose
// field 252 is the message body and field 253 its QuasarMessage member
// number.
package quasar

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"time"

	"google.golang.org/protobuf/encoding/protowire"
)

const (
	memberAudioData    = 158 // IOAudioChannelsData
	memberAudioRequest = 159 // IOChannelRequest
	memberRemoting     = 210 // Remoting
	maxFrame           = 1 << 20
)

// Ports reads service ports from quasar.cfg, falling back to the values
// seen on firmware 47.x.
func Ports(cfgPath string) (audio, aliced int) {
	audio, aliced = 9850, 9875
	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return
	}
	var cfg struct {
		Audio struct {
			Port int `json:"port"`
		} `json:"yio_audio"`
		Aliced struct {
			Port int `json:"port"`
		} `json:"aliced"`
	}
	if json.Unmarshal(b, &cfg) != nil {
		return
	}
	if cfg.Audio.Port > 0 {
		audio = cfg.Audio.Port
	}
	if cfg.Aliced.Port > 0 {
		aliced = cfg.Aliced.Port
	}
	return
}

func writeFrame(w io.Writer, member protowire.Number, body []byte) error {
	p := make([]byte, 4, 16+len(body))
	p = protowire.AppendTag(p, 252, protowire.BytesType)
	p = protowire.AppendBytes(p, body)
	p = protowire.AppendTag(p, 253, protowire.VarintType)
	p = protowire.AppendVarint(p, uint64(member))
	binary.LittleEndian.PutUint32(p, uint32(len(p)-4))
	_, err := w.Write(p)
	return err
}

// readFrame returns the member number and body of the next frame; buf is
// reused between calls.
func readFrame(r io.Reader, buf *[]byte) (protowire.Number, []byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return 0, nil, err
	}
	n := binary.LittleEndian.Uint32(hdr[:])
	if n > maxFrame {
		return 0, nil, fmt.Errorf("quasar: frame of %d bytes", n)
	}
	if cap(*buf) < int(n) {
		*buf = make([]byte, n)
	}
	b := (*buf)[:n]
	if _, err := io.ReadFull(r, b); err != nil {
		return 0, nil, err
	}
	var member protowire.Number
	var body []byte
	for len(b) > 0 {
		num, typ, l := protowire.ConsumeTag(b)
		if l < 0 {
			return 0, nil, protowire.ParseError(l)
		}
		b = b[l:]
		switch {
		case num == 252 && typ == protowire.BytesType:
			v, l := protowire.ConsumeBytes(b)
			if l < 0 {
				return 0, nil, protowire.ParseError(l)
			}
			body, b = v, b[l:]
		case num == 253 && typ == protowire.VarintType:
			v, l := protowire.ConsumeVarint(b)
			if l < 0 {
				return 0, nil, protowire.ParseError(l)
			}
			member, b = protowire.Number(v), b[l:]
		default:
			l := protowire.ConsumeFieldValue(num, typ, b)
			if l < 0 {
				return 0, nil, protowire.ParseError(l)
			}
			b = b[l:]
		}
	}
	return member, body, nil
}

// AudioStream subscribes to yio_audio and calls fn with every chunk of the
// named channel (16 kHz s16le mono samples; the slice is reused). Returns
// when ctx is done or the connection fails.
func AudioStream(ctx context.Context, port int, channel string, fn func(pcm []int16)) error {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()

	// IOChannelRequest.request_all (2): vqe_0 alone would be request_vqe, but
	// all channels lets the caller pick vqe_gain/raw ones too.
	req := protowire.AppendTag(nil, 2, protowire.BytesType)
	req = protowire.AppendBytes(req, nil)
	if channel == "vqe_0" {
		req = protowire.AppendTag(nil, 3, protowire.BytesType)
		req = protowire.AppendBytes(req, nil)
	}
	if err := writeFrame(conn, memberAudioRequest, req); err != nil {
		return err
	}
	var buf []byte
	var pcm []int16
	for {
		conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		member, body, err := readFrame(conn, &buf)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		if member != memberAudioData {
			continue
		}
		// IOAudioChannelsData { repeated IOAudioChannelData channels = 1 }
		for b := body; len(b) > 0; {
			num, typ, l := protowire.ConsumeTag(b)
			if l < 0 {
				return protowire.ParseError(l)
			}
			b = b[l:]
			if num != 1 || typ != protowire.BytesType {
				if l = protowire.ConsumeFieldValue(num, typ, b); l < 0 {
					return protowire.ParseError(l)
				}
				b = b[l:]
				continue
			}
			ch, l := protowire.ConsumeBytes(b)
			if l < 0 {
				return protowire.ParseError(l)
			}
			b = b[l:]
			name, data, ok := channelData(ch)
			if !ok || name != channel {
				continue
			}
			pcm = pcm[:0]
			for i := 0; i+1 < len(data); i += 2 {
				pcm = append(pcm, int16(binary.LittleEndian.Uint16(data[i:])))
			}
			fn(pcm)
		}
	}
}

// channelData extracts name (2) and data (3) of an IOAudioChannelData,
// requiring 16 kHz PCM when the fields are present.
func channelData(b []byte) (name string, data []byte, ok bool) {
	rate, format := uint64(16000), uint64(1)
	for len(b) > 0 {
		num, typ, l := protowire.ConsumeTag(b)
		if l < 0 {
			return "", nil, false
		}
		b = b[l:]
		switch {
		case num == 2 && typ == protowire.BytesType:
			v, l := protowire.ConsumeBytes(b)
			if l < 0 {
				return "", nil, false
			}
			name, b = string(v), b[l:]
		case num == 3 && typ == protowire.BytesType:
			v, l := protowire.ConsumeBytes(b)
			if l < 0 {
				return "", nil, false
			}
			data, b = v, b[l:]
		case (num == 7 || num == 8) && typ == protowire.VarintType:
			v, l := protowire.ConsumeVarint(b)
			if l < 0 {
				return "", nil, false
			}
			if num == 7 {
				rate = v
			} else {
				format = v
			}
			b = b[l:]
		default:
			l := protowire.ConsumeFieldValue(num, typ, b)
			if l < 0 {
				return "", nil, false
			}
			b = b[l:]
		}
	}
	return name, data, rate == 16000 && format == 1
}

// AliceMethod values of AliceCapabilityMethod.method.
const (
	StartConversation = 2
	StopConversation  = 3
)

// CallAlice sends one AliceCapability remoting call to aliced, the same
// call the stock `maind --cli` makes. START_CONVERSATION opens a plain voice
// request (as the app/button would): Alice listens from this moment on.
func CallAlice(port, method int) error {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 2*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(2 * time.Second))
	m := protowire.AppendTag(nil, 1, protowire.VarintType)
	m = protowire.AppendVarint(m, uint64(method))
	r := protowire.AppendTag(nil, 7, protowire.BytesType)
	r = protowire.AppendString(r, "AliceCapability")
	r = protowire.AppendTag(r, 8, protowire.BytesType)
	r = protowire.AppendBytes(r, m)
	if err := writeFrame(conn, memberRemoting, r); err != nil {
		return err
	}
	// aliced answers with state frames; waiting briefly for one keeps the
	// socket open until the call has been read. No answer is not an error.
	conn.SetReadDeadline(time.Now().Add(500 * time.Millisecond))
	var buf []byte
	readFrame(conn, &buf)
	return nil
}
