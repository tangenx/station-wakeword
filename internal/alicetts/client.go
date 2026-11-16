// Package alicetts generates short training recordings without sending audio
// to the station's player. The zero-value Client talks directly to UniProxy.
package alicetts

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"google.golang.org/protobuf/encoding/protowire"
)

const (
	endpointURL     = "wss://uniproxy.alice.yandex.net/uni.ws"
	maxPCMBytes     = 16000 * 2 * 10  // ten seconds of signed 16-bit mono PCM
	maxMessageBytes = maxPCMBytes + 4 // one complete bounded audio frame
)

// Client needs no station credentials or quasar-bridge installation. Endpoint
// and HTTP transport are deliberately not public configuration options.
type Client struct {
	endpoint   string
	httpClient *http.Client
}

// Synthesize returns 16 kHz mono samples only after a matching, successful
// stream close. The caller controls the network timeout through ctx.
func (c Client) Synthesize(ctx context.Context, text string) ([]int16, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(text) == "" || len(text) > 1024 || !utf8.ValidString(text) {
		return nil, errors.New("alice TTS: text must be nonempty UTF-8 and at most 1024 bytes")
	}
	endpoint := c.endpoint
	if endpoint == "" {
		endpoint = endpointURL
	}
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "wss" || u.Host == "" || u.User != nil {
		return nil, errors.New("alice TTS: secure WebSocket endpoint required")
	}
	ws, _, err := websocket.Dial(ctx, endpoint, &websocket.DialOptions{HTTPClient: c.httpClient})
	if err != nil {
		return nil, fmt.Errorf("alice TTS connection: %w", err)
	}
	defer ws.CloseNow()
	ws.SetReadLimit(maxMessageBytes)
	id := strings.ReplaceAll(uuid.NewString(), "-", "")
	id = id[:8] + "-" + id[8:16] + "-" + id[16:24] + "-" + id[24:]
	if err := ws.Write(ctx, websocket.MessageBinary, synthesisRequest(id, text)); err != nil {
		return nil, fmt.Errorf("alice TTS request: %w", err)
	}
	var stream uint32
	var pcm []byte
	for {
		kind, data, err := ws.Read(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("alice TTS stream ended before completion: %w", err)
		}
		if kind == websocket.MessageBinary {
			if bytes.HasPrefix(data, []byte("AAPI")) {
				return nil, errors.New("alice TTS: unsupported binary control response")
			}
			if len(data) < 4 {
				return nil, errors.New("alice TTS: malformed audio frame")
			}
			if stream == 0 || binary.BigEndian.Uint32(data[:4]) != stream {
				continue
			}
			if len(pcm)+len(data)-4 > maxPCMBytes {
				return nil, errors.New("alice TTS: audio is too long")
			}
			pcm = append(pcm, data[4:]...)
			continue
		}
		var message controlMessage
		if err := json.Unmarshal(data, &message); err != nil {
			return nil, errors.New("alice TTS: malformed control response")
		}
		if directive := message.Directive; directive != nil {
			h := directive.Header
			if strings.Contains(strings.ToLower(h.Name), "error") && (h.RefMessageID == "" || h.RefMessageID == id) {
				return nil, errors.New("alice TTS: server rejected the synthesis request")
			}
			if h.RefMessageID == id && h.Name == "Speak" && h.Namespace == "TTS" {
				if h.StreamID == 0 || (stream != 0 && h.StreamID != stream) {
					return nil, errors.New("alice TTS: invalid or changing audio stream")
				}
				if err := validatePCMFormat(directive.Payload.Format); err != nil {
					return nil, err
				}
				stream = h.StreamID
			}
		}
		if close := message.StreamControl; close != nil && stream != 0 && close.StreamID == stream {
			if close.Action == nil {
				return nil, errors.New("alice TTS: missing stream control action")
			}
			if *close.Action != 0 {
				continue
			}
			if close.Reason == nil || *close.Reason != 0 {
				return nil, errors.New("alice TTS: synthesis did not complete successfully")
			}
			if len(pcm) == 0 || len(pcm)%2 != 0 {
				return nil, errors.New("alice TTS: empty or truncated PCM audio")
			}
			out := make([]int16, len(pcm)/2)
			for i := range out {
				out[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
			}
			return out, nil
		}
	}
}

type controlMessage struct {
	Directive *struct {
		Header struct {
			Name         string `json:"name"`
			Namespace    string `json:"namespace"`
			RefMessageID string `json:"refMessageId"`
			StreamID     uint32 `json:"streamId"`
		} `json:"header"`
		Payload struct {
			Format string `json:"format"`
		} `json:"payload"`
	} `json:"directive"`
	StreamControl *struct {
		StreamID uint32 `json:"streamId"`
		Action   *int   `json:"action"`
		Reason   *int   `json:"reason"`
	} `json:"streamcontrol"`
}

func validatePCMFormat(format string) error {
	media, params, err := mime.ParseMediaType(format)
	if err != nil || media != "audio/x-pcm" || params["bit"] != "16" || params["rate"] != "16000" {
		return errors.New("alice TTS: response must declare 16 kHz 16-bit PCM")
	}
	if channels, present := params["channels"]; present && channels != "1" {
		return errors.New("alice TTS: response must be mono PCM")
	}
	return nil
}

func synthesisRequest(id, text string) []byte {
	field := func(data []byte, number protowire.Number, value string) []byte {
		data = protowire.AppendTag(data, number, protowire.BytesType)
		return protowire.AppendString(data, value)
	}
	// The small Alice API subset avoids importing all generated AliceRequest
	// descriptors. Field numbers follow the versioned quasar-go-genproto schema.
	tts := field(nil, 1, "ru-RU")
	tts = field(tts, 3, "pcm")
	tts = field(tts, 7, text)
	header := field(nil, 1, id)
	event := protowire.AppendTag(nil, 1, protowire.BytesType)
	event = protowire.AppendBytes(event, header)
	event = protowire.AppendTag(event, 13, protowire.BytesType)
	event = protowire.AppendBytes(event, tts)
	message := protowire.AppendTag([]byte("AAPI"), 1, protowire.BytesType)
	return protowire.AppendBytes(message, event)
}
