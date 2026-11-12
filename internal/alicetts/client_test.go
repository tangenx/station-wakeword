package alicetts

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/encoding/protowire"
)

func requestField(t *testing.T, data []byte, want protowire.Number) []byte {
	t.Helper()
	for len(data) > 0 {
		number, kind, n := protowire.ConsumeTag(data)
		if n < 0 || kind != protowire.BytesType {
			t.Fatal("request contains malformed protobuf")
		}
		value, m := protowire.ConsumeBytes(data[n:])
		if m < 0 {
			t.Fatal("request contains a truncated protobuf field")
		}
		if number == want {
			return value
		}
		data = data[n+m:]
	}
	t.Fatalf("request is missing protobuf field %d", want)
	return nil
}

func testClient(t *testing.T, respond func(context.Context, *websocket.Conn, string, string)) Client {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept TTS request: %v", err)
			return
		}
		defer ws.CloseNow()
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()
		kind, data, err := ws.Read(ctx)
		if err != nil || kind != websocket.MessageBinary || len(data) < 4 || string(data[:4]) != "AAPI" {
			t.Errorf("invalid request frame: kind=%v bytes=%d error=%v", kind, len(data), err)
			return
		}
		event := requestField(t, data[4:], 1)
		id := string(requestField(t, requestField(t, event, 1), 1))
		parts := strings.Split(id, "-")
		if len(parts) != 4 || len(parts[0]) != 8 || len(parts[1]) != 8 || len(parts[2]) != 8 || len(parts[3]) != 8 {
			t.Errorf("message ID is not a Yandex TGUID: %q", id)
			return
		}
		request := requestField(t, event, 13)
		if string(requestField(t, request, 1)) != "ru-RU" || string(requestField(t, request, 3)) != "pcm" {
			t.Error("request must select Russian raw PCM")
			return
		}
		respond(ctx, ws, id, string(requestField(t, request, 7)))
	}))
	t.Cleanup(server.Close)
	return Client{endpoint: strings.Replace(server.URL, "https:", "wss:", 1), httpClient: server.Client()}
}

func speak(ctx context.Context, ws *websocket.Conn, id, format string, stream int) {
	_ = ws.Write(ctx, websocket.MessageText, []byte(fmt.Sprintf(`{"directive":{"payload":{"format":%q},"header":{"name":"Speak","namespace":"TTS","refMessageId":%q,"streamId":%d}}}`, format, id, stream)))
}

func audio(ctx context.Context, ws *websocket.Conn, stream uint32, data []byte) {
	_ = ws.Write(ctx, websocket.MessageBinary, append(binary.BigEndian.AppendUint32(nil, stream), data...))
}

func closeStream(ctx context.Context, ws *websocket.Conn, stream, reason int) {
	// The close messageId is intentionally different from the request ID.
	_ = ws.Write(ctx, websocket.MessageText, []byte(fmt.Sprintf(`{"streamcontrol":{"messageId":"server-owned-id","streamId":%d,"action":0,"reason":%d}}`, stream, reason)))
}

func TestSynthesizeCorrelatesPCMAndDecodesAcrossChunks(t *testing.T) {
	text := `<speaker voice="zahar">Джарвис`
	c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
		if requestText != text {
			t.Errorf("request text changed: %q", requestText)
		}
		// An unrelated Opus response must neither be accepted nor reject ours.
		speak(ctx, ws, "other-request", "audio/opus", 17)
		audio(ctx, ws, 17, []byte("unrelated audio"))
		speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000;channels=1", 2)
		audio(ctx, ws, 2, []byte{1})
		audio(ctx, ws, 2, []byte{0, 255, 127, 0, 128, 255, 255})
		closeStream(ctx, ws, 17, 0)
		closeStream(ctx, ws, 2, 0)
	})
	pcm, err := c.Synthesize(t.Context(), text)
	if err != nil || !slices.Equal(pcm, []int16{1, 32767, -32768, -1}) {
		t.Fatalf("synthesis returned %v, %v", pcm, err)
	}
}

func TestSynthesizeRejectsWrongFormatOrPCMMetadata(t *testing.T) {
	for _, format := range []string{"audio/opus", "pcm", "audio/x-pcm;bit=16;rate=48000", "audio/x-pcm;bit=32;rate=16000", "audio/x-pcm;bit=16;rate=16000;channels=2"} {
		t.Run(format, func(t *testing.T) {
			c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
				speak(ctx, ws, id, format, 2)
				audio(ctx, ws, 2, []byte{1, 0})
				closeStream(ctx, ws, 2, 0)
			})
			if _, err := c.Synthesize(t.Context(), "Джарвис"); err == nil {
				t.Fatalf("accepted unsupported response format %q", format)
			}
		})
	}
}

func TestSynthesizeRequiresCleanCompletionAndWholeSamples(t *testing.T) {
	for _, tc := range []struct {
		name   string
		bytes  []byte
		reason int
		closed bool
	}{
		{name: "odd PCM", bytes: []byte{1, 0, 2}, closed: true},
		{name: "empty PCM", closed: true},
		{name: "error close", bytes: []byte{1, 0}, reason: 1, closed: true},
		{name: "truncated stream", bytes: []byte{1, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
				speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000", 2)
				audio(ctx, ws, 2, tc.bytes)
				if tc.closed {
					closeStream(ctx, ws, 2, tc.reason)
				}
			})
			if pcm, err := c.Synthesize(t.Context(), "Джарвис"); err == nil || pcm != nil {
				t.Fatalf("accepted %s: %v, %v", tc.name, pcm, err)
			}
		})
	}
}

func TestSynthesizeBoundsTotalPCM(t *testing.T) {
	c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
		speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000", 2)
		for range 4 {
			audio(ctx, ws, 2, make([]byte, 100000))
		}
		closeStream(ctx, ws, 2, 0)
	})
	if _, err := c.Synthesize(t.Context(), "Джарвис"); err == nil || !strings.Contains(err.Error(), "long") {
		t.Fatalf("unbounded PCM accepted: %v", err)
	}
}

func TestSynthesizeAllowsOneBoundedPCMFrame(t *testing.T) {
	c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
		speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000", 2)
		audio(ctx, ws, 2, make([]byte, 200000))
		closeStream(ctx, ws, 2, 0)
	})
	pcm, err := c.Synthesize(t.Context(), "Джарвис")
	if err != nil || len(pcm) != 100000 {
		t.Fatalf("valid bounded audio frame rejected: samples=%d, error=%v", len(pcm), err)
	}
}

func TestSynthesizeRejectsMalformedControls(t *testing.T) {
	for _, data := range []string{"{", "[]", `{"streamcontrol":{"streamId":2,"action":0}}`} {
		t.Run(data, func(t *testing.T) {
			c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
				speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000", 2)
				audio(ctx, ws, 2, []byte{1, 0})
				_ = ws.Write(ctx, websocket.MessageText, []byte(data))
			})
			if _, err := c.Synthesize(t.Context(), "Джарвис"); err == nil {
				t.Fatal("accepted malformed stream control")
			}
		})
	}
}

func TestSynthesizeHonorsCancellation(t *testing.T) {
	ready := make(chan struct{})
	c := testClient(t, func(ctx context.Context, ws *websocket.Conn, id, requestText string) {
		speak(ctx, ws, id, "audio/x-pcm;bit=16;rate=16000", 2)
		close(ready)
		_, _, _ = ws.Read(ctx)
	})
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { _, err := c.Synthesize(ctx, "Джарвис"); done <- err }()
	select {
	case <-ready:
		cancel()
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("synthesis request did not start")
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation was lost: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled synthesis did not finish")
	}
}

func TestVariantsKeepOriginalVoiceAndRejectMarkup(t *testing.T) {
	variants, err := Variants(" Джарвис ")
	if err != nil || len(variants) != 8 || variants[0].Text != "Джарвис" {
		t.Fatalf("presets: %+v, %v", variants, err)
	}
	ids := make(map[string]bool)
	for _, v := range variants {
		if ids[v.ID] || v.ID == "" || strings.Count(v.Text, "Джарвис") != 1 {
			t.Fatalf("invalid or repeated preset: %+v", v)
		}
		ids[v.ID] = true
	}
	for _, word := range []string{"", "я", strings.Repeat("я", 41), "два слова", `<speaker voice="zahar">слово`, "слово<[...]>ол", "https://example.org", "а\x00б", "-слово", "слово-", "а--б"} {
		if got, err := Variants(word); err == nil || got != nil {
			t.Fatalf("accepted invalid word %q: %+v, %v", word, got, err)
		}
	}
}
