package daemon

import (
	"encoding/binary"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
)

// subscribers fans a value out to every listener; a listener that cannot keep
// up loses values instead of stalling the audio path.
type subscribers[T any] struct {
	mu    sync.Mutex
	subs  map[chan T]struct{}
	drops int
}

func (s *subscribers[T]) add(buf int) chan T {
	ch := make(chan T, buf)
	s.mu.Lock()
	if s.subs == nil {
		s.subs = map[chan T]struct{}{}
	}
	s.subs[ch] = struct{}{}
	s.mu.Unlock()
	return ch
}

func (s *subscribers[T]) remove(ch chan T) {
	s.mu.Lock()
	delete(s.subs, ch)
	close(ch)
	s.mu.Unlock()
}

func (s *subscribers[T]) publish(v T) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for ch := range s.subs {
		select {
		case ch <- v:
		default:
			if s.drops++; s.drops%100 == 1 {
				log.Printf("stream: slow consumer dropped %d chunk(s) total", s.drops)
			}
		}
	}
}

func (s *subscribers[T]) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.subs)
}

// serveEvents streams hits as Server-Sent Events: one JSON object per event,
// a comment every 15 s keeps idle connections alive.
func (d *Daemon) serveEvents(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	fl.Flush()

	ch := d.events.add(32)
	defer d.events.remove(ch)
	tick := time.NewTicker(15 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-tick.C:
			if _, err := w.Write([]byte(": ping\n\n")); err != nil {
				return
			}
			fl.Flush()
		case ev := <-ch:
			b, err := json.Marshal(ev)
			if err != nil {
				continue
			}
			if _, err := w.Write(append(append([]byte("data: "), b...), '\n', '\n')); err != nil {
				return
			}
			fl.Flush()
		}
	}
}

// serveAudio streams the processed mic audio (16 kHz mono s16le) until the
// client disconnects. This is what a Home Assistant / Wyoming satellite needs;
// maind already fans the same stream out to every subscriber.
func (d *Daemon) serveAudio(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "audio/L16;rate=16000;channels=1")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	fl.Flush()

	ctl := http.NewResponseController(w)
	ch := d.audio.add(64)
	defer d.audio.remove(ch)
	for {
		select {
		case <-r.Context().Done():
			return
		case pcm := <-ch:
			// Do not let a stalled client pin a goroutine forever.
			if err := ctl.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				return
			}
			buf := make([]byte, 2*len(pcm))
			for i, s := range pcm {
				binary.LittleEndian.PutUint16(buf[2*i:], uint16(s))
			}
			if _, err := w.Write(buf); err != nil {
				return
			}
			fl.Flush()
		}
	}
}
