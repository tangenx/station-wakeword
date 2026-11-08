package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func newTestDaemon(t *testing.T) *Daemon {
	t.Helper()
	d, err := New(Options{DataDir: t.TempDir(), Negatives: filepath.Join(t.TempDir(), "neg.bin"), QuasarCfg: "/nonexistent"})
	if err != nil {
		t.Fatal(err)
	}
	d.startListen = func(string) error { return nil }
	return d
}

func TestActionsFor(t *testing.T) {
	cases := []struct {
		name string
		dry  bool
		url  string
		exec string
		word []string
		want []string
	}{
		{name: "default wakes Alice", want: []string{"alice"}},
		{name: "dry run silences Alice", dry: true, want: nil},
		{name: "hooks join by default", url: "http://x/hook", exec: "/tmp/h.sh", want: []string{"alice", "webhook", "exec"}},
		{name: "dry run keeps hooks", dry: true, url: "http://x/hook", exec: "/tmp/h.sh", want: []string{"webhook", "exec"}},
		{name: "explicit list replaces the default", word: []string{"webhook"}, url: "http://x/hook", want: []string{"webhook"}},
		{name: "unknown and duplicate actions are dropped", word: []string{"nope", "ALICE", "alice"}, want: []string{"alice"}},
		{name: "hook without a target is dropped", word: []string{"webhook"}, want: nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := newTestDaemon(t)
			d.mu.Lock()
			d.settings.DryRun, d.settings.OnHitURL, d.settings.OnHitExec = c.dry, c.url, c.exec
			d.settings.Words["w"] = &WordSettings{Enabled: true, Actions: c.word}
			got := d.actionsFor("w")
			d.mu.Unlock()
			if strings.Join(got, ",") != strings.Join(c.want, ",") {
				t.Fatalf("got %v, want %v", got, c.want)
			}
		})
	}
}

// A hit fires exactly the configured actions: Alice only when allowed, the
// hook with the hit in argv and environment.
func TestHitRunsActions(t *testing.T) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var ev hitEvent
		if err := json.NewDecoder(r.Body).Decode(&ev); err != nil {
			t.Errorf("webhook body: %v", err)
		}
		mu.Lock()
		got = append(got, "webhook:"+ev.Word)
		mu.Unlock()
	}))
	defer srv.Close()

	dir := t.TempDir()
	envFile := filepath.Join(dir, "env.txt")
	script := filepath.Join(dir, "hook.sh")
	err := os.WriteFile(script, []byte("#!/bin/sh\nprintf '%s %s %s\\n' \"$WAKE_WORD\" \"$WAKE_SCORE\" \"$1\" > "+envFile+"\n"), 0o755)
	if err != nil {
		t.Fatal(err)
	}

	d := newTestDaemon(t)
	alice := make(chan string, 1)
	d.startListen = func(word string) error { alice <- word; return nil }
	d.mu.Lock()
	d.settings.OnHitURL, d.settings.OnHitExec = srv.URL, script
	d.words = []word{{ID: "w", Model: nil}}
	d.mu.Unlock()

	h := &Hit{ID: 1, Word: "w", Name: "W", Score: 0.9, At: time.Now(), Actions: []string{"alice", "webhook", "exec"}}
	d.fireActions(h, h.Actions)

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(got)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(envFile); err == nil && len(b) > 0 {
			if f := strings.Fields(string(b)); len(f) < 3 || f[0] != "w" || f[2] != "w" {
				t.Fatalf("hook saw %q", b)
			}
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	mu.Lock()
	actions := strings.Join(got, ",")
	mu.Unlock()
	if actions != "webhook:w" {
		t.Fatalf("webhook calls: %q", actions)
	}
	select {
	case word := <-alice:
		if word != "" {
			t.Fatalf("unexpected wakeword tag for %q", word)
		}
	case <-time.After(time.Second):
		t.Fatal("Alice action did not run")
	}
}

func TestAliceWakewordTag(t *testing.T) {
	const plain = "1c000000e20f153a0f416c6963654361706162696c69747942020802e80fd201"
	const tagged = `{"tag_alice_wakeword":true}`
	for _, tc := range []struct {
		name, settings, word, frame string
	}{
		{"default", `{}`, "джарвис", plain},
		{"first word", tagged, "джарвис", "55000000e20f4e3a0f416c6963654361706162696c697479423b0802a201360a340a321a0230015a2c080510021802222431666233373961332d613431352d353637642d396263352d393737636434343932386233e80fd201"},
		{"second word", tagged, "домовёнок", "55000000e20f4e3a0f416c6963654361706162696c697479423b0802a201360a340a321a0230015a2c080510021802222462346431333731352d396435312d353565342d623362652d663236633738353432383461e80fd201"},
		{"manual listen", tagged, "", plain},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ln, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
			if err != nil {
				t.Fatal(err)
			}
			defer ln.Close()
			deadline := time.Now().Add(3 * time.Second)
			ln.SetDeadline(deadline)
			dir := t.TempDir()
			cfg := filepath.Join(dir, "quasar.cfg")
			if err := os.WriteFile(cfg, fmt.Appendf(nil, `{"aliced":{"port":%d}}`, ln.Addr().(*net.TCPAddr).Port), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte(tc.settings), 0o600); err != nil {
				t.Fatal(err)
			}
			d, err := New(Options{DataDir: dir, QuasarCfg: cfg})
			if err != nil {
				t.Fatal(err)
			}
			var status chan int
			if tc.word == "" {
				status = make(chan int, 1)
				go func() {
					rec := httptest.NewRecorder()
					d.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/listen", nil))
					status <- rec.Code
				}()
			} else {
				hit := &Hit{Word: tc.word}
				d.fireActions(hit, d.actionsFor(hit.Word))
			}
			conn, err := ln.AcceptTCP()
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			conn.SetDeadline(deadline)
			var header [4]byte
			if _, err := io.ReadFull(conn, header[:]); err != nil {
				t.Fatal(err)
			}
			size := binary.LittleEndian.Uint32(header[:])
			if size > 256 {
				t.Fatalf("unexpected IPC frame length %d", size)
			}
			body := make([]byte, size)
			if _, err := io.ReadFull(conn, body); err != nil {
				t.Fatal(err)
			}
			conn.Close()
			if status != nil && <-status != http.StatusOK {
				t.Fatal("manual listen failed")
			}
			want, err := hex.DecodeString(tc.frame)
			if err != nil {
				t.Fatal(err)
			}
			if got := append(header[:], body...); !bytes.Equal(got, want) {
				t.Fatalf("IPC frame\n got %x\nwant %x", got, want)
			}
		})
	}
}

func TestServeEvents(t *testing.T) {
	d := newTestDaemon(t)
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/api/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}
	lines := make(chan string, 4)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			if line := sc.Text(); strings.HasPrefix(line, "data: ") {
				lines <- strings.TrimPrefix(line, "data: ")
				return
			}
		}
	}()
	// The subscriber registers on connect; give it a moment, then publish.
	time.Sleep(100 * time.Millisecond)
	d.events.publish(hitEvent{Word: "w", Name: "W", Score: 0.8, At: time.Now()})
	select {
	case line := <-lines:
		var ev hitEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("%q: %v", line, err)
		}
		if ev.Word != "w" || ev.Score != 0.8 {
			t.Fatalf("event %+v", ev)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no event")
	}
}

func TestServeAudio(t *testing.T) {
	d := newTestDaemon(t)
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/api/stream", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "audio/L16") {
		t.Fatalf("content type %q", ct)
	}
	time.Sleep(100 * time.Millisecond)
	d.audio.publish([]int16{1, -2, 300})

	buf := make([]byte, 6)
	done := make(chan error, 1)
	go func() {
		_, err := io.ReadFull(resp.Body, buf)
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no audio")
	}
	want := []int16{1, -2, 300}
	for i, w := range want {
		if got := int16(binary.LittleEndian.Uint16(buf[2*i:])); got != w {
			t.Fatalf("sample %d = %d, want %d", i, got, w)
		}
	}
}

func TestNormalizeActions(t *testing.T) {
	got := normalizeActions([]string{" Webhook ", "alice", "alice", "bogus", "EXEC"})
	if strings.Join(got, ",") != "alice,exec,webhook" {
		t.Fatalf("got %v", got)
	}
	if normalizeActions(nil) != nil {
		t.Fatal("empty list must stay empty (defaults apply)")
	}
}
