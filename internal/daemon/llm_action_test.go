package daemon

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The llm action marks the request for the local bridge before Alice starts
// listening, and does not start listening when marking fails.
func TestLLMActionWritesRouteMarker(t *testing.T) {
	d := newTestDaemon(t)
	marker := filepath.Join(t.TempDir(), "route")
	d.mu.Lock()
	d.settings.RouteMarker = marker
	d.mu.Unlock()
	listened := make(chan bool, 1)
	d.startListen = func() error {
		b, err := os.ReadFile(marker)
		listened <- err == nil && string(b) == "llm\n"
		return nil
	}
	d.fireActions(&Hit{Word: "w"}, []string{actLLM})
	select {
	case ok := <-listened:
		if !ok {
			t.Fatal("marker was not written before listening")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("listening was not started")
	}

	d.settings.RouteMarker = filepath.Join(t.TempDir(), "missing", "route")
	d.startListen = func() error { t.Error("started listening without a marker"); return nil }
	h := &Hit{Word: "w"}
	d.fireActions(h, []string{actLLM})
	time.Sleep(200 * time.Millisecond)
	d.mu.Lock()
	defer d.mu.Unlock()
	if h.Error == "" {
		t.Fatal("marker failure was not reported")
	}
}

func TestLLMActionRespectsDryRun(t *testing.T) {
	d := newTestDaemon(t)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.settings.DryRun = true
	d.settings.Words["w"] = &WordSettings{Enabled: true, Actions: []string{"llm"}}
	if got := d.actionsFor("w"); len(got) != 0 {
		t.Fatalf("dry run kept %v", got)
	}
}
