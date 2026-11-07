package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Actions run when a word fires. They are listed per word in settings.json:
//
//	"words": {"jarvis": {"actions": ["alice", "webhook"]}}
//
// DryRun is a global "do not wake Alice": it drops the alice action from every
// word and leaves the rest intact, so a hook-based setup keeps working.
const (
	actAlice   = "alice"   // ask Alice to listen (stock behaviour)
	actLLM     = "llm"     // like alice, but mark the request for the local bridge's LLM
	actWebhook = "webhook" // POST settings.on_hit_url
	actExec    = "exec"    // run settings.on_hit_exec
)

var knownActions = map[string]bool{actAlice: true, actLLM: true, actWebhook: true, actExec: true}

// defaultRouteMarker is the file a local bridge (for example Quasar Bridge in
// hybrid mode) reads to learn that the next voice request came from a custom
// word and should go to its own LLM instead of Yandex.
const defaultRouteMarker = "/tmp/wakeword-route"

func (d *Daemon) routeMarker() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.settings.RouteMarker != "" {
		return d.settings.RouteMarker
	}
	return defaultRouteMarker
}

// defaultActions is used when a word has no explicit list: stock behaviour,
// plus hooks once they are configured.
func (d *Daemon) defaultActions() []string {
	out := []string{actAlice}
	if d.settings.OnHitURL != "" {
		out = append(out, actWebhook)
	}
	if d.settings.OnHitExec != "" {
		out = append(out, actExec)
	}
	return out
}

// actionsFor resolves and filters a word's actions; caller holds d.mu.
func (d *Daemon) actionsFor(id string) []string {
	actions := d.defaultActions()
	if w := d.settings.Words[id]; w != nil && len(w.Actions) > 0 {
		actions = w.Actions
	}
	out := make([]string, 0, len(actions))
	seen := map[string]bool{}
	for _, a := range actions {
		a = strings.TrimSpace(strings.ToLower(a))
		if !knownActions[a] || seen[a] {
			continue
		}
		if (a == actAlice || a == actLLM) && d.settings.DryRun {
			continue // dry run: never wake Alice
		}
		if a == actWebhook && d.settings.OnHitURL == "" {
			continue
		}
		if a == actExec && d.settings.OnHitExec == "" {
			continue
		}
		seen[a] = true
		out = append(out, a)
	}
	return out
}

// fireActions runs the actions in the background; the hit is already recorded.
func (d *Daemon) fireActions(h *Hit, actions []string) {
	d.mu.Lock()
	url, script := d.settings.OnHitURL, d.settings.OnHitExec
	d.mu.Unlock()

	for _, a := range actions {
		switch a {
		case actLLM:
			go func() {
				marker := d.routeMarker()
				// Do not start a conversation if marking failed: the request
				// would silently go to Yandex instead of the LLM.
				if err := os.WriteFile(marker, []byte("llm\n"), 0600); err != nil {
					log.Printf("action llm marker: %v", err)
					d.setHitError(h, err.Error())
					return
				}
				if err := d.startListen(); err != nil {
					os.Remove(marker)
					log.Printf("action llm: %v", err)
					d.setHitError(h, err.Error())
				}
			}()
		case actAlice:
			go func() {
				if err := d.startListen(); err != nil {
					log.Printf("action alice: %v", err)
					d.setHitError(h, err.Error())
				}
			}()
		case actWebhook:
			go func() {
				if err := postHit(url, h); err != nil {
					log.Printf("action webhook: %v", err)
					d.setHitError(h, err.Error())
				}
			}()
		case actExec:
			go func() {
				if err := runHook(script, h); err != nil {
					log.Printf("action exec: %v", err)
					d.setHitError(h, err.Error())
				}
			}()
		}
	}
}

func (d *Daemon) setHitError(h *Hit, msg string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if h.Error == "" {
		h.Error = msg
	}
}

// hitEvent is the JSON handed to webhooks and the /api/events stream.
type hitEvent struct {
	Word    string    `json:"word"`
	Name    string    `json:"name"`
	Score   float32   `json:"score"`
	At      time.Time `json:"at"`
	Actions []string  `json:"actions"`
}

func eventOf(h *Hit, actions []string) hitEvent {
	return hitEvent{Word: h.Word, Name: h.Name, Score: h.Score, At: h.At, Actions: actions}
}

func postHit(url string, h *Hit) error {
	body, err := json.Marshal(eventOf(h, h.Actions))
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "wakeword")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}
	return nil
}

// runHook executes the configured script. It runs as the daemon user (root on
// the station), so the path is fixed in settings.json and never comes from the
// network; the hit details are passed as argv and environment.
func runHook(script string, h *Hit) error {
	if !filepath.IsAbs(script) {
		return fmt.Errorf("%q: path must be absolute", script)
	}
	fi, err := os.Stat(script)
	if err != nil {
		return err
	}
	if !fi.Mode().IsRegular() {
		return fmt.Errorf("%s: not a regular file", script)
	}
	if fi.Mode().Perm()&0o111 == 0 {
		return fmt.Errorf("%s: not executable", script)
	}
	if fi.Mode().Perm()&0o002 != 0 {
		log.Printf("action exec: warning: %s is world-writable", script)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, script, h.Word, fmt.Sprintf("%.3f", h.Score))
	cmd.Env = append(os.Environ(),
		"WAKE_WORD="+h.Word,
		"WAKE_NAME="+h.Name,
		"WAKE_SCORE="+fmt.Sprintf("%.3f", h.Score),
		"WAKE_AT="+h.At.UTC().Format(time.RFC3339Nano),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if tail := strings.TrimSpace(stderr.String()); tail != "" {
			return fmt.Errorf("%v: %s", err, lastLine(tail))
		}
		return err
	}
	return nil
}

func lastLine(s string) string {
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	}
	if len(s) > 200 {
		s = s[:200] + "…"
	}
	return s
}

// normalizeActions filters a user-supplied list for settings.json.
func normalizeActions(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, 0, len(in))
	seen := map[string]bool{}
	for _, a := range in {
		a = strings.TrimSpace(strings.ToLower(a))
		if !knownActions[a] || seen[a] {
			continue
		}
		seen[a] = true
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}
