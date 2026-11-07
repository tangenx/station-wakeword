package daemon

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

// playBeep plays a short sine tone through the station speaker (default ALSA
// device) so the owner can hear the start and end of a recording from afar.
func playBeep(freq float64, ms int) error {
	n := kws.SampleRate * ms / 1000
	pcm := make([]int16, n)
	for i := range pcm {
		env := math.Min(1, math.Min(float64(i), float64(n-i))/160)
		pcm[i] = int16(14000 * env * math.Sin(2*math.Pi*freq*float64(i)/float64(kws.SampleRate)))
	}
	f, err := os.CreateTemp("", "owner-beep-*.wav")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = f.Write(wav.Encode(pcm, kws.SampleRate))
	f.Close()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "aplay", "-q", f.Name()).CombinedOutput(); err != nil {
		return fmt.Errorf("aplay: %v: %s", err, out)
	}
	return nil
}

func (d *Daemon) handleBeep(w http.ResponseWriter, r *http.Request) {
	d.mu.Lock()
	enabled := d.settings.Beeps
	d.mu.Unlock()
	if !enabled {
		writeJSON(w, map[string]bool{"ok": true})
		return
	}
	var err error
	switch r.URL.Query().Get("kind") {
	case "start":
		err = playBeep(1000, 250)
	case "stop":
		err = playBeep(500, 400)
	default:
		err = fmt.Errorf("unknown beep kind")
	}
	if err != nil {
		httpErr(w, err)
		return
	}
	writeJSON(w, map[string]bool{"ok": true})
}
