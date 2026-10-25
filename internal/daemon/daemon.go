// Package daemon runs on the station: it listens to the processed mic
// stream, detects the user's words, opens an Alice voice request on a hit,
// and serves the web UI that records and trains new words.
package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/quasar"
	"wakeword/internal/wav"
)

type Options struct {
	DataDir    string // words, settings
	Negatives  string // bundled negatives.bin
	QuasarCfg  string
	UserConfig string // maind's synced config, for the station name
	Listen     string
}

// Settings is persisted in <data>/settings.json.
type Settings struct {
	Channel string                   `json:"channel"` // yio_audio channel
	Gain    float64                  `json:"gain"`    // applied before everything else
	DryRun  bool                     `json:"dry_run"` // log hits without waking Alice
	Words   map[string]*WordSettings `json:"words"`
}

type WordSettings struct {
	Enabled     bool    `json:"enabled"`
	Sensitivity float32 `json:"sensitivity"` // threshold offset, -0.3..0.3
}

type word struct {
	ID    string
	Model *kws.Model
}

// Hit is one detection, kept for the UI (with audio for false-alarm marking).
type Hit struct {
	ID    int       `json:"id"`
	Word  string    `json:"word"`
	Name  string    `json:"name"`
	Score float32   `json:"score"`
	At    time.Time `json:"at"`
	Fired bool      `json:"fired"` // Alice was asked to listen
	Error string    `json:"error,omitempty"`
	audio []int16
}

type Daemon struct {
	opt         Options
	audioPort   int
	alicedPort  int
	startListen func() error

	mu       sync.Mutex
	settings Settings
	words    []word
	det      *kws.Detector
	detWords []string // word id per detector model
	paused   int      // >0 while enrolling: no detection
	ring     *ring
	level    float64 // dBFS of the last chunk
	audioOK  bool
	audioErr string
	capture  *capture
	hits     []*Hit
	hitSeq   int
	enroll   *enrollment
	train    trainState
	name     cachedName
}

// cachedName is the app-given station name, re-read when maind rewrites
// the synced config (the owner renamed the station).
type cachedName struct {
	name  string
	mtime time.Time
}

// stationName returns the station's name; caller holds d.mu.
func (d *Daemon) stationName() string {
	fi, err := os.Stat(d.opt.UserConfig)
	if err == nil && !fi.ModTime().Equal(d.name.mtime) {
		d.name = cachedName{quasar.StationName(d.opt.UserConfig), fi.ModTime()}
	}
	if d.name.name != "" {
		return d.name.name
	}
	h, _ := os.Hostname()
	return h
}

type trainState struct {
	Active bool    `json:"active"`
	Word   string  `json:"word,omitempty"` // id
	Name   string  `json:"name,omitempty"`
	Stage  string  `json:"stage,omitempty"`
	Frac   float64 `json:"frac"`
	Error  string  `json:"error,omitempty"`
	Done   string  `json:"done,omitempty"` // id of the last trained word
}

const ringSecs = 12

func New(opt Options) (*Daemon, error) {
	if err := os.MkdirAll(filepath.Join(opt.DataDir, "words"), 0o755); err != nil {
		return nil, err
	}
	d := &Daemon{opt: opt, ring: newRing(ringSecs * kws.SampleRate)}
	d.audioPort, d.alicedPort = quasar.Ports(opt.QuasarCfg)
	d.startListen = func() error { return quasar.CallAlice(d.alicedPort, quasar.StartConversation) }
	d.settings = Settings{Channel: "vqe_0", Gain: 32, Words: map[string]*WordSettings{}}
	if b, err := os.ReadFile(d.settingsPath()); err == nil {
		if err := json.Unmarshal(b, &d.settings); err != nil {
			return nil, fmt.Errorf("settings: %w", err)
		}
	}
	if d.settings.Words == nil {
		d.settings.Words = map[string]*WordSettings{}
	}
	if err := d.loadWords(); err != nil {
		return nil, err
	}
	return d, nil
}

func (d *Daemon) settingsPath() string { return filepath.Join(d.opt.DataDir, "settings.json") }
func (d *Daemon) wordDir(id string) string {
	return filepath.Join(d.opt.DataDir, "words", id)
}

// saveSettings writes settings atomically; caller holds d.mu.
func (d *Daemon) saveSettings() error {
	b, _ := json.MarshalIndent(d.settings, "", "  ")
	tmp := d.settingsPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, d.settingsPath())
}

func (d *Daemon) loadWords() error {
	dirs, err := os.ReadDir(filepath.Join(d.opt.DataDir, "words"))
	if err != nil {
		return err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.words = nil
	for _, e := range dirs {
		if !e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(d.wordDir(e.Name()), "model.json"))
		if err != nil {
			continue // enrolled but never trained
		}
		m, err := kws.ReadModel(b)
		if err != nil {
			log.Printf("word %s: bad model: %v", e.Name(), err)
			continue
		}
		d.words = append(d.words, word{e.Name(), m})
		if d.settings.Words[e.Name()] == nil {
			d.settings.Words[e.Name()] = &WordSettings{Enabled: true}
		}
	}
	sort.Slice(d.words, func(i, j int) bool { return d.words[i].ID < d.words[j].ID })
	return d.rebuildDetector()
}

// rebuildDetector applies the enabled words; caller holds d.mu.
func (d *Daemon) rebuildDetector() error {
	var models []*kws.Model
	var sens []float32
	d.detWords = nil
	for _, w := range d.words {
		s := d.settings.Words[w.ID]
		if s == nil || !s.Enabled {
			continue
		}
		models = append(models, w.Model)
		sens = append(sens, -s.Sensitivity) // higher sensitivity = lower threshold
		d.detWords = append(d.detWords, w.ID)
	}
	if len(models) == 0 {
		d.det = nil
		return nil
	}
	det, err := kws.NewDetector(models, sens)
	if err != nil {
		return err
	}
	d.det = det
	return nil
}

// Run streams audio until ctx ends, reconnecting to maind as needed.
func (d *Daemon) Run(ctx context.Context) {
	backoff := time.Second
	for ctx.Err() == nil {
		start := time.Now()
		err := quasar.AudioStream(ctx, d.audioPort, d.settings.Channel, d.onAudio)
		if ctx.Err() != nil {
			return
		}
		d.mu.Lock()
		d.audioOK = false
		d.audioErr = fmt.Sprint(err)
		d.mu.Unlock()
		if time.Since(start) > time.Minute {
			backoff = time.Second
		}
		log.Printf("audio stream: %v (retry in %s)", err, backoff)
		select {
		case <-ctx.Done():
		case <-time.After(backoff):
		}
		backoff = min(2*backoff, 30*time.Second)
	}
}

func (d *Daemon) onAudio(raw []int16) {
	d.mu.Lock()
	defer d.mu.Unlock()
	pcm := make([]int16, len(raw))
	var e float64
	for i, s := range raw {
		v := float64(s) * d.settings.Gain
		pcm[i] = int16(max(-32768, min(32767, v)))
		e += float64(pcm[i]) * float64(pcm[i])
	}
	d.level = 10*math.Log10(e/float64(max(len(pcm), 1))+1e-9) - 90.3 // dBFS
	d.audioOK, d.audioErr = true, ""
	d.ring.write(pcm)
	if c := d.capture; c != nil {
		c.buf = append(c.buf, pcm...)
		if len(c.buf) >= c.want {
			close(c.done)
			d.capture = nil
		}
	}
	if d.det == nil || d.paused > 0 {
		return
	}
	d.det.Push(pcm, func(i int, score float32) {
		id := d.detWords[i]
		h := &Hit{Word: id, Name: d.wordName(id), Score: score, At: time.Now()}
		h.audio = d.ring.last(3 * kws.SampleRate)
		if !d.settings.DryRun {
			h.Fired = true
			go func() {
				if err := d.startListen(); err != nil {
					log.Printf("start conversation: %v", err)
					d.mu.Lock()
					h.Error = err.Error()
					d.mu.Unlock()
				}
			}()
		}
		d.hitSeq++
		h.ID = d.hitSeq
		d.hits = append(d.hits, h)
		if len(d.hits) > 20 {
			d.hits = d.hits[len(d.hits)-20:]
		}
		log.Printf("hit %s score %.3f", id, score)
	})
}

func (d *Daemon) wordName(id string) string {
	for _, w := range d.words {
		if w.ID == id {
			return w.Model.Name
		}
	}
	return id
}

// capture records the next n samples (plus pre samples already in the
// ring) from the live stream.
type capture struct {
	buf  []int16
	want int
	done chan struct{}
}

var errBusy = errors.New("another recording is in progress")

func (d *Daemon) record(ctx context.Context, secs, preSecs float64) ([]int16, error) {
	pre := int(preSecs * kws.SampleRate)
	d.mu.Lock()
	if d.capture != nil {
		d.mu.Unlock()
		return nil, errBusy
	}
	if !d.audioOK {
		d.mu.Unlock()
		return nil, errors.New("no audio from the station: " + d.audioErr)
	}
	c := &capture{want: pre + int(secs*kws.SampleRate), done: make(chan struct{})}
	c.buf = d.ring.last(pre)
	d.capture = c
	d.mu.Unlock()
	select {
	case <-c.done:
		return c.buf[:c.want], nil
	case <-ctx.Done():
		d.mu.Lock()
		if d.capture == c {
			d.capture = nil
		}
		d.mu.Unlock()
		return nil, ctx.Err()
	case <-time.After(time.Duration(secs+5) * time.Second):
		d.mu.Lock()
		if d.capture == c {
			d.capture = nil
		}
		d.mu.Unlock()
		return nil, errors.New("audio stream stalled")
	}
}

var slugRe = regexp.MustCompile(`[^a-z0-9а-яё]+`)

// wordID derives a directory name from the word's spelling.
func wordID(name string) string {
	s := slugRe.ReplaceAllString(strings.ToLower(strings.TrimSpace(name)), "-")
	s = strings.Trim(s, "-")
	if s == "" {
		s = "word"
	}
	return s
}

func readWAVs(paths []string) ([][]int16, error) {
	var out [][]int16
	for _, p := range paths {
		pcm, err := wav.ReadFile(p, kws.SampleRate)
		if err != nil {
			return nil, err
		}
		out = append(out, pcm)
	}
	return out, nil
}
