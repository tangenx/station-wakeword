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
	"sort"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

// enrollment is the word being recorded in the web UI.
type enrollment struct {
	ID      string
	Name    string
	Takes   [][]int16
	Ambient []int16
	Speech  []int16
	Last    []int16 // most recent take, kept or not, for "listen to it"
}

const (
	takeSecs    = 3.0
	takePreSecs = 0.3
	ambientSecs = 8.0
	minTakes    = 6
)

type TakeResult struct {
	OK       bool    `json:"ok"`
	Message  string  `json:"message,omitempty"`
	Duration float64 `json:"duration"` // seconds of detected speech
	Level    float64 `json:"level"`    // dBFS of the word
	Takes    int     `json:"takes"`
}

func (d *Daemon) startEnroll(name string) (*enrollment, error) {
	if len([]rune(name)) < 2 || len([]rune(name)) > 40 {
		return nil, errors.New("введите слово (2–40 символов)")
	}
	e := &enrollment{ID: wordID(name), Name: name}
	// Re-enrolling an existing word continues from its recordings.
	dir := d.wordDir(e.ID)
	takes, _ := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	sort.Strings(takes)
	if pcm, err := readWAVs(takes); err == nil {
		e.Takes = pcm
	}
	if a, err := wav.ReadFile(filepath.Join(dir, "ambient.wav"), kws.SampleRate); err == nil {
		e.Ambient = a
	}
	if s, err := wav.ReadFile(filepath.Join(dir, "speech.wav"), kws.SampleRate); err == nil {
		e.Speech = s
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.train.Active {
		return nil, errors.New("идёт обучение, подождите")
	}
	if d.enroll == nil {
		d.paused++
	}
	d.enroll = e
	return e, nil
}

func (d *Daemon) endEnroll() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.enroll != nil {
		d.enroll = nil
		d.paused--
	}
}

func (d *Daemon) currentEnroll() (*enrollment, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.enroll == nil {
		return nil, errors.New("запись не начата")
	}
	return d.enroll, nil
}

func dbfs(pcm []int16) float64 {
	var e float64
	for _, s := range pcm {
		e += float64(s) * float64(s)
	}
	return 10*math.Log10(e/float64(max(len(pcm), 1))+1e-9) - 90.3
}

func (d *Daemon) recordAmbient(ctx context.Context) (float64, error) {
	e, err := d.currentEnroll()
	if err != nil {
		return 0, err
	}
	pcm, err := d.record(ctx, ambientSecs, 0)
	if err != nil {
		return 0, err
	}
	d.mu.Lock()
	e.Ambient = pcm
	d.mu.Unlock()
	return dbfs(pcm), nil
}

func (d *Daemon) recordTake(ctx context.Context) (TakeResult, error) {
	e, err := d.currentEnroll()
	if err != nil {
		return TakeResult{}, err
	}
	pcm, err := d.record(ctx, takeSecs, takePreSecs)
	if err != nil {
		return TakeResult{}, err
	}
	r := TakeResult{}
	s, end, ok := kws.FindSpeech(pcm)
	switch {
	case !ok:
		r.Message = "не услышала слово — скажите громче или ближе к станции"
	default:
		r.Duration = float64(end-s) / kws.SampleRate
		r.Level = dbfs(pcm[s:end])
		switch {
		case r.Duration < 0.2:
			r.Message = "слишком коротко — это был щелчок или шум?"
		case r.Duration > 2.0:
			r.Message = "слишком длинно — скажите только слово, без пауз и фраз"
		case s < kws.SampleRate/20 || end > len(pcm)-kws.SampleRate/20:
			r.Message = "слово обрезано — начинайте после сигнала и уложитесь в 2 секунды"
		default:
			r.OK = true
		}
	}
	d.mu.Lock()
	e.Last = pcm
	if r.OK {
		e.Takes = append(e.Takes, pcm)
	}
	r.Takes = len(e.Takes)
	d.mu.Unlock()
	return r, nil
}

func (d *Daemon) recordSpeech(ctx context.Context, secs float64) (float64, error) {
	e, err := d.currentEnroll()
	if err != nil {
		return 0, err
	}
	pcm, err := d.record(ctx, min(max(secs, 5), 60), 0)
	if err != nil {
		return 0, err
	}
	d.mu.Lock()
	e.Speech = pcm
	d.mu.Unlock()
	return dbfs(pcm), nil
}

func (d *Daemon) deleteTake(i int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.enroll == nil || i < 0 || i >= len(d.enroll.Takes) {
		return errors.New("нет такой записи")
	}
	d.enroll.Takes = append(d.enroll.Takes[:i], d.enroll.Takes[i+1:]...)
	return nil
}

// saveEnrollment writes the recordings to the word directory, replacing
// the previous takes.
func (d *Daemon) saveEnrollment(e *enrollment) error {
	dir := d.wordDir(e.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	old, _ := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	for _, p := range old {
		os.Remove(p)
	}
	for i, t := range e.Takes {
		if err := wav.WriteFile(filepath.Join(dir, fmt.Sprintf("take-%02d.wav", i+1)), t, kws.SampleRate); err != nil {
			return err
		}
	}
	if err := wav.WriteFile(filepath.Join(dir, "ambient.wav"), e.Ambient, kws.SampleRate); err != nil {
		return err
	}
	if len(e.Speech) > 0 {
		if err := wav.WriteFile(filepath.Join(dir, "speech.wav"), e.Speech, kws.SampleRate); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(dir, "name.txt"), []byte(e.Name), 0o644)
}

// finishEnroll saves the session and trains it in the background.
func (d *Daemon) finishEnroll() error {
	e, err := d.currentEnroll()
	if err != nil {
		return err
	}
	switch {
	case len(e.Takes) < minTakes:
		return fmt.Errorf("нужно минимум %d удачных записей слова, сейчас %d", minTakes, len(e.Takes))
	case len(e.Ambient) == 0:
		return errors.New("сначала запишите тишину в комнате")
	}
	if err := d.saveEnrollment(e); err != nil {
		return err
	}
	d.endEnroll()
	return d.startTraining(e.ID)
}

// startTraining (re)trains a word from its saved recordings.
func (d *Daemon) startTraining(id string) error {
	dir := d.wordDir(id)
	name, err := os.ReadFile(filepath.Join(dir, "name.txt"))
	if err != nil {
		return fmt.Errorf("слово %s не найдено", id)
	}
	var en kws.Enrollment
	takes, _ := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	sort.Strings(takes)
	if en.Takes, err = readWAVs(takes); err != nil {
		return err
	}
	if en.Ambient, err = wav.ReadFile(filepath.Join(dir, "ambient.wav"), kws.SampleRate); err != nil {
		return err
	}
	en.Speech, _ = wav.ReadFile(filepath.Join(dir, "speech.wav"), kws.SampleRate)
	falses, _ := filepath.Glob(filepath.Join(dir, "false-*.wav"))
	if en.False, err = readWAVs(falses); err != nil {
		return err
	}

	d.mu.Lock()
	if d.train.Active {
		d.mu.Unlock()
		return errors.New("уже идёт обучение")
	}
	d.train = trainState{Active: true, Word: id, Name: string(name), Stage: "load"}
	d.mu.Unlock()

	go func() {
		start := time.Now()
		m, err := d.trainWord(string(name), en)
		d.mu.Lock()
		d.train.Active = false
		if err != nil {
			d.train.Error = err.Error()
			d.mu.Unlock()
			log.Printf("train %s: %v", id, err)
			return
		}
		d.train.Done = id
		d.mu.Unlock()
		log.Printf("trained %s in %s: threshold %.2f recall %.2f", id, time.Since(start).Round(time.Second), m.Threshold, m.Recall)
	}()
	return nil
}

func (d *Daemon) trainWord(name string, en kws.Enrollment) (*kws.Model, error) {
	b, err := os.ReadFile(d.opt.Negatives)
	if err != nil {
		return nil, fmt.Errorf("negatives: %w", err)
	}
	neg, err := kws.ReadNegatives(b)
	if err != nil {
		return nil, err
	}
	m, err := kws.TrainWord(name, en, neg, uint64(time.Now().UnixNano()), func(stage string, frac float64) {
		d.mu.Lock()
		d.train.Stage, d.train.Frac = stage, frac
		d.mu.Unlock()
	})
	if err != nil {
		return nil, err
	}
	id := wordID(name)
	mb, _ := json.Marshal(m)
	path := filepath.Join(d.wordDir(id), "model.json")
	if err := os.WriteFile(path+".tmp", mb, 0o644); err != nil {
		return nil, err
	}
	if err := os.Rename(path+".tmp", path); err != nil {
		return nil, err
	}
	d.mu.Lock()
	if d.settings.Words[id] == nil {
		d.settings.Words[id] = &WordSettings{Enabled: true}
	}
	d.saveSettings()
	d.mu.Unlock()
	return m, d.loadWords()
}
