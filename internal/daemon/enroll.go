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
	"slices"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

// enrollment is the word being recorded in the web UI.
type enrollment struct {
	ID           string
	Name         string
	Takes        [][]int16
	TakesChanged bool // saved failure indices no longer refer to this edited take list
	AliceTTS     bool // explicit opt-in to online synthetic training examples
	Ambient      []int16
	Speech       []int16
	Confusers    [][]int16 // optional phrases that sound similar but are not the word
	Last         []int16   // most recent take, kept or not, for "listen to it"
}

const (
	takeSecs     = 3.0
	takePreSecs  = 0.3
	ambientSecs  = 8.0
	confuserSecs = 8.0
	minTakes     = 6
)

type TakeResult struct {
	OK       bool    `json:"ok"`
	Message  string  `json:"message,omitempty"`
	Duration float64 `json:"duration"` // seconds of detected speech
	Level    float64 `json:"level"`    // dBFS of the word
	Takes    int     `json:"takes"`
}

func (d *Daemon) startEnroll(name string) (*enrollment, error) {
	e, err := d.readEnrollment(name)
	if err != nil {
		return nil, err
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

func (d *Daemon) readEnrollment(name string) (*enrollment, error) {
	if len([]rune(name)) < 2 || len([]rune(name)) > 40 {
		return nil, errors.New("введите слово (2–40 символов)")
	}
	e := &enrollment{ID: wordID(name), Name: name}
	// Re-enrolling an existing word continues from its recordings.
	dir := d.wordDir(e.ID)
	opt, _ := readTrainingOptions(dir)
	e.AliceTTS = opt.AliceTTS
	takes, _ := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	slices.Sort(takes)
	var err error
	if e.Takes, err = readWAVs(takes); err != nil {
		return nil, err
	}
	if a, err := wav.ReadFile(filepath.Join(dir, "ambient.wav"), kws.SampleRate); err == nil {
		e.Ambient = a
	}
	if s, err := wav.ReadFile(filepath.Join(dir, "speech.wav"), kws.SampleRate); err == nil {
		e.Speech = s
	}
	confusers, _ := filepath.Glob(filepath.Join(dir, "negative-*.wav"))
	slices.Sort(confusers)
	pcm, err := readWAVs(confusers)
	if err != nil {
		return nil, err
	}
	e.Confusers = pcm
	return e, nil
}

// resumeTraining only reopens saved recordings after an explicit user action.
// Keep the failure available until the user starts another training attempt.
func (d *Daemon) resumeTraining() (*enrollment, error) {
	d.mu.Lock()
	failed := d.train
	if failed.Active || failed.Error == "" || failed.Word == "" {
		d.mu.Unlock()
		return nil, errors.New("нет завершившегося с ошибкой обучения")
	}
	if d.enroll != nil {
		e := d.enroll
		d.mu.Unlock()
		if e.ID != failed.Word {
			return nil, errors.New("сначала завершите или отмените запись другого слова")
		}
		return e, nil
	}
	d.mu.Unlock()
	name, err := os.ReadFile(filepath.Join(d.wordDir(failed.Word), "name.txt"))
	if err != nil {
		return nil, fmt.Errorf("не удалось открыть сохранённые записи: %w", err)
	}
	if wordID(string(name)) != failed.Word {
		return nil, errors.New("имя сохранённых записей не совпадает со словом")
	}
	e, err := d.readEnrollment(string(name))
	if err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.train.Active || d.train.Word != failed.Word || d.train.Error != failed.Error {
		return nil, errors.New("состояние обучения изменилось, обновите страницу")
	}
	if d.enroll != nil {
		if d.enroll.ID != failed.Word {
			return nil, errors.New("сначала завершите или отмените запись другого слова")
		}
		return d.enroll, nil
	}
	d.enroll = e
	d.paused++
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
	r := validateTake(pcm)
	d.mu.Lock()
	e.Last = pcm
	if r.OK {
		e.Takes = append(e.Takes, pcm)
		e.TakesChanged = true
	}
	r.Takes = len(e.Takes)
	d.mu.Unlock()
	return r, nil
}

func validateTake(pcm []int16) TakeResult {
	r := TakeResult{}
	s, end, ok := kws.FindSpeech(pcm)
	switch {
	case !ok:
		r.Message = "не услышала слово — скажите громче или ближе к станции"
	default:
		r.Duration = float64(end-s) / kws.SampleRate
		r.Level = dbfs(pcm[s:end])
		switch {
		case clippedSpeech(pcm[s:end]):
			r.Message = "запись перегружена — скажите тише или отойдите от станции и повторите"
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
	return r
}

// Check only the detected word: silence around it must not dilute clipping.
// Isolated full-scale transients are tolerated; persistent saturation is not.
func clippedSpeech(pcm []int16) bool {
	var saturated int
	for _, s := range pcm {
		if s >= 32760 || s <= -32760 {
			saturated++
		}
	}
	return saturated >= 8 && saturated*100 >= len(pcm)
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
	d.enroll.TakesChanged = true
	return nil
}

func (d *Daemon) recordConfuser(ctx context.Context) (float64, error) {
	e, err := d.currentEnroll()
	if err != nil {
		return 0, err
	}
	pcm, err := d.record(ctx, confuserSecs, 0)
	if err != nil {
		return 0, err
	}
	d.mu.Lock()
	e.Confusers = append(e.Confusers, pcm)
	d.mu.Unlock()
	return dbfs(pcm), nil
}

func (d *Daemon) deleteConfuser(i int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.enroll == nil || i < 0 || i >= len(d.enroll.Confusers) {
		return errors.New("нет такой записи")
	}
	d.enroll.Confusers = slices.Delete(d.enroll.Confusers, i, i+1)
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
	old, _ = filepath.Glob(filepath.Join(dir, "negative-*.wav"))
	for _, p := range old {
		if err := os.Remove(p); err != nil {
			return err
		}
	}
	for i, pcm := range e.Confusers {
		if err := wav.WriteFile(filepath.Join(dir, fmt.Sprintf("negative-%02d.wav", i+1)), pcm, kws.SampleRate); err != nil {
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
	if err := saveTrainingOptions(dir, trainingOptions{AliceTTS: e.AliceTTS}); err != nil {
		return err
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
func (d *Daemon) startTraining(id string) (err error) {
	d.mu.Lock()
	if d.train.Active {
		d.mu.Unlock()
		return errors.New("уже идёт обучение")
	}
	// Claim the new attempt before loading saved files: a synchronous read
	// error must replace diagnostics from the previous, possibly edited list.
	d.train = trainState{Active: true, Word: id, Stage: "load"}
	d.mu.Unlock()
	defer func() {
		if err != nil {
			d.completeTraining(id, err)
		}
	}()

	dir := d.wordDir(id)
	name, err := os.ReadFile(filepath.Join(dir, "name.txt"))
	if err != nil {
		return fmt.Errorf("слово %s не найдено", id)
	}
	d.mu.Lock()
	d.train.Name = string(name)
	d.mu.Unlock()
	var en kws.Enrollment
	takes, _ := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	slices.Sort(takes)
	if en.Takes, err = readWAVs(takes); err != nil {
		return err
	}
	if en.Ambient, err = wav.ReadFile(filepath.Join(dir, "ambient.wav"), kws.SampleRate); err != nil {
		return err
	}
	en.Speech, _ = wav.ReadFile(filepath.Join(dir, "speech.wav"), kws.SampleRate)
	confusers, _ := filepath.Glob(filepath.Join(dir, "negative-*.wav"))
	slices.Sort(confusers)
	if en.Confusers, err = readWAVs(confusers); err != nil {
		return err
	}
	falses, _ := filepath.Glob(filepath.Join(dir, "false-*.wav"))
	if en.False, err = readWAVs(falses); err != nil {
		return err
	}

	go func() {
		start := time.Now()
		m, err := d.trainWord(string(name), en)
		d.completeTraining(id, err)
		if err != nil {
			return
		}
		log.Printf("trained %s in %s: threshold %.2f recall %.2f", id, time.Since(start).Round(time.Second), m.Threshold, m.Recall)
	}()
	return nil
}

// completeTraining publishes a result without changing recordings, settings or
// the loaded model on failure. trainWord installs only an accepted model.
func (d *Daemon) completeTraining(id string, err error) {
	d.mu.Lock()
	d.train.Active = false
	if err == nil {
		d.train.Done, d.train.Error, d.train.Recovery = id, "", nil
		d.mu.Unlock()
		return
	}
	r := &trainRecovery{Code: "training_error"}
	var trainingErr *kws.TrainingError
	if errors.As(err, &trainingErr) {
		r.Code = trainingErr.Code
		r.TakeIndices = slices.Clone(trainingErr.TakeIndices)
	}
	for _, w := range d.words {
		if w.ID == id && w.Model != nil {
			r.PreviousModel = true
			break
		}
	}
	d.train.Error, d.train.Recovery = err.Error(), r
	d.mu.Unlock()
	if trainingErr != nil {
		log.Printf("train %s seed=%d: %v", id, trainingErr.Seed, err)
	} else {
		log.Printf("train %s: %v", id, err)
	}
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
	var notes []string
	en.Synthetic, notes = d.prepareAliceSynthetic(context.Background(), name)
	m, err := kws.TrainWord(name, en, neg, uint64(time.Now().UnixNano()), func(stage string, frac float64) {
		d.mu.Lock()
		d.train.Stage, d.train.Frac = stage, frac
		d.mu.Unlock()
	})
	if err != nil {
		return nil, err
	}
	if len(en.Synthetic) > 0 {
		if m.SyntheticTakes > 0 {
			notes = append(notes, fmt.Sprintf("В обучении использованы записи Алисы: %d.", m.SyntheticTakes))
		} else {
			notes = append(notes, "Записи Алисы не подошли для обучения; использованы только ваши записи.")
		}
	}
	m.TrainNotes = append(m.TrainNotes, notes...)
	if len(notes) > 0 {
		d.noteAlicePreparation(notes)
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
