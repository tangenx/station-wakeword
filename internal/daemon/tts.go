package daemon

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"wakeword/internal/alicetts"
	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

const (
	aliceCacheVersion = "alice-training-v1-pcm-s16le-16000"
	maxAliceSources   = 8
	maxAliceSamples   = 10 * kws.SampleRate
	maxAliceWAVBytes  = 44 + 2*maxAliceSamples
)

type trainingOptions struct {
	AliceTTS bool `json:"alice_tts"`
}

func readTrainingOptions(dir string) (trainingOptions, error) {
	var opt trainingOptions
	b, err := os.ReadFile(filepath.Join(dir, "training.json"))
	if errors.Is(err, os.ErrNotExist) {
		return opt, nil
	}
	if err != nil {
		return opt, err
	}
	if err := json.Unmarshal(b, &opt); err != nil {
		return trainingOptions{}, err
	}
	return opt, nil
}

func saveTrainingOptions(dir string, opt trainingOptions) error {
	b, err := json.Marshal(opt)
	if err != nil {
		return err
	}
	path := filepath.Join(dir, "training.json")
	if err := os.WriteFile(path+".tmp", b, 0o644); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// prepareAliceSynthetic is optional preparation only: the real recordings and
// held-out validation remain the trainer's authority. Failure returns notes,
// not a replacement error, so an unavailable online service cannot block it.
func (d *Daemon) prepareAliceSynthetic(ctx context.Context, name string) (clips [][]int16, notes []string) {
	dir := d.wordDir(wordID(name))
	opt, err := readTrainingOptions(dir)
	if err != nil {
		notes = []string{"Дополнение голосом Алисы недоступно. Обучение продолжится по вашим записям."}
		d.noteAlicePreparation(notes)
		return nil, notes
	}
	if !opt.AliceTTS {
		return nil, nil
	}
	defer func() { d.noteAlicePreparation(notes) }()
	variants, err := alicetts.Variants(name)
	if err != nil || len(variants) == 0 {
		return nil, []string{"Для этого слова дополнение голосом Алисы недоступно. Обучение продолжится по вашим записям."}
	}
	variants = variants[:min(len(variants), maxAliceSources)]
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	synthesize := d.synthesize
	if synthesize == nil {
		synthesize = alicetts.Client{}.Synthesize
	}
	cacheDir := filepath.Join(dir, "synthetic")
	cacheWriteFailed := os.MkdirAll(cacheDir, 0o755) != nil
	seen := map[[32]byte]bool{}
	var cached, unavailable int
	for i, variant := range variants {
		if ctx.Err() != nil {
			unavailable++
			break
		}
		d.mu.Lock()
		d.train.Stage, d.train.Frac = "synthesize", float64(i)/float64(len(variants))
		d.train.Note = fmt.Sprintf("Готовим записи Алисы: %d из %d.", i+1, len(variants))
		d.mu.Unlock()
		// The key describes the complete pronunciation request and format.
		// Only canonical, bounded and usable WAVs are accepted from the cache.
		key := sha256.Sum256([]byte(aliceCacheVersion + "\x00" + variant.Text))
		path := filepath.Join(cacheDir, fmt.Sprintf("%x.wav", key))
		raw, fromCache := readAliceCache(path)
		padded, usable := prepareAliceClip(raw)
		if !fromCache || !usable {
			requestCtx, requestCancel := context.WithTimeout(ctx, 5*time.Second)
			raw, err = synthesize(requestCtx, variant.Text)
			requestCancel()
			padded, usable = prepareAliceClip(raw)
			if err != nil || !usable {
				unavailable++
				continue
			}
			fromCache = false
			if !cacheWriteFailed {
				if err := writeAliceCache(path, raw); err != nil {
					cacheWriteFailed = true
				}
			}
		}
		digest := sha256.Sum256(wav.Encode(raw, kws.SampleRate))
		if !seen[digest] {
			seen[digest] = true
			clips = append(clips, padded)
			if fromCache {
				cached++
			}
		}
	}
	if len(clips) == 0 {
		notes = append(notes, "Не удалось получить подходящие записи Алисы. Обучение продолжится по вашим записям.")
		return clips, notes
	}
	if cached == len(clips) {
		notes = append(notes, fmt.Sprintf("Получены сохранённые записи Алисы: %d.", len(clips)))
	} else {
		notes = append(notes, fmt.Sprintf("Получены разные записи Алисы: %d.", len(clips)))
	}
	if unavailable > 0 {
		notes = append(notes, "Часть голосов недоступна; обучение продолжится по доступным примерам.")
	}
	if cacheWriteFailed {
		notes = append(notes, "Часть записей Алисы не сохранена для следующего обучения.")
	}
	return clips, notes
}

func (d *Daemon) noteAlicePreparation(notes []string) {
	d.mu.Lock()
	d.train.Note = strings.Join(notes, " ")
	d.mu.Unlock()
}

func prepareAliceClip(raw []int16) ([]int16, bool) {
	if len(raw) == 0 || len(raw) > maxAliceSamples {
		return nil, false
	}
	const padding = kws.SampleRate / 5
	padded := make([]int16, len(raw)+2*padding)
	copy(padded[padding:], raw)
	start, end, ok := kws.FindSpeech(padded)
	if !ok || end-start < kws.SampleRate/5 || end-start > 2*kws.SampleRate {
		return nil, false
	}
	return padded, true
}

func readAliceCache(path string) ([]int16, bool) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxAliceWAVBytes+1))
	if err != nil || len(b) > maxAliceWAVBytes {
		return nil, false
	}
	pcm, rate, err := wav.Decode(b)
	if err != nil || rate != kws.SampleRate || !bytes.Equal(b, wav.Encode(pcm, rate)) {
		return nil, false
	}
	return pcm, true
}

func writeAliceCache(path string, pcm []int16) error {
	if err := wav.WriteFile(path+".tmp", pcm, kws.SampleRate); err != nil {
		os.Remove(path + ".tmp")
		return err
	}
	if err := os.Rename(path+".tmp", path); err != nil {
		os.Remove(path + ".tmp")
		return err
	}
	return nil
}
