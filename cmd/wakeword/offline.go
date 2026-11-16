package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

// cmdTrain trains a word from WAV files, the same way the station does:
//
//	wakeword train -name джарвис -takes dir/ -ambient room.wav -speech talk.wav \
//	    -negatives negatives.bin -out model.json
func cmdTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	name := fs.String("name", "", "word name")
	takesDir := fs.String("takes", "", "directory of 16 kHz WAV takes, one utterance each")
	ambient := fs.String("ambient", "", "16 kHz WAV of the room")
	speech := fs.String("speech", "", "16 kHz WAV of normal speech (optional)")
	confusers := fs.String("confusers", "", "directory of similar phrases without the wake word (optional)")
	negPath := fs.String("negatives", "negatives.bin", "bundled negative set")
	out := fs.String("out", "model.json", "output model")
	seed := fs.Uint64("seed", uint64(time.Now().UnixNano()), "training seed for repeatable splits and augmentation")
	fs.Parse(args)

	files, err := takeFiles(*takesDir)
	if err != nil {
		return err
	}
	var en kws.Enrollment
	for _, f := range files {
		pcm, err := wav.ReadFile(f, kws.SampleRate)
		if err != nil {
			return err
		}
		en.Takes = append(en.Takes, pcm)
	}
	if en.Ambient, err = wav.ReadFile(*ambient, kws.SampleRate); err != nil {
		return err
	}
	if *speech != "" {
		if en.Speech, err = wav.ReadFile(*speech, kws.SampleRate); err != nil {
			return err
		}
	}
	pattern := filepath.Join(*takesDir, "negative-*.wav")
	if *confusers != "" {
		pattern = filepath.Join(*confusers, "[^.]*.wav")
	}
	for _, group := range []struct {
		pattern string
		target  *[][]int16
	}{{pattern, &en.Confusers}, {filepath.Join(*takesDir, "false-*.wav"), &en.False}} {
		paths, _ := filepath.Glob(group.pattern)
		slices.Sort(paths)
		for _, path := range paths {
			pcm, err := wav.ReadFile(path, kws.SampleRate)
			if err != nil {
				return err
			}
			*group.target = append(*group.target, pcm)
		}
	}
	nb, err := os.ReadFile(*negPath)
	if err != nil {
		return err
	}
	neg, err := kws.ReadNegatives(nb)
	if err != nil {
		return err
	}
	start := time.Now()
	last := ""
	m, err := kws.TrainWord(*name, en, neg, *seed, func(stage string, frac float64) {
		if s := fmt.Sprintf("%s %3.0f%%", stage, frac*100); s != last && int(frac*100)%10 == 0 {
			fmt.Fprintln(os.Stderr, s)
			last = s
		}
	})
	if err != nil {
		return err
	}
	b, _ := json.Marshal(m)
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	fmt.Printf("%s: threshold %.4f, calibration %d/%d original takes, %d false events over %.1f min, best epoch %d, seed %d, %s\n",
		*out, m.Threshold, m.DetectedTakes, m.EvalTakes, m.FalseTriggers, m.EvalHours*60, m.BestEpoch, m.Seed, time.Since(start).Round(time.Second))
	for _, n := range m.TrainNotes {
		fmt.Println("  ", n)
	}
	return nil
}

// Station directories contain both labels. Generic positive filenames remain
// supported, but reserved background/negative files must never become words.
func takeFiles(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "take-*.wav"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		files, err = filepath.Glob(filepath.Join(dir, "[^.]*.wav"))
		if err != nil {
			return nil, err
		}
		files = slices.DeleteFunc(files, func(path string) bool {
			name := filepath.Base(path)
			return name == "ambient.wav" || name == "speech.wav" || strings.HasPrefix(name, "negative-") || strings.HasPrefix(name, "false-")
		})
	}
	slices.Sort(files)
	return files, nil
}

// cmdScan runs models over a WAV and prints detections with timestamps.
func cmdScan(args []string) error {
	fs := flag.NewFlagSet("scan", flag.ExitOnError)
	sens := fs.Float64("sensitivity", 0, "threshold offset (negative = more sensitive)")
	fs.Parse(args)
	if fs.NArg() < 2 {
		return fmt.Errorf("usage: wakeword scan model.json... audio.wav")
	}
	var models []*kws.Model
	var offs []float32
	for _, p := range fs.Args()[:fs.NArg()-1] {
		m, err := loadModel(p)
		if err != nil {
			return err
		}
		models = append(models, m)
		offs = append(offs, float32(*sens))
	}
	pcm, err := wav.ReadFile(fs.Arg(fs.NArg()-1), kws.SampleRate)
	if err != nil {
		return err
	}
	d, err := kws.NewDetector(models, offs)
	if err != nil {
		return err
	}
	pos := 0
	start := time.Now()
	for pos < len(pcm) {
		n := min(kws.StepSamp, len(pcm)-pos)
		d.Push(pcm[pos:pos+n], func(i int, s float32) {
			fmt.Printf("%7.2fs %s %.3f\n", float64(pos+n)/kws.SampleRate, models[i].Name, s)
		})
		pos += n
	}
	fmt.Fprintf(os.Stderr, "scanned %.1fs of audio in %s\n", float64(len(pcm))/kws.SampleRate, time.Since(start).Round(time.Millisecond))
	return nil
}

func loadModel(path string) (*kws.Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := kws.ReadModel(b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}
