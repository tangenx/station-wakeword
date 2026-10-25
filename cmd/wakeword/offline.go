package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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
	negPath := fs.String("negatives", "negatives.bin", "bundled negative set")
	out := fs.String("out", "model.json", "output model")
	fs.Parse(args)

	files, _ := filepath.Glob(filepath.Join(*takesDir, "[^.]*.wav")) // skip macOS ._ files
	sort.Strings(files)
	var en kws.Enrollment
	for _, f := range files {
		pcm, err := wav.ReadFile(f, kws.SampleRate)
		if err != nil {
			return err
		}
		en.Takes = append(en.Takes, pcm)
	}
	var err error
	if en.Ambient, err = wav.ReadFile(*ambient, kws.SampleRate); err != nil {
		return err
	}
	if *speech != "" {
		if en.Speech, err = wav.ReadFile(*speech, kws.SampleRate); err != nil {
			return err
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
	m, err := kws.TrainWord(*name, en, neg, uint64(time.Now().UnixNano()), func(stage string, frac float64) {
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
	fmt.Printf("%s: threshold %.2f, held-out recall %.0f%%, %.2f false/h over %.2f h, %d takes, %s\n",
		*out, m.Threshold, m.Recall*100, m.FalsePerH, m.EvalHours, m.Takes, time.Since(start).Round(time.Second))
	for _, n := range m.TrainNotes {
		fmt.Println("  ", n)
	}
	return nil
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
