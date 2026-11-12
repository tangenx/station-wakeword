package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"wakeword/internal/kws"
)

func TestAliceTrainingOptionIsExplicitAndSavedWithEnrollment(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	path := filepath.Join(d.wordDir(e.ID), "training.json")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, err := d.startEnroll(e.Name); err != nil {
		t.Fatal(err)
	}
	option := func() bool {
		t.Helper()
		b, err := json.Marshal(d.state().Enroll)
		if err != nil {
			t.Fatal(err)
		}
		var view struct {
			AliceTTS bool `json:"alice_tts"`
		}
		if err := json.Unmarshal(b, &view); err != nil {
			t.Fatal(err)
		}
		return view.AliceTTS
	}
	if option() {
		t.Fatal("legacy enrollment enabled online synthesis")
	}
	w := httptest.NewRecorder()
	d.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/enroll/options", strings.NewReader(`{"alice_tts":true}`)))
	if w.Code != http.StatusOK || !option() {
		t.Fatalf("enable optional synthesis: %d %s", w.Code, w.Body.String())
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("option persisted before saving enrollment: %v", err)
	}
	if err := d.saveEnrollment(d.enroll); err != nil {
		t.Fatal(err)
	}
	d.endEnroll()
	if _, err := d.startEnroll(e.Name); err != nil {
		t.Fatal(err)
	}
	if !option() {
		t.Fatal("saved training option not restored")
	}
}

func TestAliceTrainingOptionRejectsInvalidRequests(t *testing.T) {
	d := newTestDaemon(t)
	h := d.Handler()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/enroll/options", strings.NewReader(`{"alice_tts":true}`)))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("changed options without enrollment: %d", w.Code)
	}
	d.enroll = &enrollment{}
	for _, body := range []string{`{}`, `{"alice_tts":null}`, `{"alice_tts":"yes"}`} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/enroll/options", strings.NewReader(body)))
		if w.Code != http.StatusBadRequest {
			t.Fatalf("accepted ambiguous option %s: %d", body, w.Code)
		}
	}
}

func TestInvalidTrainingOptionsNeverEnableOnlinePreparation(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	if err := os.WriteFile(filepath.Join(d.wordDir(e.ID), "training.json"), []byte(`{"alice_tts":true,"alice_tts":"invalid"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	opt, err := readTrainingOptions(d.wordDir(e.ID))
	if err == nil || opt.AliceTTS {
		t.Fatalf("partially decoded consent accepted: %+v, %v", opt, err)
	}
	loaded, err := d.readEnrollment(e.Name)
	if err != nil || loaded.AliceTTS {
		t.Fatalf("invalid optional settings blocked/enabled enrollment: %+v, %v", loaded, err)
	}
	calls := 0
	d.synthesize = func(context.Context, string) ([]int16, error) { calls++; return generatedTestWord(0), nil }
	clips, notes := d.prepareAliceSynthetic(t.Context(), e.Name)
	if len(clips) != 0 || calls != 0 || len(notes) == 0 {
		t.Fatalf("invalid settings requested synthesis: clips=%d requests=%d notes=%v", len(clips), calls, notes)
	}
}

func enableAliceTraining(t *testing.T, d *Daemon, name string) {
	t.Helper()
	dir := d.wordDir(wordID(name))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := saveTrainingOptions(dir, trainingOptions{AliceTTS: true}); err != nil {
		t.Fatal(err)
	}
}

// Raw TTS may contain no surrounding silence. Preparation must add enough for
// the same VAD that is used by training, without discarding this usable speech.
func generatedTestWord(variant int) []int16 {
	pcm := make([]int16, kws.SampleRate/2)
	for i := range pcm {
		pcm[i] = int16(float64(3000+variant*100) * math.Sin(2*math.Pi*float64(180+variant*13)*float64(i)/kws.SampleRate))
	}
	return pcm
}

func TestAliceSyntheticCachesAndRegeneratesCorruptClip(t *testing.T) {
	d := newTestDaemon(t)
	enableAliceTraining(t, d, "Джарвис")
	byText := map[string][]int16{}
	calls := 0
	d.synthesize = func(ctx context.Context, text string) ([]int16, error) {
		calls++
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > 5*time.Second {
			t.Error("synthesis request has no five-second deadline")
		}
		if byText[text] == nil {
			byText[text] = generatedTestWord(len(byText))
		}
		return byText[text], nil
	}
	clips, notes := d.prepareAliceSynthetic(t.Context(), "Джарвис")
	if len(clips) != 8 || calls != 8 || len(notes) == 0 {
		t.Fatalf("generation: clips=%d requests=%d notes=%v", len(clips), calls, notes)
	}
	if len(clips[0]) != kws.SampleRate/2+2*kws.SampleRate/5 || clips[0][0] != 0 || clips[0][len(clips[0])-1] != 0 {
		t.Fatal("generated speech was not padded for VAD")
	}
	clips, notes = d.prepareAliceSynthetic(t.Context(), "Джарвис")
	if len(clips) != 8 || calls != 8 || !strings.Contains(strings.Join(notes, " "), "сохранён") {
		t.Fatalf("cache miss on retry: clips=%d requests=%d notes=%v", len(clips), calls, notes)
	}
	paths, err := filepath.Glob(filepath.Join(d.wordDir("джарвис"), "synthetic", "*.wav"))
	if err != nil || len(paths) != 8 {
		t.Fatalf("cache files: %v, %v", paths, err)
	}
	if err := os.WriteFile(paths[0], []byte("broken cache"), 0o644); err != nil {
		t.Fatal(err)
	}
	clips, _ = d.prepareAliceSynthetic(t.Context(), "Джарвис")
	if len(clips) != 8 || calls != 9 {
		t.Fatalf("corrupt cache not regenerated once: clips=%d requests=%d", len(clips), calls)
	}
}

func TestAliceSyntheticDisabledOrInvalidWordNeverRequests(t *testing.T) {
	d := newTestDaemon(t)
	calls := 0
	d.synthesize = func(context.Context, string) ([]int16, error) { calls++; return generatedTestWord(0), nil }
	clips, notes := d.prepareAliceSynthetic(t.Context(), "Джарвис")
	if len(clips) != 0 || len(notes) != 0 || calls != 0 {
		t.Fatalf("default-off requested synthesis: %d %v %d", len(clips), notes, calls)
	}
	name := "<speaker>Джарвис</speaker>"
	enableAliceTraining(t, d, name)
	clips, notes = d.prepareAliceSynthetic(t.Context(), name)
	if len(clips) != 0 || len(notes) == 0 || calls != 0 {
		t.Fatalf("invalid plain word requested synthesis: %d %v %d", len(clips), notes, calls)
	}
}

func TestAliceSyntheticPartialAndDuplicateResults(t *testing.T) {
	d := newTestDaemon(t)
	enableAliceTraining(t, d, "Джарвис")
	calls := 0
	d.synthesize = func(context.Context, string) ([]int16, error) {
		calls++
		switch calls {
		case 1, 2:
			return generatedTestWord(0), nil // duplicate PCM is only one source
		case 3:
			return nil, nil
		case 4:
			return make([]int16, 11*kws.SampleRate), nil
		case 5:
			return make([]int16, kws.SampleRate), nil // silence
		default:
			return nil, errors.New("unavailable")
		}
	}
	clips, notes := d.prepareAliceSynthetic(t.Context(), "Джарвис")
	if len(clips) != 1 || calls != 8 || !strings.Contains(strings.Join(notes, " "), "Часть") {
		t.Fatalf("partial/duplicate results: clips=%d requests=%d notes=%v", len(clips), calls, notes)
	}
}

func TestAliceSyntheticTimeoutAndCacheWriteFailureFallBack(t *testing.T) {
	t.Run("timeout", func(t *testing.T) {
		d := newTestDaemon(t)
		enableAliceTraining(t, d, "Джарвис")
		calls := 0
		d.synthesize = func(ctx context.Context, _ string) ([]int16, error) {
			calls++
			<-ctx.Done()
			return nil, ctx.Err()
		}
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
		defer cancel()
		clips, notes := d.prepareAliceSynthetic(ctx, "Джарвис")
		if len(clips) != 0 || calls != 1 || len(notes) == 0 {
			t.Fatalf("timeout did not fall back: %d, %d, %v", len(clips), calls, notes)
		}
	})
	t.Run("cache write failure", func(t *testing.T) {
		d := newTestDaemon(t)
		enableAliceTraining(t, d, "Джарвис")
		if err := os.WriteFile(filepath.Join(d.wordDir("джарвис"), "synthetic"), []byte("not a directory"), 0o644); err != nil {
			t.Fatal(err)
		}
		calls := 0
		d.synthesize = func(context.Context, string) ([]int16, error) { calls++; return generatedTestWord(calls), nil }
		clips, notes := d.prepareAliceSynthetic(t.Context(), "Джарвис")
		if len(clips) != 8 || !strings.Contains(strings.Join(notes, " "), "не сохран") {
			t.Fatalf("cache failure discarded usable audio: %d, %v", len(clips), notes)
		}
	})
}

func TestAliceSupplementCannotReplaceSixRealTakes(t *testing.T) {
	d := newTestDaemon(t)
	d.enroll = &enrollment{ID: "джарвис", Name: "Джарвис", Takes: make([][]int16, 5), Ambient: []int16{1}, AliceTTS: true}
	calls := 0
	d.synthesize = func(context.Context, string) ([]int16, error) { calls++; return generatedTestWord(0), nil }
	if err := d.finishEnroll(); err == nil || !strings.Contains(err.Error(), "минимум 6") || calls != 0 {
		t.Fatalf("synthetic enabled training without real takes: %v requests=%d", err, calls)
	}
}

func TestSyntheticPreparationCannotReplaceModelWhenTrainingRejects(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d) // tiny clips contain no usable real speech
	enableAliceTraining(t, d, e.Name)
	previous := &kws.Model{Name: e.Name, Threshold: 0.7}
	d.words = []word{{ID: e.ID, Model: previous}}
	d.settings.Words[e.ID] = &WordSettings{Enabled: true}
	path := filepath.Join(d.wordDir(e.ID), "model.json")
	if err := os.WriteFile(path, []byte("previous model"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(d.opt.Negatives, []byte{'N', 'E', 'G', '1', 0, 0, 0, 0, 0, 0, 0, 0}, 0o644); err != nil {
		t.Fatal(err)
	}
	calls := 0
	d.synthesize = func(context.Context, string) ([]int16, error) { calls++; return generatedTestWord(calls), nil }
	if _, err := d.trainWord(e.Name, kws.Enrollment{Takes: e.Takes, Ambient: e.Ambient}); err == nil || !strings.Contains(err.Error(), "usable takes") || calls != 8 {
		t.Fatalf("synthetic preparation overrode real quality error: %v requests=%d", err, calls)
	}
	b, err := os.ReadFile(path)
	if err != nil || string(b) != "previous model" || d.words[0].Model != previous {
		t.Fatalf("failed training replaced the old model: %s, %v", b, err)
	}
}
