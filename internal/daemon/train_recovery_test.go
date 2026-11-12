package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"wakeword/internal/kws"
	"wakeword/internal/wav"
)

func savedRecoveryEnrollment(t *testing.T, d *Daemon) *enrollment {
	t.Helper()
	e := &enrollment{ID: "джарвис", Name: "Джарвис", Ambient: []int16{1, 2}, Speech: []int16{3, 4}, Confusers: [][]int16{{5, 6}}}
	for i := range minTakes {
		e.Takes = append(e.Takes, []int16{int16(i), 7})
	}
	if err := d.saveEnrollment(e); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestFailedNewWordResumesSavedEnrollmentWithoutModel(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	d.train = trainState{Active: true, Word: e.ID, Name: e.Name}
	typed := &kws.TrainingError{Code: "word_not_recognized", TakeIndices: []int{0, 4}, Seed: 42}
	failure := fmt.Errorf("training: %w", typed)
	d.completeTraining(e.ID, failure)
	if d.train.Active || d.train.Error != failure.Error() || d.train.Recovery == nil || d.train.Recovery.Code != "word_not_recognized" || d.train.Recovery.PreviousModel {
		t.Fatalf("new-word failure state: %+v", d.train)
	}
	if !reflect.DeepEqual(d.train.Recovery.TakeIndices, []int{0, 4}) || d.enroll != nil || d.paused != 0 || len(d.words) != 0 {
		t.Fatalf("failure changed recordings/detection automatically: %+v", d.train)
	}
	w := httptest.NewRecorder()
	d.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/train/resume", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("resume new word: %d %s", w.Code, w.Body.String())
	}
	var view enrollView
	if err := json.Unmarshal(w.Body.Bytes(), &view); err != nil || view.ID != e.ID || len(view.Takes) != minTakes {
		t.Fatalf("resumed view: %s, %v", w.Body.String(), err)
	}
	if d.train.Error != failure.Error() || d.train.Recovery == nil || d.paused != 1 || !reflect.DeepEqual(d.enroll.Takes, e.Takes) || !reflect.DeepEqual(d.enroll.Confusers, e.Confusers) {
		t.Fatalf("resume lost saved state or error: %+v, %+v", d.train, d.enroll)
	}
	first := d.enroll
	first.Takes = append(first.Takes, []int16{8, 9})
	if got, err := d.resumeTraining(); err != nil || got != first || len(got.Takes) != minTakes+1 || d.paused != 1 {
		t.Fatalf("repeat resume replaced unsaved changes: %+v, %v", got, err)
	}
	d.endEnroll()
	if d.train.Recovery == nil || d.train.Error == "" || d.paused != 0 {
		t.Fatalf("cancel discarded recovery: %+v", d.train)
	}
}

func TestFailedTrainingPreservesPreviousModelAndSettings(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	previous := &kws.Model{Name: e.Name, Threshold: 0.7}
	d.words = []word{{ID: e.ID, Model: previous}}
	settings := &WordSettings{Enabled: true, Sensitivity: 0.2, Actions: []string{"webhook"}}
	d.settings.Words[e.ID] = settings
	modelPath := filepath.Join(d.wordDir(e.ID), "model.json")
	if err := os.WriteFile(modelPath, []byte("previous model"), 0o644); err != nil {
		t.Fatal(err)
	}
	d.train = trainState{Active: true, Word: e.ID, Name: e.Name}
	d.completeTraining(e.ID, &kws.TrainingError{Code: "background_confusion", Seed: 43})
	if d.train.Recovery == nil || !d.train.Recovery.PreviousModel || d.train.Recovery.Code != "background_confusion" {
		t.Fatalf("previous model not reported: %+v", d.train)
	}
	if _, err := d.resumeTraining(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(modelPath)
	if err != nil || string(b) != "previous model" || d.words[0].Model != previous || d.settings.Words[e.ID] != settings || settings.Sensitivity != 0.2 {
		t.Fatalf("failed training replaced previous working state: %s, %v", b, err)
	}
}

func TestGenericTrainingFailureCanResume(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	d.train = trainState{Active: true, Word: e.ID, Name: e.Name}
	d.completeTraining(e.ID, errors.New("negatives file unavailable"))
	if d.train.Recovery == nil || d.train.Recovery.Code != "training_error" {
		t.Fatalf("generic failure has no recovery: %+v", d.train)
	}
	if _, err := d.resumeTraining(); err != nil {
		t.Fatal(err)
	}
}

func TestResumeRejectsUnavailableFailureOrAnotherEnrollment(t *testing.T) {
	for _, tc := range []struct {
		name   string
		active bool
		error  string
		other  bool
	}{
		{name: "no failure"},
		{name: "active training", active: true, error: "old failure"},
		{name: "another enrollment", error: "failed", other: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := newTestDaemon(t)
			e := savedRecoveryEnrollment(t, d)
			d.train = trainState{Word: e.ID, Name: e.Name, Active: tc.active, Error: tc.error}
			if tc.other {
				d.enroll = &enrollment{ID: "компьютер", Name: "Компьютер", Takes: [][]int16{{1}}}
				d.paused = 1
			}
			before := d.enroll
			w := httptest.NewRecorder()
			d.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/train/resume", nil))
			if w.Code != http.StatusBadRequest || d.enroll != before {
				t.Fatalf("resume overwrote an unrelated session: %d, %+v", w.Code, d.enroll)
			}
		})
	}
}

func TestEnrollmentBackgroundPlaybackReturnsRecordedPCM(t *testing.T) {
	d := newTestDaemon(t)
	h := d.Handler()
	for _, path := range []string{"/api/enroll/ambient", "/api/enroll/speech"} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		if w.Code != http.StatusNotFound {
			t.Fatalf("playback before enrollment: %s status %d", path, w.Code)
		}
	}
	d.enroll = &enrollment{Ambient: []int16{-32768, -1, 0, 32767}, Speech: []int16{12, -45, 678}}
	for path, pcm := range map[string][]int16{
		"/api/enroll/ambient": d.enroll.Ambient,
		"/api/enroll/speech":  d.enroll.Speech,
	} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "audio/wav" || !bytes.Equal(w.Body.Bytes(), wav.Encode(pcm, kws.SampleRate)) {
			t.Fatalf("background playback changed PCM: %s status %d", path, w.Code)
		}
	}
	d.enroll.Ambient, d.enroll.Speech = nil, nil
	for _, path := range []string{"/api/enroll/ambient", "/api/enroll/speech"} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		if w.Code != http.StatusNotFound {
			t.Fatalf("playback without a recording: %s status %d", path, w.Code)
		}
	}
}

func TestDeletingTakeInvalidatesFailureIndicesUntilReload(t *testing.T) {
	d := newTestDaemon(t)
	saved := savedRecoveryEnrollment(t, d)
	d.train = trainState{Active: true, Word: saved.ID, Name: saved.Name}
	d.completeTraining(saved.ID, &kws.TrainingError{Code: "word_not_recognized", TakeIndices: []int{1, 4}, Seed: 1})
	e, err := d.resumeTraining()
	if err != nil {
		t.Fatal(err)
	}
	if e.TakesChanged || d.state().Enroll.TakesChanged {
		t.Fatal("saved recordings already marked changed")
	}
	if err := d.deleteTake(0); err != nil {
		t.Fatal(err)
	}
	if !e.TakesChanged || !d.state().Enroll.TakesChanged || !reflect.DeepEqual(e.Takes[0], saved.Takes[1]) || !reflect.DeepEqual(d.train.Recovery.TakeIndices, []int{1, 4}) {
		t.Fatal("deletion did not mark shifted take indices stale while preserving failure")
	}
	d.endEnroll()
	e, err = d.resumeTraining()
	if err != nil || e.TakesChanged || d.state().Enroll.TakesChanged || !reflect.DeepEqual(e.Takes, saved.Takes) {
		t.Fatalf("cancel/resume did not restore saved take indices: %+v, %v", e, err)
	}
}

func TestAcceptedTakeMarksRecordingsChanged(t *testing.T) {
	d := newTestDaemon(t)
	d.audioOK = true
	d.settings.Gain = 1
	d.enroll = &enrollment{}
	d.ring.write(make([]int16, int(takePreSecs*kws.SampleRate)))
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	type result struct {
		take TakeResult
		err  error
	}
	done := make(chan result, 1)
	go func() {
		take, err := d.recordTake(ctx)
		done <- result{take, err}
	}()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		d.mu.Lock()
		ready := d.capture != nil
		d.mu.Unlock()
		if ready {
			break
		}
		select {
		case <-tick.C:
		case <-ctx.Done():
			t.Fatal("take recording did not start")
		}
	}
	pcm := make([]int16, int(takeSecs*kws.SampleRate))
	for i := kws.SampleRate; i < kws.SampleRate+kws.SampleRate/2; i++ {
		pcm[i] = int16(600 * math.Sin(2*math.Pi*220*float64(i)/kws.SampleRate))
	}
	d.onAudio(pcm)
	select {
	case got := <-done:
		if got.err != nil || !got.take.OK || !d.enroll.TakesChanged || !d.state().Enroll.TakesChanged || len(d.enroll.Takes) != 1 {
			t.Fatalf("accepted take did not mark changed: %+v, %v", got.take, got.err)
		}
	case <-ctx.Done():
		t.Fatal("take recording did not complete")
	}
}

func TestRetryLoadFailureReplacesIndicesForRenumberedTakes(t *testing.T) {
	d := newTestDaemon(t)
	saved := savedRecoveryEnrollment(t, d)
	saved.Takes = append(saved.Takes, []int16{9, 10})
	if err := d.saveEnrollment(saved); err != nil {
		t.Fatal(err)
	}
	d.train = trainState{Active: true, Word: saved.ID, Name: saved.Name}
	d.completeTraining(saved.ID, &kws.TrainingError{Code: "word_not_recognized", TakeIndices: []int{1, 4}, Seed: 1})
	e, err := d.resumeTraining()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.deleteTake(0); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(d.wordDir(saved.ID), "false-corrupt.wav"), []byte("broken wav"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := d.finishEnroll(); err == nil {
		t.Fatal("retry accepted a corrupt false-trigger clip")
	}
	if d.train.Active || d.train.Recovery == nil || d.train.Recovery.Code != "training_error" || len(d.train.Recovery.TakeIndices) != 0 || d.train.Word != saved.ID {
		t.Fatalf("retry preserved stale failure indices: %+v", d.train)
	}
	reloaded, err := d.resumeTraining()
	if err != nil || reloaded.TakesChanged || len(reloaded.Takes) != minTakes || !reflect.DeepEqual(reloaded.Takes, e.Takes) {
		t.Fatalf("could not resume newly saved take numbering: %+v, %v", reloaded, err)
	}
}

func TestSynchronousTrainingReadFailuresHaveFreshRecovery(t *testing.T) {
	for _, missing := range []string{"name.txt", "ambient.wav"} {
		t.Run(missing, func(t *testing.T) {
			d := newTestDaemon(t)
			e := savedRecoveryEnrollment(t, d)
			d.train = trainState{Word: "old-word", Error: "old failure", Recovery: &trainRecovery{Code: "word_not_recognized", TakeIndices: []int{0}}}
			if err := os.Remove(filepath.Join(d.wordDir(e.ID), missing)); err != nil {
				t.Fatal(err)
			}
			if err := d.startTraining(e.ID); err == nil {
				t.Fatal("missing training input accepted")
			}
			if d.train.Word != e.ID || d.train.Active || d.train.Error == "" || d.train.Error == "old failure" || d.train.Recovery == nil || d.train.Recovery.Code != "training_error" || len(d.train.Recovery.TakeIndices) != 0 {
				t.Fatalf("synchronous input failure left stale state: %+v", d.train)
			}
		})
	}
}

func TestStartTrainingCannotCompleteAnotherActiveJob(t *testing.T) {
	d := newTestDaemon(t)
	d.train = trainState{Active: true, Word: "already-training", Name: "Already training", Stage: "train", Frac: 0.5}
	before := d.train
	if err := d.startTraining("missing-word"); err == nil || !strings.Contains(err.Error(), "идёт обучение") {
		t.Fatalf("did not reject the active job first: %v", err)
	}
	if d.train != before {
		t.Fatalf("another request changed the active job: before %+v after %+v", before, d.train)
	}
}

func TestMissingNegativesPublishesRecoverableAsyncFailure(t *testing.T) {
	d := newTestDaemon(t)
	e := savedRecoveryEnrollment(t, d)
	if err := d.startTraining(e.ID); err != nil {
		t.Fatal(err)
	}
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		d.mu.Lock()
		state := d.train
		d.mu.Unlock()
		if !state.Active {
			if state.Word != e.ID || state.Recovery == nil || state.Recovery.Code != "training_error" || len(state.Recovery.TakeIndices) != 0 {
				t.Fatalf("missing negatives has no fresh recovery: %+v", state)
			}
			if _, err := d.resumeTraining(); err != nil {
				t.Fatal(err)
			}
			return
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("missing negatives did not complete the background job")
		}
	}
}
