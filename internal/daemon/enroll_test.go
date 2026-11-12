package daemon

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"wakeword/internal/kws"
)

func TestEnrollmentKeepsConfusersSeparateFromSpeech(t *testing.T) {
	d := newTestDaemon(t)
	e := &enrollment{ID: "джарвис", Name: "Джарвис", Ambient: []int16{1, 2}, Speech: []int16{3, 4}, Confusers: [][]int16{{5, 6}, {7, 8}}}
	if err := d.saveEnrollment(e); err != nil {
		t.Fatal(err)
	}
	got, err := d.startEnroll(e.Name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Speech, e.Speech) || !reflect.DeepEqual(got.Confusers, e.Confusers) {
		t.Fatalf("loaded speech=%v confusers=%v", got.Speech, got.Confusers)
	}
	if err := d.deleteConfuser(1); err != nil {
		t.Fatal(err)
	}
	if err := d.saveEnrollment(got); err != nil {
		t.Fatal(err)
	}
	paths, _ := filepath.Glob(filepath.Join(d.wordDir(e.ID), "negative-*.wav"))
	if len(paths) != 1 {
		t.Fatalf("deleted negative survived save: %v", paths)
	}
	d.endEnroll()
	reloaded, err := d.startEnroll(e.Name)
	if err != nil || len(reloaded.Confusers) != 1 || !reflect.DeepEqual(reloaded.Speech, e.Speech) {
		t.Fatalf("reload after delete: %+v, %v", reloaded, err)
	}
}

func TestConfuserAPIRequiresEnrollmentAndRejectsInvalidIndex(t *testing.T) {
	d := newTestDaemon(t)
	h := d.Handler()
	for _, path := range []string{"/api/enroll/confuser", "/api/enroll/confuser/0"} {
		method := http.MethodDelete
		if path == "/api/enroll/confuser" {
			method = http.MethodPost
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(method, path, nil))
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s %s: status %d", method, path, w.Code)
		}
	}
	d.enroll = &enrollment{Confusers: [][]int16{{1}, {2}}}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodDelete, "/api/enroll/confuser/no-index", nil))
	if w.Code != http.StatusBadRequest || len(d.enroll.Confusers) != 2 {
		t.Fatalf("invalid index removed a clip: %d, %v", w.Code, d.enroll.Confusers)
	}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodDelete, "/api/enroll/confuser/1", nil))
	if w.Code != http.StatusOK || len(d.enroll.Confusers) != 1 {
		t.Fatalf("delete confuser: %d, %s", w.Code, w.Body.String())
	}
	var view enrollView
	if err := json.Unmarshal(w.Body.Bytes(), &view); err != nil || view.ConfuserClips != 1 {
		t.Fatalf("view: %s, %v", w.Body.String(), err)
	}
}

func TestConfuserRecordingAppendsWithoutReplacingOrdinarySpeech(t *testing.T) {
	d := newTestDaemon(t)
	d.audioOK = true
	d.settings.Gain = 1
	d.enroll = &enrollment{Speech: []int16{10, 20}, Confusers: [][]int16{{30, 40}}}
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		w := httptest.NewRecorder()
		d.Handler().ServeHTTP(w, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/api/enroll/confuser", nil))
		done <- w
	}()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for {
		d.mu.Lock()
		ready := d.capture != nil
		d.mu.Unlock()
		if ready {
			break
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatal("confuser capture did not start")
		}
	}
	pcm := make([]int16, int(confuserSecs*kws.SampleRate))
	for i := range pcm {
		pcm[i] = int16(i % 100)
	}
	d.onAudio(pcm)
	select {
	case w := <-done:
		if w.Code != http.StatusOK {
			t.Fatalf("record confuser: %d %s", w.Code, w.Body.String())
		}
	case <-deadline.C:
		t.Fatal("confuser recording did not complete")
	}
	if !reflect.DeepEqual(d.enroll.Speech, []int16{10, 20}) || len(d.enroll.Confusers) != 2 || !reflect.DeepEqual(d.enroll.Confusers[1], pcm) {
		t.Fatalf("confuser did not append separately: %+v", d.enroll)
	}
}

func TestValidateTakeRejectsClippingButKeepsQuietSpeech(t *testing.T) {
	quiet := make([]int16, 3*kws.SampleRate)
	for i := kws.SampleRate; i < kws.SampleRate+kws.SampleRate/2; i++ {
		quiet[i] = int16(600 * math.Sin(2*math.Pi*220*float64(i)/kws.SampleRate))
	}
	if r := validateTake(quiet); !r.OK {
		t.Fatalf("quiet valid speech rejected: %+v", r)
	}
	clipped := append([]int16(nil), quiet...)
	for i := kws.SampleRate; i < kws.SampleRate+kws.SampleRate/2; i++ {
		if i%2 == 0 {
			clipped[i] = 32767
		} else {
			clipped[i] = -32768
		}
	}
	if r := validateTake(clipped); r.OK || !strings.Contains(r.Message, "перегружена") {
		t.Fatalf("saturated take accepted: %+v", r)
	}
	// A single full-scale transient must not reject an otherwise valid take.
	quiet[kws.SampleRate+20] = 32767
	if r := validateTake(quiet); !r.OK {
		t.Fatalf("one transient rejected valid speech: %+v", r)
	}
}

func TestStateDistinguishesLegacyAndEventValidation(t *testing.T) {
	d := newTestDaemon(t)
	var legacy, current kws.Model
	if err := json.Unmarshal([]byte(`{"name":"old","recall":1,"false_per_h":0,"eval_hours":0.2}`), &legacy); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"name":"new","recall":0.5,"false_per_h":0,"eval_hours":0.2,"validation_version":1,"eval_takes":2,"detected_takes":1,"false_triggers":0}`), &current); err != nil {
		t.Fatal(err)
	}
	d.words = []word{{ID: "old", Model: &legacy}, {ID: "new", Model: &current}}
	d.settings.Words["old"] = &WordSettings{Enabled: true}
	d.settings.Words["new"] = &WordSettings{Enabled: true, Sensitivity: 0.1}
	b, err := json.Marshal(d.state())
	if err != nil {
		t.Fatal(err)
	}
	var view struct {
		Words []struct {
			ValidationVersion int     `json:"validation_version"`
			EvalTakes         int     `json:"eval_takes"`
			DetectedTakes     int     `json:"detected_takes"`
			EvalHours         float64 `json:"eval_hours"`
		}
	}
	if err := json.Unmarshal(b, &view); err != nil {
		t.Fatal(err)
	}
	if view.Words[0].ValidationVersion != 0 || view.Words[0].EvalTakes != 0 {
		t.Fatalf("legacy metrics presented as current: %s", b)
	}
	if v := view.Words[1]; v.ValidationVersion != 1 || v.EvalTakes != 2 || v.DetectedTakes != 1 || v.EvalHours != 0.2 {
		t.Fatalf("event validation missing: %s", b)
	}
}
