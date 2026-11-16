package kws

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestTrainingQualityErrorDistinguishesBackgroundConfusion(t *testing.T) {
	for _, detected := range []int{0, 1} {
		err := trainingQualityError(calibration{Feasible: false, Detected: detected}, []int{0, 2, 3}, []int{1}, 42)
		var training *TrainingError
		if !errors.As(err, &training) {
			t.Fatalf("got %T %v, want structured training error", err, err)
		}
		if training.Code != "background_confusion" || training.Seed != 42 || len(training.TakeIndices) != 0 {
			t.Fatalf("background failure = %+v, want background code, seed 42 and no blamed take", training)
		}
		if !strings.Contains(training.Error(), "фоном") {
			t.Fatalf("background failure does not explain the cause: %s", training)
		}
	}
}

func TestTrainingQualityErrorKeepsOriginalHeldOutTakeIndices(t *testing.T) {
	// Original takes 0, 2 and 5 were unusable. Held-out usable takes 1 and 3
	// therefore refer to original takes 3 and 6, not display positions 1 and 3.
	original := []int{1, 3, 4, 6}
	check := []int{1, 3}
	err := trainingQualityError(calibration{Feasible: true}, original, check, 1234)
	var training *TrainingError
	if !errors.As(err, &training) {
		t.Fatalf("got %T %v, want structured training error", err, err)
	}
	if training.Code != "word_not_recognized" || training.Seed != 1234 || !slices.Equal(training.TakeIndices, []int{3, 6}) {
		t.Fatalf("word failure = %+v, want original indices [3 6] and seed 1234", training)
	}
	if !strings.Contains(training.Error(), "проверочные записи") {
		t.Fatalf("word failure does not explain the cause: %s", training)
	}
	original[1], check[0] = 99, 0
	if !slices.Equal(training.TakeIndices, []int{3, 6}) {
		t.Fatalf("error take indices alias caller data: %v", training.TakeIndices)
	}
}

func TestTrainingQualityErrorAllowsFeasibleRecognizedWord(t *testing.T) {
	if err := trainingQualityError(calibration{Feasible: true, Detected: 1}, []int{0, 1, 2}, []int{1}, 42); err != nil {
		t.Fatalf("valid calibration rejected: %v", err)
	}
}
