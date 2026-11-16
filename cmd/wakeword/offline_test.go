package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestTakeFilesExcludeNegativeRecordings(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.wav", "b.wav", "negative-01.wav", "false-1.wav", "ambient.wav", "speech.wav", "._a.wav"} {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	paths, err := takeFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, path := range paths {
		names = append(names, filepath.Base(path))
	}
	if !slices.Equal(names, []string{"a.wav", "b.wav"}) {
		t.Fatalf("non-word audio was included as positive takes: %v", names)
	}
}
