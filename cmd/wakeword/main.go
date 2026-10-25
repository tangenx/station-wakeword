// Command wakeword adds user-trained wake words to Yandex Stations.
package main

import (
	"fmt"
	"os"
	"runtime/debug"
)

const usage = `usage:
  wakeword run [flags]        station daemon: detect words, web UI, training
  wakeword train [flags]      train a word from WAV files
  wakeword scan model.json... audio.wav
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	// Minis have ~120 MB free next to maind; training holds ~45 MB live, so
	// keep the GC from letting the heap double on top of that.
	if os.Getenv("GOMEMLIMIT") == "" {
		debug.SetMemoryLimit(64 << 20)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = cmdRun(os.Args[2:])
	case "train":
		err = cmdTrain(os.Args[2:])
	case "scan":
		err = cmdScan(os.Args[2:])
	default:
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "wakeword:", err)
		os.Exit(1)
	}
}
