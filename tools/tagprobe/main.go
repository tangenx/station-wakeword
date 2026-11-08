package main

// tagprobe sends one AliceCapability START_CONVERSATION to the local aliced,
// optionally with a wake-word tag, so the resulting uniproxy request can be
// inspected. Development aid for the tag_alice_wakeword integration; not part
// of the released binary.
//
//	./tagprobe <aliced-port> [word]

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"wakeword/internal/quasar"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: tagprobe <aliced-port> [word]")
		os.Exit(2)
	}
	port, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	word := ""
	if len(os.Args) > 2 {
		word = os.Args[2]
	}
	if err := quasar.CallAlice(port, quasar.StartConversation, word); err != nil {
		fmt.Fprintln(os.Stderr, "start:", err)
		os.Exit(1)
	}
	fmt.Printf("started (word=%q)\n", word)
	time.Sleep(3 * time.Second)
	if err := quasar.CallAlice(port, quasar.StopConversation, ""); err != nil {
		fmt.Fprintln(os.Stderr, "stop:", err)
		os.Exit(1)
	}
	fmt.Println("stopped")
}
