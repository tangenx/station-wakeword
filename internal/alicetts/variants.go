package alicetts

import (
	"errors"
	"strings"
	"unicode"
)

// Variant is one bounded, deterministic pronunciation preset for a plain word.
type Variant struct {
	ID   string
	Text string
}

// Variants includes the original Alice voice, six alternative voices, and one
// quieter/slower pronunciation. Only plain single words may enter TTS markup.
func Variants(word string) ([]Variant, error) {
	word = strings.TrimSpace(word)
	runes := []rune(word)
	if len(runes) < 2 || len(runes) > 40 {
		return nil, errors.New("alice TTS: enter one word of 2–40 letters")
	}
	for i, r := range runes {
		if unicode.IsLetter(r) {
			continue
		}
		if r == '-' && i > 0 && i+1 < len(runes) && unicode.IsLetter(runes[i-1]) && unicode.IsLetter(runes[i+1]) {
			continue
		}
		return nil, errors.New("alice TTS: use one plain word without spaces, URLs or markup")
	}
	variants := []Variant{{ID: "alice", Text: word}}
	for _, voice := range []string{"alyss", "jane", "ermil", "zahar", "kostya", "oksana"} {
		variants = append(variants, Variant{ID: voice, Text: `<speaker voice="` + voice + `">` + word})
	}
	return append(variants, Variant{ID: "alice-whisper-slow", Text: `<speaker speed="0.9" is_whisper="true">` + word}), nil
}
