package quasar

import (
	"bytes"
	"encoding/hex"
	"net"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// Frames captured from the stock `maind --cli` / accepted live by maind.
// A wrong field number or length byte order is silently
// ignored by maind, so pin the exact bytes.
func TestStartConversationFrame(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	got := make(chan []byte, 1)
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		var buf [64]byte
		n, _ := c.Read(buf[:])
		got <- buf[:n]
	}()
	if err := CallAlice(ln.Addr().(*net.TCPAddr).Port, StartConversation); err != nil {
		t.Fatal(err)
	}
	want, _ := hex.DecodeString("1c000000e20f153a0f416c6963654361706162696c69747942020802e80fd201")
	if b := <-got; !bytes.Equal(b, want) {
		t.Fatalf("frame\n got %x\nwant %x", b, want)
	}
}

func TestReadAudioFrame(t *testing.T) {
	// IOAudioChannelsData with one vqe_0 chunk of two samples (1, -2).
	ch := protowire.AppendTag(nil, 2, protowire.BytesType)
	ch = protowire.AppendString(ch, "vqe_0")
	ch = protowire.AppendTag(ch, 3, protowire.BytesType)
	ch = protowire.AppendBytes(ch, []byte{1, 0, 0xfe, 0xff})
	ch = protowire.AppendTag(ch, 7, protowire.VarintType)
	ch = protowire.AppendVarint(ch, 16000)
	body := protowire.AppendTag(nil, 1, protowire.BytesType)
	body = protowire.AppendBytes(body, ch)
	var wire bytes.Buffer
	if err := writeFrame(&wire, memberAudioData, body); err != nil {
		t.Fatal(err)
	}
	var buf []byte
	member, b, err := readFrame(&wire, &buf)
	if err != nil || member != memberAudioData || !bytes.Equal(b, body) {
		t.Fatalf("readFrame: member %d err %v", member, err)
	}
	inner, _ := protowire.ConsumeBytes(b[1:])
	name, data, ok := channelData(inner)
	if !ok || name != "vqe_0" || !bytes.Equal(data, []byte{1, 0, 0xfe, 0xff}) {
		t.Fatalf("channelData: %q %x %v", name, data, ok)
	}
}
