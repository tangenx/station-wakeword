package quasar

import (
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// user_config.dat as maind writes it: one entry per account; an account
// that has not synced yet carries the "quasar" placeholder.
func TestStationNameSkipsPlaceholder(t *testing.T) {
	entry := func(uid, cfg string) []byte {
		e := protowire.AppendTag(nil, 1, protowire.BytesType)
		e = protowire.AppendString(e, uid)
		e = protowire.AppendTag(e, 2, protowire.BytesType)
		e = protowire.AppendString(e, cfg)
		b := protowire.AppendTag(nil, 1, protowire.BytesType)
		return protowire.AppendBytes(b, e)
	}
	var b []byte
	b = append(b, entry("1", `{"device_config":{"name":"quasar"}}`)...)
	b = append(b, entry("2", `{"account_config":{},"device_config":{"locale":"ru-RU","name":"Станция на кухне"}}`)...)
	b = protowire.AppendTag(b, 3, protowire.Fixed32Type) // trailing fields maind also writes
	b = protowire.AppendFixed32(b, 7)
	p := filepath.Join(t.TempDir(), "user_config.dat")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := StationName(p); got != "Станция на кухне" {
		t.Fatalf("got %q", got)
	}
}
