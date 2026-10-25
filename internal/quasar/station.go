package quasar

import (
	"encoding/json"
	"os"

	"google.golang.org/protobuf/encoding/protowire"
)

// UserConfigPath is where maind keeps the last sync config per account:
// a protobuf whose repeated field 1 is {1: passport uid, 2: config JSON}.
const UserConfigPath = "/data/quasar/data/user_config.dat"

// StationName returns the name the owner gave the station in the app
// (device_config.name of the synced config), or "" if none is stored.
// "quasar" is the firmware's placeholder before the first full sync.
func StationName(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return ""
		}
		b = b[n:]
		if num != 1 || typ != protowire.BytesType {
			if n = protowire.ConsumeFieldValue(num, typ, b); n < 0 {
				return ""
			}
			b = b[n:]
			continue
		}
		entry, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return ""
		}
		b = b[n:]
		if name := entryName(entry); name != "" && name != "quasar" {
			return name
		}
	}
	return ""
}

func entryName(b []byte) string {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return ""
		}
		b = b[n:]
		if num == 2 && typ == protowire.BytesType {
			cfg, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return ""
			}
			var c struct {
				Device struct {
					Name string `json:"name"`
				} `json:"device_config"`
			}
			if json.Unmarshal(cfg, &c) == nil {
				return c.Device.Name
			}
			return ""
		}
		if n = protowire.ConsumeFieldValue(num, typ, b); n < 0 {
			return ""
		}
		b = b[n:]
	}
	return ""
}
