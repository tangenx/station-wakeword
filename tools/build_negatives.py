#!/usr/bin/env python3
"""Build the bundled negative set (negatives.bin) from openWakeWord's
precomputed speech-embedding features (CC BY-NC-SA 4.0,
huggingface.co/datasets/davidscripka/openwakeword_features).

    python tools/build_negatives.py out/negatives.bin

Only byte ranges are fetched: ~3000 ACAV100M windows (multilingual speech,
music, noise from real-world video) and ~53 min of the false-positive
validation stream (dinner parties, conversations, reverberant music).
Format read by kws.ReadNegatives: "NEG1" u32 nWindows u32 nStreams
u32 streamLen[nStreams], then float16 windows (n,16,96) and streams (len,96).
"""
import ast
import struct
import sys
import urllib.request

import numpy as np

BASE = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
ACAV = BASE + "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VAL = BASE + "validation_set_features.npy"


def fetch(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if len(data) != length:
        raise RuntimeError(f"short read {len(data)} != {length} from {url}")
    return data


def npy_header(url):
    head = fetch(url, 0, 512)
    assert head[:6] == b"\x93NUMPY"
    hlen = struct.unpack("<H", head[8:10])[0]
    meta = ast.literal_eval(head[10 : 10 + hlen].decode())
    return 10 + hlen, meta


def main(out):
    rng = np.random.default_rng(20261026)

    off, meta = npy_header(ACAV)
    n, ctx, dim = meta["shape"]
    assert (ctx, dim) == (16, 96) and meta["descr"] == "<f2"
    row = ctx * dim * 2
    chunks, per = 60, 50
    windows = []
    for s in np.sort(rng.choice(n // per, chunks, replace=False)):
        raw = fetch(ACAV, off + int(s) * per * row, per * row)
        windows.append(np.frombuffer(raw, "<f2").reshape(per, ctx, dim))
    windows = np.concatenate(windows)

    off, meta = npy_header(VAL)
    n, dim = meta["shape"]
    assert dim == 96 and meta["descr"] == "<f4"
    seg = 10000
    streams = []
    for s in np.sort(rng.choice(n // seg, 4, replace=False)):
        raw = fetch(VAL, off + int(s) * seg * dim * 4, seg * dim * 4)
        streams.append(np.frombuffer(raw, "<f4").reshape(seg, dim).astype("<f2"))

    with open(out, "wb") as f:
        f.write(b"NEG1")
        f.write(struct.pack("<II", len(windows), len(streams)))
        f.write(struct.pack(f"<{len(streams)}I", *[len(s) for s in streams]))
        f.write(windows.astype("<f2").tobytes())
        for s in streams:
            f.write(s.tobytes())
    print(f"{out}: {len(windows)} windows, {sum(len(s) for s in streams)} stream embeddings")


if __name__ == "__main__":
    main(sys.argv[1])
