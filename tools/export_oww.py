#!/usr/bin/env python3
"""Export openWakeWord feature models (melspectrogram + speech embedding) into
the flat tensor file read by internal/kws, plus golden vectors for its tests.

    python tools/export_oww.py melspectrogram.onnx embedding_model.onnx

Writes internal/kws/assets/oww.bin and internal/kws/testdata/golden.bin.
Needs numpy, onnx, onnxruntime.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent


def write_tensors(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"OWW1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype="<f4")
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<B", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.tobytes())


def main(mel_path, emb_path):
    mel = onnx.load(mel_path)
    mi = {t.name: numpy_helper.to_array(t) for t in mel.graph.initializer}
    tensors = [("mel.filters", mi["1.melW"])]  # (257, 32)

    emb = onnx.load(emb_path)
    ei = {t.name: numpy_helper.to_array(t) for t in emb.graph.initializer}
    convs = [n for n in emb.graph.node if n.op_type == "Conv"]
    for i, n in enumerate(convs):
        w = ei[n.input[1]]
        b = ei[n.input[2]] if len(n.input) > 2 else np.zeros(w.shape[0], np.float32)
        tensors.append((f"emb.conv{i}.w", w))  # OIHW
        tensors.append((f"emb.conv{i}.b", b))
    write_tensors(ROOT / "internal/kws/assets/oww.bin", tensors)

    # Golden vectors: 3 s of band-limited noise bursts at int16 scale.
    rng = np.random.default_rng(7)
    n = 16000 * 3
    audio = rng.normal(0, 1, n).astype(np.float32)
    env = (np.sin(np.linspace(0, 9 * np.pi, n)) > 0).astype(np.float32) * 3000 + 30
    audio = (np.convolve(audio, np.ones(4) / 4, mode="same") * env).astype(np.float32)
    audio = np.clip(np.round(audio), -32768, 32767).astype(np.float32)

    ms = ort.InferenceSession(mel_path)
    es = ort.InferenceSession(emb_path)

    def mel(x):
        return ms.run(None, {"input": x[None, :]})[0].squeeze(axis=(0, 1)) / 10 + 2

    # Streaming exactly as internal/kws does it: 480 samples of context
    # (zeros at start) + each 1280-sample step -> 8 frames; one embedding of
    # the last 76 frames per step once 76 frames exist.
    hist = np.concatenate([np.zeros(480, np.float32), audio])
    frames, embs = [], []
    for i in range(len(audio) // 1280):
        frames.append(mel(hist[i * 1280 : i * 1280 + 1760]))
        f = np.concatenate(frames)
        if len(f) >= 76:
            win = f[-76:][None, :, :, None].astype(np.float32)
            embs.append(es.run(None, {"input_1": win})[0].reshape(96))
    step = frames[-1]
    e = np.stack(embs)
    write_tensors(
        ROOT / "internal/kws/testdata/golden.bin",
        [("audio", audio), ("mel.step", step), ("emb.stream", e)],
    )
    print("mel step", step.shape, "emb", e.shape)


if __name__ == "__main__":
    main(*sys.argv[1:3])
