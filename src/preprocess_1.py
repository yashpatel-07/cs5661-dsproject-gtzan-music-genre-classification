#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import scipy.io.wavfile as wav
import numpy as np
import pathlib
from scipy.signal import resample
import tqdm
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description="Preprocess audio files.")
parser.add_argument("--data_dir", type=str, default=pathlib.Path.cwd() / "data",
                    help="Directory containing audio files.")
parser.add_argument("--target_sr", type=int, default=22050,
                    help="Target sample rate.")
parser.add_argument("--target_length", type=int,
                    default=10, help="Target sample length.")
parser.add_argument("--sample_multiplier", type=int, default=1,
                    help="Number of samples to split the inputs into.")
parser.add_argument("--mmap", action="store_true",
                    help="Use memory mapping for large files.")
parser.add_argument("--output_dir", type=str, default=pathlib.Path.cwd() / "output",
                    help="Directory to save the processed files.")

args = parser.parse_args()

data_dir = pathlib.Path(args.data_dir)
output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

id = itertools.count()
audio_df = pd.DataFrame(columns=["genre", "audio"])

for file in tqdm.tqdm(data_dir.rglob("*.wav")):
    sr, audio = wav.read(file, mmap=args.mmap)

    # remap to -1 to 1
    if np.iinfo(audio.dtype).min < 0:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        audio = (audio - 0.5) * 2

    # Resample if necessary
    n_samples = int(len(audio) * args.target_sr / sr)  # resample to 16 kHz
    if sr != args.target_sr:
        audio = resample(audio, int(n_samples))

    # Pad or truncate the audio to the desired length
    target_length = args.target_length * args.target_sr

    # Split the audio into smaller segments
    segment_length = target_length
    stride = len(audio) // (args.sample_multiplier + 1)
    segments = np.lib.stride_tricks.sliding_window_view(audio, segment_length)[
        ::stride, :]

    # Save each segment as a separate file
    for i, segment in enumerate(segments):
        audio_df = pd.concat([pd.DataFrame([{
            "genre": file.parent.name,
            "audio": segment.astype(np.float32)
        }]), audio_df], ignore_index=True)


audio_df["genre"] = audio_df["genre"].astype("category")
audio_df["audio"] = audio_df["audio"].astype("object")
audio_df["audio"] = audio_df["audio"].apply(lambda x: x.astype("float32"))

audio_df.to_feather(output_dir / "audio.feather")
