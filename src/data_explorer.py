#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import keras
import keras_tuner

from matplotlib.widgets import Button

from preprocess import MelSpectrogram_PreprocessingLayer, MultiSTFT_PreprocessingLayer, SingleSTFT_PreprocessingLayer
import pandas as pd

args = argparse.ArgumentParser(description="Audio Data Explorer")
args.add_argument(
    "--data_dir",
    type=str,
    default=Path.cwd() / "output",
    help="Directory containing the audio data.",
)

parsed_args = args.parse_args()

def gen_model(model):
    input_layer = keras.layers.Input(shape=(None,))
    return keras.Model(inputs=input_layer, outputs=model(keras_tuner.HyperParameters())(input_layer), name=f"{model.__name__}")

models = [gen_model(layer) for layer in [
    SingleSTFT_PreprocessingLayer,
    MultiSTFT_PreprocessingLayer,
    MelSpectrogram_PreprocessingLayer
]]

dataset = pd.read_feather(parsed_args.data_dir / "audio.feather")

# Load a sample audio file
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
mel, stft, multi_stft = axs
mel.set_title("Mel Spectrogram")
stft.set_title("STFT")
multi_stft.set_title("Multi STFT")

fig.subplots_adjust(bottom=0.2)
mel_img, = mel.imshow(models[0](dataset[0][0]).numpy()[0, :, :, 0], aspect='auto')
stft_img, = stft.imshow(models[1](dataset[0][0]).numpy()[0, :, :, 0], aspect='auto')
multi_stft_img, = multi_stft.imshow(models[2](dataset[0][0]).numpy()[0, :, :, 0], aspect='auto')

def update_imgs(data):
    mel_img.set_array(models[0](data[0]).numpy()[0, :, :, 0])
    stft_img.set_array(models[1](data[1]).numpy()[0, :, :, 0])
    multi_stft_img.set_array(models[2](data[2]).numpy()[0, :, :, 0])
    plt.draw()

class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(dataset)
        update_imgs(dataset.iloc[i][["audio", "audio"]].values)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(dataset)
        update_imgs(dataset.iloc[i][["audio", "audio"]].values)
        plt.draw()

callback = Index()
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()