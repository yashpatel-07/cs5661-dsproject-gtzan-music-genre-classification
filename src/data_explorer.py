#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import keras
import keras_tuner

from matplotlib.widgets import Button, TextBox

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

dataset = pd.read_feather(Path(parsed_args.data_dir) / "audio.feather")

# Load a sample audio file
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
mel, stft, multi_stft = axs
mel.set_title("Mel Spectrogram")
stft.set_title("STFT")
multi_stft.set_title("Multi STFT")

def get_audio_data(i):
    audio = dataset.iloc[i]["audio"]
    audio = np.expand_dims(audio, axis=0)
    return audio

fig.subplots_adjust(bottom=0.2)
mel_img = mel.imshow(models[0](get_audio_data(0)).numpy()[0, :, :], aspect='auto')
stft_img = stft.imshow(models[1](get_audio_data(0)).numpy()[0, :, :], aspect='auto')
multi_stft_img = multi_stft.imshow(models[2](get_audio_data(0)).numpy()[0, :, :], aspect='auto')

def update_imgs(indx):
    data = get_audio_data(indx)
    plt.suptitle(f"Audio {indx} - {dataset.iloc[indx]['genre']}")
    mel_img.set_array(models[0](data).numpy()[0, :, :])
    stft_img.set_array(models[1](data).numpy()[0, :, :] * 255)
    multi_stft_img.set_array(models[2](data).numpy()[0, :, :])
    plt.draw()

class Index:
    ind = 0
    textbox = None

    def next(self, event):
        self.ind += 1
        i = self.ind % len(dataset)
        update_imgs(i)
        self.textbox.set_val(str(self.ind))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(dataset)
        update_imgs(i)
        self.textbox.set_val(str(self.ind))
        plt.draw()
    def submit(self, text):
        if self.ind == int(text):
            return # No change

        self.ind = int(text)
        if self.ind < 0:
            self.ind = 0
        elif self.ind >= len(dataset):
            self.ind = len(dataset) - 1
        update_imgs(self.ind)
        plt.draw()

callback = Index()
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
axbox = fig.add_axes([0.1, 0.05, 0.5, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
text_box = TextBox(axbox, "Index", textalignment="center")
text_box.on_submit(callback.submit)
callback.textbox = text_box
text_box.set_val("1")  # Trigger `submit` with the initial string.


plt.show()