#!/usr/bin/env python
# # -*- coding: utf-8 -*-

from typing import List
import keras
import keras_tuner
import math

# from keras import backend as K


def SingleSTFT_PreprocessingLayer(trial: keras_tuner.HyperParameters = None, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    
    frame_length_ms = trial.Int("frame_length_ms", 80, 200, default=10)
    frame_step_ms = trial.Int("frame_step_ms", 10, 80, default=30)
    
    if frame_length_ms is None:
        frame_length_ms = 80
    if frame_step_ms is None:
        frame_step_ms = 10
    if frame_length_ms < frame_step_ms:
        frame_length_ms = frame_step_ms + 10
    
    return keras.layers.Pipeline([
        keras.layers.Reshape((None, 1)),
        keras.layers.STFTSpectrogram(
            mode="log",
            frame_length=SAMPLE_RATE * frame_length_ms // 1000,
            frame_step=SAMPLE_RATE * frame_step_ms // 1000,
            fft_length=2 ** math.ceil(math.log2(SAMPLE_RATE * frame_length_ms // 1000)),
            trainable=False,
            expand_dims=False,
        )
    ], name="STFT_Spectogram")

class MultiSTFT(keras.layers.Layer):
    """
    A custom Keras layer for computing multiple STFT of an audio signal.
    """
    def __init__(self, frame_lengths: List[int], frame_step: int, SAMPLE_RATE: int, **kwargs):
        
        # Remove any STFT-specific arguments from kwargs, since keras.Layer does not accept them
        stft_args = kwargs.copy()
        stft_args.pop("frame_length", None)
        stft_args.pop("frame_step", None)
        stft_args.pop("fft_length", None)
        stft_args.pop("mode", None)
        stft_args.pop("trainable", None)
        stft_args.pop("expand_dims", None)
        stft_args.pop("name", None)
        super(MultiSTFT, self).__init__(**stft_args)
        self.frame_lengths = frame_lengths
        self.frame_step = frame_step
        self.SAMPLE_RATE = SAMPLE_RATE
        self.stft_layers = [
            keras.layers.STFTSpectrogram(
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=2 ** math.ceil(math.log2(max(frame_lengths))),
                padding="same",
                **kwargs,
            ) for frame_length in frame_lengths
        ]

    def call(self, inputs, training=None):
        x = keras.ops.expand_dims(inputs, axis=-1)
        spectrograms = [spectrogram(x, training=training) for spectrogram in self.stft_layers]
        spectrograms = [keras.ops.expand_dims(spectrogram, axis=-1) for spectrogram in spectrograms]
        return keras.ops.concatenate(spectrograms, axis=-1)

    def build(self, input_shape):
        for stft in self.stft_layers:
            stft.build(input_shape + (1, ))
        super(MultiSTFT, self).build(input_shape)
    def get_config(self):
        config = super(MultiSTFT, self).get_config()
        config.update({
            "frame_length_ms": self.frame_lengths,
            "frame_step_ms": self.frame_step,
        })
        return config
    def compute_output_shape(self, input_shape):
        output_shapes = [stft.compute_output_shape(input_shape + (1, )) for stft in self.stft_layers]
        return output_shapes[0] + (3, )
    def compute_length_from_ms(self, length: int):
        return int(length * self.SAMPLE_RATE / 1000)

def MultiSTFT_PreprocessingLayer(hp = keras_tuner.HyperParameters, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    frame_lengths = [hp.Int(f"frame_length_ms_{i}", 30, 250, step = 5) for i in range(3)]
    frame_step = hp.Int("frame_step", 10, 50, step=5)
    
    frame_lengths = [(frame_length if frame_length is not None else 30) for frame_length in frame_lengths]
    if frame_step is None:
        frame_step = 5
    
    frame_lengths = [SAMPLE_RATE * length // 1000 for length in frame_lengths]

    if frame_step > min(frame_lengths):
        frame_step = min(frame_lengths) - 1
    
    return MultiSTFT(
        frame_lengths=frame_lengths,
        frame_step=frame_step,
        SAMPLE_RATE=SAMPLE_RATE,
        mode="log",
        trainable=False,
        expand_dims=False,
        **kwargs,
    )


def MelSpectrogram_PreprocessingLayer(hp: keras_tuner.HyperParameters = None, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    mel_bins = hp.Int("num_mel_bins", 64, 1024, default=128, sampling="log")
    
    if mel_bins is None:
        mel_bins = 128
    
    return keras.layers.Pipeline([
        # keras.layers.Reshape((None, )),
        keras.layers.MelSpectrogram(
            sampling_rate=SAMPLE_RATE,
            fft_length=2048,
            sequence_stride=512,
            num_mel_bins=mel_bins,
        ),
    ], name="Mel_Spectrogram")


class RGBFaker( keras.Layer ) :
    def call( self, inputs ) :
        # expand your input from gray scale to rgb
        # if your inputs.shape = (None,None,1)
        if len(inputs.shape) == 3:
            x = keras.ops.expand_dims( inputs, axis=-1 )
        else:
            x = inputs
        fake_rgb = keras.ops.concatenate([x for _ in range(3)], axis=-1)
        return fake_rgb
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (3,)

def HyperPreprocessingLayer(hp: keras_tuner.HyperParameters = None, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    
    style = hp.Choice("PreprocessingLayer", [
        "MelSpectrogram",
        "SingleSTFT",
        "MultiSTFT",
    ])
    
    if style == "MelSpectrogram":
        with hp.conditional_scope("PreprocessingLayer", "MelSpectrogram"):
            layer = MelSpectrogram_PreprocessingLayer(hp, SAMPLE_RATE)
    elif style == "SingleSTFT":
        with hp.conditional_scope("PreprocessingLayer", "SingleSTFT"):
            layer = SingleSTFT_PreprocessingLayer(hp, SAMPLE_RATE)
    elif style == "MultiSTFT":
        with hp.conditional_scope("PreprocessingLayer", "MultiSTFT"):
            layer = MultiSTFT_PreprocessingLayer(hp, SAMPLE_RATE)
    else:
        raise ValueError(f"Unknown PreprocessingLayer: {style}")
    
    if style != "MultiSTFT":
        return keras.layers.Pipeline([
            layer,
            RGBFaker(),
        ], name="PreprocessingLayer")
    else:
        return layer