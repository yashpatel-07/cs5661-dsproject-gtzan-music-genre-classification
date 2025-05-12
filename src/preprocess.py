#!/usr/bin/env python
# # -*- coding: utf-8 -*-

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


def MultiSTFT_PreprocessingLayer(hp = keras_tuner.HyperParameters, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    start_frame_length = hp.Int("start_frame_length_ms", 30, 70, step = 5)
    end_frame_length = hp.Int("end_frame_length_ms", 70, 100, step = 5)
    frame_step = hp.Int("frame_step", 5, 25, step=5)

    if start_frame_length is None:
        start_frame_length = 30
    if end_frame_length is None:
        end_frame_length = 70
    if frame_step is None:
        frame_step = 5
    if start_frame_length < frame_step:
        start_frame_length = frame_step + 10
    if end_frame_length < frame_step:
        end_frame_length = frame_step + 10
    if start_frame_length > end_frame_length:
        end_frame_length = start_frame_length + 10
    
    return lambda input_layer: keras.layers.Concatenate(axis=-1)([
        keras.layers.Pipeline([
            keras.layers.Reshape(input_layer.shape[1:] + (1, )),
            # keras.layers.MaxPooling1D(pool_size=100),
            # keras.layers.Reshape((input_layer.input.shape[0] // 100, ) + (1, )),
            keras.layers.STFTSpectrogram(
                mode="log",
                frame_length=SAMPLE_RATE * frame_size // 1000,
                frame_step=SAMPLE_RATE * frame_step // 1000,
                padding="same",
                fft_length=2 ** math.ceil(math.log2(SAMPLE_RATE * end_frame_length // 1000)),
                trainable=False,
                expand_dims=True,
                name=f"STFT_{frame_size}",
            )
        ],
        name=f"STFT_Pipeline_{frame_size}")(input_layer)
        for frame_size in [start_frame_length, (start_frame_length + end_frame_length) // 2, end_frame_length]
    ])


def MelSpectrogram_PreprocessingLayer(hp: keras_tuner.HyperParameters = None, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    mel_bins = hp.Int("num_mel_bins", 64, 1024, default=128, sampling="log")
    
    if mel_bins is None:
        mel_bins = 128
    
    return keras.layers.Pipeline([
        keras.layers.Reshape((None, )),
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
    def compute_output_shape( self, input_shape ) :
        print(input_shape)
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
        return lambda x: RGBFaker()(layer(x))
    else:
        return layer