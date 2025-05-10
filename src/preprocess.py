#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import keras
import keras_tuner
import math


def SingleSTFT_PreprocessingLayer(trial: keras_tuner.HyperParameters = None, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    return keras.layers.Pipeline([
        keras.layers.Reshape((None, 1)),
        keras.layers.STFTSpectrogram(
            mode="log",
            frame_length=SAMPLE_RATE * trial.Int("frame_length_ms", 20, 100) // 1000,
            frame_step=SAMPLE_RATE * trial.Int("frame_step_ms", 10, 80) // 1000,
            fft_length=2048,
            trainable=False,
        )
    ])


def MultiSTFT_PreprocessingLayer(hp = keras_tuner.HyperParameters, SAMPLE_RATE: int = 22500, **kwargs):
    """
    A custom Keras layer for preprocessing audio data.
    """
    start_frame_length = hp.Int("start_frame_length_ms", 30, 70, step = 5)
    end_frame_length = hp.Int("end_frame_length_ms", 70, 100, step = 5)
    frame_step = hp.Int("frame_step", 5, 25, step=5)
    
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
    return keras.layers.Pipeline([
        keras.layers.Reshape((None, )),
        keras.layers.MelSpectrogram(
            sampling_rate=SAMPLE_RATE,
            fft_length=2048,
            sequence_stride=512,
            num_mel_bins=hp.Int("num_mel_bins", 64, 256, sampling="log"),
        ),
    ])
