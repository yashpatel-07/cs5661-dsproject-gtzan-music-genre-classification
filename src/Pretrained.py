#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import keras_tuner
from preprocess import HyperPreprocessingLayer

from Logger import TensorBoard as ConfusionMatrixCallback


def PretrainedModel(hp: keras_tuner.HyperParameters, SAMPLE_RATE: int, input_shape: tuple = (None,)):
    """
    A custom Keras model for audio classification.
    """
    image_model = hp.Choice(
        "image_model", ["MobileNet", "ResNet50", "ResNet101", "ResNet50V2", "ResNet101V2"])

    if image_model == "MobileNet":
        image_model = keras.applications.MobileNetV2(
            include_top=False,
            input_shape=(None, None, 3),
            pooling="max",
        )
    elif image_model == "ResNet50":
        image_model = keras.applications.ResNet50(
            include_top=False,
            input_shape=(None, None, 3),
            pooling="max",
        )
    elif image_model == "ResNet50V2":
        image_model = keras.applications.ResNet50V2(
            include_top=False,
            input_shape=(None, None, 3),
            pooling="max",
        )
    elif image_model == "ResNet101":
        image_model = keras.applications.ResNet101(
            include_top=False,
            input_shape=(None, None, 3),
            pooling="max",
        )
    elif image_model == "ResNet101V2":
        image_model = keras.applications.ResNet101V2(
            include_top=False,
            input_shape=(None, None, 3),
            pooling="max",
        )
    else:
        raise ValueError(f"Unknown image model: {image_model}")

    preprocessing_layer = HyperPreprocessingLayer(hp, SAMPLE_RATE)

    hidden_size = hp.Int("hidden_size", 128, 512, step=64)

    input = keras.layers.Input(input_shape, name="input")

    with keras.RematScope(
        # This is a custom scope to allow for the use of Keras' `Remat` feature
        # which allows for the reuse of intermediate tensors to save memory.

        mode="larger_than",
        output_size_threshold=1024 ** 4,
    ):
        x = preprocessing_layer(input)
        x = keras.layers.Resizing(
            height=224,
            width=224,
            interpolation="bilinear",
            name="Resizing",
        )(x) # Resizing to 224x224, the input size for most pretrained models
    # x = keras.layers.Reshape((None, None, 3))(x)
    x = image_model(x)

    hidden_layer_count = hp.Int("hidden_layers", 1, 3, step=1, default=1)
    pipeline = keras.layers.Pipeline([
        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.5),

        *[keras.layers.Dense(units=hidden_size, activation="relu") for _ in range(hidden_layer_count)],

        keras.layers.Dense(units=10, activation="softmax"),
    ], name="Pretrained_Model_Pipeline")(x)

    model = keras.models.Model(inputs=input, outputs=pipeline)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float(
            "learning_rate", 1e-5, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # return keras.models.Sequential([
    #     preprocessing_layer,
    #     image_model,
    #     keras.layers.GlobalAveragePooling2D(),
    #     keras.layers.Dropout(rate=0.5),
    #     keras.layers.Dense(units=hidden_size, activation="relu"),
    #     keras.layers.Dense(units=hidden_size, activation="relu"),
    #     keras.layers.Dense(units=10, activation="softmax"),
    # ], name="Pretrained_Model_Pipeline")

    return model


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    import pandas as pd
    import argparse
    import sklearn.model_selection

    parser = argparse.ArgumentParser(description="Audio Data Explorer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=Path.cwd() / "output",
        help="Directory containing the audio data.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="The sample rate of the processed audio files.",
        required=True
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=Path.cwd() / "logs",
        help="Directory to save the logs.",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parsed_args = parser.parse_args()

    dataset = pd.read_feather(Path(parsed_args.data_dir) / "audio.feather")
    tuner = keras_tuner.Hyperband(
        lambda x: PretrainedModel(
            x, parsed_args.sample_rate, dataset["audio"].iloc[0].shape),
        objective='val_loss',
        max_epochs=50,
        factor=3,
        hyperband_iterations=parsed_args.iterations,
        directory=Path.cwd() / "hyper_parameter_tuning",
        project_name="Pretrained",
        seed=42,
    )

    dataset["genre_id"] = dataset["genre"].astype("category").cat.codes

    print(dataset["genre_id"])

    x_train, x_test, y_train, y_test, labels_train, labels_test = \
        sklearn.model_selection.train_test_split(
            np.stack(dataset["audio"]),
            dataset["genre_id"].to_numpy(),
            dataset["genre"].to_numpy(),

            test_size=0.2,
            random_state=42,
            stratify=dataset["genre_id"].to_numpy()
        )

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=parsed_args.batch_size,
        callbacks=[
            # keras.callbacks.TensorBoard(parsed_args.log_dir),
            ConfusionMatrixCallback(
                validation_data=(x_test, y_test),
                log_dir=parsed_args.log_dir,
                batch_size=parsed_args.batch_size,
                categories=dataset["genre"].dtype,
                sample_rate=parsed_args.sample_rate,
            ),
            keras.callbacks.EarlyStopping(  # Stop `return 1` classifiers earlier
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(  # stop spiraling around optimal point
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ],
    )
