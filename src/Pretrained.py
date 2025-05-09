#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import keras_tuner
from preprocess import MultiSTFT_PreprocessingLayer, SingleSTFT_PreprocessingLayer, MelSpectrogram_PreprocessingLayer
# import optuna

def PretrainedModel(hp: keras_tuner.HyperParameters):
    """
    A custom Keras model for audio classification.
    """
    
    spectogram_type = hp.Choice("spectrogram_type", ["MultiSTFT"])
    
    image_model = hp.Choice("image_model", ["MobileNet", "ResNet50"])
    
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
    else:
        raise ValueError(f"Unknown image model: {image_model}")
    
    if spectogram_type == "SingleSTFT":
        preprocessing_layer = SingleSTFT_PreprocessingLayer(hp)
    elif spectogram_type == "MultiSTFT":
        preprocessing_layer = MultiSTFT_PreprocessingLayer(hp)
    elif spectogram_type == "MelSpectrogram":
        preprocessing_layer = MelSpectrogram_PreprocessingLayer(hp)
    else:
        raise ValueError(f"Unknown spectrogram type: {spectogram_type}")
    
    hidden_size = hp.Int("hidden_size", 128, 512, step = 64)
    
    input = keras.layers.Input((None, ))

    x = preprocessing_layer(input)
    # x = keras.layers.Reshape((None, None, 3))(x)
    x = image_model(x)
    # x = keras.layers.Flatten()(x)


    pipeline = keras.layers.Pipeline([
        keras.layers.Dropout(rate=0.5),

        keras.layers.Dense(units=hidden_size, activation="relu"),
        keras.layers.Dense(units=hidden_size, activation="relu"),
        
        keras.layers.Dense(units=10, activation="softmax"),
    ], name="Pretrained_Model_Pipeline")(x)
    
    model = keras.models.Model(inputs=input, outputs=pipeline)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

if __name__ == "__main__":
    import numpy as np
    # hyperparams = {'spectrogram_type': 'MultiSTFT', 'image_model': 'MobileNet', 'start_frame_length_ms': 35, 'end_frame_length_ms': 55, 'frame_length_step': 10, 'frame_step': 0.4603676629120844, 'hidden_size': 256, 'learning_rate': 2.0116853501886966e-05}
    # hyperparams = optuna.trial.FixedTrial(hyperparams)
    # model = PretrainedModel(hyperparams, (220500, ))
    # model.fit(GTZAN_Dataset('/Users/eliasschablowski/Desktop/CSULA/5661/p/output'))
    from pathlib import Path
    
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study_name = "example-study"  # Unique identifier of the study.
    # storage_name = "sqlite:///{}.db".format(study_name)
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    
    # def runTrial(trial: optuna.Trial):
    #     model: keras.Model = PretrainedModel(trial, (220500, ))
    #     model.fit(GTZAN_Dataset('/Users/eliasschablowski/Desktop/CSULA/5661/p/output'), epochs=trial.suggest_int("epochs", 10, 100, log=True))
    #     score = model.evaluate(GTZAN_Dataset('/Users/eliasschablowski/Desktop/CSULA/5661/p/output'), verbose=0)
    #     return score[1]
    
    # study.optimize(
    #     runTrial,
    #     n_trials=100,
    #     timeout=600,
    # )

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
    parsed_args = parser.parse_args()


    dataset = pd.read_feather(parsed_args.data_dir / "audio.feather")
    tuner = keras_tuner.Hyperband(
        lambda x: PretrainedModel(x),
        objective='val_loss',
        max_epochs=20,
        directory=Path.cwd() / "hyper_parameter_tuning",
        project_name="Pretrained"
    )
    
    dataset["genre_id"] = dataset["genre"].astype("category").cat.codes
    
    print(dataset["genre_id"])
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(np.stack(dataset["audio"]), dataset["genre_id"].to_numpy(), test_size=0.2, random_state=42, stratify=dataset["genre_id"].to_numpy())
    
    tuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=4)

    # hp = keras_tuner.HyperParameters()
    # for x, y in {
    #     "spectrogram_type": "MultiSTFT",
    #     "image_model": "MobileNet",
    #     "start_frame_length_ms": 30,
    #     "end_frame_length_ms": 50,
    #     "frame_step": 15,
    #     "hidden_size": 256,
    #     "learning_rate": 1e-2
    # }.items():
    #     hp.Fixed(x, y)

    
    # model = PretrainedModel(hp)
    # model.fit(np.stack(dataset["audio"]), np.array([0 for x in range(len(dataset))]), epochs=10, validation_split=0.2)