import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import tensorflow_addons as tfa
import yaml

from callbacks import DecayHistory, KerasCheckpoint, OnnxCheckpoint
from generate_datasets import load_dataset
from models import NeMoCoModel, load_model
from training_parameters import DatasetConfig, TrainingParameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--name", type=str, help="optional name to give")
    parser.add_argument("--restart", type=Path, help="restart the training with a pretrained model")
    parser.add_argument("--optimize-train", action="store_true", help="use loss instead of validation loss for best models")
    parser.add_argument("--configure", action="store_true", help="open a default train configuration and edit it")
    parser.add_argument("--load-config", type=Path, nargs="*", default=None)
    args = parser.parse_args()

    c = DatasetConfig.from_yaml(args.dataset_directory / "dataset_config.yaml")

    # instantiate model
    if args.load_config is not None:
        if args.load_config:
            config_path = args.load_config
        else:
            # get latest config file
            checkpoint_dir = Path("checkpoints")
            latest_checkpoint_dir = max(checkpoint_dir.iterdir(), key=os.path.getctime)
            config_path = latest_checkpoint_dir / "train_config.yaml"
        p = TrainingParameters.from_yaml(config_path)
    else:
        p = TrainingParameters(
            dataset_config=c,
            num_experts=4,
            batch_size=128,
            gating_layer_units=[32, 32, 32],
            expert_layer_units=[512, 512, 512],
            dropout_prob=0.5,
            optimizer="AdaBelief",
        )

    if args.configure:
        EDITOR = os.environ.get('EDITOR', 'vim')
        with NamedTemporaryFile(prefix='config_', mode='w', suffix='.yaml') as temp:
            p.to_yaml(temp.name)
            ret = subprocess.call([EDITOR, temp.name])
            if ret != 0:
                sys.exit("Cancelled training")
            p = TrainingParameters.from_yaml(temp.name)

    if args.restart:
        print(f"retraining checkpoint {args.restart}")
        p = TrainingParameters.from_yaml(args.restart.parent / "train_config.yaml")
        model = load_model(args.restart)
    else:
        # instantiate model
        model = NeMoCoModel(
            p.gating_input_features, p.gating_layer_units,
            p.expert_input_features, p.expert_layer_units,
            p.num_experts, p.expert_output_features,
            p.dropout_prob
        )
    model.summary()

    # handle optimizer initalization
    optimizer = p.optimizer(**p.optimizer_settings)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # pick dataset and prepare subsets for training
    dataset_dir = args.dataset_directory
    train_ds = load_dataset(dataset_dir / "train.tfrecords", p.batch_size, is_train=True)
    test_ds = load_dataset(dataset_dir / "test.tfrecords", p.batch_size)
    val_ds = load_dataset(dataset_dir / "val.tfrecords", p.batch_size)

    # define callbacks used during training
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.name:
        name = args.name
    else:
        pattern = r"(?:\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_)(.*)|$"  # matches YYYY-MM-DD_hh_mm
        name = re.findall(pattern, c.name)[0]

    monitor = "loss" if args.optimize_train else "val_loss"
    file_stem = f"ep-{{epoch:05d}}_vl-{{{monitor}:.5f}}"
    train_dir_name = f"{date_str}_{name.upper()}"
    train_dir = Path("checkpoints") / train_dir_name
    cloud_dir = Path("/cloud/checkpoints") / train_dir_name
    train_dir.mkdir(exist_ok=True, parents=True)
    cloud_dir.mkdir(exist_ok=True, parents=True)

    keras_filepath = train_dir / f"{file_stem}.h5"
    keras_checkpoint_cb = KerasCheckpoint(filepath=keras_filepath, save_best_only=True, monitor=monitor)
    onnx_filepath = cloud_dir / f"{file_stem}.onnx"
    onnx_checkpoint_cb = OnnxCheckpoint(filepath=onnx_filepath, save_best_only=True, monitor=monitor)

    log_dir = train_dir / "logs"
    tensorboard_cb = TensorBoard(log_dir=log_dir)

    p.to_yaml(train_dir / "train_config.yaml")
    c.to_yaml(train_dir / "dataset_config.yaml")

    p.to_yaml(cloud_dir / "train_config.yaml")
    c.to_yaml(cloud_dir / "dataset_config.yaml")

    # copy norming information to cloud directory
    norm_files = args.dataset_directory.glob("norm_*.*")
    for filepath in norm_files:
        shutil.copy(filepath, cloud_dir / filepath.name)
    test_sequence_files = args.dataset_directory.glob("test_sequence*.*")
    for filepath in test_sequence_files:
        shutil.copy(filepath, cloud_dir / filepath.name)

    # accumulate all callbacks
    callbacks=[
        keras_checkpoint_cb,
        onnx_checkpoint_cb,
        tensorboard_cb
    ]

    # compile model and start training
    model.fit(
        train_ds,
        batch_size=p.batch_size,
        epochs=p.num_epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks
    )
