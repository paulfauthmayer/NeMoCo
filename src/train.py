import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow_addons as tfa
from generate_datasets import load_dataset

from callbacks import KerasCheckpoint, OnnxCheckpoint
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
    args = parser.parse_args()

    c = DatasetConfig().from_yaml(args.dataset_directory / "dataset_config.yaml")

    # instantiate model
    if args.restart:
        print(f"retraining checkpoint {args.restart}")
        p = TrainingParameters(1, 1, 1).from_yaml(args.restart.parent / "train_config.yaml")
        model = load_model(args.restart)
    else:
        p = TrainingParameters(
            dataset_config=c,
            num_experts=4,
            batch_size=128,
            gating_layer_units=[32, 32, 32],
            expert_layer_units=[512, 512, 512],
            dropout_prob=0.5,
            optimizer="AdamW"
        )

        # instantiate model
        model = NeMoCoModel(
            p.gating_input_features, p.gating_layer_units,
            p.expert_input_features, p.expert_layer_units,
            p.num_experts, p.expert_output_features,
            p.dropout_prob
        )
    model.summary()

    # handle optimizer initalization
    if p.optimizer.__name__ == "AdamW":
        lr_schedule = tf.optimizers.schedules.ExponentialDecay(1e-4, 100, 0.9)
        wd_schedule = tf.optimizers.schedules.ExponentialDecay(5e-5, 100, 0.9)
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=lambda : None)
        optimizer.weight_decay = lambda : wd_schedule(optimizer.iterations)
    else:
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

    if args.optimize_train:
        monitor = "loss"
        file_stem = "ep-{epoch:05d}_tl-{loss:.5f}"
    else:
        monitor = "val_loss"
        file_stem = "ep-{epoch:05d}_vl-{val_loss:.5f}"
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

    # compile model and start training
    model.fit(
        train_ds,
        batch_size=p.batch_size,
        epochs=p.num_epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[
            keras_checkpoint_cb,
            onnx_checkpoint_cb,
            tensorboard_cb
        ],
    )
