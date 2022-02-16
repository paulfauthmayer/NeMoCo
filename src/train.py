import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow_addons import optimizers
from generate_datasets import load_dataset

from callbacks import OnnxCheckpointCallback
from models import NeMoCoModel, load_model
from training_parameters import DatasetConfig, TrainingParameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--name", type=str, help="optional name to give")
    parser.add_argument("--restart", type=Path, help="restart the training with a pretrained model")
    args = parser.parse_args()

    c = DatasetConfig().from_yaml(args.dataset_directory / "dataset_config.yaml")
    if args.restart:
        print(f"retraining checkpoint {args.restart}")
        p = TrainingParameters(1, 1, 1).from_yaml(args.restart.parent /
                                                  "train_config.yaml")
        model = load_model(args.restart)
    else:
        p = TrainingParameters(
            dataset_config=c,
            num_experts=4,
            batch_size=128,
            gating_layer_units=[32, 32, 32],
            expert_layer_units=[512, 512, 512],
            dropout_prob=0.5
        )

        # instantiate model
        model = NeMoCoModel(
            p.gating_input_features, p.gating_layer_units,
            p.expert_input_features, p.expert_layer_units,
            p.num_experts, p.expert_output_features,
            p.dropout_prob
        )
    model.summary()
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
        pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_(.*)"  # matches YYYY-MM-DD_hh_mm
        name = next(re.finditer(pattern, c.name), "").group(0)

    train_dir_name = f"{date_str}_{name.upper()}"
    file_stem = "ep-{epoch:02d}_vl-{val_loss:.5f}"
    train_dir = Path("checkpoints") / train_dir_name
    cloud_dir = Path("/cloud/checkpoints") / train_dir_name
    train_dir.mkdir(exist_ok=True, parents=True)
    cloud_dir.mkdir(exist_ok=True, parents=True)

    keras_filepath = train_dir / f"{file_stem}.h5"
    keras_checkpoint_cb = ModelCheckpoint(filepath=keras_filepath, save_best_only=True, verbose=True)
    onnx_filepath = cloud_dir / f"{file_stem}.onnx"
    onnx_checkpoint_cb = OnnxCheckpointCallback(filepath=onnx_filepath, save_best_only=True)

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
