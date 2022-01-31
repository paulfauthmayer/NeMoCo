import argparse
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow_addons import optimizers
from generate_datasets import load_dataset

from models import NeMoCoModel
from training_parameters import DatasetConfig, TrainingParameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_directory", type=Path)
    args = parser.parse_args()

    c = DatasetConfig().from_yaml(args.dataset_directory / "dataset_config.yaml")
    p = TrainingParameters(dataset_config=c)

    # instantiate model
    model = NeMoCoModel(
        p.gating_input_features,
        p.gating_layer_units,
        p.expert_input_features,
        p.expert_layer_units,
        p.num_experts,
        p.expert_output_features,
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
    callbacks = []
    train_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%d_%H-%M")
    train_dir.mkdir(parents=True, exist_ok=True)

    filepath = train_dir / "ep-{epoch:02d}_vl-{val_loss:.5f}.h5"
    filepath.parent.mkdir(exist_ok=True, parents=True)
    callbacks.append(ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=True))

    log_dir = train_dir / "logs"
    callbacks.append(TensorBoard(log_dir=log_dir))

    p.to_yaml(train_dir / "train_config.yaml")

    # compile model and start training
    model.fit(
        train_ds,
        batch_size=p.batch_size,
        epochs=p.num_epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks,
    )
