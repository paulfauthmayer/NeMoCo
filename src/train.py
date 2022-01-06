import argparse
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint
from generate_datasets import load_dataset

from models import create_model
from training_parameters import TrainingParameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("dataset_norm_path", type=Path)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()

    p = TrainingParameters(args.dataset_path, args.dataset_norm_path)

    # instantiate model
    model = create_model(p)
    model.summary()
    optimizer = p.optimizer(**p.optimizer_settings)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # pick dataset and prepare subsets for training
    # TODO: automatically select the correct dataset - if not found, generate!
    dataset_dir = Path("/code/src/datasets/trainable_datasets/2022-01-03_17-38_INITIAL")
    train_ds = load_dataset(dataset_dir / "train.tfrecords", p, is_train=True)
    test_ds = load_dataset(dataset_dir / "test.tfrecords", p)
    val_ds = load_dataset(dataset_dir / "val.tfrecords", p)

    # define callbacks used during training
    callbacks = []

    filepath = Path("checkpoints") / datetime.now().strftime("%Y-%m-%d_%H-%M") / "epoch-{epoch:02d}_vl-{val_loss:.5f}.h5"
    filepath.parent.mkdir(exist_ok=True, parents=True)
    callbacks.append(ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=True))

    # compile model and start training
    model.fit(
        train_ds,
        batch_size=p.batch_size,
        epochs=p.num_epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks,
    )
