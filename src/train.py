import argparse
import math
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
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

    # shared variables for train and test steps
    optimizer = p.optimizer(**p.optimizer_settings)
    loss_object = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name="test_accuracy")

    @tf.function
    def train_step(X_gating, X_expert, Y):
        with tf.GradientTape() as tape:

            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model([X_gating, X_expert], training=True)
            loss = loss_object(Y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(Y, predictions)

    @tf.function
    def test_step(X_gating, X_expert, Y):
        predictions = model([X_gating, X_expert], training=False)
        loss = loss_object(Y, predictions)
        test_loss(loss)
        test_accuracy(Y, predictions)

    # pick dataset and prepare subsets for training
    # TODO: automatically select the correct dataset - if not found, generate!
    dataset_dir = Path("/code/src/datasets/trainable_datasets/2022-01-03_17-38_INITIAL")
    train_ds = load_dataset(dataset_dir / "train.tfrecords", p, is_train=True)
    test_ds = load_dataset(dataset_dir / "test.tfrecords", p)
    val_ds = load_dataset(dataset_dir / "val.tfrecords", p)

    # compile model and start training
    model.compile(optimizer=optimizer)

    for epoch in tqdm(range(p.num_epochs)):
        # Reset the metrics at the start of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # TRAINING
        for i, batch in tqdm(
            enumerate(train_ds),
            total=math.ceil(p.num_samples * p.train_val_test_ratios[0] / p.batch_size),
        ):
            X_expert = tf.sparse.to_dense(batch["expert_input"])
            X_gating = tf.sparse.to_dense(batch["gating_input"])
            Y = tf.sparse.to_dense(batch["output"])
            train_step(X_gating, X_expert, Y)

        # VALIDATION
        for i, batch in enumerate(val_ds):
            X_expert = tf.sparse.to_dense(batch["expert_input"])
            X_gating = tf.sparse.to_dense(batch["gating_input"])
            Y = tf.sparse.to_dense(batch["output"])
            test_step(X_gating, X_expert, Y)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result():.6f}, "
            f"Mean Absolute Error: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result():.6f}, "
            f"Test MAE: {test_accuracy.result() * 100}"
        )
