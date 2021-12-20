import argparse
from pathlib import Path

import numpy as np
from numpy.core.numeric import full
import pandas as pd
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import exp


class TrainingParameters:
    def __init__(self, dataset_path) -> None:
        # architecture
        self.num_experts = 8

        # training specific
        self.learn_rate = 0.0001
        self.num_epochs = 200
        self.batch_size = 30
        self.dropout_prob = 0.5
        self.optimizer = Adam
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)

        # data handling
        self._data_head = pd.read_csv(dataset_path, nrows=0)

        self.train_val_test_ratios = [0.70, 0.15, 0.15]
        self.dataset_path = dataset_path

        self.gating_input_cols = self._data_head.filter(
            regex=r"velocity_\w_([7-9]|1[0-2])$"
        ).columns
        self.output_cols = self._data_head.filter(regex=r"^out_").columns
        self.expert_input_cols = (
            self._data_head
            .drop(self.gating_input_cols, axis=1)
            .drop(self.output_cols, axis=1)
            .columns
        )

        with open(dataset_path, "r") as f:
            self.num_samples = sum(1 for _ in f) - 1  # -1 because we use csvs

    @property
    def expert_input_idx(self):
        return [self._data_head.columns.get_loc(col) for col in self.expert_input_cols]

    @property
    def gating_input_idx(self):
        return [self._data_head.columns.get_loc(col) for col in self.gating_input_cols]

    @property
    def output_idx(self):
        return [self._data_head.columns.get_loc(col) for col in self.output_cols]
    
    def summarize(self):
        print("TODO: summarize the training parameters")


def create_model():
    model = None
    return model


@tf.function
def train_step(model, expert_in, gating_in, expected_out):
    with tf.GradientTape() as tape:
        predictions = model(expert_in, expected_out)


def train(dataset_path, num_experts=8):

    p = TrainingParameters(dataset_path)

    # assemble datasets
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=str(p.dataset_path),
        batch_size=2,
    )

    model = create_model()

    for epoch in range(p.num_epochs):
        for step in range(p.num_samples // p.batch_size):
            batch = next(dataset)
            expert_in = batch[p.expert_input_cols]
            gating_in = batch[p.expert_input_cols]
            expected_out = batch[p.output_cols]
            train_step(model, expert_in, gating_in, expected_out)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
