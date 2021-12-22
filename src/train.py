import argparse
from pathlib import Path

import numpy as np
from numpy.core.numeric import full
import pandas as pd
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import exp

from models import create_model
from training_parameters import TrainingParameters

@tf.function
def train_step(model, expert_in, gating_in, expected_out):
    with tf.GradientTape() as tape:
        predictions = model(expert_in, expected_out)

@tf.function
def test_step():
    return None

def train(dataset_path, num_experts=8):

    p = TrainingParameters(dataset_path)

    model = create_model(p)
    model.summary()

    gating_in = tf.ones((p.batch_size, len(p.gating_input_cols)))
    expert_in = tf.ones((p.batch_size, len(p.expert_input_cols)))
    print("=+"*10,'builduing',"+="*10)
    model([gating_in, expert_in])
    print("=+"*10,'calling',"+="*10)
    model([gating_in, expert_in])
    print(model([gating_in, expert_in]).shape)

    # # assemble datasets
    # dataset = tf.data.experimental.make_csv_dataset(
    #     file_pattern=str(p.dataset_path),
    #     batch_size=2,
    # )

    # for epoch in range(p.num_epochs):
    #     for step in range(p.num_samples // p.batch_size):
    #         batch = next(dataset)
    #         expert_in = batch[p.expert_input_cols]
    #         gating_in = batch[p.expert_input_cols]
    #         expected_out = batch[p.output_cols]
    #         train_step(model, expert_in, gating_in, expected_out)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
