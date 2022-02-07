import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from globals import EPS, SEED
from training_parameters import DatasetConfig

FEATURE_DESCRIPTION = {
    "gating_input": tf.io.VarLenFeature(np.float32),
    "expert_input": tf.io.VarLenFeature(np.float32),
    "output": tf.io.VarLenFeature(np.float32),
}


def load_dataset(
    tfrecords_path: Path, batch_size: int, is_train: bool = False
) -> tf.data.Dataset:
    raw_ds = tf.data.TFRecordDataset(str(tfrecords_path))
    # TODO: parse_example vs parse_single_example ??
    ds = raw_ds.map(lambda x: tf.io.parse_single_example(x, FEATURE_DESCRIPTION))
    ds = ds.batch(batch_size)
    if is_train:
        ds = ds.shuffle(buffer_size=int(1e7), reshuffle_each_iteration=True)
    return ds


def _float_feature(value):
    """Returns a tf float_list from a float / double list"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def nemoco_example(sample: np.array, norm_data: np.array, c: DatasetConfig) -> tf.train.Example:

    sample_data = np.array(sample)
    # TODO: check for exploding values
    # for now this shouldn't be an issue as we only expect the epsilon to take effect
    # for all-zero columns
    sample_data = (sample_data - norm_data[0]) / (norm_data[1] - EPS)

    gating_input = np.array(sample_data[c.gating_input_idx], dtype=np.float32)
    expert_input = np.array(sample_data[c.expert_input_idx], dtype=np.float32)
    output = np.array(sample_data[c.output_idx], dtype=np.float32)

    feature = {
        "gating_input": _float_feature(gating_input),
        "expert_input": _float_feature(expert_input),
        "output": _float_feature(output),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_dataset(
    data_path: Path,
    norm_data_path: Path,
    output_directory: Path,
    name: str = "",
):

    c = DatasetConfig(data_path, norm_data_path)
    norm_data_ds = pd.read_csv(c.dataset_norm_csv_path)
    norm_data = norm_data_ds.to_numpy()

    dataset_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + (
        f"_{name}".upper() if name else ""
    )
    dataset_directory = output_directory / dataset_name
    c.dataset_directory = dataset_directory
    c.name = dataset_name
    dataset_directory.mkdir(exist_ok=True, parents=True)
    record_file_train = dataset_directory / "train.tfrecords"
    record_file_test = dataset_directory / "test.tfrecords"
    record_file_val = dataset_directory / "val.tfrecords"

    with open(c.dataset_csv_path, "r") as f, tf.io.TFRecordWriter(
        str(record_file_train)
    ) as train_writer, tf.io.TFRecordWriter(
        str(record_file_test)
    ) as test_writer, tf.io.TFRecordWriter(
        str(record_file_val)
    ) as val_writer:
        # skip the header row
        _ = f.readline()

        counter = defaultdict(int)
        rng = np.random.default_rng(SEED)

        for sample in tqdm(f.readlines(), total=c.num_samples):
            sample = sample.split(",")
            sample = np.array([np.float32(x) for x in sample])
            example = nemoco_example(sample, norm_data, c)
            example = example.SerializeToString()

            x = rng.random()
            if x <= c.split_ratios["train"]:
                counter["train"] += 1
                train_writer.write(example)
            elif x <= c.split_ratios["train"] + c.split_ratios["val"]:
                counter["val"] += 1
                val_writer.write(example)
            else:
                counter["test"] += 1
                test_writer.write(example)

        c.num_samples_per_split = dict(counter)
        print("Splits: ", end="")
        for key, value in counter.items():
            print(f"[{key} : {value} ({value/c.num_samples:.1%})] ", end="")
        print()

    # save norm data in the same split
    norm_data_ds.iloc[:, c.expert_input_idx].to_csv(dataset_directory / "norm_expert.csv", index=False, header=True)
    norm_data_ds.iloc[:, c.gating_input_idx].to_csv(dataset_directory / "norm_gating.csv", index=False, header=True)
    norm_data_ds.iloc[:, c.output_idx].to_csv(dataset_directory / "norm_output.csv", index=False, header=True)

    # save dataset configuration
    summary_file = dataset_directory / "dataset_config.yaml"
    c.to_yaml(summary_file)

    return summary_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("norm_data_path", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("datasets/trainable_datasets"))
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    generate_dataset(**vars(args))
