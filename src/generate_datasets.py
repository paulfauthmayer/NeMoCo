import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from posixpath import split
import shutil

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from globals import EPS, SEED
from globals import TRAIN, VAL, TEST
from globals import GATING_INPUT, EXPERT_INPUT, OUTPUT
from prepare_data import prepare_data
from training_parameters import DatasetConfig

FEATURE_DESCRIPTION = {
    GATING_INPUT: tf.io.VarLenFeature(np.float32),
    EXPERT_INPUT: tf.io.VarLenFeature(np.float32),
    OUTPUT: tf.io.VarLenFeature(np.float32),
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
    sample_data = (sample_data - norm_data[0]) / (norm_data[1] + EPS)

    gating_input = np.array(sample_data[c.gating_input_idx], dtype=np.float32)
    expert_input = np.array(sample_data[c.expert_input_idx], dtype=np.float32)
    output = np.array(sample_data[c.output_idx], dtype=np.float32)

    feature = {
        GATING_INPUT: _float_feature(gating_input),
        EXPERT_INPUT: _float_feature(expert_input),
        OUTPUT: _float_feature(output),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_dataset(
    input_data_path: Path,
    output_directory: Path,
    name: str = "",
    reuse_preparation: Path = None,
):
    # prepare output directory
    dataset_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + (
        f"_{name}".upper() if name else ""
    )
    dataset_directory = output_directory / dataset_name
    dataset_directory.mkdir(exist_ok=True, parents=True)
    norm_data_path = dataset_directory / "motion_norm.csv"

    # prepare data
    if reuse_preparation:
        data_path = dataset_directory / reuse_preparation.name
        shutil.copy(reuse_preparation, data_path)
        data = pd.read_csv(data_path, engine="pyarrow")
    else:
        data_path, data = prepare_data([input_data_path], [], output_directory=dataset_directory)

    # generate dataset config and define input and output features
    split_ratios = {TRAIN: .7, VAL: .15, TEST: .15}
    c = DatasetConfig(data_path, norm_data_path, split_ratios=split_ratios)
    print(
        "Features > "
        f"gating: {len(c.gating_input_cols):3d} "
        f"expert: {len(c.expert_input_cols):3d} "
        f"output: {len(c.output_cols):3d}"
        )
    c.dataset_directory = dataset_directory
    c.name = dataset_name

    # split data into subsets
    rng = np.random.default_rng(SEED)
    def subset_from_random_number():
        x = rng.random()
        if x <= c.split_ratios[TRAIN]:
            return TRAIN
        elif x <= c.split_ratios[TRAIN] + c.split_ratios[VAL]:
            return VAL
        else:
            return TEST
    data["subset"] = [subset_from_random_number() for _ in range(len(data))]

    counts = data["subset"].value_counts()
    counts = {k: int(v) for k, v in counts.items()}
    print("Splits: ", end="")
    for key, value in counts.items():
        print(f"[{key} : {value} ({value/c.num_samples:.1%})] ", end="")
    print()
    c.num_samples_per_split = counts

    # generate standardization data from train data
    train_data = data[data["subset"] == TRAIN]
    norm_data_df = train_data.drop(["subset", "sequence_name"], axis=1, errors="ignore").agg(["mean", "std"])
    norm_data = norm_data_df.to_numpy()

    # save dataset configuration
    summary_file = dataset_directory / "dataset_config.yaml"
    c.to_yaml(summary_file)

    # write a tfrecord file for each subset
    for subset, group in data.groupby("subset"):
        record_file = dataset_directory / f"{subset}.tfrecords"
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for _, row in tqdm(group.iterrows(), total=c.num_samples_per_split[subset]):
                sample = row.drop(["subset", "sequence_name"]).to_numpy().astype(np.float32)
                example = nemoco_example(sample, norm_data, c)
                example = example.SerializeToString()
                writer.write(example)


    # save norm data in the same split
    norm_data_df.to_csv(norm_data_path, index=False, header=True)
    norm_expert = norm_data_df.iloc[:, c.expert_input_idx]
    norm_expert.to_csv(dataset_directory / "norm_expert.csv", index=False, header=True)
    norm_gating = norm_data_df.iloc[:, c.gating_input_idx]
    norm_gating.to_csv(dataset_directory / "norm_gating.csv", index=False, header=True)
    norm_output = norm_data_df.iloc[:, c.output_idx]
    norm_output.to_csv(dataset_directory / "norm_output.csv", index=False, header=True)

    # save test sequences for debug purposes
    data.iloc[:50, c.gating_input_idx].to_csv(dataset_directory / "test_sequence_gating.csv", index=False)
    data.iloc[:50, c.expert_input_idx].to_csv(dataset_directory / "test_sequence_expert.csv", index=False)
    data.iloc[:50, c.output_idx].to_csv(dataset_directory / "test_sequence_output.csv", index=False)

    # save dataset configuration
    summary_file = dataset_directory / "dataset_config.yaml"
    c.to_yaml(summary_file)

    return summary_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_data_path", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("datasets/trainable_datasets"))
    parser.add_argument("--name", type=str)
    parser.add_argument("--reuse-preparation", type=Path)
    args = parser.parse_args()

    generate_dataset(**vars(args))
