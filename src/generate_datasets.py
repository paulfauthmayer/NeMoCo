import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from training_parameters import TrainingParameters
from collections import defaultdict

FEATURE_DESCRIPTION = {
    "gating_input": tf.io.VarLenFeature(np.float32),
    "expert_input": tf.io.VarLenFeature(np.float32),
    "output": tf.io.VarLenFeature(np.float32),
}


def load_dataset(
    tfrecords_path: Path, p: TrainingParameters, is_train: bool = False
) -> tf.data.Dataset:
    raw_ds = tf.data.TFRecordDataset(str(tfrecords_path))
    # TODO: parse_example vs parse_single_example ??
    ds = raw_ds.map(lambda x: tf.io.parse_single_example(x, FEATURE_DESCRIPTION))
    ds = ds.batch(p.batch_size)
    if is_train:
        ds = ds.shuffle(buffer_size=int(1e5), reshuffle_each_iteration=True)
    return ds


def _float_feature(value):
    """Returns a tf float_list from a float / double list"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def nemoco_example(sample: np.array) -> tf.train.Example:

    sample_data = np.array(sample)
    # TODO: check for exploding values
    # for now this shouldn't be an issue as we only expect the epsilon to take effect
    # for all-zero columns
    sample_data = (sample_data - norm_data[0]) / (norm_data[1] + 1e-7)

    gating_input = np.array(sample_data[p.gating_input_idx], dtype=np.float32)
    expert_input = np.array(sample_data[p.expert_input_idx], dtype=np.float32)
    output = np.array(sample_data[p.output_idx], dtype=np.float32)

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
    training_parameters: TrainingParameters = None,
    name: str = "",
):

    global norm_data
    global p

    p = (
        training_parameters
        if training_parameters is not None
        else TrainingParameters(data_path, norm_data_path)
    )
    norm_data = pd.read_csv(p.dataset_norm_path).to_numpy()

    dataset_name = datetime.now().strftime("%Y-%m-%d_%H-%M") + (
        f"_{name}".upper() if name else ""
    )
    dataset_dir = output_directory / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)
    record_file_train = dataset_dir / "train.tfrecords"
    record_file_test = dataset_dir / "test.tfrecords"
    record_file_val = dataset_dir / "val.tfrecords"

    with open(p.dataset_path, "r") as f, tf.io.TFRecordWriter(
        str(record_file_train)
    ) as train_writer, tf.io.TFRecordWriter(
        str(record_file_test)
    ) as test_writer, tf.io.TFRecordWriter(
        str(record_file_val)
    ) as val_writer:
        # skip the header row
        _ = f.readline()

        counter = defaultdict(int)
        rng = np.random.default_rng(p.seed)

        for sample in tqdm(f.readlines(), total=p.num_samples):
            sample = sample.split(",")
            sample = np.array([np.float32(x) for x in sample])
            example = nemoco_example(sample)
            example = example.SerializeToString()

            x = rng.random()
            if x <= p.train_val_test_ratios[0]:
                counter["train"] += 1
                train_writer.write(example)
            elif x <= sum(p.train_val_test_ratios[:2]):
                counter["val"] += 1
                val_writer.write(example)
            else:
                counter["test"] += 1
                test_writer.write(example)

        print("Splits: ", end="")
        for key, value in counter.items():
            print(f"[{key} : {value} ({value/p.num_samples:.1%})] ", end="")
        print()

    summary_file = dataset_dir / "training_paramters.yml"
    p.summarize(summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("norm_data_path", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path())
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    generate_dataset(**vars(args))
