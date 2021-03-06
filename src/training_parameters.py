from collections import defaultdict
import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Any, List
import yaml

from globals import TRAIN, VAL, TEST

optimizers = {
    # Default Tensorflow Optimizers
    "Adadelta": tf.keras.optimizers.Adam,
    "Adagrad": tf.keras.optimizers.Adagrad,
    "Adam": tf.keras.optimizers.Adam,
    "Adamax": tf.keras.optimizers.Adamax,
    "Nadam": tf.keras.optimizers.Nadam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
    # Tensorflow Addons Optimizers
    "AdamW": tfa.optimizers.AdamW,
    "AdaBelief": tfa.optimizers.AdaBelief,
    # Others
}
optimizers_inv = {v: k for k, v in optimizers.items()}


@dataclass
class Conversion:
    real: Any = lambda x: x
    yaml: Any = lambda x: x


class BaseConfig:
    def __init__(self):
        self.conversions = defaultdict(Conversion)

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        # create new class instance as placeholder
        config = cls()

        # load config from dist
        with open(yaml_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        loaded_config = {k: config.conversions[k].real(v) for k, v in loaded_config.items()}

        # overwrite own attributes if they're included in the config
        for attr, value in config.__dict__.items():
            new_value = loaded_config.get(attr, value)
            setattr(config, attr, new_value)

        return config

    def readable(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "conversions"}
        d = {k: self.conversions[k].yaml(v) for k, v in d.items()}
        return d

    def to_yaml(self, yaml_path: Path) -> None:
        d = self.readable()
        with open(yaml_path, "w") as f:
            yaml.safe_dump(d, f, indent=4, sort_keys=False)


class DatasetConfig(BaseConfig):
    def __init__(
        self,
        dataset_csv_path: Path = None,
        dataset_norm_csv_path: Path = None,
        split_ratios: dict = {TRAIN: 0.70, "val": 0.15, "test": 0.15},
    ) -> None:

        super().__init__()
        self.conversions.update({
            "dataset_directory": Conversion(Path, str),
            "dataset_csv_path": Conversion(Path, str),
            "dataset_norm_csv_path": Conversion(Path, str),
        })

        # name and dataset_directory will be set when running generate_dataset.py
        self.name = None
        self.dataset_directory: Path = None

        self.dataset_csv_path = None
        self.dataset_norm_csv_path = None

        self.num_samples = None
        self.split_ratios = split_ratios
        self.num_samples_per_split = None

        self.gating_input_cols = None
        self.expert_input_cols = None
        self.output_cols = None

        self.gating_input_idx = None
        self.expert_input_idx = None
        self.output_idx = None

        if bool(dataset_csv_path) ^ bool(dataset_norm_csv_path):
            raise RuntimeError("Invalid input for dataset config")

        elif dataset_csv_path and dataset_norm_csv_path:
            self.dataset_csv_path = dataset_csv_path.absolute()
            self.dataset_norm_csv_path = dataset_norm_csv_path.absolute()

            data_head = pd.read_csv(dataset_csv_path, nrows=0)

            excluded_cols = data_head.filter(regex=r"effector_vel|mode_|sequence_name|placeholder|phase").columns
            self.output_cols = list(
                data_head
                .filter(regex=r"^out_")  # matches all that start with "out_"
                .drop(excluded_cols, axis=1, errors="ignore")
                .columns
            )
            self.gating_input_cols = list(
                data_head
                .filter(regex=r"root_(position|velocity|direction|angle|length)|(?<!_)effector_ball_[lr]")
                .drop(data_head.filter("^out_").columns, axis=1, errors="ignore")
                .drop(self.output_cols, axis=1, errors="ignore")
                .drop(excluded_cols, axis=1, errors="ignore")
                .columns
            )
            self.expert_input_cols = list(
                data_head
                .drop(data_head.filter(regex=r"foot_[rl]_contact").columns, axis=1)
                .drop(data_head.filter("^out_").columns, axis=1, errors="ignore")
                .drop(data_head.filter(regex="pose").columns, axis=1, errors="ignore")
                .drop(self.output_cols, axis=1, errors="ignore")
                .drop(excluded_cols, axis=1, errors="ignore")
                .columns
            )

            unused_columns = [
                col for col in data_head.columns
                if col not in [*excluded_cols, *self.output_cols, *self.gating_input_cols, *self.expert_input_cols]
            ]
            if unused_columns:
                print("Warning: The following columns weren't explicitly included or excluded:")
                print("\n".join([f"\t* {col}" for col in unused_columns]))

            self.expert_input_idx = [
                data_head.columns.get_loc(col) for col in self.expert_input_cols
            ]
            self.gating_input_idx = [
                data_head.columns.get_loc(col) for col in self.gating_input_cols
            ]
            self.output_idx = [
                data_head.columns.get_loc(col) for col in self.output_cols
            ]

            with open(dataset_csv_path, "r") as f:
                self.num_samples = sum(1 for _ in f) - 1  # -1 because we use csvs

    def is_valid(self):
        return any([v is not None for k, v in self.__dict__.items()])


class TrainingParameters(BaseConfig):
    def __init__(
        self,
        gating_input_features: int = None,
        expert_input_features: int = None,
        expert_output_features: int = None,
        gating_layer_units: List[int] = [128, 128],
        expert_layer_units: List[int] = [512, 512],
        num_experts: int = 8,
        learn_rate: float = 0.0001,
        weight_decay : float = 0.0025,
        num_epochs: int = 1000,
        batch_size: int = 256,
        dropout_prob: float = 0.5,
        optimizer: str = "Adam",
        optimizer_settings: dict = {},
        dataset_config: DatasetConfig = None,
    ) -> None:

        super().__init__()
        self.conversions.update({
            "optimizer": Conversion(lambda x: optimizers[x], lambda x: optimizers_inv[x]),
            "dataset_directory": Conversion(Path, str),
        })

        # architecture
        self.gating_input_features = gating_input_features
        self.gating_layer_units = gating_layer_units
        self.expert_input_features = expert_input_features
        self.expert_layer_units = expert_layer_units
        self.num_experts = num_experts
        self.expert_output_features = expert_output_features

        # training specific
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.optimizer = optimizers[optimizer]
        self.optimizer_settings = optimizer_settings

        # relevant for testing
        self.dataset_directory: Path = None

        if dataset_config is not None:
            self.gating_input_features = len(dataset_config.gating_input_cols)
            self.expert_input_features = len(dataset_config.expert_input_cols)
            self.expert_output_features = len(dataset_config.output_cols)
            self.dataset_directory = dataset_config.dataset_directory

    def __str__(self) -> str:
        attrs = copy.deepcopy(self.__dict__)
        attrs.pop("conversions", None)

        out = " Training Parameters ".center(60, "~") + "\n\n"
        fmt = "{:<22} | {:40}\n"

        for key, value in attrs.items():
            if isinstance(value, (list, pd.Index)):
                value = [str(x) for x in value]
                if len(value) > 3:
                    value = f"[ {', '.join(value[:3])}, ...]"
                else:
                    value = f"[ {', '.join(value)} ]"

            if isinstance(value, Path):
                value = value.absolute()

            if key == "optimizer":
                value = optimizers_inv[value]

            out += fmt.format(key, str(value))
        return out
