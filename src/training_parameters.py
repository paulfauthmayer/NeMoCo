import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.optimizers import Adam
import yaml


class TrainingParameters:
    def __init__(self, dataset_path: Path, dataset_norm_path: Path) -> None:
        # architecture
        self.num_experts = 8
        self.expert_layer_shapes = [512, 512]
        self.gating_layer_shapes = [128, 128]

        # training specific
        self.learn_rate = 0.0001
        self.num_epochs = 200
        self.batch_size = 30
        self.dropout_prob = 0.5
        self.optimizer = Adam
        self.optimizer_settings = {}
        self.seed = 42

        # data handling
        self._data_head = pd.read_csv(dataset_path, nrows=0)

        self.train_val_test_ratios = [0.70, 0.15, 0.15]
        self.dataset_path = dataset_path
        self.dataset_norm_path = dataset_norm_path

        self.gating_input_cols = self._data_head.filter(
            regex=r"velocity_\w_([7-9]|1[0-2])$"
        ).columns
        self.output_cols = self._data_head.filter(regex=r"^out_").columns
        self.expert_input_cols = (
            self._data_head.drop(self.gating_input_cols, axis=1)
            .drop(self.output_cols, axis=1)
            .columns
        )

        self.expert_input_idx = [
            self._data_head.columns.get_loc(col) for col in self.expert_input_cols
        ]
        self.gating_input_idx = [
            self._data_head.columns.get_loc(col) for col in self.gating_input_cols
        ]
        self.output_idx = [
            self._data_head.columns.get_loc(col) for col in self.output_cols
        ]

        # self.num_samples = 73425
        with open(dataset_path, "r") as f:
            self.num_samples = sum(1 for _ in f) - 1  # -1 because we use csvs

    def __repr__(self) -> None:
        attrs = self.__dict__
        attrs.pop("_data_head", None)

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

            out += fmt.format(key, str(value))

        return out

    def summarize(self, out_path: Path = None) -> dict:
        attrs = self.__dict__
        attrs.pop("_data_head", None)
        if out_path is not None:
            out_path.parent.mkdir(exist_ok=True, parents=True)
            with open(str(out_path), "w") as f:
                yaml.dump(attrs, f)
        print(self)
        return attrs
