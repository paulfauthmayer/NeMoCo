import argparse
from pathlib import Path

import pandas as pd

from prepare_data import prepare_data, normalize_dataframe
from generate_datasets import generate_dataset
from training_parameters import DatasetConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("name", type=str)
    parser.add_argument("--test-sequence-dir", type=Path)
    args = parser.parse_args()

    dataset_path = Path("datasets")

    data_path, norm_path = prepare_data(
        input_motions=[args.input_csv],
        output_motions=[args.output_csv],
        output_directory=dataset_path / "prepared_datasets",
        store_mann_version=False,
        prefix=args.name
    )

    config_path = generate_dataset( 
        data_path,
        norm_path,
        output_directory=dataset_path / "trainable_datasets",
        name=args.name.upper()
    )

    sequence_dir = args.test_sequence_dir if args.test_sequence_dir else config_path.parent

    c = DatasetConfig().from_yaml(config_path)
    ds = pd.read_csv(c.dataset_csv_path)
    norm = pd.read_csv(c.dataset_norm_csv_path)

    ds_normalized = normalize_dataframe(ds)

    ds_normalized.iloc[:50, c.gating_input_idx].to_csv(sequence_dir / "test_sequence_gating.csv", index=False)
    norm.iloc[:, c.gating_input_idx].to_csv(sequence_dir / "test_sequence_gating_norm.csv", index=False)
    ds_normalized.iloc[:50, c.expert_input_idx].to_csv(sequence_dir / "test_sequence_expert.csv", index=False)
    norm.iloc[:, c.expert_input_idx].to_csv(sequence_dir / "test_sequence_expert_norm.csv", index=False)
    ds_normalized.iloc[:50, c.output_idx].to_csv(sequence_dir / "test_sequence_output.csv", index=False)
    norm.iloc[:, c.output_idx].to_csv(sequence_dir / "test_sequence_output_norm.csv", index=False)