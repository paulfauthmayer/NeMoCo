import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from globals import MODE_CROUCH, MODE_WALK, MODE_RUN, MODE_IDLE, MODE_JUMP


def normalize_dataframe(
    df: pd.DataFrame, reference_df: pd.DataFrame = None
) -> pd.DataFrame:

    # store columns that can't be normalized
    non_numerics = df.select_dtypes(exclude=np.number)
    df = df.drop(non_numerics.columns, axis=1)

    # if no reference dataframe is specified, we normalize against itself
    if reference_df is None:
        df = df.astype(np.float64)
        reference_df = df
    # select only the columns relevant for the dataframe
    else:
        reference_df = reference_df[list(df.columns)]
        df = df.astype(np.float64)
        reference_df = reference_df.astype(np.float64)

    # perform normalization via standardization
    df = (df - reference_df.mean()) / reference_df.std()
    df.fillna(0.0, inplace=True)

    # add the dropped columns back into the dataframe
    df = pd.concat([df, non_numerics], axis=1)

    return df


def get_mode(sequence_name: str) -> str:
    name = sequence_name.lower()
    if MODE_RUN in name:
        return MODE_RUN
    elif MODE_WALK in name:
        return MODE_WALK
    elif MODE_JUMP in name:
        return MODE_JUMP
    elif MODE_CROUCH in name:
        return MODE_CROUCH
    elif MODE_IDLE in name:
        return MODE_IDLE
    # case: avoid and sidestep
    # all rather slow, thus we label it as walk
    else:
        return MODE_WALK


def write_mann_data(
    input_df,
    output_df,
    input_df_norm,
    output_df_norm,
    output_directory: Path,
    prefix=None
):
    input_df.drop("sequence_name", axis=1).to_csv(
        output_directory / f"{prefix}Input.txt", sep=" ", header=False, index=False
    )
    output_df.to_csv(
        output_directory / f"{prefix}Output.txt", sep=" ", header=False, index=False
    )

    input_df_norm.to_csv(
        output_directory / f"{prefix}InputNorm.txt",
        sep=" ",
        header=False,
        index=False,
    )
    output_df_norm.to_csv(
        output_directory / f"{prefix}OutputNorm.txt",
        sep=" ",
        header=False,
        index=False,
    )

    if prefix:
        input_df.drop("sequence_name", axis=1).to_csv(
            output_directory / f"{prefix}Input.csv", sep=",", index=False
        )

        output_df.to_csv(
            output_directory / f"{prefix}Output.csv", sep=",", index=False
        )


def load_data(
    input_motions: List[Path],
    output_motions: List[Path] = [],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # if needed, find matching output_motions
    if not output_motions:
        output_motions = []
        for input_motion in input_motions:
            matching_output = input_motion.parent / f"{input_motion.stem}_Output.csv"
            if not matching_output.exists():
                raise RuntimeError(f"Could not identify matching output to input {input_motion}")
            output_motions.append(matching_output)

    # get headers and datatypes
    shared_params = {"sep": ",", "encoding": "utf-8-sig"}
    input_headers = pd.read_csv(input_motions[0], nrows=0, **shared_params).columns
    input_dtypes = {
        col: str if col == "sequence_name" else np.float64 for col in input_headers
    }
    output_headers = pd.read_csv(output_motions[0], nrows=0, **shared_params).columns
    output_dtypes = {col: np.float64 if i != 0 else str for i, col in enumerate(output_headers)}

    # pyarrow is faster but doesn't support reading only part of the rows,
    # thus we only use it after reading the header rows
    shared_params.update({"engine": "pyarrow"})

    # read datasets
    output_df = pd.concat(pd.read_csv(path, dtype=output_dtypes, **shared_params) for path in output_motions)
    input_df = pd.concat(
        pd.read_csv(path, dtype=input_dtypes, **shared_params) for path in input_motions
    )

    # add prefix to output data for easier selection
    output_df = output_df.add_prefix("out_")

    # clean up byte order marks inserted by UE
    input_df['sequence_name'] = input_df['sequence_name'].str.replace('\ufeff', '')
    output_df.iloc[:,0] = output_df.iloc[:,0].str.replace('\ufeff', '')
    output_df = output_df.astype(np.float64)

    return input_df, output_df


def prepare_data(
    input_motions: List[Path],
    output_motions: List[Path],
    output_directory: Path,
    store_mann_version: bool,
    prefix: str = "",
    use_fingers: bool = False,
    use_twist: bool = False,
) -> Tuple[Path, Path]:

    # load input and output data from disk
    input_data, output_data = load_data(input_motions, output_motions)

    # drop columns we don't need
    if not use_fingers:
        finger_regex = r"_(thumb)|(index)|(middle)|(ring)|(pinky)_"
        input_data = input_data.drop(input_data.filter(regex=finger_regex).columns, axis=1)
        output_data = output_data.drop(output_data.filter(regex=finger_regex).columns, axis=1)
    if not use_twist:
        twist_regex = r"_twist_"
        input_data = input_data.drop(
            input_data.filter(regex=twist_regex).columns, axis=1)
        output_data = output_data.drop(
            output_data.filter(regex=twist_regex).columns, axis=1)

    # translate sequences to modes
    modes = input_data["sequence_name"].apply(get_mode)
    one_hot_modes = pd.get_dummies(modes, prefix="mode")
    input_data = pd.concat([input_data, one_hot_modes], axis=1)
    input_data = input_data.drop("sequence_name", axis=1)

    # get metrics required for normalization
    metrics = ["mean", "std"]
    input_df_norm = input_data.agg(metrics)
    output_df_norm = output_data.agg(metrics)

    # combine dataframes into a single file
    combined_df = pd.concat([input_data, output_data], axis=1)
    combined_norm = pd.concat([input_df_norm, output_df_norm], axis=1)

    # save dataframes to disk
    # using pyarrow instead of pandas speeds up the writing
    # of our csv files by a factor of 5
    prefix_snake = f"{prefix}{'_' if prefix else ''}"
    data_path = output_directory / f"{prefix_snake}motion_data.csv"
    norm_path = output_directory / f"{prefix_snake}motion_norm.csv"
    pa.csv.write_csv(pa.Table.from_pandas(combined_df, preserve_index=False), data_path)
    pa.csv.write_csv(pa.Table.from_pandas(combined_norm, preserve_index=False), norm_path)

    # this is the format required by the original implementation
    if store_mann_version:
        write_mann_data(
            input_data, output_data,
            input_df_norm, output_df_norm,
            output_directory,
            prefix=prefix
        )

    return data_path, norm_path


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-motions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-motions", type=Path, nargs="+", default=[])
    parser.add_argument("--output-directory", type=Path, default=Path("."))
    parser.add_argument("--store-mann-version", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--use-fingers", action="store_true")
    parser.add_argument("--use-twist", action="store_true")
    args = parser.parse_args()

    prepare_data(**vars(args))


if __name__ == "__main__":
    main()
