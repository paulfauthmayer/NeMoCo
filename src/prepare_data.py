import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def prepare_data(
    input_motions: List[Path],
    output_motions: List[Path],
    output_directory: Path,
    store_mann_version: bool,
    prefix: str = "",
    use_fingers: bool = False,
):

    # get headers and datatypes
    shared_params = {"sep": ",", "engine": "python", "encoding": "utf-8-sig"}
    input_headers = pd.read_csv(input_motions[0], nrows=0, **shared_params).columns
    input_dtypes = {
        col: str if col == "sequence_name" else np.float64 for col in input_headers
    }
    output_dtypes = np.float64

    # read datasets
    output_df = pd.concat(pd.read_csv(path, **shared_params) for path in output_motions)

    # check for trash data hidden in first columns
    print(f"Checking file {output_motions} for invalid data")
    for i, val in tqdm(enumerate(output_df["pose_root_pos_x"]), total=len(output_df)):
        try:
            np.float64(val)
        except:
            print(f"Incorrect value {val} in line {i}")

    input_df = pd.concat(
        pd.read_csv(path, dtype=input_dtypes, **shared_params) for path in input_motions
    )

    # check for trash data hidden in first columns
    print(f"Checking file {input_motions} for invalid data")
    for i, val in tqdm(enumerate(input_df["pose_root_pos_x"])):
        try:
            np.float64(val)
        except:
            print(f"Incorrect value {val} in line {i}")

    # drop columns we don't need
    print("Dropping unnecessary columns")
    input_df = input_df.drop("sequence_name", axis=1)
    if not use_fingers:
        finger_regex = r"_(thumb)|(index)|(middle)|(ring)|(pinky)_"
        input_df = input_df.drop(input_df.filter(regex=finger_regex).columns, axis=1)
        output_df = output_df.drop(output_df.filter(regex=finger_regex).columns, axis=1)

    # get metrics required for normalization
    print("Calculate standardization parameters")
    input_df_norm = input_df.agg(["mean", "std"])
    output_df_norm = output_df.agg(["mean", "std"])

    # combine dataframes into a single file
    print("combine dataframes")
    combined_df = pd.concat([input_df, output_df.add_prefix("out_")], axis=1)
    combined_norm = pd.concat([input_df_norm, output_df_norm], axis=1)

    print("store dataframes")
    prefix_snake = f"{prefix}{' ' if prefix else ''}"
    combined_df.to_csv(
        output_directory / f"{prefix_snake}motion_data.csv", header=True, index=False
    )
    combined_norm.to_csv(
        output_directory / f"{prefix_snake}motion_norm.csv", header=True, index=False
    )

    # this is the format required by the original implementation
    if store_mann_version:
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


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-motions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-motions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("."))
    parser.add_argument("--store-mann-version", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--use-fingers", action="store_true")
    args = parser.parse_args()
    prepare_data(**vars(args))


if __name__ == "__main__":
    main()
