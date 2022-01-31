import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path

from generate_datasets import load_dataset
from models import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nemoco_model_path", type=Path)
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()
    
    test_ds_path = args.dataset_path / "test.tfrecords"

    model = load_model(args.nemoco_model_path)
    test_ds = load_dataset(test_ds_path, batch_size=1, is_train=False)

    evaluation = model.evaluate(test_ds)
