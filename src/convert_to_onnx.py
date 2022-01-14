import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
import tf2onnx

from models import NeMoCoModel, DenseExpert


def convert_model_to_onnx(model: Model, output_path: Path):
    tf2onnx.convert.from_keras(model, output_path=output_path, opset=15)


def convert_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path):
    custom_objects = {"NeMoCoModel": NeMoCoModel, "DenseExpert": DenseExpert}
    model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    convert_model_to_onnx(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="The path to the checkpoint you want to convert"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="The path where you want the ONNX model to be stored",
    )
    args = parser.parse_args()

    convert_checkpoint_to_onnx(args.checkpoint_path, args.output_path)
