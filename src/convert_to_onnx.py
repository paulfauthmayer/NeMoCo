import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from onnxruntime import InferenceSession
import tensorflow as tf
from tensorflow.keras.models import Model
import tf2onnx

from models import NeMoCoModel, DenseExpert


def test_onnx(onnx_path: Path, model: Model):
    onnx_session = InferenceSession(str(onnx_path))

def none_to_num(array: np.array, num: int) -> Iterable:
    array = np.asarray(array)
    for i, e in enumerate(array):
        if isinstance(e, Iterable):
            array[i] = none_to_num(e, num)
        else:
            array[i] = e if e is not None else num
    return array


def convert_model_to_onnx(model: Model, output_path: Path):
    input_signature = (
        tf.TensorSpec((1, model.input_shape[0][1]), tf.float32, name="gating_input"),
        tf.TensorSpec((1, model.input_shape[1][1]), tf.float32, name="expert_input")
    )
    tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=output_path, opset=15)


def convert_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path):
    custom_objects = {"NeMoCoModel": NeMoCoModel, "DenseExpert": DenseExpert}
    model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    convert_model_to_onnx(model, output_path)
    test_onnx(output_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="The path to the checkpoint you want to convert."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="The path where you want the ONNX model to be stored.",
        default=None,
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = args.checkpoint.with_suffix(".onnx")

    convert_checkpoint_to_onnx(args.checkpoint, output_path)
