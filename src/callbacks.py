from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from convert_to_onnx import convert_model_to_onnx

class OnnxCheckpointCallback(keras.callbacks.Callback):
    def __init__(
        self,
        filepath: Path,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode="min",
    ) -> None:
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        if mode == "min":
            self.monitor_op = np.less 
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf

        self.filepath.parent.mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs=None):

        # check if current epoch is best so far
        if self.save_best_only:
            score = logs.get(self.monitor)
            if not self.monitor_op(score, self.best):
                return
            self.best = score

        # convert model to onnx 
        output_path = str(self.filepath).format(epoch=epoch+1, **logs)
        print(f"Saving ONNX model to {output_path}")
        convert_model_to_onnx(self.model, output_path)


        