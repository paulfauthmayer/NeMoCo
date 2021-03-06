from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from convert_to_onnx import convert_model_to_onnx

class CheckpointCallback(keras.callbacks.Callback):
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

    def _save(self, filepath):
        raise RuntimeError("This function should be overridden")

    def on_epoch_end(self, epoch, logs=None):

        # check if current epoch is best so far
        if self.save_best_only:
            score = logs.get(self.monitor)
            if not self.monitor_op(score, self.best):
                return
            self.best = score

        # convert model to onnx
        filepath = str(self.filepath).format(epoch=epoch+1, **logs)
        self._save(filepath)


class KerasCheckpoint(CheckpointCallback):
    def __init__(
        self,
        filepath: Path,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode="min",
    ) -> None:
        super().__init__(filepath, save_best_only, monitor, mode)

    def _save(self, filepath):
        print(f"\nSaving keras model to {filepath}")
        self.model.save(filepath, include_optimizer=False)


class OnnxCheckpoint(CheckpointCallback):
    def __init__(
        self,
        filepath: Path,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode="min",
    ) -> None:
        super().__init__(filepath, save_best_only, monitor, mode)

    def _save(self, filepath):
        print(f"Saving ONNX model to {filepath}")
        convert_model_to_onnx(self.model, filepath)


class DecayHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.lr = []
        self.wd = []

    def on_batch_end(self, batch, logs={}):
        self.lr.append(self.model.optimizer.lr(self.model.optimizer.iterations))
        self.wd.append(self.model.optimizer.weight_decay)