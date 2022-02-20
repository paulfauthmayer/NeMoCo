from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, ELU, Dropout
from tensorflow.keras import activations

from globals import GATING_INPUT, EXPERT_INPUT, OUTPUT

class DenseExpert(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        experts,
        activation=None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )
        self.units = units
        self.experts = experts
        self.activation = activations.get(activation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "experts": self.experts,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        '''alpha and beta are the pool of weights over all experts at the given layer'''
        self.alpha = self.add_weight(
            shape=(self.experts, self.units, input_shape[0][-1]),
            initializer="random_normal",
            name="weights",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(self.experts, self.units),
            initializer="random_normal",
            name="biases",
            trainable=True,
        )

    def call(self, inputs):
        x, g = inputs                                           # x: {batch_size, input}, g: {batch_size, num_experts}

        x = tf.expand_dims(x, -1)                               # x: {batch_size, input, 1}
        w = self.get_expert_weights(g)                          # w: {batch_size, ouput, input}
        b = self.get_expert_biases(g)                           # b: {batch_size, output}

        r = w @ x                                               # r: {batch_size, output, 1}
        r = tf.squeeze(r, -1)                                   # r: {batch_size, output}
        r = r + b

        if self.activation is not None:
            r = self.activation(r)

        return r

    def get_expert_weights(self, gate_perc):
        a = tf.expand_dims(self.alpha, 0)                       # a: {1, num_experts, output, input}
        g = tf.expand_dims(tf.expand_dims(gate_perc, -1), -1)   # g: {batch_size, num_experts, 1, 1}

        r = g * a                                               # r: {batch_size, num_experts, output, input}
        r = tf.reduce_sum(r, axis=1)                            # r: {batch_size, output, input}

        return r

    def get_expert_biases(self, gate_perc):
        b = tf.expand_dims(self.beta, 0)                        # b: {1, num_experts, output}
        g = tf.expand_dims(gate_perc, -1)                       # g: {batch_size, num_experts, 1}

        r = g * b                                               # r: {batch_size, num_experts, output}
        r = tf.reduce_sum(r, axis=1)                            # r: {batch_size, output}

        return r


class NeMoCoModel(tf.keras.Model):

    def __init__(
        self,
        gating_input_features: int,
        gating_layer_units: List[int],
        expert_input_features: int,
        expert_layer_units: List[int],
        num_experts: int,
        expert_output_features: int,
        dropout_prob: float
    ) -> None:

        self.build_instructions = {
            "gating_layer_units": gating_layer_units,
            "gating_input_features": gating_input_features,
            "expert_layer_units": expert_layer_units,
            "expert_input_features": expert_input_features,
            "num_experts": num_experts,
            "expert_output_features": expert_output_features,
            "dropout_prob": dropout_prob,
        }

        inputs, outputs = self.create_nemoco_graph(**self.build_instructions)
        super().__init__(
            inputs=inputs,
            outputs=outputs
        )

    def get_config(self):
        config = super().get_config()
        config.update({"build_instructions": self.build_instructions})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config["build_instructions"])

    @classmethod
    def create_nemoco_graph(
        cls,
        gating_layer_units: List[int],
        gating_input_features: int,
        expert_layer_units: List[int],
        expert_input_features: int,
        num_experts: int,
        expert_output_features: int,
        dropout_prob: float
    ):
        gating_input = Input(shape=(gating_input_features,), name=GATING_INPUT)
        expert_input = Input(shape=(expert_input_features,), name=EXPERT_INPUT)

        # Gating Network
        x = gating_input
        for units in gating_layer_units:
            x = Dense(units, activation='elu')(x)
            x = Dropout(dropout_prob)(x)
        gating_out = Dense(num_experts, activation='softmax', name="gating_output")(x)

        # Expert Network
        x = expert_input
        for units in expert_layer_units:
            x = DenseExpert(units, num_experts, "elu")([x, gating_out])
            x = Dropout(dropout_prob)(x)
        y = DenseExpert(expert_output_features, num_experts, name="final_output")([x, gating_out])

        return [gating_input, expert_input], [y]

    def train_step(self, data):
        # unpack the data
        x_expert = tf.sparse.to_dense(data[EXPERT_INPUT])
        x_gating = tf.sparse.to_dense(data[GATING_INPUT])
        y = tf.sparse.to_dense(data[OUTPUT])

        with tf.GradientTape() as tape:

            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_pred = self([x_gating, x_expert], training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # unpack the data
        x_expert = tf.sparse.to_dense(data[EXPERT_INPUT])
        x_gating = tf.sparse.to_dense(data[GATING_INPUT])
        y = tf.sparse.to_dense(data[OUTPUT])

        # compute predictions
        y_pred = self([x_gating, x_expert], training=False)

        # updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

def load_model(checkpoint_path: Path) -> tf.keras.Model:
    custom_objects = {"NeMoCoModel": NeMoCoModel, "DenseExpert": DenseExpert}
    model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    return model
