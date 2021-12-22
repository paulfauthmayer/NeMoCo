import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Input

from training_parameters import TrainingParameters


class DenseExpert(Layer):
    def __init__(
        #TODO: define activation function for layer
        self,
        units,
        experts,
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

    def build(self, input_shape):
        '''alpha and beta are the pool of weights over all experts at the given layer'''
        self.alpha = self.add_weight(
            shape=(self.experts, input_shape[0][-1], self.units),
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
        print(f"Experts build {self.name}: {input_shape=} {self.alpha.shape=} {self.beta.shape=}")

    def call(self, inputs):
        x, gate_perc = inputs
        w = self.get_expert_weights(gate_perc)
        b = self.get_expert_biases(gate_perc)
        print(f"Expert call {self.name}: {x.shape=} {gate_perc.shape=} {w.shape=} {b.shape=} ")
        result = tf.matmul(x, w) # TODO: this screws up the shape, why?
        result = result + b
        return result

    def get_expert_weights(self, gate_perc):
        a = tf.expand_dims(self.alpha, 0)   # 1 * n_exp * neurons_in * neurons_out
        gate_perc = tf.expand_dims(tf.expand_dims(gate_perc, -1), -1) # bs * n_exp * 1 * 1
        r = gate_perc * a
        return tf.reduce_sum(r, axis=1)

    def get_expert_biases(self, gate_perc):
        b = tf.expand_dims(self.beta, 0)            # 1 * n_exp * neurons
        gate_perc = tf.expand_dims(gate_perc, -1) # bs * n_exp * 1 * 1
        r = gate_perc * b
        return tf.reduce_sum(r, axis=1)


def create_model(p: TrainingParameters):

    gating_input = Input(shape=(len(p.gating_input_cols,)))
    expert_input = Input(shape=(len(p.expert_input_cols,)))

    # Gating Network
    x = Dense(p.gating_layer_shapes[0])(gating_input)
    x = Dense(p.gating_layer_shapes[1])(x)
    gating_perc = Dense(p.num_experts)(x)

    # Expert Network
    x = DenseExpert(p.expert_layer_shapes[0], p.num_experts)([expert_input, gating_perc])
    x = DenseExpert(p.expert_layer_shapes[1], p.num_experts)([x, gating_perc])
    y = DenseExpert(len(p.output_cols), p.num_experts)([x, gating_perc])

    model = tf.keras.Model(
        inputs=[gating_input, expert_input],
        outputs=[y]
    )

    return model
