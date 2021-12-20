import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer

from training_parameters import TrainingParameters


class DenseExpert(Layer):
    def __init__(
        #TODO: define activation function for layer
        self,
        units=32,
        experts=8,
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
        print(f"input_shape: {input_shape}")
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

    def call(self, inputs):
        x, gate_perc = inputs
        w = self.get_expert_weights(gate_perc)
        b = self.get_expert_biases(gate_perc)
        return tf.matmul(x, w) + b

    def get_expert_weights(self, gate_perc):
        print(f"{self.alpha.shape=}")
        print(f"{gate_perc.shape=}")
        a = self.alpha                      # n_exp * neurons_in * neurons_out
        a = tf.expand_dims(self.alpha, 0)   # 1 * n_exp * neurons_in * neurons_out
        gate_perc                           # bs * n_exp
        gate_perc = tf.expand_dims(tf.expand_dims(gate_perc, -1), -1) # bs * n_exp * 1 * 1
        r = a * gate_perc
        print(f"{r.shape=}")
        return tf.reduce_sum(r, axis=1)

    def get_expert_biases(self, gate_perc):
        print(f"{self.beta.shape=}")
        print(f"{gate_perc.shape=}")
        b = self.beta                       # n_exp * neurons
        b = tf.expand_dims(b, 0)            # 1 * n_exp * neurons
        gate_perc                           # bs * n_exp
        gate_perc = tf.expand_dims(gate_perc, -1)
        r = b * gate_perc
        print(f"{r.shape=}")
        return tf.reduce_sum(r, axis=1)


class Gating(Model):
    def __init__(self, p: TrainingParameters, *args, **kwargs):

        super(Gating, self).__init__(*args, **kwargs)

        self.my_layers = []

        self.my_layers.append(Dense(p.gating_layer_shapes[0]))
        self.my_layers.append(Dense(p.gating_layer_shapes[1]))
        self.my_layers.append(Dense(p.num_experts))

    def call(self, x):
        for layer in self.my_layers:
            x = layer(x)
        return x


class Motion(Model):
    def __init__(self, p: TrainingParameters, *args, **kwargs):

        super(Motion, self).__init__(*args, **kwargs)

        self._layers = []

        self._layers.append(DenseExpert(p.expert_layer_shapes[0]))
        self._layers.append(DenseExpert(p.expert_layer_shapes[1]))
        self._layers.append(DenseExpert(len(p.output_cols)))

    def call(self, x):
        # set weights like this:
        # self.dense1.set_weights([x * 3 for x in self.dense1.get_weights()])
        x_experts, gating_perc = x
        for layer in self.layers:
            x = layer([x_experts, gating_perc])

        return x_experts


class NeMoCo(Model):
    def __init__(self, p, *args, **kwargs):

        super(NeMoCo, self).__init__(*args, **kwargs)

        self.gating = Gating(p)
        self.experts = Motion(p)

    def call(self, x):
        x_gating, x_experts = x
        gating_perc = self.gating(x_gating)
        print(f"{gating_perc=}")
        print(f"{self.experts.weights=}")
        y = self.experts([x_experts, gating_perc])
        print(f"{y.shape=}")
        return y
    
    def summary(self):
        self.gating.summary()
        self.experts.summary()
