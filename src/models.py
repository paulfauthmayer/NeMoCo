import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Input, ELU

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
        # print(f"Experts build {self.name}: {input_shape=} {self.alpha.shape=} {self.beta.shape=}")

    def call(self, inputs):
        x, gate_perc = inputs
        w = self.get_expert_weights(gate_perc)
        b = self.get_expert_biases(gate_perc)
        # print(f"Expert call {self.name}: {x.shape=} {gate_perc.shape=} {w.shape=} {b.shape=} ")
        result = (w @ x[..., None])[..., 0]
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

class NeMoCoModel(tf.keras.Model):
    def train_step(self, data):
        # unpack the data
        x_expert = tf.sparse.to_dense(data["expert_input"])
        x_gating = tf.sparse.to_dense(data["gating_input"])
        y = tf.sparse.to_dense(data["output"])

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
        x_expert = tf.sparse.to_dense(data["expert_input"])
        x_gating = tf.sparse.to_dense(data["gating_input"])
        y = tf.sparse.to_dense(data["output"])

        # Compute predictions
        y_pred = self([x_gating, x_expert], training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def create_model(p: TrainingParameters) -> Model:

    gating_input = Input(shape=(len(p.gating_input_cols,)), name="gating_input")
    expert_input = Input(shape=(len(p.expert_input_cols,)), name="expert_input")

    # Gating Network
    x = Dense(p.gating_layer_shapes[0], activation='elu')(gating_input)
    x = Dense(p.gating_layer_shapes[1], activation='elu')(x)
    gating_perc = Dense(p.num_experts, activation='softmax')(x)

    # Expert Network
    x = DenseExpert(p.expert_layer_shapes[0], p.num_experts,)([expert_input, gating_perc])
    x = ELU()(x)
    x = DenseExpert(p.expert_layer_shapes[1], p.num_experts)([x, gating_perc])
    x = ELU()(x)
    y = DenseExpert(len(p.output_cols), p.num_experts)([x, gating_perc])

    model = NeMoCoModel(
        inputs=[gating_input, expert_input],
        outputs=[y]
    )

    return model
