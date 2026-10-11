import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Layer


class MultiLagLSTM(Layer):
    """LSTM cell with feedback from multiple prior cell states, as in the paper."""

    def __init__(self, units, lookback, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lookback = lookback

    def build(self, input_shape):
        input_size = int(input_shape[-1])
        init = "glorot_uniform"
        zeros = "zeros"

        self.W_i = self.add_weight(name="W_i", shape=(input_size, self.units), initializer=init)
        self.W_o = self.add_weight(name="W_o", shape=(input_size, self.units), initializer=init)
        self.W_c = self.add_weight(name="W_candidate", shape=(input_size, self.units), initializer=init)
        self.b_i = self.add_weight(name="b_i", shape=(self.units,), initializer=zeros)
        self.b_o = self.add_weight(name="b_o", shape=(self.units,), initializer=zeros)
        self.b_c = self.add_weight(name="b_candidate", shape=(self.units,), initializer=zeros)

        self.V_i, self.V_o, self.V_c = [], [], []
        self.W_f, self.V_f, self.b_f = [], [], []
        for k in range(self.lookback):
            self.V_i.append(self.add_weight(name=f"V_i_{k + 1}", shape=(self.units, self.units), initializer=init))
            self.V_o.append(self.add_weight(name=f"V_o_{k + 1}", shape=(self.units, self.units), initializer=init))
            self.V_c.append(self.add_weight(name=f"V_candidate_{k + 1}", shape=(self.units, self.units), initializer=init))
            self.W_f.append(self.add_weight(name=f"W_forget_{k + 1}", shape=(input_size, self.units), initializer=init))
            self.V_f.append(self.add_weight(name=f"V_forget_{k + 1}", shape=(self.units, self.units), initializer=init))
            self.b_f.append(self.add_weight(name=f"b_forget_{k + 1}", shape=(self.units,), initializer=zeros))

        super().build(input_shape)

    def call(self, inputs):
        steps = inputs.shape[1]
        if steps is None:
            raise ValueError("The LSTM sequence length must be fixed.")

        zero = tf.zeros((tf.shape(inputs)[0], self.units), dtype=inputs.dtype)
        cell_states = []
        hidden = zero

        for t in range(steps):
            x_t = inputs[:, t, :]
            gate_i_sum = zero
            gate_o_sum = zero
            candidate_sum = zero
            cell_update = zero

            for k in range(1, min(t, self.lookback) + 1):
                c_prev = cell_states[-k]
                idx = k - 1
                forget = tf.sigmoid(
                    x_t @ self.W_f[idx] + c_prev @ self.V_f[idx] + self.b_f[idx]
                )
                gate_i_sum += c_prev @ self.V_i[idx]
                gate_o_sum += c_prev @ self.V_o[idx]
                candidate_sum += c_prev @ self.V_c[idx]
                cell_update += forget * c_prev

            gate_i = tf.sigmoid(x_t @ self.W_i + gate_i_sum + self.b_i)
            gate_o = tf.sigmoid(x_t @ self.W_o + gate_o_sum + self.b_o)
            candidate = tf.tanh(x_t @ self.W_c + candidate_sum + self.b_c)
            cell = gate_i * candidate + cell_update
            hidden = tf.tanh(cell) * gate_o
            cell_states.append(cell)

        return hidden

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "lookback": self.lookback})
        return config


def build_lstm(sequence_length=5, hidden_size=12, learning_rate=0.001):
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    inputs = Input(shape=(sequence_length, 2))
    # A sequence of length m contains at most m-1 earlier cell states.
    hidden = MultiLagLSTM(hidden_size, sequence_length - 1)(inputs)
    output = Dense(1, activation="linear")(hidden)
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
