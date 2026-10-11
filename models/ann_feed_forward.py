import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


def build_ann(input_size, hidden_size=12, learning_rate=0.009):
    model = Sequential([
        Input(shape=(input_size,)),
        Dense(hidden_size, activation="sigmoid"),
        Dense(1, activation="linear")
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model