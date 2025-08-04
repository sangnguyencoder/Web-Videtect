import tensorflow as tf
try:
    from tensorflow.keras import layers, models
except ImportError:
    from keras import layers, models


def build_model(input_shape=(16, 112, 112, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling3D((1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(128, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape((x.shape[1], -1))(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
