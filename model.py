import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    Input,
    GlobalAveragePooling2D,
    Concatenate,
    Activation,
)
from loss import custom_mse
from tensorflow.keras.optimizers import Adam

from constant import DAVE2_PATH, DIM, LR, BATCH_SIZE

def build_model():
    imagenet_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    dave2 = tf.keras.models.load_model(DAVE2_PATH)

    X_input1 = Input(shape=(*DIM, 3))
    for i, layer in enumerate(dave2.layers[:5]):
        layer.trainable = False
        if i == 0:
            X1 = layer(X_input1)
        else:
            X1 = layer(X1)
    X1 = GlobalAveragePooling2D()(X1)

    imagenet_model.trainable = False

    X_input2 = Input(shape=(224, 224, 3))
    X2 = imagenet_model(X_input2)
    X2 = GlobalAveragePooling2D()(X2)

    X = Concatenate()([X1, X2])
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)

    X = Dense(256)(X)
    X = BatchNormalization()(X)
    X = Activation("elu")(X)
    X = Dropout(0.2)(X)

    X = Dense(64)(X)
    X = BatchNormalization()(X)
    X = Activation("elu")(X)
    X = Dropout(0.2)(X)

    X = Dense(2, activation="linear")(X)

    model = tf.keras.Model([X_input1, X_input2], X)
    model.compile(
        Adam(learning_rate=LR, weight_decay=LR / BATCH_SIZE),
        loss=custom_mse,
        metrics=["mse"],
    )
    model.summary()
