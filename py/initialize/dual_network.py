from py.config import INPUT_SHAPE, OUTPUT_SHAPE, RESNET_N_SIZE, RESNET_RESIDUAL_SIZE, RESNET_BEST_FILE, \
    IMPROVED_RESNET_BEST_FILE
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as bk
import os


def conv(filters):
    return Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal",
                  kernel_regularizer=l2(0.0005))


def residual_block():
    def f(x):
        sc = x
        x = conv(RESNET_N_SIZE)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = conv(RESNET_N_SIZE)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation("relu")(x)
        return x

    return f


def make_resnet_model(path):
    if os.path.exists(path):
        return
    inp = Input(shape=INPUT_SHAPE)

    x = conv(RESNET_N_SIZE)(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for _ in range(RESNET_RESIDUAL_SIZE):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(OUTPUT_SHAPE, activation="softmax", kernel_regularizer=l2(0.0005), name="pi")(x)
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name='v')(v)

    model = Model(inputs=inp, outputs=[p, v])
    model.save(path)

    bk.clear_session()
    del model


def create_model():
    inp = Input(shape=INPUT_SHAPE)

    x = conv(RESNET_N_SIZE)(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for _ in range(RESNET_RESIDUAL_SIZE):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(OUTPUT_SHAPE, activation="softmax", kernel_regularizer=l2(0.0005), name="pi")(x)
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name='v')(v)

    model = Model(inputs=inp, outputs=[p, v])
    return model


def initialize_network():
    make_resnet_model(RESNET_BEST_FILE)
    make_resnet_model(IMPROVED_RESNET_BEST_FILE)


if __name__ == '__main__':
    initialize_network()
