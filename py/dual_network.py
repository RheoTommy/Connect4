from config import INPUT_SHAPE, OUTPUT_SHAPE, DENSE_BEST_FILE, CNN_BEST_FILE, \
    DENSE_N_SIZE, CNN_N_SIZE, RESNET_N_SIZE, RESNET_RESIDUAL_SIZE, RESNET_BEST_FILE, IMPROVED_RESNET_BEST_FILE
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, \
    Dropout
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


# Dense
def make_dense_model():
    if os.path.exists(DENSE_BEST_FILE):
        return

    inp = Input(shape=INPUT_SHAPE)

    x = Dense(DENSE_N_SIZE, activation="relu")(inp)
    x = Dropout(0.5)(x)
    x = Dense(DENSE_N_SIZE, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(DENSE_N_SIZE, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(DENSE_N_SIZE, activation="relu")(x)
    x = GlobalAveragePooling2D()(x)

    p = Dense(OUTPUT_SHAPE, activation="softmax", kernel_regularizer=l2(0.0005), name="pi")(x)
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name='v')(v)

    model = Model(inputs=inp, outputs=[p, v])
    model.save(DENSE_BEST_FILE)

    bk.clear_session()
    del model


# CNN
def make_cnn_model():
    if os.path.exists(CNN_BEST_FILE):
        return

    inp = Input(shape=INPUT_SHAPE)

    x = conv(CNN_N_SIZE)(inp)
    x = Activation("relu")(x)
    x = conv(CNN_N_SIZE)(x)
    x = Activation("relu")(x)
    x = conv(CNN_N_SIZE)(x)
    x = Activation("relu")(x)
    x = conv(CNN_N_SIZE)(x)
    x = Activation("relu")(x)
    x = conv(CNN_N_SIZE)(x)
    x = Activation("relu")(x)
    x = conv(CNN_N_SIZE)(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)

    p = Dense(OUTPUT_SHAPE, activation="softmax", kernel_regularizer=l2(0.0005), name="pi")(x)
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation("tanh", name='v')(v)

    model = Model(inputs=inp, outputs=[p, v])
    model.save(CNN_BEST_FILE)

    bk.clear_session()
    del model


# ResNet
def make_resnet_model():
    if os.path.exists(RESNET_BEST_FILE):
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
    model.save(RESNET_BEST_FILE)

    bk.clear_session()
    del model


# Improved ResNet
def make_improved_resnet_model():
    if os.path.exists(IMPROVED_RESNET_BEST_FILE):
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
    model.save(IMPROVED_RESNET_BEST_FILE)

    bk.clear_session()
    del model


if __name__ == '__main__':
    make_dense_model()
    make_cnn_model()
    make_resnet_model()
