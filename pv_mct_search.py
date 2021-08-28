import numpy as np
from tensorflow.keras.models import Model

# ある盤面に対する方策と価値を推論する
from config import INPUT_SHAPE
from game import State


def predict(model: Model, st: State):
    a, b, c = INPUT_SHAPE
    x = np.array([st.pieces, st.enemy_pieces, st.block])
    x = x.reshape((c, a, b)).transpose((1, 2, 0)).reshape((1, a, b, c))


# ニューラルネットワーク近似を活用したモンテカルロ木探索
def pv_mct_search_action(model: Model):
