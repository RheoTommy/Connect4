import numpy as np
from tensorflow.keras.models import Model, load_model
from pure_mct_search import pure_mct_search_action
from uct_mct_search import uct_mct_search_action
from pv_mct_search import pv_mct_search
from config import PURE_MCT_SEARCH_NUM, UCT_MCT_SEARCH_NUM, PV_MCT_SEARCH_NUM
from numba import cuda
from tensorflow.keras import backend as bk

from game import State
import timeit
import time


def pure_mct_time(st: State):
    def f():
        pure_mct_search_action(100)(st)

    return f


def uct_mct_time(st: State):
    def f():
        uct_mct_search_action(100, 10)(st)

    return f


def pv_mct_time(st: State, md: Model):
    def f():
        pv_mct_search(md, 0.0)[1](st)

    return f


if __name__ == '__main__':
    cuda.get_current_device().reset()

    model: Model = load_model("models/cnn_best.h5")

    state = State()

    print("PV")
    print(timeit.Timer(pv_mct_time(state, model)).timeit(25))
    print("PURE")
    print(timeit.Timer(pure_mct_time(state)).timeit(25))
    print("UCT")
    print(timeit.Timer(uct_mct_time(state)).timeit(25))

    bk.clear_session()
    del model
