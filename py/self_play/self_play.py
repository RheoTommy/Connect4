import pickle
from datetime import datetime

import numpy as np
import ray

from py.config import SELF_PLAY_TEMP, OUTPUT_SHAPE, RESNET_BEST_FILE
from py.game.game import State

from py.game.pv_mct_search import pv_mct_search
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as bk
from sys import argv
import tensorflow as tf

from py.initialize.dual_network import create_model


# 先手にとってのゲーム結果
def first_player_value(st: State):
    if st.is_lose():
        return -1 if st.is_first_player() else 1
    return 0


# 学習データの保存
def write_data(history):
    now = datetime.now()
    path = "../data/{:04}{:02}{:02}{:02}{:02}{:02}.history".format(
        now.year, now.month, now.day, now.hour, now.minute, now.second
    )
    with open(path, mode="wb") as f:
        pickle.dump(history, f)


def play(model: Model, is_improved):
    history = []
    st = State()
    while not st.is_done():
        scores, value = pv_mct_search(model, SELF_PLAY_TEMP)[0](st)
        policies = [0] * OUTPUT_SHAPE
        for ac, policy in zip(st.legal_actions(), scores):
            policies[ac] = policy
        history.append(
            [[st.pieces, st.enemy_pieces, st.block], policies, None if not is_improved else value])
        action = np.random.choice(st.legal_actions(), p=scores)
        st = st.next_state(action)

    value = first_player_value(st)
    for i in range(len(history)):
        if is_improved:
            history[i][2] += value
            history[i][2] /= 2
        else:
            history[i][2] = value
        value *= -1
    return history


# 指定回数自己対戦を行い学習データを作成する
def self_play(path, is_improved, num):
    history = []
    model: Model = load_model(path)
    print("Self Play Started")
    for i in range(num):
        print("\rSelf Play {}/{}".format(i + 1, num), end=" ")
        h = play(model, is_improved)
        history.extend(h)
    print()

    bk.clear_session()
    del model

    return history


@ray.remote(num_cpus=1, num_gpus=0)
def self_play_parallel(is_improved, num, weights):
    model = create_model()
    model.set_weights(weights)
    history = []
    print("Self Play Started")
    for i in range(num):
        print("\rSelf Play {}/{}".format(i + 1, num), end=" ")
        h = play(model, is_improved)
        history.extend(h)
    print()

    bk.clear_session()
    del model

    return history


def process_parallel(path, is_improved, num):
    model: Model = load_model(path)
    weights = model.get_weights()
    weights = ray.put(weights)
    res = [self_play_parallel.remote(is_improved, num // 16, weights) for _ in range(16)]
    history = []
    for r in res:
        history.extend(ray.get(r))
    write_data(history)


def process(path, is_improved, num):
    history = self_play(path, is_improved, num)
    write_data(history)


if __name__ == '__main__':
    # for device in tf.config.experimental.list_physical_devices("GPU"):
    #     tf.config.experimental.set_memory_growth(device, True)
    # model_path = argv[0]
    # improved = int(argv[1]) == 1
    # count = int(argv[2])
    # process(model_path, improved, count)
    process_parallel(RESNET_BEST_FILE, False, 16)
