from game import State
from pv_mct_search import pv_mct_search
from config import OUTPUT_SHAPE, SELF_PLAY_TEMP, SELF_PLAY_COUNT, DENSE_BEST_FILE, RESNET_BEST_FILE, INPUT_SHAPE, W, H
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as bk
import numpy as np
import pickle


# データサイズの修正
def resize_block(blocks):
    a, b, _ = INPUT_SHAPE
    res = np.zeros((a, b))
    for i in range(H):
        for j in range(W):
            res[i, j] = blocks[i, j]
    return res


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


# 1 ゲーム分の学習
def play(model):
    history = []
    st = State()
    while not st.is_done():
        scores, _ = pv_mct_search(model, SELF_PLAY_TEMP)[0](st)
        policies = [0] * OUTPUT_SHAPE
        for ac, policy in zip(st.legal_actions(), scores):
            policies[ac] = policy
        history.append(
            [[resize_block(st.pieces), resize_block(st.enemy_pieces), resize_block(st.block)], policies, None])
        action = np.random.choice(st.legal_actions(), p=scores)
        st = st.next_state(action)

    value = first_player_value(st)
    for i in range(len(history)):
        history[i][2] = value
        value *= -1
    return history


# 指定回数自己対戦を行い学習データを作成する
def self_play(best_model_path, debug=False):
    history = []
    model: Model = load_model(best_model_path)
    if debug:
        print("Self Play Started")
    for i in range(SELF_PLAY_COUNT):
        if debug:
            print("\rSelf Play {}/{}".format(i + 1, SELF_PLAY_COUNT), end=" ")
        h = play(model)
        history.extend(h)
    if debug:
        print()

    write_data(history)

    bk.clear_session()
    del model


# 1 ゲーム分の学習
def improved_play(model):
    history = []
    st = State()
    while not st.is_done():
        scores, value = pv_mct_search(model, SELF_PLAY_TEMP)[0](st)
        policies = [0] * OUTPUT_SHAPE
        for ac, policy in zip(st.legal_actions(), scores):
            policies[ac] = policy
        history.append(
            [[resize_block(st.pieces), resize_block(st.enemy_pieces), resize_block(st.block)], policies, value])
        action = np.random.choice(st.legal_actions(), p=scores)
        st = st.next_state(action)

    value = first_player_value(st)
    for i in range(len(history)):
        history[i][2] += value
        history[i][2] /= 2
        value *= -1
    return history


# 指定回数自己対戦を行い学習データを作成する
def improved_self_play(best_model_path, debug=False):
    history = []
    model: Model = load_model(best_model_path)
    if debug:
        print("Self Play Started")
    for i in range(SELF_PLAY_COUNT):
        if debug:
            print("\rSelf Play {}/{}".format(i + 1, SELF_PLAY_COUNT), end=" ")
        h = improved_play(model)
        history.extend(h)
    if debug:
        print()

    write_data(history)

    bk.clear_session()
    del model


if __name__ == '__main__':
    improved_self_play(RESNET_BEST_FILE, True)
