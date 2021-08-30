import random

from config import H, W, EVAL_COUNT, PURE_MCT_SEARCH_NUM, UCT_MCT_SEARCH_NUM
from game import State, random_action
from py.pure_mct_search import pure_mct_search_action
from py.pv_mct_search import pv_mct_search
from py.uct_mct_search import uct_mct_search_action
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as bk


# ランダムなブロック生成
def random_block():
    block_num = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0]
    random.shuffle(block_num)
    block = [0] * (H * W)
    for j in range(W):
        for i in range(H):
            if block_num[j] != 0:
                block[(H - i - 1) * W + j] = 1
                block_num[j] -= 1
    return block


# ランダムな初期状態作成
def random_state():
    return State(block=random_block())


# ランダムな初期状態で対戦
def play_random(next_actions, debug=False):
    st = random_state()
    while not st.is_done():
        if debug:
            print(st)
        next_action = next_actions[0] if st.is_first_player() else next_actions[1]
        st = st.next_state(next_action(st))
    if debug:
        print(st)
    if st.is_draw():
        return 0
    else:
        return -1 if st.is_first_player() else 1


# 2 つのアルゴリズム同士での勝率計算
def evaluate_algo_random(label, next_actions):
    total_point = 0
    for i in range(EVAL_COUNT):
        if i % 2 == 0:
            total_point += play_random(next_actions)
        else:
            total_point -= play_random(list(reversed(next_actions)))
        print("\rEvaluate {}/{}".format(i + 1, EVAL_COUNT), end=" ")
    print()

    ave_point = total_point / EVAL_COUNT
    print(label, ave_point)
    return ave_point


# アルゴリズムの評価
def evaluate_player_random(best_file):
    model: Model = load_model(best_file)
    pv_action = pv_mct_search(model, 0.0)[1]

    next_actions = [pv_action, random_action]
    vs_random = evaluate_algo_random("{} VS Random".format(best_file), next_actions)

    next_actions = [pv_action, pure_mct_search_action(PURE_MCT_SEARCH_NUM)]
    vs_pure = evaluate_algo_random("{} VS Pure mct search".format(best_file), next_actions)

    next_actions = [pv_action, uct_mct_search_action(UCT_MCT_SEARCH_NUM, 10)]
    vs_uct = evaluate_algo_random("{} VS UCT mct search action".format(best_file), next_actions)

    bk.clear_session()
    del model
    return vs_random, vs_pure, vs_uct
