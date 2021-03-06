from shutil import copy

from tensorflow.keras.models import Model, load_model
from py.game.pv_mct_search import pv_mct_search
from py.config import EVAL_COUNT, PURE_MCT_SEARCH_NUM, UCT_MCT_SEARCH_NUM
from py.game.game import play, random_action
import tensorflow.keras.backend as bk

from py.game.pure_mct_search import pure_mct_search_action
from py.game.uct_mct_search import uct_mct_search_action


def update_player(best_file, latest_file):
    copy(latest_file, best_file)
    print("Change Best Player")


def evaluate_change(best_file, latest_file):
    model0: Model = load_model(latest_file)
    model1: Model = load_model(best_file)

    next_action0 = pv_mct_search(model0, 1.0)[1]
    next_action1 = pv_mct_search(model1, 1.0)[1]
    next_actions = [next_action0, next_action1]

    total_point = 0
    for i in range(EVAL_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point -= play(list(reversed(next_actions)))
        print("\rEvaluate {}/{}".format(i + 1, EVAL_COUNT), end=" ")
    print()

    ave_point = total_point / EVAL_COUNT
    print("Average Point", ave_point)

    bk.clear_session()
    del model0
    del model1

    if ave_point > 0:
        update_player(best_file, latest_file)
        return True
    else:
        return False


# 2 つのアルゴリズム同士での勝率計算
def evaluate_algo(label, next_actions, is_random):
    total_point = 0
    for i in range(EVAL_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions, is_random=is_random)
        else:
            total_point -= play(list(reversed(next_actions)), is_random=is_random)
        print("\rEvaluate {}/{}".format(i + 1, EVAL_COUNT), end=" ")
    print()

    ave_point = total_point / EVAL_COUNT
    print(label, ave_point)
    return ave_point


# アルゴリズムの評価
def evaluate_player(best_file, is_random):
    model: Model = load_model(best_file)
    pv_action = pv_mct_search(model, 0.0)[1]

    next_actions = [pv_action, random_action]
    vs_random = evaluate_algo("{} VS Random".format(best_file), next_actions, is_random)

    next_actions = [pv_action, pure_mct_search_action(PURE_MCT_SEARCH_NUM)]
    vs_pure = evaluate_algo("{} VS Pure mct search".format(best_file), next_actions, is_random)

    next_actions = [pv_action, uct_mct_search_action(UCT_MCT_SEARCH_NUM, 10)]
    vs_uct = evaluate_algo("{} VS UCT mct search action".format(best_file), next_actions, is_random)

    bk.clear_session()
    del model
    return vs_random, vs_pure, vs_uct
