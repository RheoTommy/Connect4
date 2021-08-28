from shutil import copy

from tensorflow.keras.models import Model, load_model
from pv_mct_search import pv_mct_search
from py.config import EVAL_COUNT, DENSE_LATEST_FILE, DENSE_BEST_FILE
from py.game import play
import tensorflow.keras.backend as bk


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
            total_point -= play(next_actions)
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


if __name__ == '__main__':
    evaluate_change(DENSE_BEST_FILE, DENSE_LATEST_FILE)
