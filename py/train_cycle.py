import tensorflow.keras

from dual_network import make_cnn_model, make_dense_model, make_resnet_model, make_improved_resnet_model
from py.pv_mct_search import pv_mct_search
from py.randomized_game import evaluate_player_random, evaluate_algo_random
from self_play import self_play, improved_self_play
from train_network import train_network
from evaluating import evaluate_change, evaluate_player, evaluate_algo
from config import DENSE_BEST_FILE, CNN_BEST_FILE, RESNET_BEST_FILE, DENSE_LATEST_FILE, RESNET_LATEST_FILE, \
    CNN_LATEST_FILE, CYCLE_NUM, IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as bk


def train(files):
    self_play(files[0], True)
    train_network(files[0], files[1])
    evaluate_change(files[0], files[1])
    vs_random, vs_pure, vs_uct = evaluate_player(files[0])
    log_eval("../scores/{}_vs_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search".format(files[2]), vs_uct)
    vs_random, vs_pure, vs_uct = evaluate_player_random(files[0])
    log_eval("../scores/{}_vs_random_in_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search_in_random".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search_in_random".format(files[2]), vs_uct)


def improved_train(files):
    improved_self_play(files[0], True)
    train_network(files[0], files[1])
    evaluate_change(files[0], files[1])
    vs_random, vs_pure, vs_uct = evaluate_player(files[0])
    log_eval("../scores/{}_vs_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search".format(files[2]), vs_uct)
    vs_random, vs_pure, vs_uct = evaluate_player_random(files[0])
    log_eval("../scores/{}_vs_random_in_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search_in_random".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search_in_random".format(files[2]), vs_uct)


def log_eval(file_name, d):
    with open(file_name, mode="a") as f:
        f.write("{}\n".format(d))


if __name__ == '__main__':
    make_resnet_model()
    make_improved_resnet_model()
    resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE, "ResNet"]
    improved_resnet_files = [IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE, "ImprovedResNet"]

    for i in range(CYCLE_NUM):
        print("Train Cycle {}".format(i + 1))
        train(resnet_files)
        improved_train(improved_resnet_files)

        model0: Model = load_model(RESNET_BEST_FILE)
        model1: Model = load_model(IMPROVED_RESNET_BEST_FILE)
        next_actions = [pv_mct_search(model0, 0.0)[1], pv_mct_search(model1, 0.0)[1]]
        improved_vs_normal = evaluate_algo("Improved VS Normal", next_actions)
        log_eval("../scores/improved_vs_normal", improved_vs_normal)
        improved_vs_normal_in_random = evaluate_algo_random("Improved VS Normal in random", next_actions)
        log_eval("../scores/improved_vs_normal_in_random", improved_vs_normal_in_random)

        bk.clear_session()
        del model0
        del model1
