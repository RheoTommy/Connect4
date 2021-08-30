import time

import ray

from py.initialize.dual_network import initialize_network
from py.game.pv_mct_search import pv_mct_search

from py.self_play.self_play import process, process_parallel
from py.train.train_network import train_network
from py.evaluate.evaluating import evaluate_change, evaluate_player, evaluate_algo
from config import RESNET_BEST_FILE, RESNET_LATEST_FILE, \
    CYCLE_NUM, IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE, SELF_PLAY_COUNT
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as bk
import tensorflow as tf


def log_eval(file_name, d):
    with open(file_name, mode="a") as f:
        f.write("{}\n".format(d))


def train(files, is_improved):
    start = time.perf_counter()
    process_parallel(files[0], is_improved, SELF_PLAY_COUNT)
    end = time.perf_counter()
    print(end - start)

    start = time.perf_counter()
    train_network(files[0], files[1])
    end = time.perf_counter()
    print(end - start)

    start = time.perf_counter()
    evaluate_change(files[0], files[1])
    end = time.perf_counter()
    print(end - start)

    start = time.perf_counter()
    vs_random, vs_pure, vs_uct = evaluate_player(files[0], False)
    log_eval("../scores/{}_vs_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search".format(files[2]), vs_uct)
    end = time.perf_counter()
    print(end - start)

    start = time.perf_counter()
    vs_random, vs_pure, vs_uct = evaluate_player(files[0], True)
    log_eval("../scores/{}_vs_random_in_random".format(files[2]), vs_random)
    log_eval("../scores/{}_vs_pure_mct_search_in_random".format(files[2]), vs_pure)
    log_eval("../scores/{}_vs_uct_search_in_random".format(files[2]), vs_uct)
    end = time.perf_counter()
    print(end - start)


if __name__ == '__main__':
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    initialize_network()
    ray.init(num_cpus=16, num_gpus=1)

    resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE, "ResNet"]
    improved_resnet_files = [IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE, "ImprovedResNet"]

    for i in range(CYCLE_NUM):
        print("Train Cycle {}".format(i + 1))

        train(resnet_files, False)
        train(improved_resnet_files, True)

        model0: Model = load_model(IMPROVED_RESNET_BEST_FILE)
        model1: Model = load_model(RESNET_BEST_FILE)
        next_actions = [pv_mct_search(model0, 0.0)[1], pv_mct_search(model1, 0.0)[1]]
        improved_vs_normal = evaluate_algo("Improved VS Normal", next_actions, False)
        log_eval("../scores/improved_vs_normal", improved_vs_normal)
        improved_vs_normal_in_random = evaluate_algo("Improved VS Normal in random", next_actions, True)
        log_eval("../scores/improved_vs_normal_in_random", improved_vs_normal_in_random)

        bk.clear_session()
        del model0
        del model1
