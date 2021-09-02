import time

import ray

from py.game.game import random_action
from py.game.pure_mct_search import pure_mct_search_action
from py.game.uct_mct_search import uct_mct_search_action
from py.initialize.dual_network import initialize_network, create_model
from py.game.pv_mct_search import pv_mct_search

from py.self_play.self_play import process_parallel
from py.train.train_network import train_network
from py.evaluate.evaluating import evaluate_change, evaluate_algo
from config import RESNET_BEST_FILE, RESNET_LATEST_FILE, \
    CYCLE_NUM, IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE, SELF_PLAY_COUNT, PURE_MCT_SEARCH_NUM, \
    UCT_MCT_SEARCH_NUM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as bk
import tensorflow as tf


@ray.remote(num_cpus=1, num_gpus=0)
def eval_parallel(weights, action, is_random, file, label):
    model = create_model()
    model.set_weights(weights)
    actions = [pv_mct_search(model, 0.0)[1], action]
    score = evaluate_algo(label, actions, is_random)
    log_eval(file, score)
    del model


@ray.remote(num_cpus=1, num_gpus=0)
def sub_battle(weights0, weights1, is_random, file, label):
    m0 = create_model()
    m1 = create_model()
    m0.set_weights(weights0)
    m1.set_weights(weights1)
    actions = [pv_mct_search(m0, 0.0)[1], pv_mct_search(m1, 0.0)[1]]
    score = evaluate_algo(label, actions, is_random)
    log_eval(file, score)
    del m0
    del m1


def log_eval(file_name, d):
    with open(file_name, mode="a") as f:
        f.write("{}\n".format(d))


def train(files, is_improved):
    start = time.perf_counter()
    process_parallel(files[0], is_improved, SELF_PLAY_COUNT)
    end = time.perf_counter()
    print("Self Play", end - start)

    start = time.perf_counter()
    train_network(files[0], files[1])
    end = time.perf_counter()
    print("Train", end - start)

    start = time.perf_counter()
    evaluate_change(files[0], files[1])
    end = time.perf_counter()
    print("Evaluate Change", end - start)

    start = time.perf_counter()

    model: Model = load_model(files[0])
    weights = model.get_weights()
    weights = ray.put(weights)
    actions = [random_action, pure_mct_search_action(PURE_MCT_SEARCH_NUM),
               uct_mct_search_action(UCT_MCT_SEARCH_NUM, 10)]
    randoms = [False, False, False, True, True, True]
    labels = ["VS Random", "VS Pure", "VS Uct", "VS Random in Random", "VS Pure in Random", "VS Uct in Random"]
    file = ["_vs_random", "_vs_pure_mct_search", "_vs_uct_search", "_vs_random_in_random",
            "_vs_pure_mct_search_in_random", "_vs_uct_search_in_random"]
    hs = []
    for x in range(6):
        h = eval_parallel.remote(weights, actions[x % 3], randoms[x], "../scores/{}{}".format(files[2], file[x]),
                                 labels[x])
        hs.append(h)
    for h in hs:
        ray.get(h)

    end = time.perf_counter()
    print("Evaluate", end - start)


if __name__ == '__main__':
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    initialize_network()
    ray.init(num_cpus=16, num_gpus=1)

    resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE, "ResNet"]
    improved_resnet_files = [IMPROVED_RESNET_BEST_FILE, IMPROVED_RESNET_LATEST_FILE, "ImprovedResNet"]

    for i in range(CYCLE_NUM):
        print("Train Cycle {}".format(i + 1))

        if i != 0:
            train(resnet_files, False)
        train(improved_resnet_files, True)

        s = time.perf_counter()
        model0: Model = load_model(IMPROVED_RESNET_BEST_FILE)
        model1: Model = load_model(RESNET_BEST_FILE)
        weight0 = model0.get_weights()
        weight1 = model1.get_weights()
        weight0 = ray.put(weight0)
        weight1 = ray.put(weight1)
        label0 = "Improved VS Normal"
        label1 = "Improved VS Normal in random"
        file0 = "../scores/improved_vs_normal"
        file1 = "../scores/improved_vs_normal_in_random"
        handles = [sub_battle.remote(weight0, weight1, False, file0, label0),
                   sub_battle.remote(weight0, weight1, True, file1, label1)]
        for handle in handles:
            ray.get(handle)
        e = time.perf_counter()
        print("Battle", e - s)

        bk.clear_session()
        del model0
        del model1
