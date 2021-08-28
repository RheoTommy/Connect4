from dual_network import make_cnn_model, make_dense_model, make_resnet_model
from self_play import self_play
from train_network import train_network
from evaluating import evaluate_change, evaluate_player
from config import DENSE_BEST_FILE, CNN_BEST_FILE, RESNET_BEST_FILE, DENSE_LATEST_FILE, RESNET_LATEST_FILE, \
    CNN_LATEST_FILE, CYCLE_NUM
from joblib import Parallel, delayed


def cycle(files):
    for i in range(CYCLE_NUM):
        print("Train", i)
        self_play(files[0])
        train_network(files[0], files[1])
        updated = evaluate_change(files[0], files[1])
        if updated:
            evaluate_player(files[0])


if __name__ == '__main__':
    make_dense_model()
    make_cnn_model()
    make_resnet_model()
    dense_files = [DENSE_BEST_FILE, DENSE_LATEST_FILE]
    cnn_files = [CNN_BEST_FILE, CNN_LATEST_FILE]
    resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE]
    cycle(dense_files)
    cycle(cnn_files)
    cycle(resnet_files)
