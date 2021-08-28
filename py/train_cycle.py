from dual_network import make_cnn_model, make_dense_model, make_resnet_model
from self_play import self_play
from train_network import train_network
from evaluating import evaluate_change, evaluate_player
from config import DENSE_BEST_FILE, CNN_BEST_FILE, RESNET_BEST_FILE, DENSE_LATEST_FILE, RESNET_LATEST_FILE, \
    CNN_LATEST_FILE, CYCLE_NUM


def cycle(files):
    for i in range(CYCLE_NUM):
        print("Train", i)
        self_play(files[0], True)
        train_network(files[0], files[1])
        evaluate_change(files[0], files[1])
        vs_random, vs_pure, vs_uct = evaluate_player(files[0])
        log_eval("../scores/{}_vs_random".format(files[2]), vs_random)
        log_eval("../scores/{}_vs_pure_mct_search".format(files[2]), vs_pure)
        log_eval("../scores/{}_vs_uct_search".format(files[2]), vs_uct)


def log_eval(file_name, d):
    with open(file_name, mode="a") as f:
        f.write("{}\n".format(d))


if __name__ == '__main__':
    make_resnet_model()
    resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE, "ResNet"]
    cycle(resnet_files)
    # make_dense_model()
    # make_cnn_model()
    # make_resnet_model()
    # dense_files = [DENSE_BEST_FILE, DENSE_LATEST_FILE]
    # cnn_files = [CNN_BEST_FILE, CNN_LATEST_FILE]
    # resnet_files = [RESNET_BEST_FILE, RESNET_LATEST_FILE]
    # cycle(dense_files)
    # cycle(cnn_files)
    # cycle(resnet_files)
