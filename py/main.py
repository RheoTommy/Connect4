import time

from py.config import RESNET_BEST_FILE
from py.self_play import self_play

if __name__ == '__main__':
    start = time.perf_counter()
    self_play(RESNET_BEST_FILE)
    end = time.perf_counter()
    print(end - start)
