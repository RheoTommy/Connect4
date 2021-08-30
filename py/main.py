import time

from py.config import RESNET_BEST_FILE
from py.self_play import self_play_parallel
import tensorflow as tf

if __name__ == '__main__':
    for device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    start = time.perf_counter()
    self_play_parallel(RESNET_BEST_FILE)
    end = time.perf_counter()
    print(end - start)
