from config import INPUT_SHAPE, RN_EPOCHS, BATCH_SIZE, DENSE_LATEST_FILE, DENSE_BEST_FILE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as bk
from pathlib import Path
import numpy as np
import pickle


# 学習データの読み込み
def load_data():
    history_path = sorted(Path("../data").glob("*.history"))[-1]
    with history_path.open(mode="rb") as f:
        return pickle.load(f)


# 学習率
def step_decay(epoch):
    if epoch >= 80:
        return 0.00025
    elif epoch >= 50:
        return 0.0005
    else:
        return 0.001


def train_network(best_model_path, latest_model_path):
    history = load_data()
    xs, y_policies, y_values = zip(*history[0])

    a, b, c = INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape((len(xs), c, a, b)).transpose((0, 2, 3, 1))
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    model: Model = load_model(best_model_path)
    model.compile(loss=["categorical_crossentropy", "mse"], optimizer="adam")

    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print("\rTrain {}/{}".format(epoch + 1, RN_EPOCHS), end=" ")
    )

    lr_decay = LearningRateScheduler(step_decay)

    model.fit(xs, [y_policies, y_values], batch_size=BATCH_SIZE, epochs=RN_EPOCHS, verbose=0,
              callbacks=[lr_decay, print_callback])

    model.save(latest_model_path)

    bk.clear_session()
    del model


if __name__ == '__main__':
    train_network(DENSE_BEST_FILE, DENSE_LATEST_FILE)
