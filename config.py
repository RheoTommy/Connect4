# 盤面の大きさ (最大 15)
H = 10
W = 10

# ニューラルネットワークの入力シェイプ
INPUT_SHAPE = (15, 15, 3)

# ニューラルネットワークの出力シェイプ
OUTPUT_SHAPE = 15

# モデルとファイル名
DENSE_LATEST_FILE = "models/dense_latest.h5"
DENSE_BEST_FILE = "models/dense_best.h5"
CNN_LATEST_FILE = "models/cnn_latest.h5"
CNN_BEST_FILE = "models/cnn_best.h5"
RESNET_LATEST_FILE = "models/resnet_latest.h5"
RESNET_BEST_FILE = "models/resnet_best.h5"

# Dense のニューロン数
DENSE_N_SIZE = 128

# CNN の Conv のカーネル数
CNN_N_SIZE = 128

# ResNet の Conv のカーネル数
RESNET_N_SIZE = 128

# ResNet の残差ブロック数
RESNET_RESIDUAL_SIZE = 16

# 原始モンテカルロ木探索の各アクションのランダム試行回数
PURE_MCT_SEARCH_NUM = 200

# UCT モンテカルロ木探索のランダム試行回数
UCT_MCT_SEARCH_NUM = 200

# ニューラルネットワーク近似を利用したモンテカルロ木探索のランダム試行回数
PV_MCT_SEARCH_NUM = 50


# 盤面内かどうか
def is_in(x, y):
    return 0 <= x < H and 0 <= y < W
