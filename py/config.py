# 盤面の大きさ
H = 6
W = 7

# ニューラルネットワークの入力シェイプ
INPUT_SHAPE = (6, 7, 3)

# ニューラルネットワークの出力シェイプ
OUTPUT_SHAPE = 7

# モデルとファイル名
RESNET_LATEST_FILE = "../models/resnet_latest.h5"
RESNET_BEST_FILE = "../models/resnet_best.h5"
IMPROVED_RESNET_LATEST_FILE = "../models/improved_resnet_latest.h5"
IMPROVED_RESNET_BEST_FILE = "../models/improved_resnet_best.h5"

# ResNet の Conv のカーネル数
RESNET_N_SIZE = 128

# ResNet の残差ブロック数
RESNET_RESIDUAL_SIZE = 16

# 原始モンテカルロ木探索の各アクションのランダム試行回数
PURE_MCT_SEARCH_NUM = 50

# UCT モンテカルロ木探索のランダム試行回数
UCT_MCT_SEARCH_NUM = 50

# ニューラルネットワーク近似を利用したモンテカルロ木探索のランダム試行回数
PV_MCT_SEARCH_NUM = 50

# セルフプレイ一回のゲーム数
# SELF_PLAY_COUNT = 128
SELF_PLAY_COUNT = 1

# セルフプレイ時の温度パラメータ
SELF_PLAY_TEMP = 1.0

# 1 データの学習回数
# RN_EPOCHS = 100
RN_EPOCHS = 1

# 学習時のバッチサイズ
BATCH_SIZE = 128

# 評価時の対戦回数
# EVAL_COUNT = 25
EVAL_COUNT = 1

# 学習サイクルの実行回数
CYCLE_NUM = 10


# 盤面内かどうか
def is_in(x, y):
    return 0 <= x < H and 0 <= y < W
