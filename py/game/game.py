import copy
import random
import numpy as np
from py.config import H, W, is_in


# Connect4 の盤面を管理
class State:
    def __init__(self, pieces=None, enemy_pieces=None, block=None):
        self.pieces = pieces if pieces is not None else [0] * (H * W)
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [0] * (H * W)
        self.block = block if block is not None else [0] * (H * W)

    @staticmethod
    def piece_count(pieces):
        count = 0
        for i in range(H):
            for j in range(W):
                count += pieces[i * W + j] == 1
        return count

    def is_lose(self):
        # 4 並びかどうかの判定
        def is_comp(x, y, dx, dy):
            for _ in range(4):
                if not is_in(x, y) or self.enemy_pieces[x * W + y] == 0:
                    return False
                x += dx
                y += dy
            return True

        for i in range(H):
            for j in range(W):
                if is_comp(i, j, 1, 0) or is_comp(i, j, 0, 1) or is_comp(i, j, 1, -1) or is_comp(i, j, -1, 1):
                    return True
        return False

    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) + self.piece_count(
            self.block) == H * W and not self.is_lose()

    def is_done(self):
        return self.is_draw() or self.is_lose()

    # action は 0..W
    def next_state(self, action):
        pieces = copy.deepcopy(self.pieces)
        for i in range(H - 1, -1, -1):
            if self.pieces[i * W + action] == 0 and self.enemy_pieces[i * W + action] == 0 \
                    and self.block[i * W + action] == 0:
                pieces[i * W + action] = 1
                break
            assert i != 0
        return State(copy.deepcopy(self.enemy_pieces), pieces, copy.deepcopy(self.block))

    def legal_actions(self):
        actions = []
        for j in range(W):
            if self.pieces[0 * W + j] == 0 and self.enemy_pieces[0 * W + j] == 0 and self.block[0 * W + j] == 0:
                actions.append(j)
        return actions

    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        s = ""
        for i in range(H):
            for j in range(W):
                if self.pieces[i * W + j] == 1:
                    s += ox[0]
                elif self.enemy_pieces[i * W + j] == 1:
                    s += ox[1]
                elif self.block[i * W + j] == 1:
                    s += '#'
                else:
                    s += '-'
            s += '\n'
        s += str(self.legal_actions())
        return s


# ランダムなブロック生成
def random_block():
    block_num = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0]
    random.shuffle(block_num)
    block = [0] * (H * W)
    for j in range(W):
        for i in range(H):
            if block_num[j] != 0:
                block[(H - i - 1) * W + j] = 1
                block_num[j] -= 1
    return block


# ランダムな初期状態作成
def random_state():
    return State(block=random_block())


# 完全ランダム
def random_action(st: State):
    legal_actions = st.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# コンソールから選択
def player_action(st: State):
    print(st)
    print(st.legal_actions())
    action = int(input())
    return action


# 完全ランダムでプレイアウトし，結果を返す
def play_out(st: State):
    if st.is_lose():
        return -1
    elif st.is_done():
        return 0
    else:
        return -play_out(st.next_state(random_action(st)))


# プレイヤーを二人指定し，結果を返す
def play(next_actions, debug=False, is_random=False):
    st = State()
    if is_random:
        st = random_state()
    while not st.is_done():
        if debug:
            print(st)
        next_action = next_actions[0] if st.is_first_player() else next_actions[1]
        st = st.next_state(next_action(st))
    if debug:
        print(st)
    if st.is_draw():
        return 0
    else:
        return -1 if st.is_first_player() else 1


# プレイヤーを二人指定し，指定回数だけ対戦させたときの勝率を計算する
def battle_players(next_actions, num, debug=False):
    scores = np.zeros(2)
    for i in range(num):
        if debug:
            print("\rBattle {}/{}".format(i + 1, num), end=" ")
        score = play(next_actions)
        scores[0] += score
        scores[1] -= score
    if debug:
        print()

    scores /= num
    return scores
