import random
import numpy as np
from config import H, W, is_in, PURE_MCT_SEARCH_NUM


# Connect4 の盤面を管理
class State:
    def __init__(self, pieces=None, enemy_pieces=None, block=None):
        self.pieces = pieces if pieces is not None else np.zeros(H * W).reshape((H, W))
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else np.zeros(H * W).reshape((H, W))
        self.block = block if block is not None else np.zeros(H * W).reshape((H, W))

    @staticmethod
    def piece_count(pieces):
        count = 0
        for i in range(H):
            for j in range(W):
                count += pieces[i, j] == 1
        return count

    def is_lose(self):
        # 4 並びかどうかの判定
        def is_comp(x, y, dx, dy):
            for _ in range(4):
                if not is_in(x, y) or self.enemy_pieces[x, y] == 0:
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
        pieces = self.pieces.copy()
        for i in range(H - 1, -1, -1):
            if self.pieces[i, action] == 0 and self.enemy_pieces[i, action] == 0 and self.block[i, action] == 0:
                pieces[i, action] = 1
                break
            assert i != 0, "{} にコマを置くことはできません".format(action)
        return State(self.enemy_pieces.copy(), pieces, self.block.copy())

    def legal_actions(self):
        actions = []
        for j in range(W):
            if self.pieces[0, j] == 0 and self.enemy_pieces[0, j] == 0 and self.block[0, j] == 0:
                actions.append(j)
        return actions

    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        s = ""
        for i in range(H):
            for j in range(W):
                if self.pieces[i, j] == 1:
                    s += ox[0]
                elif self.enemy_pieces[i, j] == 1:
                    s += ox[1]
                elif self.block[i, j] == 1:
                    s += '#'
                else:
                    s += '-'
            s += '\n'
        return s


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


# 原始モンテカルロ木探索
def pure_mct_search_action(st: State):
    legal_actions = st.legal_actions()
    scores = np.zeros(len(legal_actions))
    for i, action in enumerate(legal_actions):
        next_st = st.next_state(action)
        for _ in range(PURE_MCT_SEARCH_NUM):
            scores[i] -= play_out(next_st)

    return legal_actions[np.argmax(scores)]


# プレイヤーを二人指定し，結果を返す
def play(next_actions):
    st = State()
    while not st.is_done():
        next_action = next_actions[0] if st.is_first_player() else next_actions[1]
        st = st.next_state(next_action(st))
    if st.is_draw():
        return 0
    else:
        return -1 if st.is_first_player() else 1


if __name__ == '__main__':
    play([player_action, pure_mct_search_action])
