# 原始モンテカルロ木探索
import numpy as np

from py.game.game import State, play_out


# 原始モンテカルロ木探索
def pure_mct_search_action(num):
    def f(st: State):
        legal_actions = st.legal_actions()
        scores = np.zeros(len(legal_actions))
        for i, action in enumerate(legal_actions):
            next_st = st.next_state(action)
            for _ in range(num // len(legal_actions)):
                scores[i] -= play_out(next_st)

        return legal_actions[np.argmax(scores)]

    return f
