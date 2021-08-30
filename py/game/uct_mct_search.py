# UCB モンテカルロ木探索
import math

import numpy as np

from py.game.game import State, play_out


# UCT スコアを使ったモンテカルロ木探索
def uct_mct_search_action(num, expand_num):
    class Node:
        def __init__(self, st: State):
            self.state = st
            self.n = 0
            self.w = 0
            self.child_nodes = []

        def expand(self):
            for ac in self.state.legal_actions():
                self.child_nodes.append(Node(self.state.next_state(ac)))

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0
                self.w += value
                self.n += 1
                return value

            if len(self.child_nodes) == 0:
                value = play_out(self.state)
                self.w += value
                self.n += 1
                if self.n == expand_num:
                    self.expand()
                return value
            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value

        def next_child_node(self):
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            t = 0
            for c in self.child_nodes:
                t += c.n
            uct_values = []
            for child_node in self.child_nodes:
                uct_values.append(-child_node.w / child_node.n + (2 * math.log(t) / child_node.n) ** 0.5)

            return self.child_nodes[np.argmax(np.array(uct_values))]

    def f(st: State):
        root_node = Node(st)
        root_node.expand()
        for _ in range(num):
            root_node.evaluate()
        legal_actions = st.legal_actions()
        n_list = []
        for c in root_node.child_nodes:
            n_list.append(c.n)
        return legal_actions[np.argmax(np.array(n_list))]

    return f
