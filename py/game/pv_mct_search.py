from math import sqrt

import numpy as np

from py.config import INPUT_SHAPE, PV_MCT_SEARCH_NUM
from py.game.game import State


# ボルツマン分布
def boltzman(xs, temp):
    xs = [x ** (1 / temp) for x in xs]
    return [x / sum(xs) for x in xs]


# モデルと盤面を指定して方策と盤面評価値を返す
def predict(model, st: State):
    a, b, c = INPUT_SHAPE
    x = np.array([st.pieces, st.enemy_pieces, st.block])
    x = x.reshape((c, a, b)).transpose((1, 2, 0)).reshape((1, a, b, c))

    y = model.predict(x, batch_size=1)

    policies = y[0][0][list(st.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1

    value = y[1][0][0]

    return policies, value


# ニューラルネットワーク近似を活用したモンテカルロ木探索
# ボルツマン分布に従ってランダム性を加える
def pv_mct_search(model, temp):
    class Node:
        def __init__(self, st: State, p):
            self.state = st
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = []

        @staticmethod
        def nodes_to_scores(nodes):
            scores = []
            for node in nodes:
                scores.append(node.n)
            return scores

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0
                self.w += value
                self.n += 1
                return value

            if len(self.child_nodes) == 0:
                polices, value = predict(model, self.state)
                self.w += value
                self.n += 1

                for ac, policy in zip(self.state.legal_actions(), polices):
                    self.child_nodes.append((Node(self.state.next_state(ac), policy)))
                return value
            else:
                value = -self.next_child_node().evaluate()
                self.w += value
                self.n += 1
                return value

        def next_child_node(self):
            t = sum(self.nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append(
                    ((-child_node.w / child_node.n) if child_node.n else 0.0) + child_node.p * sqrt(t) / (
                            1 + child_node.n))
            return self.child_nodes[np.argmax(pucb_values)]

    def score(st: State):
        root_node = Node(st, 0)

        for _ in range(PV_MCT_SEARCH_NUM):
            root_node.evaluate()

        scores = root_node.nodes_to_scores(root_node.child_nodes)
        if temp == 0.0:
            ac = np.argmax(scores)
            scores = np.zeros(len(scores))
            scores[ac] = 1
        else:
            scores = boltzman(scores, temp)
        return scores, root_node.w / root_node.n

    def action(st: State):
        return np.random.choice(st.legal_actions(), p=score(st)[0])

    return score, action
