import random
import time
import math
from copy import deepcopy

from connected_four import ConnectState, GameMeta, MCTSMeta


class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}
        self.outcome = GameMeta.PLAYERS['none']

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child
    
    def get_exploration(self) -> float:
        # compute a dynamic exploration constant c based on the number of visits
        #base exploration constant
        c0 = MCTSMeta.EXPLORATION #1.4
        if self.parent is not None:
            root_visits = self.parent.N 
            alpha = 0.2
            return c0 / (1 +  alpha * math.log(1 + root_visits))
        return c0
        #decrease c -> more exploitation (attack)
        #increase c -> more exploration (defense)

    def value(self, explore: float = MCTSMeta.EXPLORATION):
        if self.N == 0:
            return GameMeta.INF
        else: #dynamic c
            c_now = self.get_exploration()
            return self.Q / self.N + c_now * math.sqrt(math.log(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state=ConnectState()):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.game_over():
            moves = state.get_legal_moves()
            player = state.to_play
            opp = 3 - player
            # Immediate win check
            for m in moves:
                s2 = deepcopy(state)
                s2.move(m)
                if s2.game_over() and s2.get_outcome() == player:
                    state.move(m)
                    break
            else:
                #Immediate threat blocking
                blocked = False
                for m in moves:
                    s2 = deepcopy(state)
                    s2.move(m)
                    for m2 in s2.get_legal_moves():
                        s3 = deepcopy(s2)
                        s3.move(m2)
                        if s3.game_over() and s3.get_outcome() == opp:
                            state.move(m)
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    state.move(random.choice(moves))

        return state.get_outcome()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == GameMeta.OUTCOMES['draw']:
                reward = 0
            else:
                reward = 1 - reward

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move

    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time
