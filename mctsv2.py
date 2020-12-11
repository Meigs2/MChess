import random
import chess
import numpy as np
import chess
import math
# Used to create unique zorbit hashes of the current board state.
from chess import polyglot
from tqdm import tqdm
from matplotlib import pyplot as plt

from network import Network
import adapter

import multiprocessing
from multiprocessing import Pool, Manager

end_states = {'1-0': 1, '0-1': -1, '1/2-1/2' : 0}

class Mcts():
    def __init__(self, network: Network):
        self.N_vals = {} # Number of times current node has been visited.
        self.N_vals_edges = {} # Number of times current board and action have been visited.
        self.Policy_vectors = {} # list of moves 
        self.Q_vals = {} # Q_vals
        
        self.c = 5
        self.network = network

    # Function that quickly determines value of the next move.
    def get_ucb(self, board_hash: int, move: chess.Move, move_prob: float):
        if (board_hash, move) in self.Q_vals:
            Q = self.Q_vals[(board_hash, move)]
            n_vals = self.N_vals[board_hash]
            edges = self.N_vals_edges[(board_hash, move)]
            sqrt = np.cbrt(n_vals)
            ret = Q + (self.c * move_prob * sqrt/(1 + edges))
            # ret = Q + (self.c * move_prob)
            return float(ret)
        else:
            n_vals = self.N_vals[board_hash]
            sqrt = np.cbrt(n_vals + 1e-2)
            return float(self.c * move_prob * sqrt)

    def update_values(self, board_hash: int, move: chess.Move, value: float):
        if (board_hash, move) in self.Q_vals:
            self.Q_vals[(board_hash, move)] = (self.N_vals_edges[(board_hash, move)] * self.Q_vals[(board_hash, move)] + value) / (self.N_vals_edges[(board_hash, move)] + 1)
            self.N_vals_edges[(board_hash, move)] += 1
        else:
            self.Q_vals[(board_hash, move)] = value
            self.N_vals_edges[(board_hash, move)] = 1
        
    def search(self, board: chess.Board):
        """
        Recursevly search a tree using UCT and the NNet to intellegently select the proper tree to search.
        """
        if board.is_game_over():
            result = board.result()
            return -end_states[result]

        board_hash = polyglot.zobrist_hash(board)
        
        if board_hash not in self.Policy_vectors:
            # Get nnet prediction of the current board for use in the 
            # Determine which branch to explore next.
            # If black to play, mirror the board, unmirror at end. NNet always sees current player as white.
            # Make a copy of the board as to preserve our move stack.
            temp = None
            if not board.turn:
                # If black to play, flip and mirror board.
                temp = board.transform(chess.flip_vertical)
                temp.apply_mirror()
            else:
                temp = board.copy()

            cannonical = adapter.get_cannonical(temp)
            policy_vector, nnet_value = self.network.predict(cannonical)
            
            # Mask out invalid moves
            valids = adapter.moves_to_policy_mask(list(board.legal_moves))
            policy_vector *= valids
            
            # Normalize vector, add valid moves if needed.
            if np.sum(policy_vector > 0):
                policy_vector /= np.linalg.norm(policy_vector)
                self.Policy_vectors[board_hash] = policy_vector
            else:
                print("All valid moves were masked. Adding valids. Warning if lots of these messages.")
                policy_vector += valids
                policy_vector /= np.linalg.norm(policy_vector)

            self.N_vals[board_hash] = 0
            del temp
            
            # Return the esimate until we actually reach the end.
            return -nnet_value

        # Iterate over legal moves and get the probability of making that move, according to the nnet.
        action_heuristic_dict = {}
        curr_move_policy = self.Policy_vectors[board_hash]
        for move in list(board.legal_moves):
            move_prob = adapter.get_move_prob(curr_move_policy, move)
            action_heuristic_dict[move] = self.get_ucb(board_hash, move, move_prob * 10.0)
        
        # Pick move with max value, make it bigger
        move = max(action_heuristic_dict, key=action_heuristic_dict.get)
        
        # action_heuristic_dict[max_move] *= 50.0
        # # Normalize
        # values = np.array(list(action_heuristic_dict.values()))
        # values += np.abs(values.min())
        # values /= np.linalg.norm(values)
        # p = self.fix_p(values)

        # move = np.random.choice(list(action_heuristic_dict.keys()), p=p)

        board.push(move)
        value = self.search(board)
        board.pop()

        # We've done our search, now we're back-propigating values for the next search.
        self.update_values(board_hash, move, value)
        self.N_vals[board_hash] += 1
        return -value

    def fix_p(self, p: np.ndarray):
        ret = np.nan_to_num(p, True, 1e-7)
        if ret.sum() != 1.0:
            ret = ret*(1./ret.sum())
        return ret

    def show_heatmap(self, board_hash, moves: list):
        array = np.zeros(64)
        for move in moves:
            array[move.to_square] = adapter.get_move_prob(self.Policy_vectors[board_hash], move)
        plt.cla()
        plt.imshow(np.flipud(array.reshape((8,8))))
        plt.pause(3)

    def select_move(self, board: chess.Board, n = 5, t = 1):
        # Perform number of MCTS's

        board_hash = polyglot.zobrist_hash(board)
        moves = list(board.legal_moves)
        
        # bar = tqdm(moves, desc="Searching all moves...")
        # for move in bar:
        #     board.push(move)
        #     for _ in range(n):
        #         self.search(board)
        #     board.pop()
        # bar.close()

        bar = tqdm(range(200), desc="Search")
        for _ in bar:
            self.search(board)
        bar.close()
        counts = [self.N_vals_edges[(board_hash, move)] if (board_hash, move) in self.N_vals_edges else 0 for move in moves]

        # [print(f'Move {move} had chance: {adapter.get_move_prob(self.Policy_vectors[board_hash], move)}') for move in moves]

        move = None
        if len(board.move_stack) < 0:
            # Play stocastically for first 4 full moves
            counts = [x ** (1.0 / t) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            index = np.random.choice(list(range(len(counts))), p=probs)
            move = moves[index]
        else:
            # Play Deterministically after move 7
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            move = moves[np.random.choice(bestAs)]
        
        # self.show_heatmap(board_hash, moves)
        print('-------------------------------------')
        print(f'Next move Q Value: {self.Q_vals[(board_hash, move)]}')
        print(f'Next move Pi Value: {adapter.get_move_prob(self.Policy_vectors[board_hash], move)}')
        print(f'Move: {move}')
        return move, adapter.move_to_policy(move)