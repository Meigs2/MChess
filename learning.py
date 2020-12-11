import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from collections import deque
from random import shuffle

from network import Network
from mctsv2 import Mcts
import adapter

end_states = {'1-0': 1, '0-1': -1, '1/2-1/2' : 0}

class Coach():
    """
    docstring
    """
    def __init__(self, nnet: Network):
        self.nnet = nnet
        self.pnet = self.nnet.__class__()  # the competitor network
        self.mcts = Mcts(self.nnet)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def learn(self):
        # load current nnet
        network = Network()
        for i in range(1, 200 + 1):
            print(i)
            iterationTrainExamples = deque([], maxlen=100000)

            # Play 10 games of self play.
            for _ in tqdm(range(50), desc="Self Play"):
                self.mcts = Mcts(network)  # reset search tree
                iterationTrainExamples += self.self_play(network)
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > 75:
                print(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory[::2]:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder='./temp/', filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder='./temp/', filename='temp.pth.tar')
            prev_mcts = Mcts(self.pnet)

            self.nnet.train(trainExamples)
            new_mcts = Mcts(self.nnet)

            nwins, pwins, draws = self.play_against(new_mcts, prev_mcts)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder='./temp/', filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder='./temp/', filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder='./temp/', filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def play_against(self, p1: Mcts, p2: Mcts, num = 20):
        board = chess.Board()
        num = int(num / 2)
        p1Won = 0
        p2Won = 0
        draws = 0

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            board.reset()
            while not board.is_game_over():
                if board.turn:
                    move, _ = p1.select_move(board)
                    board.push(move)
                else:
                    move, _ = p2.select_move(board)
                    board.push(move)
            result = board.result()
            r = end_states[result]
            if r == 1:
                p1 += 1
            if r == -1:
                p2 += 1
            else:
                draws += 1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            board.reset()
            while not board.is_game_over():
                if board.turn:
                    move, _ = p2.select_move(board)
                    board.push(move)
                else:
                    move, _ = p1.select_move(board)
                    board.push(move)
            result = board.result()
            r = end_states[result]
            if r == 1:
                p2 += 1
            if r == -1:
                p1 += 1
            else:
                draws += 1

        return p1Won, p2Won, draws

    def self_play(self, network: Network):
        """
        self-play action. 2 engines play against each other and record their results
        to be used for training purposes. No need to generalize the algorithm here, 
        we're only concerned with chess. Adapter class is used to adapt python-chess
        to positions the algorithm:tm: can understand. Opening book can be given to have the
        network train on a specific opening, or multiple openings instead of the standard
        chess opening. (implementing later)
        """
        examples = []
        board = chess.Board()

        # play game and record steps
        while True:
            move, pi = self.mcts.select_move(board)
            temp = None
            if not board.turn:
                temp = board.transform(chess.flip_vertical)
                temp.apply_mirror()
            else:
                temp = board.copy()

            cannonical = adapter.get_cannonical(temp)
            examples.append((cannonical, board.turn, pi, None))
            
            board.push(move)
            if board.is_game_over():
                result = board.result()
                r = end_states[result]
                print(board.result())
                return [(x[0], x[1], r * (1 if x[2] else -1)) for x in examples]

    def train_from_games(self, path):
        gameHistory = []
        shuffled_history = []
        pgn = open(path)
        game = chess.pgn.read_game(pgn)
        game_count = 1
        while game:
            if game_count > 128:
                shuffle(gameHistory)
                shuffled_history.extend(gameHistory[:512])
                game_count = 1
                gameHistory.clear()
            # Train on 10000 states of chess
            if len(shuffled_history) >= 131072:
                self.nnet.train(shuffled_history)
                self.nnet.save_checkpoint()
                return

            # Load game otherwise
            board = chess.Board()
            result = game.headers['Result']
            termination = game.headers['Termination']

            if termination != 'Normal':
                game = chess.pgn.read_game(pgn)
                continue

            moves = list(game.mainline_moves())
            examples = []
            for move in moves:
                # If black to play, flip and mirror board so current player is always white.
                temp = None
                if not board.turn:
                    temp = board.transform(chess.flip_vertical)
                    temp.apply_mirror()
                else:
                    temp = board.copy()

                examples.append((adapter.get_cannonical(temp), adapter.move_to_policy(move), board.turn, None))
                board.push(move)
            
            r = end_states[result]
            for x in examples:
                state = (x[0], x[1], r * (1 if x[2] else -1))
                gameHistory.append(state)
            game = chess.pgn.read_game(pgn)
            game_count += 1


    def retrain(self, history: [], network: Network):
        """
        docstring
        """
        pass
