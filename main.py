from chess import Board
import os
import chess
import tensorflow as tf
from tensorflow.python.client import device_lib

from networkv2 import NetworkV2
from mctsv2 import Mcts
import adapter
from learning import Coach

network = NetworkV2()
network.load_checkpoint()

c = Coach(network)
c.train_from_games('F:\Data Science\MChess\games\lichess_elite_2020-04.pgn')
