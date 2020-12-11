from chess import Board
import chess
import chess.pgn
from mctsv2 import Mcts
import adapter
from networkv2 import NetworkV2
import numpy as np

np.random.seed()
network = NetworkV2()
network.load_checkpoint()

board = Board()
board
count = 1
while True:
    mcts = Mcts(network)
    board.reset()

    # Create a game and set headers.
    game = chess.pgn.Game()
    game.headers["White"] = 'MChess'
    game.headers["Black"] = 'MChess'
    game.setup(board)

    node = game

    while not board.is_game_over():
        move, _ = mcts.select_move(board)
        board.push(move)
        print(board)
        node = node.add_variation(move) # Add game node

    print('------------------------------')
    print(board)
    print(board.fen())
    print(board.result())

    game.headers["Result"] = board.result()
    new_pgn = open(f'game.pgn', 'w')
    exporter = chess.pgn.FileExporter(new_pgn)
    game.accept(exporter)
    count += 1
    new_pgn.close()