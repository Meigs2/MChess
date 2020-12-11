import chess
import numpy as np

row_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
col_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
promotion_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}

def get_cannonical(board: chess.Board):
    """
    Given a python-chess board, convert it to a bit representation
    of the current gamestate.
    """
    array = np.zeros([8,8,12])

    m = board.piece_map()
    # magic >:)
    for i in m:
        row = int(i / 8)
        col = i % 8
        scale = int(not m[i].color) * 6
        z = (m[i].piece_type + scale) - 1
        array[row, col, z] = 1

    return array

def get_move_prob(pi: np.ndarray, move: chess.Move):
    """
    Extracts the probablilty, according to the policy vector, of making a legal move
    at the current position. Scalar from 0 to 1.
    """
    uci_move = str(move.uci())
    parts = [uci_move[i:i+2] for i in range(0, len(uci_move), 2)]
    
    c1, r1 = col_map[parts[0][:1]], row_map[parts[0][1:]]
    c2, r2 = col_map[parts[1][:1]], row_map[parts[0][1:]]

    layer = move.from_square * 64 # layer number
    pos = move.to_square #* 64 # cell number in 8x8 (normal moves)

    # break move into constituants.
    if len(uci_move) == 4:
        return pi[layer + pos]
    elif len(uci_move) == 5:
        r3 = promotion_map[parts[2]]
        r3 += 0 if r2 == 1 else 4
        return pi[layer + 64 + (((c2 - 8) + (r3 * 8) - 1))]
    else:
        raise Exception("I dont know what you did, but its BAD.")

def move_to_policy(move: chess.Move):
    mask = np.zeros(8192)
    
    uci_move = str(move.uci())
    parts = [uci_move[i:i+2] for i in range(0, len(uci_move), 2)]
    
    c1, r1 = col_map[parts[0][:1]], row_map[parts[0][1:]]
    c2, r2 = col_map[parts[1][:1]], row_map[parts[0][1:]]

    layer = move.from_square * 64 # layer number
    pos = move.to_square #* 64 # cell number in 8x8 (normal moves)

    # break move into constituants.
    if len(uci_move) == 4:
        mask[layer + pos] = 1
        return mask
    elif len(uci_move) == 5:
        r3 = promotion_map[parts[2]]
        r3 += 0 if r2 == 1 else 4
        mask[layer + 64 + (((c2 - 8) + (r3 * 8) - 1))] = 1
        return mask
    else:
        raise Exception("I dont know what you did, but its BAD.")
    pass

def moves_to_policy_mask(moves: list):
    mask = np.zeros(8192)
    for move in moves:
        mask += move_to_policy(move)
    return mask