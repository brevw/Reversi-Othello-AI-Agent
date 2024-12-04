# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves, get_directions

class TimeoutException(Exception):
    pass

PLAYER_WINS, TIE, OPPONENT_WINS = -1.0, 0.0, +1.0

_6_BY_6_POSITIONAL_WEIGHTS = np.array([
    [ 20,  0,  10,  10,  0,  20],
    [  0,  0,   5,   5,  0,   0],
    [ 10,  5,   2,   2,  5,  10],
    [ 10,  5,   2,   2,  5,  10],
    [  0,  0,   2,   2,  0,   0],
    [ 20,  0,   5,   5,  0,  20]
], dtype=int)

_8_BY_8_POSITIONAL_WEIGHTS = np.array([
    [ 30,   0,  20,   5,   5,  20,   0,  30],
    [  0,   0,   5,   5,   5,   5,   0,   0],
    [ 20,   5,   2,   2,   2,   2,   5,  20],
    [  1,   5,   2,   2,   2,   2,   5,   1],
    [  1,   5,   2,   2,   2,   2,   5,   1],
    [ 20,   5,   2,   2,   2,   2,   5,  20],
    [  0,   0,   5,   5,   5,   5,   0,   0],
    [ 30,   0,  20,   1,   1,  20,   0,  30]
], dtype=int)

_10_BY_10_POSITIONAL_WEIGHTS = np.array([
    [ 40,   0,  20,  10,   1,   1,  10,  20,   0,  40],
    [  0,   0,   5,   5,   5,   5,   5,   5,   0,   0],
    [ 20,   5,   1,   2,   2,   2,   2,   1,   5,  20],
    [ 10,   5,   1,   2,   2,   2,   2,   1,   5,  10],
    [  1,   5,   1,   2,   2,   2,   2,   1,   5,   1],
    [  1,   5,   1,   2,   2,   2,   2,   1,   5,   1],
    [ 10,   5,   1,   2,   2,   2,   2,   1,   5,  10],
    [ 20,   5,   1,   1,   1,   1,   1,   1,   5,  20],
    [  0,   0,   5,   5,   5,   5,   5,   5,   0,   0],
    [ 40,   0,  20,   10,  1,   1,   10,  20,  0,  40]
])

_12_BY_12_POSITIONAL_WEIGHTS = np.array([
    [ 50,  70,  50,  20,  10,   1,   1,  10,  20,  30,   0, 100],
    [  0,  10,  10,  10,  10,  10,  10,  10,  10,  10,   0,   0],
    [ 30,  10,   5,   5,   5,   5,   5,   5,   5,   5,  10,  30],
    [ 20,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,  20],
    [ 10,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,  10],
    [  1,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,   1],
    [  1,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,   1],
    [ 10,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,  10],
    [ 20,  10,   5,   2,   2,   2,   2,   2,   2,   5,  10,  20],
    [ 30,  10,   5,   5,   5,   5,   5,   5,   5,   5,  10,  30],
    [  0,   0,  10,  10,  10,  10,  10,  10,  10,  10,   0,   0],
    [100,   0,  50,  20,  10,   1,   1,  10,  20,  30,   0, 100]
])

STATIC_WEIGHTS = {
  6  : _6_BY_6_POSITIONAL_WEIGHTS,
  8  : _8_BY_8_POSITIONAL_WEIGHTS,
  10 : _10_BY_10_POSITIONAL_WEIGHTS,
  12 : _12_BY_12_POSITIONAL_WEIGHTS
}

STEP_SIZE = {
  6  : 1,
  8  : 1,
  10 : 1,
  12 : 1
}

STARTING_DEPTH = {
  6  : 3,
  8  : 2,
  10 : 2,
  12 : 2
}

DEBUG = True

# piece_advantage - actual_mobility_advantage - positional_advantage - corner_occupancy - stability
EVAL_WEIGHTS_START = {
    6: np.array([2, 3, 5, 4, 6]),
    8: np.array([1, 1, 1, 1, 1]),
    10: np.array([2, 3, 5, 4, 6]),
    12: np.array([2, 3, 5, 4, 6]),
}
EVAL_WEIGHTS_MID = {
    6: np.array([2, 3, 5, 4, 6]),
    8: np.array([1, 3, 5, 10, 5]),
    10: np.array([2, 3, 5, 4, 6]),
    12: np.array([2, 3, 5, 4, 6]),
}
EVAL_WEIGHTS_END = {
    6: np.array([2, 3, 5, 4, 6]),
    8: np.array([5, 1, 0, 5, 10]),
    10: np.array([2, 3, 5, 4, 6]),
    12: np.array([2, 3, 5, 4, 6]),
}

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.time_limit = 1.99 # 2 second at max for the search 

  def step(self, chess_board, player, opponent):
    """
    Finds the next best move for the player
    """

    start_time = time.time()

    step_size = STEP_SIZE[chess_board.shape[0]]
    max_depth = STARTING_DEPTH[chess_board.shape[0]]
    best_move = None
    temp = 1
    while temp:
      temp = self.alpha_beta_pruning_depth_limited(chess_board, player, opponent, start_time, max_depth)
      if temp: 
        best_move = temp
      else:
        break
      max_depth += step_size
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, f"seconds, best move found at depth {max_depth}")
    chess_board_copy = chess_board.copy()
    print(f"board:\n {chess_board}")
    print(f"best_move: {best_move}")
    execute_move(chess_board_copy, best_move, player)
    if DEBUG:
      print(self.evaluate_board(chess_board_copy, player, opponent, None, debug=DEBUG))
      print(self.evaluate_board(chess_board_copy, player, opponent, None, debug=False), '\n')
    return best_move

  # evaluation metrics
  def piece_advantage(self, board: np.array, player, opponent) -> float: 
      """
      Heuristic based on the number of captured pieces of the player compared to the opponent
      """
      player_pieces, opponent_pieces = np.sum(board == player), np.sum(board == opponent)
      return (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
  
  def actual_mobility_advantage(self, board: np.array, player, opponent) -> float:
      """
      Heuristic based on the number of moves the player can make compared to the opponent
      """
      player_actual_mobility, opponent_actual_mobility = len(get_valid_moves(board, player)), len(get_valid_moves(board, opponent))
      return 0 if player_actual_mobility + opponent_actual_mobility == 0 else (player_actual_mobility - opponent_actual_mobility) / (player_actual_mobility + opponent_actual_mobility)

  def potential_mobility_advantage(self, board: np.array, player, opponent) -> float: 
    """
    Heuristic based on long term mobility. It looks at the number of empty cells around each player or opponent piece
    """
    player_potential_mobility, opponent_potential_mobility = 0, 0 
    directions = get_directions()
    rows, cols = board.shape
    for r in rows: 
      for c in cols:
        if board[r, c] == player:
          for dr, dc in directions: 
            nr, nc = r + dr, c + dc 
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1] and board[nr, nc] == 0: 
              opponent_potential_mobility += 1
        elif board[r, c] == opponent: 
          for dr, dc in directions: 
            nr, nc = r + dr, c + dc 
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1] and board[nr, nc] == 0:
              player_potential_mobility += 1
    
    return 0 if player_potential_mobility + opponent_potential_mobility == 0 else (player_potential_mobility - opponent_potential_mobility) / (player_potential_mobility + opponent_potential_mobility)
  
  def positional_advantage(self, chess_board, player, opponent) -> float:
    """
      Heuristic based on the static weight matrices
    """
    weights = STATIC_WEIGHTS[chess_board.shape[0]]
    player_poisitional, opponent_poisitional = np.sum((chess_board == player) * weights), np.sum((chess_board == opponent) * weights)
    return 0 if player_poisitional + opponent_poisitional == 0 else ((player_poisitional - opponent_poisitional) / (player_poisitional + opponent_poisitional))

  def corner_occupancy(self, board: np.array, player, opponent) -> float:
      """
      Heuristic based on the number of corners occupied by the player compared to the opponent
      """
      rows, cols = [0, 0, -1, -1], [0, -1, 0, -1]
      player_corners, opponent_corners = np.sum(board[rows, cols] == player), np.sum(board[rows, cols] == opponent)
      return (player_corners - opponent_corners) / 4.0
  
  def stability(self, board: np.array, player, opponent) -> float:
    """
    Heuristic based on the number of corners occupied by the player compared to the opponent
    """
    
    player_stability, opponent_stability = 0, 0
    directions = [
        (0, 1),  
        (1, 0),  
        (1, 1),  
        (1, -1), 
    ]
    rows, cols = board.shape
    count = 0
    for r in range(rows):
      for c in range(cols):
          current = board[r, c]
          if current == 0:  
              continue
          count = 0
          for dr, dc in directions:
            count = 0
            for i in [-1, 1]:
              dr_ = dr * i
              dc_ = dc * i
              rr, cc = r + dr_, c + dc_
              while 0 <= rr < rows and 0 <= cc < cols:
                if board[rr, cc] != current:  
                  count += 1
                  break
                rr += dr_
                cc += dc_
              if count == 0: 
                break
            if count == 2: 
              break
                
          if count != 2: 
              if current == player:
                  player_stability += 1
              else:
                  opponent_stability += 1
    return 0 if player_stability + opponent_stability == 0 else (player_stability - opponent_stability) / (player_stability + opponent_stability)
  
  def evaluate_board(self, board: np.array, player, opponent, end_state, debug = False) -> float:
    """
    Evaluates the board based on a weighted sum of heuristics
    """
    if end_state is not None: 
      return end_state
    
    size = board.shape[0]
    total_pieces = np.sum(board != 0) 
    end_game = total_pieces > (size * size * 2 // 3)
    mid_game = total_pieces > (size * size // 3) 
        
    weights = EVAL_WEIGHTS_END[size] if end_game else (EVAL_WEIGHTS_MID[size] if mid_game else EVAL_WEIGHTS_START[size])
    
    eval = [
            0 if weights[0] == 0 else self.piece_advantage(board, player, opponent),
            0 if weights[1] == 0 else self.actual_mobility_advantage(board, player, opponent),
            0 if weights[2] == 0 else self.positional_advantage(board, player, opponent),
            0 if weights[3] == 0 else self.corner_occupancy(board, player, opponent),
            0 if weights[4] == 0 else self.stability(board, player, opponent)
          ]
    
    return eval if debug else np.sum(eval * weights) / np.sum(weights)
  
  
  

  def move_ordering_evaluator(self, board, color, player_score, opponent_score):
      # Corner positions are highly valuable
      rows, cols = [0, 0, -1, -1], [0, -1, 0, -1]
      corner_score = np.sum(board[rows, cols] == color) * 10
      corner_penalty = np.sum(board[rows, cols] == 3 - color) * 10 * -10

      # Mobility: the number of moves the opponent can make
      opponent_moves = len(get_valid_moves(board, 3 - color))
      mobility_score = -opponent_moves

      # Combine scores
      total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
      return total_score

  def alpha_beta_pruning_depth_limited(self, chess_board: np.array, player, opponent, start_time: float, depth_limit: int):
    """
    Depth limited tree search via alpha beta pruning 
    """
    def max_value(board, alpha, beta, depth) -> float:
      if time.time() - start_time > self.time_limit:
        raise TimeoutException()
      # do not exceed max depth
      end_game, player_score, opponent_score = check_endgame(board, player, opponent)
      end_state = None 
      if end_game:
        end_state = PLAYER_WINS if player_score > opponent_score else (TIE if player_score == opponent_score else OPPONENT_WINS)
      if depth >= depth_limit or end_game:
        return self.evaluate_board(board, player, opponent, end_state)
      
      valid_moves = get_valid_moves(chess_board, player)
      if not valid_moves:
        # switch to other player
        return min_value(chess_board, alpha, beta, depth)
      
      for move in valid_moves: 
        board_copy = board.copy()
        execute_move(board_copy, move, player)
        min_val = min_value(board_copy, alpha, beta, depth + 1)
        # equiv to alpha = max(alpha, ...)
        if alpha < min_val: 
          alpha = min_val

        if alpha >= beta: 
          return beta
        
      return alpha

    def min_value(board, alpha, beta, depth) -> float: 
      if time.time() - start_time > self.time_limit:
        raise TimeoutException()
      # do not exceed max depth
      end_game, opponent_score, player_score  = check_endgame(board, opponent, player)
      end_state = None 
      if end_game:
        end_state = PLAYER_WINS if player_score > opponent_score else (TIE if player_score == opponent_score else OPPONENT_WINS)
      if depth >= depth_limit or end_game:
        return self.evaluate_board(board, player, opponent, end_state)
      
      valid_moves = get_valid_moves(chess_board, opponent)
      if not valid_moves:
        # switch to other player
        return max_value(chess_board, alpha, beta, depth)
      
      for move in valid_moves: 
        board_copy = board.copy()
        execute_move(board_copy, move, opponent)
        max_val = max_value(board_copy, alpha, beta, depth + 1)
        beta = min(beta, max_val)

        if alpha >= beta: 
          return alpha
      return beta
    
    def entry_point(board):
      INF = sys.maxsize
      alpha, beta = -INF, INF
      if time.time() - start_time > self.time_limit:
        raise TimeoutException()
      # do not exceed max depth
      best_move = None
      end_game, player_score, opponent_score = check_endgame(board, player, opponent)
      if end_game:
        return None
      
      valid_moves = get_valid_moves(chess_board, player)
      move_orderings = valid_moves
      move_orderings = [(move, self.move_ordering_evaluator(board, player, player_score, opponent_score)) for move in valid_moves]
      sorted(move_orderings, key=lambda x: x[1])
      for move, _ in move_orderings: 
        board_copy = board.copy()
        execute_move(board_copy, move, player)
        min_val = min_value(board_copy, alpha, beta, 1)
        # equiv to alpha = max(alpha, ...)
        if alpha < min_val: 
          alpha = min_val
          best_move = move
      return best_move
    
    try:
      return entry_point(chess_board)
    except TimeoutException:
      return None 