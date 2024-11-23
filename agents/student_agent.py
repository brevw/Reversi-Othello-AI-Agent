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
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()

    i = 100
    best_move = 1
    temp = 1
    while temp:
      temp = self.alpha_beta_pruning_depth_limited(chess_board, player, opponent, start_time, i)
      if temp: 
        best_move = temp
      else:
        break
      i += 1
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    return best_move

  def evaluate(self, chess_board, player, opponent) -> float:

    def count_stable_discs(chess_board, player):
      stable_discs = 0
      directions = get_directions()
      rows, cols = chess_board.shape
      for r in range(rows):
          for c in range(cols):
              if chess_board[r, c] == player:
                  stable = True
                  # to be stable has to stable over all directions
                  for dr, dc in directions:
                      rr, cc = r, c
                      while 0 <= rr < rows and 0 <= cc < cols:
                          if chess_board[rr, cc] == 0:
                              stable = False
                              break
                          rr += dr
                          cc += dc
                      if not stable:
                          break
                  if stable:
                      stable_discs += 1
      return stable_discs

    def edge_occupancy(chess_board, player):
       rows, cols = chess_board.shape 
       return np.sum(chess_board[0] == player) + np.sum(chess_board[rows - 1] == player) + np.sum(chess_board[:, 0] == player)+ np.sum(chess_board[:, cols - 1] == player)

    player_pieces = np.sum(chess_board == player)
    opponent_pieces = np.sum(chess_board == opponent)
    piece_advantage = player_pieces - opponent_pieces
    
    player_mobility = len(get_valid_moves(chess_board, player))
    opponent_mobility = len(get_valid_moves(chess_board, opponent))
    mobility_advantage = player_mobility - opponent_mobility

    player_stablility = count_stable_discs(chess_board, player)
    opponent_stability = count_stable_discs(chess_board, opponent)
    stability_advantage = player_stablility - opponent_stability

    player_edge_occupancy = edge_occupancy(chess_board, player)
    opponent_edge_occupancy = edge_occupancy(chess_board, opponent)
    edge_occupancy_advantage = player_edge_occupancy - opponent_edge_occupancy


    eval =  piece_advantage * 1.0           + \
            mobility_advantage * 1.0        + \
            stability_advantage * 3.0       + \
            edge_occupancy_advantage * 4.0
    return eval


  def alpha_beta_pruning_depth_limited(self, chess_board: np.array, player, opponent, start_time: float, depth_limit: int):

    def max_value(board, alpha, beta, depth) -> float:
      if time.time() - start_time > self.time_limit:
        raise TimeoutException()
      # do not exceed max depth
      best_move = None
      if depth == depth_limit or check_endgame(board, player, opponent)[0]:
        return self.evaluate(board, player, opponent)
      
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
          if depth == 0: 
            # record best move
            best_move = move

        if alpha >= beta: 
          return beta
      return best_move if depth == 0 else alpha

    def min_value(board, alpha, beta, depth) -> float: 
      if time.time() - start_time > self.time_limit:
        raise TimeoutException()
      # do not exceed max depth
      if depth == depth_limit or check_endgame(board, opponent, player):
        return self.evaluate(board, player, opponent)
      
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
    
    try:
      INF = sys.maxsize
      return max_value(chess_board, -INF, +INF, 0)
    except TimeoutException:
      return None 
  

