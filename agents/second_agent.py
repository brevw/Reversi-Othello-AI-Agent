# Second agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("second_agent")
class SecondAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(SecondAgent, self).__init__()
    self.name = "StudentAgent"

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
    CUT_OFF = sys.maxsize  

    def max_value(chess_board, a, b, depth) -> float:
      end_game, player_score, opponent_score = check_endgame(chess_board, 1, 2)
      if depth == CUT_OFF or end_game:
        return self.heuristic(chess_board, 1, player_score, opponent_score)
      
      valid_moves = get_valid_moves(chess_board, 1)
      if not valid_moves: 
        return min_value(chess_board, a, b, depth)
      
      for move in valid_moves: 
        chess_board_copy = chess_board.copy()
        execute_move(chess_board_copy, move, 1)
        a = max(a, min_value(chess_board_copy, a, b, depth + 1))
        if a >= b: 
          return b
      return a
    
    def min_value(chess_board, a, b, depth) -> float: 
      end_game, player_score, opponent_score = check_endgame(chess_board, 2, 1)
      if depth == CUT_OFF or end_game:
        return self.heuristic(chess_board, 2, player_score, opponent_score)
      
      valid_moves = get_valid_moves(chess_board, 2)
      if not valid_moves:
        return max_value(chess_board, a, b, depth)

      for move in valid_moves: 
        chess_board_copy = chess_board.copy()
        execute_move(chess_board_copy, move, 2)
        b = min(b, max_value(chess_board_copy, a, b, depth + 1))
        if a >= b: 
          return a
      return b
    
    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()

    a = sys.maxsize
    b = -a - 1
    best_move = None
    valid_moves_1 = get_valid_moves(chess_board, 1)
    valid_moves_2 = get_valid_moves(chess_board, 2)
    
    if (player == 1 and valid_moves_1) or (player == 2 and not valid_moves_2): 
      for move in valid_moves_1: 
        chess_board_copy = chess_board.copy()
        execute_move(chess_board_copy, move, 1)
        a_ = min_value(chess_board_copy, a, b, 1)
        if a > a_: 
          best_move = move
          a = a_
    else: 
      for move in get_valid_moves(chess_board, 2): 
        chess_board_copy = chess_board.copy()
        execute_move(chess_board_copy, move, 2)
        b_ = max_value(chess_board_copy, a, b, 1)
        if b < b_: 
          best_move = move
          b = b_

    time_taken = time.time() - start_time

    #print("My AI's turn took ", time_taken, "seconds.")
    
    return best_move
  
  def heuristic(self, board, color, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
        return total_score

