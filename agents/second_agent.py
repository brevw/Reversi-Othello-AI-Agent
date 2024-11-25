# Second agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

class TimeoutException(Exception):
    pass

@register_agent("second_agent")
class SecondAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(SecondAgent, self).__init__()
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

    start_time = time.time()
    root = Node(chess_board, player, opponent)

    while time.time() - start_time < self.time_limit:
      leaf_node = root.expand()

      if not check_endgame(leaf_node.chess_board, leaf_node.player, leaf_node.opponent)[0]: ## if not at end game 
        leaf_node.create_children()

      if leaf_node.children: 
        node_to_simulate = np.random.choice(leaf_node.children)
      else: 
        node_to_simulate = leaf_node

      result = node_to_simulate.simulate()

      node_to_simulate.backpropagate(result)
    
    best_child = max(root.children, key=lambda child: child.wins/child.n_visits)

    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    return best_child.move

class Node: 
  
  def __init__(self, chess_board, player, opponent, parent=None, move=None):
    self.chess_board = chess_board
    self.player = player
    self.opponent = opponent
    self.parent = parent
    self.move = move
    self.children = []
    self.n_visits = 0 
    self.wins = 0

  def is_leaf(self): 
    return not self.children 
  
  def uct_score(self, exploration_weight=1.4):
    if self.n_visits == 0: 
      return float('inf')
    parent = self
    if self.parent: 
      parent = self.parent
    return (self.wins / self.n_visits) + (exploration_weight * np.sqrt(np.log(parent.n_visits) / self.n_visits))
  
  def create_children(self):     
    for move in get_valid_moves(self.chess_board, self.player):
      new_board = self.chess_board.copy()
      execute_move(new_board, move, self.player)
      # Create a new child node for this move
      child_node = Node(new_board, self.opponent, self.player, parent=self, move=move)
      self.children.append(child_node)

  def select_child(self): 
    valid_moves = get_valid_moves(self.chess_board, self.player)
    if not valid_moves: 
      return None

    if not self.children: 
      self.create_children()
      return np.random.choice(self.children) 
    
    node_to_explore = None
    best_score = float('-inf')
    for child in self.children: 
      score = child.uct_score()
      if score > best_score: 
          best_score = score
          node_to_explore = child
    return node_to_explore
  
  def expand(self): 
    current_node = self
    while not current_node.is_leaf():
      current_node = current_node.select_child()
    return current_node
  
  def simulate(self): 
    current_board = self.chess_board.copy()
    current_player = self.player
    current_opponent = self.opponent

    while not check_endgame(current_board, current_player, current_opponent)[0]:
      valid_moves = get_valid_moves(current_board, current_player)
      if valid_moves:
          move = random_move(current_board, current_player)
          execute_move(current_board, move, current_player)
        
      current_player, current_opponent = current_opponent, current_player
    
    player_score = np.sum(current_board == self.player)
    opponent_score = np.sum(current_board == self.opponent)

    return 1 if player_score > opponent_score else 0 if player_score < opponent_score else 0.5

  def backpropagate(self, score): 
    self.n_visits += 1
    self.wins += score
    if self.parent: 
      self.parent.backpropagate(1 - score) ## opponent gets opposite score