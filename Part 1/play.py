from kneighbors_regressor import MyKNeighborsRegressor
from linear_regressor import MyLinearRegressor
from mlp_regressor import MyMLPRegressor

import numpy as np


class TicTacToe:

	def __init__(self):
		# regressor = MyKNeighborsRegressor('tictac_multi.txt')
		# regressor = MyLinearRegressor('tictac_multi.txt')
		regressor = MyMLPRegressor('./data/tictac_multi.txt')
		regressor.load()
		self.regressor = regressor

		self.board = np.array([0 for i in range(9)])
		self.chars = {-1: 'O', 0: ' ', 1: 'X'}
		self.player_piece, self.computer_piece = 1, -1
		self.is_players_move = True
		self.moves = set()
		self.winning_indices = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
		self.winner = None
		self.loop()

	def place(self, move):
		self.board[move-1] = self.player_piece if self.is_players_move else self.computer_piece
		self.moves.add(move)
		self.is_players_move = not self.is_players_move

	def check_winner(self):
		for indices in self.winning_indices:
			if self.board[indices[0]] == self.board[indices[1]] == self.board[indices[2]] == self.player_piece:
				self.winner = 'Player'
				return True
			elif self.board[indices[0]] == self.board[indices[1]] == self.board[indices[2]] == self.computer_piece:
				self.winner = 'Computer'
				return True
		return False

	def get_computer_move(self):
		moves = self.regressor.predict([self.board])
		print("My Predictions: ", list(map(lambda x: round(x, 2), moves)))
		max_idx, max_prob = -1, 0
		for i, prob in enumerate(moves):
			if not i+1 in self.moves and prob>=max_prob:
				max_idx = i
				max_prob = prob
		return max_idx+1

	def loop(self):
		self.print_board()
		while len(self.moves) != 9:
			if self.is_players_move:
				move = int(input('Your turn: '))
				self.place(move)
			else:
				move = self.get_computer_move()
				self.place(move)
			self.print_board()
			if self.check_winner(): break
		if not self.winner: print("Draw! I am as smart as you")
		elif self.winner == 'Player': print("You Win! Seems like you are smart")
		elif self.winner == 'Computer': print("You Lost! Better Luck next time")

	def print_board(self):
	    for i, c in enumerate(self.board):
	        if c != 0:
	        	print(self.chars[c], end='')
	        else:
	        	print(self.chars[c], end='')
	        if (i+1)%3:
	        	print(' | ', end='')
	        else:
	        	if i!= 8: print('\n---------')
	        	else: print('')


tictactoe = TicTacToe()