import evaluation

import numpy as np

class MyLinearRegressor:

	def __init__(self, file_path='./data/tictac_multi.txt'):
		self.weights = []
		self.file_path = file_path

		data = np.loadtxt(file_path)
		X = data[:, :9]
		self.X = np.append(X, np.ones([len(X), 1]), 1)
		self.ys = data[:, 9:]
		
		
	def fit(self):
		print("Regression Coefficients")
		for i in range(9):
			print("Model ", i)
			weight = np.matmul(np.linalg.inv(np.matmul(self.X.transpose(), self.X)), np.matmul(self.X.transpose(), self.ys[:, i]))
			self.weights.append(weight)
			print(weight)
	
	def predict(self, x):
		x = np.append(x, 1)
		predictions = [0]*9
		for i, weight in enumerate(self.weights):
			predictions[i] = np.matmul(weight.transpose(), x)
		return predictions

	def print_evaluation(self):
		print("Accuracy: ")
		for i, weight in enumerate(self.weights):	
			y_pred = []
			print("Model", i+1)
			for x in self.X:
				y_pred.append(np.matmul(weight.transpose(), x))
			print(evaluation.get_regressor_accuracy([self.ys[:, i]], [np.around(y_pred)])*100)
		print()