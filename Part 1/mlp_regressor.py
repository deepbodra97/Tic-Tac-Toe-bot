import evaluation

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

import numpy as np

import pickle

# hyperparameters
hidden_layer_sizes = (27, 27, 27) # default = (100, )
activation_function = 'logistic' # sigmoid activation function
optimizer = 'adam' # default = 'adam'
learning_rate_init = 0.01 # default=0.001
random_state=1

# kfold config
n_splits = 10
random_state = 1
shuffle = True

class MyMLPRegressor:

	def __init__(self, file_path='./data/tictac_multi.txt'):
		self.clf = MLPRegressor(
						hidden_layer_sizes=hidden_layer_sizes,
						solver=optimizer,
						activation=activation_function,
						learning_rate_init=learning_rate_init,
						random_state=random_state,
						max_iter=500
					)
		self.scores = []
		self.file_path = file_path

		data = np.loadtxt(file_path)
		self.X = data[:, :9]
		self.ys = data[:, 9:]
		
	def fit(self):
		kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
		for train_index, test_index in kf.split(self.X):
			self.clf.fit(self.X[train_index], self.ys[train_index])
			self.scores.append(evaluation.get_regressor_accuracy(np.around(self.clf.predict(self.X[test_index])), self.ys[test_index]))
		self.print_cross_val_scores()
		self.scores = []

	def predict(self, x):
		return self.clf.predict(x)[0]

	def save(self, filename='mlpregressor'):
		pickle.dump(self.clf, open("./models/"+filename, 'wb'))

	def load(self, filename='mlpregressor'):
		self.clf = pickle.load(open("./models/"+filename, 'rb'))
			
	def print_cross_val_scores(self):
		print("Cross Validation accuracy: ", list(map(lambda x: round(100*x, 2), self.scores)))
		print("Average Cross validation accuracy: ", round(100*sum(self.scores)/len(self.scores), 2))

	def print_evaluation(self):
		y_pred = self.clf.predict(self.X)
		print("Accuracy: ", evaluation.get_regressor_accuracy(self.ys, np.around(y_pred))*100)
		print()