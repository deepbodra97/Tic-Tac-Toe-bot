import evaluation

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

import numpy as np

import pickle

# hyperparameters
n_neighbors = 2

# kfold config
n_splits = 10
random_state = 1
shuffle = True


class MyKNeighborsRegressor:

	def __init__(self, file_path='./data/tictac_multi.txt'):
		self.clf = KNeighborsRegressor(n_neighbors=n_neighbors, p=2)
		self.scores = []
		self.file_path = file_path

		data = np.loadtxt(file_path)
		self.X = data[:, :9]
		self.ys = data[:, 9:]
		
		
	def fit(self):
		kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
		for train_index, test_index in kf.split(self.X):
			self.clf.fit(self.X[train_index], self.ys[train_index])
			# self.scores.append(self.clf.score(self.X[test_index], self.ys[test_index]))
			self.scores.append(evaluation.get_regressor_accuracy(np.around(self.clf.predict(self.X[test_index])), self.ys[test_index]))
		self.print_bootstrap_scores()
		self.scores = []
		print()

	def predict(self, x):
		return self.clf.predict(x)[0]

	def save(self, filename='knregressor'):
		pickle.dump(self.clf, open("./models/"+filename, 'wb'))

	def load(self, filename='knregressor'):
		self.clf = pickle.load(open("./models/"+filename, 'rb'))
			
	def print_bootstrap_scores(self):
		with np.printoptions(precision=2):	
			print("Bootstrap accuracies: ", list(map(lambda x: round(100*x, 2), self.scores)))
			print("Average Bootstrap accuracy: ", round(100*sum(self.scores)/len(self.scores), 2))

	def print_evaluation(self):
		y_pred = clf.predict(self.X)

		evaluation.print_evaluation(self.ys, np.around(y_pred))
		print()

regr = MyKNeighborsRegressor()
regr.fit()
regr.save()