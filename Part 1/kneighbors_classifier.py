import evaluation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


import numpy as np

# hyperparameters
n_neighbors = 1

class MyKNeighborsClassifier:

	def __init__(self, file_path='./data/tictac_final.txt'):
		self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
		self.scores = []
		self.file_path = file_path

		data = np.loadtxt(file_path)
		self.X = data[:, :9]
		self.y = data[:, 9:]

	def fit(self):
		kf = KFold(n_splits=10, random_state=1, shuffle=True)
		for train_index, test_index in kf.split(self.X):
			self.clf.fit(self.X[train_index], self.y[train_index].ravel())	
			self.scores.append(self.clf.score(self.X[test_index], self.y[test_index]))
		self.print_cross_val_scores()
	
	def print_cross_val_scores(self):
		print("Cross Validation accuracy: ", list(map(lambda x: round(100*x, 2), self.scores)))
		print("Average Cross validation accuracy: ", round(100*sum(self.scores)/len(self.scores), 2))

	def print_evaluation(self):
		y_pred = self.clf.predict(self.X)
		evaluation.print_confusion_matrix(self.y, y_pred)
		evaluation.print_precision_recall(self.y, y_pred)
		print("Accuracy: ", evaluation.get_classifier_accuracy(self.y, y_pred)*100)