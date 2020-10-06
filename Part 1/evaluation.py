from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import numpy as np

import warnings
warnings.filterwarnings('ignore')

def get_accuracy(y_pred, y_true):
	n_correct = 0
	for y_p, y_t in zip(y_pred, y_true):
		if y_p == y_t: n_correct += 1
	return n_correct/len(y_pred)

def print_evaluation(y, y_pred):
	with np.printoptions(precision=4):
		print("Normalized Confusion Matrix")
		cm = confusion_matrix(y, y_pred)
		cm = cm / cm.astype(np.float).sum(axis=1)
		print(cm)
		precision, recall, _, _ = precision_recall_fscore_support(y, y_pred, labels=np.unique(y_pred))
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("Accuracy: ", get_accuracy(y, y_pred)*100)