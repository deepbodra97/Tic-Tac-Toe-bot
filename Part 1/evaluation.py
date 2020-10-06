from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

import numpy as np

import warnings
warnings.filterwarnings('ignore')

def get_classifier_accuracy(y_true, y_pred, normalize=True):
	return accuracy_score(y_pred, y_true)

def get_regressor_accuracy(y_pred, y_true):
	n_correct = 0
	for y_p, y_t in zip(y_pred, y_true):
		for yp, yt in zip(y_p, y_t):
			if yp==yt: n_correct += 1
	return n_correct/(len(y_pred)*9)

def print_confusion_matrix(y, y_pred):
	with np.printoptions(precision=4, suppress=True):
		print("Normalized Confusion Matrix")
		cm = confusion_matrix(y, y_pred)
		cm = cm / cm.astype(np.float).sum(axis=1)
		print(cm)

def print_precision_recall(y, y_pred):
	with np.printoptions(precision=4):
		precision, recall, _, _ = precision_recall_fscore_support(y, y_pred, labels=np.unique(y_pred))
		print("Precision: ", precision)
		print("Recall: ", recall)