from kneighbors_classifier import MyKNeighborsClassifier
from linearsvm_classifier import MyLinearSVMClassifier
from mlp_classifier import MyMLPClassifier

print("----------Linear SVM Classifier----------")
print("Dataset: tictac_final.txt")
linearsvm_classifier = MyLinearSVMClassifier('./data/tictac_final.txt')
linearsvm_classifier.fit()
linearsvm_classifier.print_evaluation()
print()

print("Dataset: tictac_single.txt")
linearsvm_classifier = MyLinearSVMClassifier('./data/tictac_single.txt')
linearsvm_classifier.fit()
linearsvm_classifier.print_evaluation()


print("----------K-Neighbors Classifier----------")
print("tictac_final.txt")
kneighbors_classifier = MyKNeighborsClassifier('./data/tictac_final.txt')
kneighbors_classifier.fit()
kneighbors_classifier.print_evaluation()
print()

print("tictac_single.txt")
kneighbors_classifier = MyKNeighborsClassifier('./data/tictac_single.txt')
kneighbors_classifier.fit()
kneighbors_classifier.print_evaluation()
print()

print("----------MLP Classifier----------")
print("Dataset: tictac_final.txt")
mlp_classifier = MyMLPClassifier('./data/tictac_final.txt', (9, 9))
mlp_classifier.fit()
mlp_classifier.print_evaluation()
print()

print("Dataset: tictac_single.txt")
mlp_classifier = MyMLPClassifier('./data/tictac_single.txt', (27, 9))
mlp_classifier.fit()
mlp_classifier.print_evaluation()
print()