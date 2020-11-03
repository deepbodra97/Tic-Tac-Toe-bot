from kneighbors_regressor import MyKNeighborsRegressor
from linear_regressor import MyLinearRegressor
from mlp_regressor import MyMLPRegressor

print("---------- K-Neighbors Regressor----------")
print("Dataset: tictac_multi.txt")
kneighbors_regressor = MyKNeighborsRegressor('./data/tictac_multi.txt')
kneighbors_regressor.fit()
kneighbors_regressor.print_evaluation()
kneighbors_regressor.save()
print()

print("----------Linear Regressor----------")
print("Dataset: tictac_multi.txt")
linear_regressor = MyLinearRegressor('./data/tictac_multi.txt')
linear_regressor.fit()
linear_regressor.print_evaluation()
print()

print("----------MLP Regressor----------")
print("Dataset: tictac_multi.txt")
mlp_regressor = MyMLPRegressor('./data/tictac_multi.txt')
mlp_regressor.fit()
mlp_regressor.print_evaluation()
mlp_regressor.save()
print()