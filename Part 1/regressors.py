from kneighbors_regressor import MyKNeighborsRegressor
from linear_regressor import MyLinearRegressor
from mlp_regressor import MyMLPRegressor

import numpy as np

# kneighbors_regressor = MyKNeighborsRegressor('./data/tictac_multi.txt')
# kneighbors_regressor.fit()
# kneighbors_regressor.print_evaluation()

# linear_regressor = MyLinearRegressor('./data/tictac_multi.txt')
# linear_regressor.fit()
# linear_regressor.print_evaluation()

mlp_regressor = MyMLPRegressor('./data/tictac_multi.txt')
mlp_regressor.fit()
mlp_regressor.print_evaluation()
mlp_regressor.save()
# mlp_regressor.load()