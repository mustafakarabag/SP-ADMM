import numpy as np
from saddle_point_games.inner_product_game import InnerProductGameSimplex

np.random.seed(0)

N = 100
rho_a = 1e-2
rho_b = 1e-2
num_of_itr = 50
game_constants = 1*np.random.randn(N)
box_a = [[1e-3, 1]] * N
box_b = [[1e-3, 1]] * N

my_game = InnerProductGameSimplex(game_constants, box_a, box_b)

my_game.solve_saddle_point_admm(rho_a, rho_b, num_of_itr)

my_game.visualize_game_value()
my_game.visualize_game_value_aux()
my_game.visualize_game_value_no_normalization()
my_game.visualize_augmented_game_value()
my_game.visualize_primal_residuals()
pass