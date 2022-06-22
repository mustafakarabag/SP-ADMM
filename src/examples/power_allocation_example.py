import numpy as np

from ..saddle_point_games.power_allocation_game import PowerAllocationGame

np.random.seed(0)

N = 10
rho_a = 1e-2
rho_b = 1e-2
num_of_itr = 50
# game_constants = 1*np.random.randn(N)
power_lim_adversary = 10
power_lim_player = 20
box_player = [[0, power_lim_player]] * N
box_adversary = [[0, power_lim_adversary]] * N

# my_game = InnerProductGameSimplex(game_constants, box_a, box_b)
my_game = PowerAllocationGame(receiver_noises=[2, 6, 5, 8, 3, 9, 5, 6, 7, 3], power_multipliers=([1] * N),
                              box_a=box_adversary, box_b=box_player, power_lim_a=power_lim_adversary,
                              power_lim_b=power_lim_player)
my_game.saddle_point_value = 2.860
my_game.solve_saddle_point_admm(rho_a, rho_b, num_of_itr)

my_game.visualize_game_value()
my_game.visualize_game_value_aux()
my_game.visualize_game_value_no_normalization()
my_game.visualize_augmented_game_value()
my_game.visualize_primal_residuals()
pass