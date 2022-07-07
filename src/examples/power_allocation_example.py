import numpy as np
import os
from ..saddle_point_games.power_allocation_game import PowerAllocationGame

#Fix seed for any randomization
np.random.seed(0)

#SP-ADMM parameters
rho_a = 1e-3
rho_b = 1e-3
num_of_itr = 100

#Power allocation game parameters
N = 10
power_lim_adversary = 10
power_lim_player = 20
box_player = [[0, power_lim_player]] * N
box_adversary = [[0, power_lim_adversary]] * N

#Construct the game
my_game = PowerAllocationGame(receiver_noises=[2, 6, 5, 8, 3, 9, 5, 6, 7, 3], power_multipliers=([1] * N),
                              box_a=box_adversary, box_b=box_player, power_lim_a=power_lim_adversary,
                              power_lim_b=power_lim_player)
my_game.saddle_point_value = 2.860 #Value taken form https://web.stanford.edu/class/ee392o/cvxccv.pdf

#Solve the game
my_game.solve_saddle_point_admm(rho_a, rho_b, num_of_itr)

#Save directory for the plots
step_size_str = "rho_a" + "{:.2e}".format(rho_a) + "_" + "rho_b" + "{:.2e}".format(rho_b)
save_dir="src/examples/logs/power_allocation_example_plots/" + step_size_str
if not os.path.exists(save_dir): os.mkdir(save_dir)

#Plot the results
my_game.visualize_game_value(normalized=False, save_name = save_dir +"/game_value_x.tex")
my_game.visualize_game_value_aux(normalized=False, save_name = save_dir +"/game_value_z.tex")
my_game.visualize_augmented_game_value(normalized=False, save_name = save_dir +"/aug_game_value.tex")
my_game.visualize_primal_residuals(save_name= save_dir +"/primal_residuals.tex")
my_game.visualize_total_residuals(save_name= save_dir +"/total_residuals.tex")
pass