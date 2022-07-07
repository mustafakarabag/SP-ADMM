import numpy as np
import os
from ..saddle_point_games.inner_product_game import InnerProductGameBall

#Fix seed for any randomization
np.random.seed(0)

#SP-ADMM parameters
rho_a = 1e0
rho_b = 1e0
num_of_itr = 50

#Inner product game parameters
N = 100
game_constants = 1*np.random.randn(N)
box_a = [[5e-2, 1]] * N
box_b = [[5e-2, 1]] * N

#Construct the game
my_game = InnerProductGameBall(game_constants, box_a, box_b)

#Solve the game
my_game.solve_saddle_point_admm(rho_a, rho_b, num_of_itr)

#Save directory for the plots
step_size_str = "rho_a" + "{:.2e}".format(rho_a) + "_" + "rho_b" + "{:.2e}".format(rho_b)
save_dir="src/examples/logs/inner_product_game_ball_example_plots/" + step_size_str
if not os.path.exists(save_dir): os.mkdir(save_dir)

#Plot the results
my_game.visualize_game_value(normalized=False, save_name = save_dir +"/game_value_x.tex")
my_game.visualize_game_value_aux(normalized=False, save_name = save_dir +"/game_value_z.tex")
my_game.visualize_augmented_game_value(normalized=False, save_name = save_dir +"/aug_game_value.tex")
my_game.visualize_primal_residuals(save_name= save_dir +"/primal_residuals.tex")
pass