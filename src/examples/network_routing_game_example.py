from ..mdp.mdp import MDP
from ..mdp.averagemdp import AverageMDP
from ..saddle_point_games.network_routing_game import NetworkRoutingGame
import numpy as np
import os
import time

np.random.seed(0)

#Construct an MDP using a random Erdos-Renyi graph

number_of_expected_edges = 5 # The expected number of outgoing edges from a node
list_of_states_and_transitions =[]
NS = 100
game_graph = np.random.rand(NS, NS) > (1-number_of_expected_edges/NS)
game_graph = game_graph.astype(int)
for s in range(NS):
    succ_states = np.nonzero(game_graph[s,:])[0]
    list_of_states_and_transitions.append([np.reshape(succ_states, (1,len(succ_states))), np.eye(len(succ_states))])

initial_state_dist = np.ones(NS)/NS

my_mdp = MDP(list_of_states_and_transitions, initial_state_dist)
print(my_mdp.NSA)
#The density of State 1 is at least 0.1
visit_const = {1 : 0.1}



#Construct the game with the MDP and the visitation constraints
my_game = NetworkRoutingGame(my_mdp, visit_const)

#SP-ADMM parameters
rho_a = 1e0
rho_b = 1e0
num_of_itr = 100

#Solve the game using SP-ADMM and SP-Frank-Wolfe methods
print("MDP NSA:" + str(my_mdp.NSA) + ", " + "Number of itr:" + str(num_of_itr))
start_time = time.time()
my_game.solve_saddle_point_admm(rho_a, rho_b, num_of_itr)
end_time = time.time()
admm_time = end_time - start_time
print("ADMM solving time:" + str(admm_time) + " sec")

start_time = time.time()
my_game.solve_saddle_point_fw(num_of_itr)
end_time = time.time()
fw_time = end_time - start_time
print("FW solving time:" + str(fw_time) + " sec")


#Save directory for the plots
step_size_str = "_rho_a" + "{:.2e}".format(rho_a) + "_" + "rho_b" + "{:.2e}".format(rho_b)
save_dir="src/examples/logs/network_routing_game_example_plots/" + "NS" +str(my_mdp.NS) + "_NSA" +str(my_mdp.NSA) + step_size_str
if not os.path.exists(save_dir): os.mkdir(save_dir)


with open(save_dir+'/solve_times.txt', 'w') as f:
    f.write("ADMM solving time:" + str(admm_time) + " sec")
    f.write('\n')
    f.write("FW solving time:" + str(fw_time) + " sec")

my_game.find_best_responses_both()
my_game.find_best_responses_both_fw()
my_game.visualize_game_value_upper_lower(display_fw=True, log_y_scale=True, save_name = save_dir +"/upper_lower_values.tex")
my_game.visualize_optimality_gap(display_fw=True, log_y_scale=True, save_name =save_dir + "/optimality_gap_values.tex")
