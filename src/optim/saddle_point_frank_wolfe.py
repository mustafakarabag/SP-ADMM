import numpy as np
import matplotlib.pyplot as plt


class SaddlePointFW:
    def __init__(self):
        """
        Initializes an instance of a saddle-point Frank-Wolfe problem.
        The instance is not solved yet.
        The default saddle point value is 0. To be modified externally.
        """
        self.instance_solved_fw = False
        self.saddle_point_value = 0

    """
    Problem specific methods to be overwritten by the problem specific classes.
    """
    def df_dx_a(self, x_a, x_b):
        raise Exception("The gradient function for the minimizer is not implemented")

    def df_dx_b(self, x_a, x_b):
        raise Exception("The gradient function for the maximizer is not implemented")

    def minimizer_move_point_spfw(self, vec):
        raise Exception("The function to compute the move direction is not implemented for the minimizer.")

    def maximizer_move_point_spfw(self, vec):
        raise Exception("The function to compute the move direction is not implemented for the maximizer.")

    def initialize_vars_fw(self):
        raise Exception("The initilizer for ADMM is not implemented for the problem.")

    def compute_game_vals_fw(self):
        raise Exception("The method to compute the game values is not implemented for the problem.")

    def find_best_responses_both_fw(self):
        raise Exception("The method to compute the best responses of both players is not implemented.")

    def solve_saddle_point_fw(self, num_of_itr=100):
        #Lists to keep results
        self.x_a_list_fw = []
        self.x_b_list_fw = []
        x_a, x_b = self.initialize_vars_fw()
        for t in range(num_of_itr):
            #print('fw ' + str(t))
            self.x_a_list_fw.append(x_a)
            self.x_b_list_fw.append(x_b)
            x_a_grad = self.df_dx_a(x_a, x_b)
            x_b_grad = self.df_dx_b(x_a, x_b)
            s_a = self.minimizer_move_point_spfw(x_a_grad)
            s_b = self.maximizer_move_point_spfw(x_b_grad)
            #g = np.inner(x_a - s_a, x_a_grad) + np.inner(x_b - s_b, -x_b_grad)
            gamma = 2/(2+t)
            x_a = (1-gamma)*x_a + gamma*s_a
            x_b = (1-gamma)*x_b + gamma*s_b

        self.instance_solved_fw = True
        #Compute the value of the game
        self.compute_game_vals_fw()

    def visualize_game_value_no_normalization_fw(self):
        """
        Plots the  graph of game values.
        """
        plt.figure()
        plt.plot(abs(np.asarray(self.game_val_list_fw)))
        #plt.yscale('log')
        plt.title("Frank-Wolfe")
        plt.ylabel("Game value")
        plt.xlabel("Itr.")
        plt.show()