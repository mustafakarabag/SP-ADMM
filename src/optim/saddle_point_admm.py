import numpy as np
import matplotlib.pyplot as plt


class SaddlePointADMM:
    def __init__(self):
        """
        Initializes an instance of a saddle point ADMM problem.
        The instance is not solved yet.
        The default saddle point value is 0. To be modified externally.
        """
        self.instance_solved = False
        self.saddle_point_value = 0

    """
    Problem specific methods to be overwritten by the problem specific classes.
    """

    def initialize_vars(self):
        raise Exception("The initilizer for ADMM is not implemented for the problem.")

    def compute_game_vals(self):
        raise Exception("The method to compute the game values is not implemented for the problem.")

    def find_best_responses_both(self):
        raise Exception("The method to compute the best responses of both players is not implemented.")

    def project_z_a(self):
        raise Exception("The method to project z_a is not implemented for the problem.")

    def project_z_b(self):
        raise Exception("The method to project z_b is not implemented for the problem.")

    def solve_saddle_point_admm(self, rho_a=1, rho_b=1, num_of_itr=100):
        """
        Saddle point ADMM solver. Refer to the paper for the details of the algorithm.
        x_a: Primal variable for the minimizer
        x_b: Primal variable for the maximizer
        z_a: Auxilary primary variable for the minimizer
        z_b: Auxilary primary variable for the maximizer
        lmd_a: Dual variable for the minimizer
        lmd_b: Dual variable for the maximizer


        :param rho_a: Dual step size parameter for the minimizer
        :param rho_b: Dual step size parameter for the maximizer
        :param num_of_itr: Number of iterations of the ADMM algorithm
        """

        #Lists to keep results
        self.z_a_list = []
        self.z_b_list = []
        self.x_a_list = []
        self.x_b_list = []
        self.lmd_a_list = []
        self.lmd_b_list = []

        self.rho_a = rho_a
        self.rho_b = rho_b

        # Initialize parameters
        x_a, x_b, z_a, z_b, lmd_a, lmd_b = self.initialize_vars()

        #Iterative algorithm
        for k in range(num_of_itr):
            #Update result lists
            self.x_a_list.append(x_a)
            self.x_b_list.append(x_b)
            self.z_a_list.append(z_a)
            self.z_b_list.append(z_b)
            self.lmd_a_list.append(lmd_a)
            self.lmd_b_list.append(lmd_b)

            # Update primal variables
            x_a, x_b = self.solve_augmented_saddle_game(z_a, z_b, lmd_a, lmd_b)

            # Update auxilary primal variables
            z_a = self.project_z_a(x_a + lmd_a / rho_a)
            z_b = self.project_z_b(x_b + lmd_b / rho_b)

            # Update dual parameters
            lmd_a += rho_a * (x_a - z_a)
            lmd_b += rho_b * (x_b - z_b)


        self.instance_solved = True
        #Compute the residual values for the primal variables
        self.compute_primal_residuals()
        #Compute the value of the game using the primal (x) variables
        self.compute_game_vals()
        #Compute the value of the game using the primal (z) variables
        self.compute_game_vals_aux()
        #Compute the augmented value of the game
        self.compute_aug_game_vals()


    def compute_primal_residuals(self):
        """
        Computes the lists of primary residual norms for the minimizer and maximizer.
        """
        self.primal_residual_norm_a_list = []
        for (x_a, z_a) in zip(self.x_a_list, self.z_a_list):
            self.primal_residual_norm_a_list.append(np.linalg.norm(x_a - z_a))

        # Compute the primal residual norm for the maximizer
        self.primal_residual_norm_b_list = []
        for (x_b, z_b) in zip(self.x_b_list, self.z_b_list):
            self.primal_residual_norm_b_list.append(np.linalg.norm(x_b - z_b))

    def compute_aug_game_vals(self):
        """
        Computes the augmented value of the game for a given list of solution points
        """
        self.aug_lag_value_list = []
        for (x_a, x_b, z_a, z_b, lmd_a, lmd_b, game_val) in zip(self.x_a_list, self.x_b_list, self.z_a_list, self.z_b_list, self.lmd_a_list, self.lmd_b_list, self.game_val_list):
            aug_lag_val =  game_val \
                            + np.inner(lmd_a, x_a - z_a) + self.rho_a/2*np.inner(x_a - z_a, x_a - z_a)  \
                            - np.inner(lmd_b, x_b - z_b) + self.rho_b/2*np.inner(x_b - z_b, x_b - z_b)
            self.aug_lag_value_list.append(aug_lag_val)


    def visualize_primal_residuals(self):
        """
        Plots the log graph of primary residuals.
        """
        plt.figure()
        plt.plot(self.primal_residual_norm_a_list)
        plt.yscale('log')
        plt.ylabel("Primal residual (Minimizer)")
        plt.xlabel("Itr.")
        plt.show()

        plt.figure()
        plt.plot(self.primal_residual_norm_b_list)
        plt.yscale('log')
        plt.ylabel("Primal residual (Maximizer)")
        plt.xlabel("Itr.")
        plt.show()

    def visualize_game_value(self):
        """
        Plots the log graph of absolute game values from the saddle point value.
        """
        plt.figure()
        plt.plot(abs(np.asarray(self.game_val_list) - self.saddle_point_value))
        plt.yscale('log')
        plt.ylabel("Abs. game value (with x variables)")
        plt.xlabel("Itr.")
        plt.show()

    def visualize_game_value_no_normalization(self):
        """
        Plots the  graph of game values.
        """
        plt.figure()
        plt.plot(abs(np.asarray(self.game_val_list_aux)))
        #plt.yscale('log')
        plt.ylabel("Game value (with z variables)")
        plt.xlabel("Itr.")
        plt.show()

    def visualize_game_value_aux(self):
        """
        Plots the log graph of absolute game values from the saddle point value. Uses z variables instead of x variables.
        """
        plt.figure()
        plt.plot(abs(np.asarray(self.game_val_list_aux) - self.saddle_point_value))
        plt.yscale('log')
        plt.ylabel("Abs. game value (with z variables)")
        plt.xlabel("Itr.")
        plt.show()

    def visualize_augmented_game_value(self):
        """
        Plots the log graph of absolute game values from the saddle point value.
        """
        plt.figure()
        plt.plot(abs(np.asarray(self.aug_lag_value_list) - self.saddle_point_value))
        plt.yscale('log')
        plt.ylabel("Augmented game value")
        plt.xlabel("Itr.")
        plt.show()
