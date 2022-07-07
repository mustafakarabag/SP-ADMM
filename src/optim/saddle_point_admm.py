import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

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

    def initialize_vars_spadmm(self):
        raise Exception("The initilizer for ADMM is not implemented for the problem.")

    def compute_game_vals(self):
        raise Exception("The method to compute the game values is not implemented for the problem.")

    def find_best_responses_both(self):
        raise Exception("The method to compute the best responses of both players is not implemented.")

    def project_z_a_spadmm(self):
        raise Exception("The method to project z_a is not implemented for the problem.")

    def project_z_b_spadmm(self):
        raise Exception("The method to project z_b is not implemented for the problem.")

    def solve_augmented_saddle_game_spadmm(self):
        raise Exception("The method to solve the augmented saddle game is not implemented for the problem.")

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
        x_a, x_b, z_a, z_b, lmd_a, lmd_b = self.initialize_vars_spadmm()

        #Iterative algorithm
        for k in range(num_of_itr):
            print('admm ' + str(k))
            #Update result lists
            self.x_a_list.append(x_a)
            self.x_b_list.append(x_b)
            self.z_a_list.append(z_a)
            self.z_b_list.append(z_b)
            self.lmd_a_list.append(lmd_a)
            self.lmd_b_list.append(lmd_b)

            # Update primal variables
            x_a, x_b = self.solve_augmented_saddle_game_spadmm(z_a, z_b, lmd_a, lmd_b)

            # Update auxilary primal variables
            z_a = self.project_z_a_spadmm(x_a + lmd_a / rho_a)
            z_b = self.project_z_b_spadmm(x_b + lmd_b / rho_b)

            # Update dual parameters
            lmd_a += rho_a * (x_a - z_a)
            lmd_b += rho_b * (x_b - z_b)


        self.instance_solved = True
        #Compute the residual values for the primal variables
        self.compute_primal_residuals()
        #Compute the residual values for the dual variables
        self.compute_dual_residuals()
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
        # Compute the primal residual norm for the minimizer
        self.primal_residual_norm_a_list = []
        for (x_a, z_a) in zip(self.x_a_list, self.z_a_list):
            self.primal_residual_norm_a_list.append(np.linalg.norm(x_a - z_a))

        # Compute the primal residual norm for the maximizer
        self.primal_residual_norm_b_list = []
        for (x_b, z_b) in zip(self.x_b_list, self.z_b_list):
            self.primal_residual_norm_b_list.append(np.linalg.norm(x_b - z_b))

        # Compute the total primal residual norm
        self.primal_residual_total_norm_list = []
        for (r_a, r_b) in zip(self.primal_residual_norm_a_list, self.primal_residual_norm_b_list):
            self.primal_residual_total_norm_list.append(r_a + r_b)

    def compute_dual_residuals(self):
        """
        Computes the lists of dual residual norms for the minimizer and maximizer.
        """
        # Compute the dual residual norm for the minimizer
        self.dual_residual_norm_a_list = [1e6]

        prev_z_a = self.z_a_list[0]
        for z_a in self.z_a_list[1:]:
            self.dual_residual_norm_a_list.append(self.rho_a * np.linalg.norm(prev_z_a - z_a))
            prev_z_a = z_a

        # Compute the dual residual norm for the maximizer
        self.dual_residual_norm_b_list = [1e6]
        prev_z_b = self.z_b_list[0]
        for z_b in self.z_b_list[1:]:
            self.dual_residual_norm_b_list.append(self.rho_b * np.linalg.norm(prev_z_b - z_b))
            prev_z_b = z_b

        # Compute the total dual residual norm
        self.dual_residual_total_norm_list = []
        for (s_a, s_b) in zip(self.dual_residual_norm_a_list, self.dual_residual_norm_b_list):
            self.dual_residual_total_norm_list.append(s_a + s_b)

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


    def visualize_primal_residuals(self, save_name=None):
        """
        Plots the log graph of primary residuals.
        """
        plt.figure()
        plt.plot(self.primal_residual_norm_a_list, label="Minimizer")
        plt.plot(self.primal_residual_norm_b_list, label="Maximizer")
        plt.plot(self.primal_residual_total_norm_list, label="Total")
        plt.yscale('log')
        plt.ylabel("Primal residuals")
        plt.xlabel("Itr.")
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()

    def visualize_total_residuals(self, save_name=None):
        """
        Plots the log graph of primary residuals.
        """
        plt.figure()

        plt.plot(np.asarray(self.primal_residual_total_norm_list) + np.asarray(self.dual_residual_total_norm_list), label="Total residual")
        plt.yscale('log')
        plt.ylabel("Total residual")
        plt.xlabel("Itr.")
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()

    def visualize_game_value(self, normalized=False, save_name=None):
        """
        Plots the log graph of absolute game values from the saddle point value.
        """
        plt.figure()
        if normalized:
            plt.plot(abs(np.asarray(self.game_val_list) - self.saddle_point_value))
            plt.yscale('log')
            plt.ylabel("Normalized absolute game value (with x variables)")
        else:
            plt.plot(np.asarray(self.game_val_list))
            plt.ylabel("Game value (with x variables)")
        plt.xlabel("Itr.")
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()

    def visualize_game_value_upper_lower(self, display_fw=True, log_y_scale=True, save_name=None):
        """
        Plots the graph of game values using the primal auxiliary z variables. In addition plots the upper and lower bounds.
        """
        plt.figure()
        plt.plot(np.asarray(self.game_val_list_aux), label='Min. itr - Max itr')
        plt.plot(np.asarray(self.game_val_list_aux_min_best_max_itr), label='Min. best - Max itr')
        plt.plot(np.asarray(self.game_val_list_aux_min_itr_max_best), label='Min. itr - Max best')
        if display_fw:
            plt.plot(np.asarray(self.game_val_list_fw), label='Min. itr - Max  FW')
            plt.plot(np.asarray(self.game_val_list_min_best_max_itr_fw), label='Min. best - Max itr FW')
            plt.plot(np.asarray(self.game_val_list_min_itr_max_best_fw), label='Min. itr - Max best FW')
        plt.legend()
        plt.ylabel("Game value (with z variables)")
        plt.xlabel("Itr.")
        if log_y_scale:
            plt.yscale('log')
        if save_name is not None:
            tikzplotlib.save(save_name)
        plt.show()

    def visualize_optimality_gap(self, display_fw=True, log_y_scale=True, save_name=None):
        """
        Plots the  graph of game values.
        """
        plt.figure()
        plt.plot(np.asarray(self.optimality_gap_list), label='SP-ADMM')
        if display_fw:
            plt.plot(np.asarray(self.optimality_gap_list_fw), label='SP-FW')
        # plt.yscale('log')
        plt.legend()
        plt.ylabel("Optimality Gap")
        plt.xlabel("Itr.")
        if log_y_scale:
            plt.yscale('log')
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()

    def visualize_game_value_aux(self, normalized=False, save_name=None):
        """
        Plots the log graph of absolute game values from the saddle point value. Uses z variables instead of x variables.
        """
        plt.figure()
        if normalized:
            plt.plot(abs(np.asarray(self.game_val_list_aux) - self.saddle_point_value))
            plt.yscale('log')
            plt.ylabel("Normalized absolute game value (with z variables)")
        else:
            plt.plot(np.asarray(self.game_val_list_aux))
            plt.ylabel("Game value (with z variables)")
        plt.xlabel("Itr.")
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()

    def visualize_augmented_game_value(self, normalized=False, save_name=None):
        """
        Plots the log graph of absolute game values from the saddle point value.
        """
        plt.figure()
        if normalized:
            plt.plot(abs(np.asarray(self.aug_lag_value_list) - self.saddle_point_value))
            plt.yscale('log')
            plt.ylabel("Normalized absolute augmented game value")
        else:
            plt.plot(np.asarray(self.aug_lag_value_list))
            plt.ylabel("Augmented game value")
        plt.xlabel("Itr.")
        if save_name is not None:
            tikzplotlib.save(save_name)

        plt.show()
