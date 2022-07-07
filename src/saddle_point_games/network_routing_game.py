import numpy as np
from ..optim.projections import Projections
from ..optim.saddle_point_admm import SaddlePointADMM
from ..optim.saddle_point_frank_wolfe import SaddlePointFW
from ..mdp.mdp import MDP
from ..mdp.averagemdp import AverageMDP
import matplotlib.pyplot as plt

class NetworkRoutingGame(SaddlePointADMM, SaddlePointFW):


    def __init__(self, mdp: MDP, visit_const: dict):
        """
        Initializes a network routing game with adversarial costs.

        :param mdp: An MDP (Graph) that the congestion game is played on.
        :param visit_const: A dictionary of states and lower bounds on the visitation frequencies. Keys are states and
        the values are their respective lower bounds.
        """
        self.mdp = mdp
        self.N = sum(sum(self.mdp.NA_list))
        self.visit_const = visit_const
        super().__init__()

    #SP-ADMM related methods
    def initialize_vars_spadmm(self):
        """
        Initializes the varibales of the saddle point ADMM game.

        x_a and x_b are initialized with a zero vector.
        lmd_a and lmd_b are initialized with a vector of zeros
        z_a and z_b are initialized with the projection of x_a and x_b to X_a and X_b, respectively.
        :return: Initialized variables
        """

        x_a = np.ones(self.N)/self.N
        x_b = np.ones(self.N)/self.N
        lmd_a = np.zeros(self.N)
        lmd_b = np.zeros(self.N)

        z_a = self.project_z_a_spadmm(x_a)
        z_b = self.project_z_b_spadmm(x_b)

        return x_a, x_b, z_a, z_b, lmd_a, lmd_b

    def solve_augmented_saddle_game_spadmm(self, z_a, z_b, lmd_a, lmd_b):
        """
        Solves the quadratic, bilinear saddle point game by decomposition.
        :param z_a: Auxilary primal variable for the minimizer from the previous iteration.
        :param z_b: Auxilary primal variable for the maximizer from the previous iteration.
        :param lmd_a: Dual variable for the minimizer from the previous iteration.
        :param lmd_b: Dual variable for the minimizer from the previous iteration.
        :return: A solution for the augmented saddle point game
        """
        x_a = np.zeros(z_a.shape)
        x_b = np.zeros(z_b.shape)
        for i in range(self.N):
            x_a[i], x_b[i] = self.solve_individual_augmented_saddle_game(z_a[i], z_b[i],
                                                                         lmd_a[i],
                                                                         lmd_b[i], self.rho_a, self.rho_b)
        return x_a, x_b

    def solve_individual_augmented_saddle_game(self, z_a_i, z_b_i, lmd_a_i, lmd_b_i, rho_a, rho_b):
        f"""
        Analytically solves the individual augmented saddle point games. The game is
        min_(x_a_i) max_(x_b_i) x_a_i*(x_a_i + x_b_i)
                                + lmd_a_i*(x_a_i - z_a_i) + rho_a/2*(x_a_i - z_a_i)^2
                                - lmd_b_i*(x_b_i - z_b_i) + rho_b/2*(x_b_i - z_b_i)^2
        subject to              0 <= x_a[i] <= 1
                                0<= x_b[i] <= 1
        :return: A saddle point x_a_i and x_b_i
        """

        # Solution points from the first order optimality conditions

        #Optimal responses based on first order conditions
        find_opt_x_b = lambda x_a, x_b :( x_a + z_b_i * rho_b - lmd_b_i) / rho_b
        find_opt_x_a = lambda x_a, x_b :(-x_b + z_a_i * rho_a - lmd_a_i) / (rho_a + 2)

        #Case 1 x_a plays normal, x_b_plays normal
        x_a_i = (-rho_b * z_b_i + rho_a * rho_b * z_a_i - lmd_a_i * rho_b + lmd_b_i) / (2*rho_b + rho_a * rho_b + 1)
        x_b_i = (x_a_i - lmd_b_i + rho_b * z_b_i)/rho_b

        if (x_a_i >= 0) and (x_a_i <= 1) and (x_b_i >= 0) and (x_b_i <= 1):
            return x_a_i, x_b_i

        #Case 2 x_a plays 0, x_b plays normal
        x_a_i = 0
        x_b_i = find_opt_x_b(x_a_i, x_b_i)
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (x_b_i >= 0) and (x_b_i <= 1) and (opt_x_a_i <= 0):
            return x_a_i, x_b_i

        #Case 3 x_a plays 1, x_b plays normal
        x_a_i = 1
        x_b_i = find_opt_x_b(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        if (x_b_i >= 0) and (x_b_i <= 1) and (opt_x_a_i >= 1):
            return x_a_i, x_b_i

        #Case 4 x_a plays normal, x_b plays 0
        x_b_i = 0
        x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (x_a_i >= 0) and (x_a_i <= 1) and (opt_x_b_i <= 0):
            return x_a_i, x_b_i


        #Case 5 x_a plays 0, x_b plays 0
        x_a_i = 0
        x_b_i = 0
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (opt_x_b_i <= 0) and (opt_x_a_i <= 0):
            return x_a_i, x_b_i

        #Case 6 x_a plays 1, x_b plays 0
        x_a_i = 1
        x_b_i = 0
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (opt_x_b_i <= 0) and (opt_x_a_i >= 1):
            return x_a_i, x_b_i

        #Case 7 x_a plays normal, x_b plays 1
        x_b_i = 1
        x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (x_a_i >= 0) and (x_a_i <= 1) and (opt_x_b_i >= 1):
            return x_a_i, x_b_i

        #Case 8 x_a plays 0, x_b plays 1
        x_a_i = 0
        x_b_i = 1
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (opt_x_b_i >= 1) and (opt_x_a_i <= 0):
            return x_a_i, x_b_i

        #Case 9 x_a plays 1, x_b plays 1
        x_a_i = 1
        x_b_i = 1
        opt_x_a_i = find_opt_x_a(x_a_i, x_b_i)
        opt_x_b_i = find_opt_x_b(x_a_i, x_b_i)
        if (opt_x_b_i >= 1) and (opt_x_a_i >= 1):
            return x_a_i, x_b_i

        return x_a_i, x_b_i

    def project_z_a_spadmm(self, vec):
        vec_mat = vec.reshape((1, len(vec)))
        return np.squeeze(AverageMDP.compute_closest_policy_l2_norm(self.mdp, vec_mat, self.visit_const))

    def project_z_b_spadmm(self, vec):
        vec_mat = vec.reshape((1, len(vec)))
        return np.squeeze(AverageMDP.compute_closest_policy_l2_norm(self.mdp, vec_mat))


    def compute_game_val_itr(self, x_a, x_b):
        """
        Computes the game value for the inner product game
        :param x_a: Primal variable for the minimizer
        :param x_b: Primal variable for the maximizer
        :return: The objective value
        """
        return np.sum(np.multiply(x_a, x_a + x_b))

    def compute_game_vals(self):
        """
        Computes the game values for a given list of points
        """
        self.game_val_list = []
        for (x_a, x_b) in zip(self.x_a_list, self.x_b_list):
            self.game_val_list.append(self.compute_game_val_itr(x_a, x_b))

    def compute_game_vals_aux(self):
        """
        Computes the game values for a given list of points
        """
        self.game_val_list_aux = []
        for (z_a, z_b) in zip(self.z_a_list, self.z_b_list):
            self.game_val_list_aux.append(self.compute_game_val_itr(z_a, z_b))

    def find_best_responses_both(self):
        """
        Computes the best responses of each player given the other player's action
        """
        self.game_val_list_aux_min_best_max_itr = []
        self.game_val_list_aux_min_itr_max_best = []
        self.min_best_response_list = []
        self.max_best_response_list = []
        for (z_a, z_b) in zip(self.z_a_list, self.z_b_list):
            min_best_response = np.squeeze(AverageMDP.minimize_quadratic_cost(self.mdp, quad_cons=np.ones((1, self.mdp.NSA)), lin_cons=z_b.reshape((1, len(z_b))), visit_const=self.visit_const))
            self.game_val_list_aux_min_best_max_itr.append(self.compute_game_val_itr(min_best_response, z_b))
            max_best_response = np.squeeze(AverageMDP.maximize_linear_cost(self.mdp, lin_cons=z_a.reshape(1, len(z_a))))
            self.game_val_list_aux_min_itr_max_best.append(self.compute_game_val_itr(z_a, max_best_response))
        self.optimality_gap_list = []
        for i in range(0, len(self.game_val_list_aux_min_best_max_itr)):
            best_upper_bound = min(self.game_val_list_aux_min_itr_max_best[0:(i+1)])
            best_lower_bound = max(self.game_val_list_aux_min_best_max_itr[0:(i+1)])
            self.optimality_gap_list.append(best_upper_bound - best_lower_bound)



    #SP-FW related methods
    def initialize_vars_fw(self):
        """
        Initializes the variables of the saddle point FW game.

        x_a and x_b are initialized with the projection of zero vectors to X_a and X_b, respectively.
        :return: Initialized variables
        """
        x_a_unif = np.ones(self.N)/self.N
        x_b_unif = np.ones(self.N)/self.N
        x_a = self.project_z_a_spadmm(x_a_unif)
        x_b = self.project_z_b_spadmm(x_b_unif)
        return x_a, x_b

    def df_dx_a(self, x_a, x_b):
        """
        Computes the derivative of the objective function wrt x_a
        :param x_a: The minimizer's variable
        :param x_b: The maximizer's variable
        :return: The derivative
        """
        return 2*x_a + x_b

    def df_dx_b(self, x_a, x_b):
        """
        Computes the derivative of the objective function wrt x_b
        :param x_a: The minimizer's variable
        :param x_b: The maximizer's variable
        :return: The derivative
        """
        return x_a

    def maximizer_move_point_spfw(self, df_dx_b):
        """
        Computes the density vector for the maximizer that maximizes the first order approximation of the objective function
        :param df_dx_b: The derivative of the objective function at the current solution point
        :return: A density vector
        """
        move_point = np.squeeze(AverageMDP.maximize_linear_cost(self.mdp, lin_cons=df_dx_b.reshape(1, len(df_dx_b))))
        return move_point

    def minimizer_move_point_spfw(self, df_dx_a):
        """
        Computes the density vector for the maximizer that maximizes the first order approximation of the objective function
        :param df_dx_a: The derivative of the objective function at the current solution point
        :return: A density vector
        """
        move_point = np.squeeze(AverageMDP.maximize_linear_cost(self.mdp, lin_cons=-df_dx_a.reshape(1, len(df_dx_a)), visit_const=self.visit_const))
        return move_point

    def compute_game_vals_fw(self):
        """
        Computes the game values for a given list of solution points
        """
        self.game_val_list_fw = []
        for (x_a, x_b) in zip(self.x_a_list_fw, self.x_b_list_fw):
            self.game_val_list_fw.append(self.compute_game_val_itr(x_a, x_b))


    def find_best_responses_both_fw(self):
        """
        Computes the best responses of each player given the other player's action
        """
        self.game_val_list_min_best_max_itr_fw = []
        self.game_val_list_min_itr_max_best_fw = []
        self.min_best_response_list_fw = []
        self.max_best_response_list_fw = []
        for (x_a, x_b) in zip(self.x_a_list_fw, self.x_b_list_fw):
            min_best_response = np.squeeze(AverageMDP.minimize_quadratic_cost(self.mdp, quad_cons=np.ones((1, self.mdp.NSA)), lin_cons=x_b.reshape((1, len(x_b))), visit_const=self.visit_const))
            self.game_val_list_min_best_max_itr_fw.append(self.compute_game_val_itr(min_best_response, x_b))
            max_best_response = np.squeeze(AverageMDP.maximize_linear_cost(self.mdp, lin_cons=x_a.reshape(1, len(x_a))))
            self.game_val_list_min_itr_max_best_fw.append(self.compute_game_val_itr(x_a, max_best_response))
        self.optimality_gap_list_fw = []
        for i in range(0, len(self.game_val_list_min_best_max_itr_fw)):
            best_upper_bound = min(self.game_val_list_min_itr_max_best_fw[0:(i+1)])
            best_lower_bound = max(self.game_val_list_min_best_max_itr_fw[0:(i+1)])
            self.optimality_gap_list_fw.append(best_upper_bound - best_lower_bound)


    def visualize_game_value_no_normalization_compare(self):
        """
        Plots the  graph of game values.
        """
        plt.figure()
        plt.plot(np.asarray(self.game_val_list_fw), label='FW')
        plt.plot(np.asarray(self.game_val_list_aux), label='ADMM')
        # plt.yscale('log')
        plt.legend()
        plt.ylabel("Game value")
        plt.xlabel("Itr.")
        plt.show()