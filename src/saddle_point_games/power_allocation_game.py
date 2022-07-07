import numpy as np
from ..optim.projections import Projections
from ..optim.saddle_point_admm import SaddlePointADMM

class PowerAllocationGame(SaddlePointADMM):
    def __init__(self, receiver_noises, power_multipliers, box_a, box_b, power_lim_a, power_lim_b):
        self.receiver_noises = receiver_noises
        self.power_multipliers = power_multipliers
        self.N = len(self.receiver_noises) #The number of individual saddle point games
        self.power_lim_a = power_lim_a
        self.power_lim_b = power_lim_b
        self.box_a = box_a
        self.box_b = box_b
        super().__init__()

    #SP-ADMM related methods
    def initialize_vars_spadmm(self):
        """
        Initializes the varibales of the saddle point ADMM game.

        x_a and x_b are initialized with a random Gaussian vector.
        lmd_a and lmd_b are initialized with a vector of zeros
        z_a and z_b are initialized with the projection of x_a and x_b to X_a and X_b, respectively.
        :return: Initialized variables
        """

        x_a = np.zeros(self.N)
        x_b = np.zeros(self.N)
        lmd_a = np.zeros(self.N)
        lmd_b = np.zeros(self.N)

        #for i in range(self.N):
        #    x_a[i] = (self.box_a[i][0] + self.box_a[i][1]) / 2
        #    x_b[i] = (self.box_b[i][0] + self.box_b[i][1]) / 2

        #x_a = np.random.randn(self.N)/self.N
        #x_b = np.random.randn(self.N)/self.N

        z_a = self.project_z_a_spadmm(x_a)
        z_b = self.project_z_b_spadmm(x_b)

        return x_a, x_b, z_a, z_b, lmd_a, lmd_b

    def project_z_a_spadmm(self, vec):
        return Projections.project_onto_simplex(vec, self.power_lim_a)

    def project_z_b_spadmm(self, vec):
        return Projections.project_onto_simplex(vec, self.power_lim_b)

    def solve_augmented_saddle_game_spadmm(self, z_a, z_b, lmd_a, lmd_b):
        """
        Solves the quadratic, bilinear saddle point game by decomposition.
        :param z_a: Auxilary primal variable for the minimizer from the previous iteration.
        :param z_b: Auxilary primal variable for the maximizer from the previous iteration.
        :param lmd_a: Dual variable for the minimizer from the previous iteration.
        :param lmd_b: Dual variable for the minimizer from the previous iteration.
        :return:
        """
        x_a = np.zeros(z_a.shape)
        x_b = np.zeros(z_b.shape)
        for i in range(self.N):
            x_a[i], x_b[i] = self.solve_individual_augmented_saddle_game(self.power_multipliers[i], self.receiver_noises[i], z_a[i], z_b[i], lmd_a[i],
                                                                         lmd_b[i], self.rho_a, self.rho_b, self.box_a[i], self.box_b[i], num_of_itr=1000)
        return x_a, x_b

    def solve_individual_augmented_saddle_game(self, beta_i, rec_noise_i,  z_adversary_i, z_player_i,
                                               lmd_adversary_i, lmd_player_i, rho_adversary, rho_player,
                                               box_adversary_i, box_player_i, num_of_itr=1000, tol=1e-7):
        f"""
        Solves the individual augmented saddle point game using saddle point Frank Wolfe algorithm. 
        Refer to https://arxiv.org/pdf/1610.07797.pdf for the details of the algorithm.
         
        The game is
        min_(x_adversary_i) max_(x_player_i) log(1+(beta_i*x_player_i)/(rec_noise_i + x_adversary_i))
                                + lmd_adversary_i*(x_adversary_i - z_adversary_i) + rho_adversary/2*(x_adversary_i - z_adversary_i)^2
                                - lmd_player_i*(x_player_i - z_player_i) + rho_player/2*(x_player_i - z_player_i)^2
        subject to              box_adversary_i[0] <= x_adversary_i <= box_adversary_i[1]
                                box_player_i[0] <= x_player_i <= box_player_i[1]
        :return: A saddle point x_adversary_i and x_player_i
        """
        #the functions to compute gradients
        dLdx_n = lambda x_p, x_n, b, s, l_n, r_n, z_n: -(b*x_p)/((s+x_n)*(b*x_p+s+x_n)) + l_n + r_n*(x_n-z_n)
        dLdx_p = lambda x_p, x_n, b, s, l_p, r_p, z_p: b/(b*x_p+s+x_n) - l_p - r_p*(x_p-z_p)

        # Initialize from the mid-point of the box constraint
        x_adversary_i = (box_adversary_i[0] + box_adversary_i[1])/2
        x_player_i = (box_player_i[0] + box_player_i[1])/2

        #Current solution of the saddle point Frank Wolfe
        curr_sol = np.asarray([x_adversary_i, x_player_i])

        #Iteratively find the solution to the augmented saddle point game
        for itr in range(num_of_itr):

            #Unpack the current solution
            x_adversary_i = curr_sol[0]
            x_player_i = curr_sol[1]

            #Compute the gradient at the current point
            curr_grad = np.asarray(
                [dLdx_n(x_player_i, x_adversary_i, beta_i, rec_noise_i, lmd_adversary_i, rho_adversary, z_adversary_i),
                -dLdx_p(x_player_i, x_adversary_i, beta_i, rec_noise_i, lmd_player_i, rho_player, z_player_i)])

            #Find the point that minimizes the linear approximation
            #This step can be done analytically since the both variables, x_adversary_i and x_player_i, are one dimentional
            curr_move_point= np.asarray(
                [box_adversary_i[int(curr_grad[0] < 0)],
                 box_player_i[int(curr_grad[1] < 0)]])
            #print(curr_move_point)
            #The expected change in the objective function
            curr_change_mag = np.inner(curr_sol - curr_move_point, curr_grad)
            #print(curr_grad)
            #Return if the expected change in the objective function is below the tolerance level
            if curr_change_mag < tol:
                #print(itr)
                return x_adversary_i, x_player_i
            else:
                step_size = 2/(2+itr)
                curr_sol = (1-step_size)*curr_sol + step_size*curr_move_point
                #print(curr_sol)
        #print(itr)
        return x_adversary_i, x_player_i

    #Game value computation methods
    def compute_game_val_itr(self,x_a, x_b):
        """
        Computes the game value for the power allocation game
        :param x_a: Primal variable for the minimizer
        :param x_b: Primal variable for the maximizer
        :return: The objective value
        """
        signal = np.multiply(np.asarray(self.power_multipliers), x_b)
        noise = np.asarray(self.receiver_noises) + x_a
        return np.sum(np.log(1 + np.divide(signal, noise)))

    def compute_game_vals(self):
        """
        Computes the game values for the primal x variables
        """
        self.game_val_list = []
        for (x_a, x_b) in zip(self.x_a_list, self.x_b_list):
            self.game_val_list.append(self.compute_game_val_itr(x_a, x_b))

    def compute_game_vals_aux(self):
        """
        Computes the game values for the auxiliary primal z variables
        """
        self.game_val_list_aux = []
        for (z_a, z_b) in zip(self.z_a_list, self.z_b_list):
            self.game_val_list_aux.append(self.compute_game_val_itr(z_a, z_b))