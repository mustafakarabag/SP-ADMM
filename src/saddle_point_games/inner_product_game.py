import numpy as np
from ..optim.projections import Projections
from ..optim.saddle_point_admm import SaddlePointADMM


class InnerProductGame(SaddlePointADMM):
    def __init__(self, game_constants, box_a, box_b):
        """
        Initializes an inner product game.

        :param game_constants: c_i constants of the game where the objective function is (sum c_i*x_a_i*x_b_i)
        :param box_a: A list of box constraints for the minimizer. box_a[i][0] <= x_a[i] <= box_a[i][1]
        :param box_b: A list of box constraints for the maximizer. box_b[i][0] <= x_b[i] <= box_b[i][1]
        """
        self.game_constants = game_constants
        self.N = len(self.game_constants)  # The number of individual saddle point games
        self.box_a = box_a
        self.box_b = box_b
        super().__init__()

    def initialize_vars(self):
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

        # for i in range(self.N):
        #    x_a[i] = (self.box_a[i][0] + self.box_a[i][1]) / 2
        #    x_b[i] = (self.box_b[i][0] + self.box_b[i][1]) / 2

        x_a = np.random.randn(self.N) / self.N
        x_b = np.random.randn(self.N) / self.N

        z_a = self.project_z_a(x_a)
        z_b = self.project_z_b(x_b)

        return x_a, x_b, z_a, z_b, lmd_a, lmd_b

    def solve_augmented_saddle_game(self, z_a, z_b, lmd_a, lmd_b):
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
            x_a[i], x_b[i] = self.solve_individual_augmented_saddle_game(self.game_constants[i], z_a[i], z_b[i],
                                                                         lmd_a[i],
                                                                         lmd_b[i], self.rho_a, self.rho_b,
                                                                         self.box_a[i], self.box_b[i])
        return x_a, x_b

    def solve_individual_augmented_saddle_game(self, c_i, z_a_i, z_b_i, lmd_a_i, lmd_b_i, rho_a, rho_b, box_a_i=None,
                                               box_b_i=None):
        f"""
        Analytically solves the individual augmented saddle point games. The game is
        min_(x_a_i) max_(x_b_i) c_i*x_a_i*x_b_i
                                + lmd_a_i*(x_a_i - z_a_i) + rho_a/2*(x_a_i - z_a_i)^2
                                - lmd_b_i*(x_b_i - z_b_i) + rho_b/2*(x_b_i - z_b_i)^2
        subject to              box_a_i <= x_a[i] 
                                box_b_i <= x_b[i] 
        :return: A saddle point x_a_i and x_b_i
        """
        # TODO: Extend the solver to support the upper bounds of the box constraints.

        # Solution points from the first order optimality conditions
        x_a_i = (-c_i * rho_b * z_b_i + c_i * lmd_b_i + rho_a * rho_b * z_a_i - lmd_a_i * rho_b) / (
                c_i ** 2 + rho_a * rho_b)
        x_b_i = (c_i * rho_a * z_a_i - c_i * lmd_a_i + rho_a * rho_b * z_b_i - lmd_b_i * rho_a) / (
                c_i ** 2 + rho_a * rho_b)

        # Set the solution to 0 if it is out of bounds
        if x_a_i >= box_a_i[0]:
            if x_b_i < box_b_i[0]:
                x_b_i = box_b_i[0]
                x_a_i = (-c_i * x_b_i + z_a_i * rho_a - lmd_a_i) / rho_a
                if x_a_i < box_a_i[0]:
                    x_a_i = box_a_i[0]
        else:
            x_a_i = box_a_i[0]
            x_b_i = (c_i * x_a_i + z_b_i * rho_b - lmd_b_i) / rho_b
            if x_b_i < box_b_i[0]:
                x_b_i = box_b_i[0]

        return x_a_i, x_b_i

    def compute_game_val_itr(self, x_a, x_b):
        """
        Computes the game value for the inner product game
        :param x_a: Primal variable for the minimizer
        :param x_b: Primal variable for the maximizer
        :return: The objective value
        """
        return np.sum(np.multiply(self.game_constants, np.multiply(x_a, x_b)))

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




class InnerProductGameBall(InnerProductGame):
    """
    Inner product game on in a unit ball
    """
    def __init__(self, game_constants, box_a, box_b):
        super().__init__(game_constants, box_a, box_b)

    def project_z_a(self, vec):
        return Projections.project_onto_ball(vec)

    def project_z_b(self, vec):
        return Projections.project_onto_ball(vec)


class InnerProductGameSimplex(InnerProductGame):
    """
    Inner product game on in a unit ball
    """
    def __init__(self, game_constants, box_a, box_b):
        super().__init__(game_constants, box_a, box_b)

    def project_z_a(self, vec):
        return Projections.project_onto_simplex(vec)

    def project_z_b(self, vec):
        return Projections.project_onto_simplex(vec)
