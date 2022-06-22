import numpy as np

class SaddlePointFrankWolfe:
    def __init__(self):
        """
        Initializes an instance of a saddle point Frank-Wolfe problem.
        The instance is not solved yet.
        The default saddle point value is 0. To be modified externally.
        """
        self.instance_solved = False
        self.saddle_point_value = 0

