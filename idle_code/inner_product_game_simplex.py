from src.optim import Projections
from src.saddle_point_games.inner_product_game import InnerProductGame


class InnerProductGameSimplex(InnerProductGame):
    def __init__(self, game_constants, box_a, box_b):
        super().__init__(game_constants, box_a, box_b)

    def project_z_a(self, vec):
        return Projections.project_onto_simplex(vec)

    def project_z_b(self, vec):
        return Projections.project_onto_simplex(vec)
