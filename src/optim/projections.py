import numpy as np

class Projections:
    def project_onto_simplex(vec, total=1, tol=1e-7):
        # Projects the given input onto the probability simplex

        proj_vec = np.zeros(vec.shape)
        # Find the projection
        err = 1
        mu_max = max(vec) + total
        mu_min = min(vec) - total
        while err > tol:
            mu = (mu_max + mu_min) / 2
            proj_vec = np.maximum(vec - mu, 0)
            proj_sum = sum(proj_vec)
            err = abs(proj_sum - total)
            if proj_sum >= total:
                mu_min = mu
            else:
                mu_max = mu

        return proj_vec

    def project_onto_ball(vec, r=1):
        # Projects the given input onto the unit ball
        mag = np.linalg.norm(vec, ord=2)
        return r*(vec / max(mag, 1))
