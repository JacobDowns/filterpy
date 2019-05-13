import numpy as np
from tut_sigmas import TutSigmaPoints

points = TutSigmaPoints()
X, w, w = points.get_set(np.ones(5), np.eye(5), 'li')

print(X)
