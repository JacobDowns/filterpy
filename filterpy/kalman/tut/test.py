import numpy as np
from tut_sigmas import TutSigmaPoints

points = TutSigmaPoints()
X, wm, wc = points.get_set(np.ones(2), np.eye(2), 'simplex')

print(X)
