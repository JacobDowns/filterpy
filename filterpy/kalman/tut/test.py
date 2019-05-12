import numpy as np
from tut_sigmas import TutSigmaPoints

points = TutSigmaPoints()
points.get_set(np.ones(3), np.eye(3), 'hermite')

#print(X)
