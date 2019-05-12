import numpy as np
from sigmas import SigmaPoints
from unscented_transform import unscented_transform

points = SigmaPoints()
X, wm, wc = points.get_set(np.ones(3), np.eye(3), 'julier')

# Nonlinear function
def F(x):
    return np.array([x[1]*x[2], x[2]**2, 5*x[0]*x[1]*x[2] + 2.*x[1]])

print(X)
print()
print(wm)

print(X.shape)
y, Py, = unscented_transform(X.T, wm, wc)
print(y)
print(Py)
