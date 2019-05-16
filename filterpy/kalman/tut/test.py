import numpy as np
from tut_sigmas import TutSigmaPoints
from sigma_transform import SigmaTransform

indexes = np.arange(8)
from itertools import combinations, product

f(40)

quit()

st = SigmaTransform()

# Prior
n = 2
x = np.array([1.,1.])
Px = np.eye(n)
# Nonlinear func
def f(X):
    Y = np.array([X[0,:]**2 + 2.0*X[1,:]**4, 3.*X[0,:] + 2., -1.*X[0,:]**2 + X[1,:]**2])
    return Y
    
y, Py, Pxy, X, wm, wc = st.do_transform(x, Px, f)

