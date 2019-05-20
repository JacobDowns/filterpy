import numpy as np
import matplotlib.pyplot as plt
from tut_sigmas import TutSigmaPoints

class SigmaTransform(object):

    def __init__(self):
        self.points = TutSigmaPoints
        
    # Do an unscented transform 
    def do_transform(self, x, Px, f, mean_func=None, x_residual=None, y_residual=None,
                     sigma_set='mwer', **sigma_args):
        
        # Get sigma points
        X, wm, wc = self.points.get_sigma_set(x, Px, sigma_set, **sigma_args)
        # Transform them
        Y = self.f(X)

        kmax, n = Y.shape

        # Compute transformed mean
        try:
            if mean_fn is None:
                y = np.dot(wm, Y)    
            else:
                y = y_mean(Y, wm)
        except:
            print(Y)
            raise
        
        # Compute covariance
        if y_residual is np.subtract or y_residual is None:
            ry = Y - y[np.newaxis, :]
            Py = ry.T @ np.diag(wc) @ ry
        else:
            Py = np.zeros((n, n))
            for k in range(kmax):
                ry = y_residual(Y[k], y)
                Py += wc[k] * np.outer(ry, ry)


        # Compute cross covariance
        if x_residual is np.subtract or x_residual is None:
            rx = X - x[np.newaxis, :]
            Pxy = rx.T @ np.diag(wc) @ ry
        else:
            Py = np.zeros((n, n))
            for k in range(kmax):
                rx = x_residual(X[k], x)
                Pxy += wc[k] * np.outer(rx, ry)
        

        if noise_cov is not None:
            Py += noise_cov

        return y, Py, Pxy


       



        
                


        
        
