import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman.tut.tut_sigmas import TutSigmaPoints
from filterpy.kalman.tut.tut_normal import TutNormal


class SigmaTransform(object):
    """
    Estimates the statistics of a noisy nonlinear transformation 
        y = f(x)
    where x ~ N(x, Px).

    Parameters
    ----------

    sqrt_method : function(ndarray), default=np.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

    add : callable (x, y), optional
        Function that computes the sum of x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
    """

    def __init__(self, sqrt_method = None, add = None):
        self.points = TutSigmaPoints(sqrt_method, add)

        
    # Do an unscented transform 
    def do_transform(self, x, Px, f, mean_fn=None, residual_x=None,
                     residual_y=None, **sigma_args):
        """ 
        Estimates mean, covariance, and cross-covariance for a nonlinear
        transformation f : R^n -> R^m
        y = f(x) + q
        with x ~ N(x, Px), q~N(0,Q) using a sigma point method.

        Parameters
        ----------

        x : numpy.array(n)
            Prior mean vector

        Px : numpy.array(n,n) 
           Prior covariance matrix

        mean_fn : callable  (X, weights), optional
            Function that computes the mean of the provided sigma points
            and weights. Use this if your state variable contains nonlinear
            values such as angles which cannot be summed.

        residual_x : callable (x, y), optional
        residual_y : callable (x, y), optional
            Function that computes the residual (difference) between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars. One is for the state variable,
            the other is for the measurement state.

        sigma_args : additional keyword arguments, optional
            Sigma point arguments such as the sigma set type and scaling 
            parameters. merwe sigma points are used by default.

        Returns
        -------

        y : np.array(m)
            Estimated mean of the transformed random variable y

        Py : np.array(m,m)
            Estimated covariance of the transformed random variable y

        Pxy : np.array(n,m)
            Estimated cross-covariance of the transformation

        X : np.array(n, N)
            Sigma points used to estimate y, Py, and Pxy. The number
            of sigma points N depends on the method used.

        wm : np.array(N)
        wc : np.array(N)
            Mean and covariance weights. N depends on the method used
            to generate the sigma points. 


        References
        ----------

        .. [1] https://nbviewer.jupyter.org/github/sbitzer/UKF-exposed/blob/master/UKF.ipynb
        """

        if np.isscalar(x):
            x = np.asarray([x])
            n = 1
        else:
            # State dimension
            n = len(x)

        if  np.isscalar(Px):
            Px = np.eye(n)*Px
        else:
            Px = np.atleast_2d(Px)

        
        # Get sigma points
        X, wm, wc = self.points.get_set(x, Px, **sigma_args)
        
        # Columns are transformed sigma points
        Y = f(X)
        # Dimensions
        n, m = Y.shape

        # Compute transformed mean
        try:
            if mean_fn is None:
                y = np.dot(Y, wm)    
            else:
                y = y_mean(Y, wm)
        except:
            raise

        # Compute covariance
        if residual_y is np.subtract or residual_y is None:
            ry = Y - y[:, np.newaxis]
            Py = ry @ np.diag(wc) @ ry.T
        else:
            Py = np.zeros((n, n))
            for k in range(m):
                ry = residual_y(Y[k], y)
                Py += wc[k] * np.outer(ry, ry)


        # Compute cross covariance
        if residual_x is np.subtract or residual_x is None:
            rx = X - x[:, np.newaxis]
            Pxy = rx @ np.diag(wc) @ ry.T
        else:
            Pxy = np.zeros((m, n))
            for k in range(m):
                rx = residual_x(X[k], x)
                Pxy += wc[k] * np.outer(rx, ry)

        joint_mean = np.block([x,y])
        joint_cov = np.block([[Px, Pxy],[Pxy.T, Py]])
        joint_dist = TutNormal(joint_mean, joint_cov)
        
        return joint_dist, X, wm, wc  


    
    def get_conditional(self, x, Px, y, Py, Pxy, yo, Pyo):
        """ 
        Computes the conditional distribution x|y_m from the given the joint distribution
        [x, y] ~ N([x, mu], [[Px, Pxy], [Pxy^T, Py]])

        Parameters
        ----------

        x : numpy.array(n)
            Prior mean vector

        Px : numpy.array(n,n) 
            Prior covariance matrix

        y : numpy.array(m)
            Measurement mean

        Py : numpy.array(m,m) 
            Measurement covariance

        Pxy : numpy.array(n, m) 
            Cross covariance

        ym : numpy.array(m) 
            Observation mean 

        Pyo : numpy.array(m,m) 
            Observation covariance 

        Returns
        -------

        x_new : np.array(n)
            Mean of conditional distribution for x

        Px_new : np.array(n,n)
            Covariance of conditional distribution for x

        """

        # Add measurement noise
        Py1 = Py + Pyo
        # Kalman gain
        print(Pxy)
        K = Pxy@np.linalg.inv(Py)
        print("K", K)
        
        x_prime = x + K@(yo - y)
        Px_prime = Px - K@Py1@K.T

        return x_prime, Px_prime

    
       
