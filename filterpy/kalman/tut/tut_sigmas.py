import numpy as np
#from scipy.linalg import cholesky
from numpy.linalg import cholesky, inv
from itertools import combinations

class TutSigmaPoints(object):
    
    """
    Generates sigma points and weights according according to one of 
    several published methods. 

     Parameters
    ----------

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

    add : callable (x, y), optional
        Function that computes the sum of x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
    """

    def __init__(self, sqrt_method = None, add = None):

        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if add is None:
            self.add = np.add
        else:
            self.add = add
            
        # Available sigma point sets
        self.sigma_functions = {}
        self.sigma_functions['merwe'] = self.__get_set_merwe__
        self.sigma_functions['menegaz'] = self.__get_set_menegaz__
        self.sigma_functions['li'] = self.__get_set_li__
        self.sigma_functions['mysovskikh'] = self.__get_set_mysovskikh__
        self.sigma_functions['gauss'] = self.__get_set_gauss__
        self.sigma_functions['julier'] = self.__get_set_julier__
        self.sigma_functions['simplex'] = self.__get_set_simplex__
        self.sigma_functions['hermite'] = self.__get_set_hermite__
        
         
    def get_set(self, x, Px, set_name, **scale_args):

        """ 
        Computes the sigma points for and weights for the unscented
        transform. Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        set_name : string, default='mwer'
           The name of the sigma point set to compute. 

        **args : scalar scaling variables
           Additional parameters used to scale the sigma points.

        Returns
        -------

        X : np.array, of size (n, N)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space. N is the 
            number of sigma points, which varies depending on the method.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance
        """

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(Px):
            Px = np.eye(n)*Px
        else:
            Px = np.atleast_2d(Px)

        # State dimension
        n = len(x)
        # Get sigma points for N(0, I)
        X, wm, wc = self.sigma_functions[set_name](n, **scale_args)
        # Change variables to get sigma points for N(x, Px)
        X = self.add(x[:,None].repeat(X.shape[1], axis = 1), self.sqrt(Px)@X)

        return X, wm, wc


    def __get_set_merwe__(self, n, **scale_args):
        """
        Generates sigma points and weights according Van der Merwe's
        2004 dissertation[1]. Scaling parameters include alpha, beta, and 
        kappa. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 points will be generated.

        kappa : float, default=0.
            Scaling factor that can reduce high order errors. kappa=0 gives
            the standard unscented filter. According to [Julier], if you set
            kappa to 3-n for a Gaussian x you will minimize the fourth
            order errors.

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance

        References
        ----------

        .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
            Filter to Nonlinear Systems"

       """

        
        alpha = 0.5
        if 'alpha' in scale_args:
            alpha = scale_args['alpha']
        beta = 2.
        if 'beta' in scale_args:
            beta = scale_args['beta']
        kappa = 3. - n
        if 'kappa' in scale_args:
            kappa = scale_args['kappa']

        lambda_ = alpha**2*(n + kappa) - n
        

        ### Sigma points
        X = np.sqrt(n + lambda_)*np.block([np.zeros(n)[:,None], np.eye(n), -np.eye(n)])

        
        ### Weights
        c = 1. / (2.*(n + lambda_))
        wc = np.full(2*n + 1, c)
        wm = np.full(2*n + 1, c)
        wm[0] =  lambda_ / (n + lambda_)
        wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
        
        return X, wm, wc


    def __get_set_julier__(self, n, **scale_args):
        """
        Generates sigma points and weights according Julier's 1997 paper [1]. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 points will be generated.

        kappa : float, default=0.
            Scaling factor that can reduce high order errors. kappa=0 gives
            the standard unscented filter. According to [Julier], if you set
            kappa to 3-n for a Gaussian x you will minimize the fourth
            order errors.

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance

        References
        ----------

        .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
            Filter to Nonlinear Systems"

       """

        
        kappa = 3. - n
        if 'kappa' in scale_args:
            kappa = scale_args['kappa']
        

        # Sigma points
        X = np.sqrt(n + kappa)*np.block([np.zeros(n)[:,None], np.eye(n), -np.eye(n)])
        
        # Weights
        c = 1. / (2.*(n + kappa))
        wm = np.full(2*n + 1, c)
        wm[0] = kappa / (n + kappa)
        
        return X, wm, wm

    
    def __get_set_gauss__(n, **scale_args):
        """ 
        Generates sigma points and weights for the so called Gauss set in [1],
        which is just the mwer set with alpha = 1, beta = 0, and kappa = 3.

        Parameters
        ----------

        n : int
            Dimensionality of the state. n+1 points will be generated.

        w0 : scalar
           A scaling parameter with 0 < w0 < 1

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance


        References
        ----------

        .. [1] https://nbviewer.jupyter.org/github/sbitzer/UKF-exposed/blob/master/UKF.ipynb
        """
        return self.__get_set_mwer__(n, alpha = 1., beta = 0., kappa = 3.)
        

    
    def __get_set_menegaz__(self, n, **scale_args):
        """ 
        Computes the sigma points and weights for the unscented transform
        using the method outlined by Menegaz [1] for zero mean x and 
        identity covariance. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. n+1 points will be generated.

        w0 : scalar
           A scaling parameter with 0 < w0 < 1

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance


        References
        ----------

        .. [1] H.M. Menegaz et al. "A new smallest sigma set for the Unscented 
           Transform and its applications on SLAM" 
        """

        w0 = 0.5
        # If the first weight is defined
        if 'w0' in scale_args:
            w0 = scale_args['w0']
            if w0 >= 1.0 or w0 <= 0.0:
                raise ValueError("w0 must be between 0 and 1")


        ### Sigma point set
        alpha = np.sqrt((1. - w0) / n)
        C = self.sqrt(np.diag(np.ones(n), 0) - (alpha**2)*np.ones((n, n)))
        C_inv = inv(C)

        W = np.diag(np.diag(w0*(alpha**2)*C_inv @ np.ones((n,n)) @ C_inv.T), 0)
        W_sqrt = self.sqrt(W)

        X = np.zeros((n, n+1))
        X[:,0] =  -(alpha / np.sqrt(w0))*np.ones(n)
        X[:,1:] = C @ inv(W_sqrt)
        X = X.T

        
        ### Weights
        w = np.zeros(n+1)
        w[0] = w0
        w[1:] = np.diag(W, 0)

        return X.T, w, w


    def __get_set_simplex__(self, n, **scale_args):
        """
        Generates sigma points and weights according to the simplex
        method presented in [1].

         Parameters
        ----------

        n : int
            Dimensionality of the state. n+1 points will be generated.

        Returns
        -------

        X : np.array, of size (n, n+1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance


        References
        ----------

        .. [1] Phillippe Moireau and Dominique Chapelle "Reduced-Order
           Unscented Kalman Filtering with Application to Parameter
           Identification in Large-Dimensional Systems"
           DOI: 10.1051/cocv/2010006
        """


        # Generate sigma points
        lambda_ = n / (n + 1)
        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])
        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1))
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]

        X = np.sqrt(n)*Istar

        # Generate weights
        wm = np.full(n + 1, lambda_)
        
        return X, wm, wm


    def __get_set_li__(self, n, **scale_args):
        """ 
        Computes the sigma points and weights for a  slightly modified 
        version of the fifth order cubature rule in Li [1]. Setting 
        the scaling parameter r = sqrt(3.) recovers the original method. 
        This method requires n > 4 and n - r^2 - 1 != 0. 

        Parameters
        ----------

         n : int
            Dimensionality of the state. 2n^2 + 1 points will be generated.

        r : scalar
           A scaling parameter with n - r^2 - 1 != 0

        Returns
        -------

        X : np.array, of size (n, 2n^2 + 1)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance


        References
        ----------

        .. [1] Z. Li et al. "A Novel Fifth-Degree Cubature Kalman Filter 
           for Real-Time Orbit Determination by Radar" 
        """

        r = np.sqrt(3./2.)
        # If the first weight is defined
        if 'r' in scale_args:
            r = slace_args['r']
            if n < 5 or abs(n - r**2 - 1.) < 1e-16:
                raise ValueError("This method requires n>4 and n - r^2 - 1 != 0")
        

        ### Generate Weights

        # Coordinate for the first symmetric set
        r1 = (r*np.sqrt(n-4.))/np.sqrt(n - r**2 - 1.)
        # First symmetric set weight
        w2 = (4. - n) / (2. * r1**4)
        # Second symmetric set weight
        w3 = 1. / (4. * r**4)
        # Center point weight
        w1 = 1. - 2.*n*w2 - 2.*n*(n-1)*w3
        # Vector of weights
        w = np.block([w1, np.repeat(w2, 2*n), np.repeat(w3, 2*n*(n-1))])


        ### Generate Points
        
        # First fully symmetric set
        X0 = r1*np.eye(n)
        X0_s = np.block([X0, -X0])
        
        # Second fully symmetric set
        X1 = r*np.eye(n)
        indexes_i = []
        indexes_j = []
        for i in range(1,n):
            indexes_i.append(np.repeat([i],i))
            indexes_j.append(np.arange(0,i))
        indexes_i = np.concatenate(indexes_i).ravel()
        indexes_j = np.concatenate(indexes_j).ravel()
        P1 = X1[indexes_i, :].T + X1[indexes_j, :].T
        P2 = X1[indexes_i, :].T - X1[indexes_j, :].T
        X1_s = np.block([P1, P2, -P1, -P2])

        print(P1)
        print()
        print(P2)
        for i in range(P2.shape[1]):
            print(P2[:,i])
        quit()
        # Full set of points (columns are points)
        X = np.block([np.zeros(n)[:,None], X0_s, X1_s])

        return X, w, w


    def __get_set_hermite__(self, n, **args):
        """
        Computes the sigma points and weights for the third degree Gauss hermite
        quadrature rule. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. n^2 + 3n + 3 points will be generated.

        Returns
        -------

        X : np.array, of size (n, 2^n)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance

        References
        ----------

        .. [1] J. Lu and D.L. Darmofal "Higher-dimensional integration 
           with gaussian weight for applications in probabilistic design" 
        """

        ### Generate sigma points
        
        # First set of points
        I = (np.arange(n)[:,None] + 1).repeat(n + 1, axis = 1).T
        R = (np.arange(n + 1) + 1)[:,None].repeat(n, axis = 1)
        A = -np.sqrt((n+1.) / (n*(n-I+2.)*(n-I+1.)))
        indexes = (I == R)
        A[indexes] = np.sqrt( ((n+1.)*(n-R[indexes]+1.)) / (n*(n-R[indexes]+2.)))
        indexes = I > R
        A[indexes] = 0.
        

        # Second set of points
        ls = np.arange(n+1)[:,None].repeat(n+1)
        ks = (np.arange(n+1)[:,None].repeat(n+1, axis = 1).T).flatten() 
        indexes = ks < ls
        B = np.sqrt(n / (2.*(n-1.)))*(A[ks[indexes]] + A[ls[indexes]])

        # Full set
        #X = np.sqrt(n + 2.)*np.block([[np.zeros(n)], [A], [-A], [B], [-B]])
        X = np.block([[np.zeros(n)], [A], [-A], [B], [-B]])


        ### Generate weights
        
        w0 = 2./(n+2.)
        w1 = (n**2 * (7. - n)) / (2.*(n + 1.)**2 * (n+2.)**2)
        w2 = (2.*(n-1.)**2) / ((n+1.)**2 * (n+2.)**2)
        w = np.block([w0, np.repeat(w1, 2*len(A)), np.repeat(w2, 2*len(B))])
        
        return X.T, w, w


    def __get_set_mysovskikh__(self, n, **scale_args):
        """
        Computes the sigma points and weights for a fifth order cubature 
        rule due to Mysovskikh, and outlined in Lu and Darmofal [1]. 
        This method has no scaling parameter. 

        Parameters
        ----------

        n : int
            Dimensionality of the state. n^2 + 3n + 3 points will be generated.

        Returns
        -------

        X : np.array, of size (n, n^2 + 3n + 3)
            Two dimensional array of sigma points. Each column is a sigma 
            point.

        wm : np.array
            weight for each sigma point for the mean

        wc : np.array
            weight for each sigma point for the covariance

        References
        ----------

        .. [1] J. Lu and D.L. Darmofal "Higher-dimensional integration 
           with gaussian weight for applications in probabilistic design" 
        """

        ### Generate sigma points
        
        # First set of points
        I = (np.arange(n)[:,None] + 1).repeat(n + 1, axis = 1).T
        R = (np.arange(n + 1) + 1)[:,None].repeat(n, axis = 1)
        A = -np.sqrt((n+1.) / (n*(n-I+2.)*(n-I+1.)))
        indexes = (I == R)
        A[indexes] = np.sqrt( ((n+1.)*(n-R[indexes]+1.)) / (n*(n-R[indexes]+2.)))
        indexes = I > R
        A[indexes] = 0.
        

        # Second set of points
        ls = np.arange(n+1)[:,None].repeat(n+1)
        ks = (np.arange(n+1)[:,None].repeat(n+1, axis = 1).T).flatten() 
        indexes = ks < ls
        B = np.sqrt(n / (2.*(n-1.)))*(A[ks[indexes]] + A[ls[indexes]])

        # Full set
        #X = np.sqrt(n + 2.)*np.block([[np.zeros(n)], [A], [-A], [B], [-B]])
        X = np.block([[np.zeros(n)], [A], [-A], [B], [-B]])


        ### Generate weights
        
        w0 = 2./(n+2.)
        w1 = (n**2 * (7. - n)) / (2.*(n + 1.)**2 * (n+2.)**2)
        w2 = (2.*(n-1.)**2) / ((n+1.)**2 * (n+2.)**2)
        w = np.block([w0, np.repeat(w1, 2*len(A)), np.repeat(w2, 2*len(B))])
        
        return X.T, w, w


    def __get_set_hermite__(self, n, **scale_args):
        indexes = range(n)

        # Sigma points
        X = np.zeros((n, 2**(n+1) - 1))
        # Weights
        wm = np.zeros(2**(n+1) - 1)
        wm[0] = (2./3.)**n
        r = 0
        for i in range(1, n+1):
            comb = combinations(indexes, i)
            w = (2./3.)**(n-i) * (1./6.)**i
            for c in list(comb): 
                X[c, 2*r+1] = np.sqrt(3)
                X[c, 2*r+2] = -np.sqrt(3)
                wm[2*r+1] = w
                wm[2*r+2] = w
                r += 1

        return X, wm, wm
