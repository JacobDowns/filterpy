import numpy as np

class TutNormal(object):

    def __init__(self, u0, Pu):
        self.u0 = u0
        self.Pu = Pu


    def get_conditional(self, yo, Q):

        if np.isscalar(yo):
            yo = [yo]
            Q = np.atleast_2d(Q)
            m = 1
        else :
            m = len(yo)
            
        n = len(self.u0) - m

        # Partition the mean and covariance
        x = self.u0[0:n]
        mu = self.u0[n:]
        Px = self.Pu[0:n, 0:n]
        S = self.Pu[n:, n:] + Q
        C = self.Pu[n:, 0:m] 

        # Compute the conditional distribution
        K = C@np.linalg.inv(S)
        x_new = x + K@(yo - mu)
        Px_new = Px - K@S@K.T

        return TutNormal(x_new, Px_new)


    def get_conditional(self, yo, Q):

        if np.isscalar(yo):
            yo = [yo]
            Q = np.atleast_2d(Q)
            m = 1
        else :
            m = len(yo)
            
        n = len(self.u0) - m

        # Partition the mean and covariance
        x = self.u0[0:n]
        mu = self.u0[n:]
        Px = self.Pu[0:n, 0:n]
        S = self.Pu[n:, n:] + Q
        C = self.Pu[n:, 0:m] 

        # Compute the conditional distribution
        K = C@np.linalg.inv(S)
        x_new = x + K@(yo - mu)
        Px_new = Px - K@S@K.T

        return TutNormal(x_new, Px_new)


    

       



        
                


        
        
