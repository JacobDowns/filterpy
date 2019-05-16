import numpy as np
from itertools import combinations, product
from scipy.misc import comb


def generate_fully_symmetric_set(n, vals):
    """
    Generates a fully set of symmetric points of dimension n with 
    values given in vals. 

    Parameters
    ----------

    n: int
        Dimension of points in fully symmetric set

    vals : numpy.array(k)
        Non-zero values in the fully symmetric set

    Returns
    -------

    S : numpy.array(n, 2^k * (n choose k))
        Each column is a point in th fully symmetric set
    """
     
    indexes = np.arange(n)
    k = len(vals)
    index_combs = combinations(indexes, k)
    S = np.zeros((n, 2**k * comb(n,k, exact = True)))
    i = 0
    for index_comb in list(index_combs):
        signs = np.ones((k,2))
        signs[:,1] = -1.
        sign_combs = product(*signs)
        for sign_comb in list(sign_combs):  
            S[index_comb, i] = np.sqrt(3.)
            S[index_comb, i] *= sign_comb
            i += 1
            
    return S
