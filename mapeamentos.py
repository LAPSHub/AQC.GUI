# mapeamentos.py
# Mapping Library
#
# Luana Goncalves
# 22.fev.2018

"""
Library of performance metrics and regression functions between the metric
in relation to DMOS, including: Analysis of variance (ANOVA), Outliers ratio, Pearson correlation coefficient and
Spearman correlation coefficient, logistic and linear functions."""

import random
import numpy as np
from math import sqrt
from scipy import misc
import random
import matplotlib.pyplot as plt
from numpy import arange, sin, pi, random, array
from  scipy import stats


def anova(x,y):
    
    """Computes the Analysis of variance (ANOVA) between two arrays, metrics
    values array and difference mean score (DMOS).

    F = anova(x,y)

    Parameters
    ----------
    x        : metrics array.
    y        : DMOS array.

    Return
    ----------
    F        : ANOVA value.
    """    
    F, p = stats.f_oneway(x,y)
        
    return F

def outlier(x, y, t):
    
    """Computes the Outliers Ratio of logistic regression between two arrays
    in function of a variance window.

    cont, up, down = outlier(x,y,t)

    Parameters
    ----------
    x        : metrics array.
    y        : DMOS array.
    t        : logistic regression array.

    Return
    ----------
    cont        : outliers ratio value.
    up          : window up array.
    down        : window down array.
    """
    
    window = 59
    std = np.array([])
    for i in range(0, len(t)):
        s = np.std(t[i:(i+window)])
        std = np.insert(std, 0, s)
    up = t  + 3*(std[::-1])+10
    down = t  - 3*(std[::-1])-10

    cont = 0
    
    x_list = x.tolist()
    x_sort = np.sort(x)
    
    for j in range(0, len(x)):
        x_index = x_list.index(x_sort[j])
        if (down[j] > y[x_index]) or (up[j] < y[x_index]):
            cont = cont+1
    
    return cont, up, down 

def spearman(x,y):
    
    """Computes the Spearman correlation coefficient between two arrays, metrics
    array and difference mean score (DMOS) array.

    r = spearman(x,y)

    Parameters
    ----------
    x        : metrics array.
    y        : DMOS array.

    Return
    ----------
    r        : spearman value.
    """    
    r, d = stats.spearmanr(x,y)
    return r

def pearson(x,y):

    """Computes the Pearson correlation coefficient between two arrays, metrics
    array and difference mean score (DMOS) array.

    r = pearson(x,y)

    Parameters
    ----------
    x        : metrics array.
    y        : DMOS array.

    Return
    ----------
    r        : pearson value.
    """    
    r, d =stats.pearsonr(x,y)
    return r

def levenberg(x, y, metrica):

    """Fit a regression logistic between two arrays, metrics array and
    difference mean score (DMOS) array.

    t, p = levenberg(x,y, metric label)

    Parameters
    ------------
    x          : metrics array.
    y          : DMOS array.
    metric     : metric label value.
    

    Metric label
    ------------
    1          : PSNR.
    2          : MSE.
    3          : MSIM.
    4          : UQI.
    5          : SNR.
    6          : PBVIF.
    7          : NQM.
    8          : RMSE.


    Return
    ------------
    t        : logistic regression array.
    plsq[0]  : parameteres of logistic function array.
    """ 

    def residuals(p, y, x):

        """Calcules the residual error.

        err = residuals(p,y,x)

        Parameters
        ------------
        p          : initial conditions.
        y          : truth estimating.
        x          : worth estimating.

        Return
        ------------
        error          : residual error array.
        """ 
        b1, b2, b3, b4, b5 = p
        err = y - (b1*(0.5 - 1/(1+np.exp(b2*(x -b3)))) + b4*x +b5)
        return err


    
    def peval(x, p):

        """Calcules the estimated function.

        err = peval(p,y,x)

        Parameters
        ------------
        p          : initial conditions.
        y          : truth estimating.
        x          : worth estimating.

        Return
        ------------
        t        : logistic regression array.
        """
        return p[0]*(0.5 - 1/(1+np.exp(p[1]*(x -p[2])))) + p[3]*x +p[4]
    

    
    if (metrica == 1):
        p0 = [-60, 1.3, 130, 0.4, 70]
    if (metrica == 2):
        p0 = [8, 0.35, 8, 0.25, 10]
    if (metrica == 3):
        p0 = [-5, 1.3, 1, 0.28, 13]
    if (metrica == 4):
        p0 = [-6, 2.6, 1, 0.4, 3]
    if (metrica == 5):
        p0 = [-20, 0.1, 20, -0.05, 17]
    if (metrica == 6):
        p0 = [-6, 2.6, 1, 0.4, 3]
    if (metrica == 7):
        p0 = [-32, 0.17, 19, -0.05, 19]
    if (metrica == 8):
        p0 = [-21, 0.5, 10, -0.25, 19]

    
    from scipy.optimize import leastsq
    plsq = leastsq(residuals, p0, args=(y, x))

    
    t = peval(np.sort(x), plsq[0])
    
    
    return t, plsq[0]

def regressaoLinear(x,y):
    
    """Fit a regression linear between two arrays, metrics array and
    difference mean score (DMOS) array.

    a, b = regressaoLinear(x,y)

    Parameters
    -----------
    x        : metrics array.
    y        : DMOS array.


    Return
    -----------
    a        : angular coefficient value.
    b        : linear coefficient value.
    """ 
    
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    n = len(x)
    
    
    a = (np.sum(x*y) - (mean_y*mean_x*n))/ (np.sum(np.square(x))- n*(mean_x**2))
    b = mean_y - a*mean_x
    

    return a, b

