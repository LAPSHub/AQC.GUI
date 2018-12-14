# graficos.py
# Plot 
#
# Luana Goncalves, Leonardo de Brito
# 02.jun.2017

import matplotlib.pyplot as plt
import numpy as np

'''
Plot the DMOS versus the metric, metric versus logistic regression,
metric versus linear regression and variance windows.'''

def grafico_levenberg(x, t):
    
    """ Plots the DMOS versus  logistic regression.

    grafico_levenverg(x,t)

    Parameters
    ------------
    x          : metrics array.
    t          : logistic regression array.    
    
    """
    plt.plot(np.sort(x), t,'r',label='Logistica')
    plt.title('Otimizacao')
    plt.ylabel('DMOS')
    legend = plt.legend(loc='upper right', shadow=False)
    legend.get_frame()

def grafico_linear(a, b, x):

    """ Plots the DMOS versus  linear regression.

    grafico_linear(a,b,x)

    Parameters
    ------------
    x          : metrics array.
    a          : angular coefficient value.
    b          : linear coefficient value.
    """
    plt.plot(x, a*x+b,'b',label='Linear')
    plt.title('Otimizacao')
    plt.ylabel('DMOS')
    legend = plt.legend(loc='upper right', shadow=False)
    legend.get_frame()
    
def grafico(x, y, down, up):
    
    """ Plots the DMOS versus metric and variance windows.

    grafico(x,y,down,up)

    Parameters
    ------------
    x          : metrics array.
    y          : DMOS array
    down       : window down array.
    up         : window up array.
    """
    plt.plot(np.sort(x), down, '--b')
    plt.plot(np.sort(x), up, '--b')
    plt.plot(x, y, 'go')
    plt.grid()
