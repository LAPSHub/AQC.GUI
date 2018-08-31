# graficos.py
# Plot 
#
# Luana Goncalves, Leonardo de Brito
# 02.jun.2017

import matplotlib.pyplot as plt
import numpy as np

def grafico_levemberg(x, t):

    plt.plot(np.sort(x), t,'r',label='Logistica')
    plt.title('Otimizacao')
    plt.ylabel('DMOS')
    legend = plt.legend(loc='upper right', shadow=False)
    legend.get_frame()

def grafico_linear(a, b, x):

    plt.plot(x, a*x+b,'b',label='Linear')
    plt.title('Otimizacao')
    plt.ylabel('DMOS')
    legend = plt.legend(loc='upper right', shadow=False)
    legend.get_frame()
    
def grafico(x, y, down, up):
    plt.plot(np.sort(x), down, '--b')
    plt.plot(np.sort(x), up, '--b')
    plt.plot(x, y, 'go')
    plt.grid()
