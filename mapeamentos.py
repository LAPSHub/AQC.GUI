# mapeamentos.py
# Mapping Library
#
# Luana Goncalves
# 22.fev.2018

import random
import numpy as np
from math import sqrt
from scipy import misc
import random
import matplotlib.pyplot as plt
from numpy import arange, sin, pi, random, array
from  scipy import stats

def reject_outliers(data):
    data1 = data.astype(np.float)
    m = 2
    u = np.median(data1)
    s = np.std(data1)
    filtered = [e for e in data1 if (u - 1 * s < e < u + 1 * s)]
    return filtered

def anova(x,y):
    
    F, p = stats.f_oneway(x,y)
        
    return F


def levenberg(x, y, metrica):
    
    # ==== Calculo do erro residual =====
    def residuals(p, y, x):
        b1, b2, b3, b4, b5 = p
        err = y - (b1*(0.5 - 1/(1+np.exp(b2*(x -b3)))) + b4*x +b5)
        return err
    # ===================================

    # ==== Estimacao dos parametros =====
    def peval(x, p):
        return p[0]*(0.5 - 1/(1+np.exp(p[1]*(x -p[2])))) + p[3]*x +p[4]
    # ===================================

    # Inicializacao dos valores iniciais
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
    # ===================================

    # =========== Otimizacao ============
    from scipy.optimize import leastsq
    plsq = leastsq(residuals, p0, args=(y, x))
    # ===================================
    
    t = peval(np.sort(x), plsq[0])
    
    
    return t, plsq[0]

def regressaoLinear(x,y):
    
    # ==== Caracteristicas dos sinais ====
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    n = len(x)
    # ====================================


    # ===== Estimacao dos parametros =====
    a = (np.sum(x*y) - (mean_y*mean_x*n))/ (np.sum(np.square(x))- n*(mean_x**2))
    b = mean_y - a*mean_x
    # ====================================

    return a, b

def outlier(x, y, p, t):
    # ==== Caracteristicas da janela ====
    window = 59
    std = np.array([])
    for i in range(0, len(t)):
        s = np.std(t[i:(i+window)])
        std = np.insert(std, 0, s)
    up = t  + 3*(std[::-1])+10
    donw = t  - 3*(std[::-1])-10

    cont = 0
    # ===================================
    x_list = x.tolist()
    x_sort = np.sort(x)
    # ======= Contador de Outliers ======
    for j in range(0, len(x)):
        x_index = x_list.index(x_sort[j])
        if (donw[j] > y[x_index]) or (up[j] < y[x_index]):
            cont = cont+1
    # ===================================        
    return cont, up, donw 

def spearman(x,y):
    
    r, d = stats.spearmanr(x,y)
    return r

def pearson(x,y):

    r, d =stats.pearsonr(x,y)
    return r

