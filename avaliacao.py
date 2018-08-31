# avaliacao.py
# Metric evaluation
#
# Luana Goncalves, Leonardo Brito
# 02.jun.2017

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mapeamentos


def avaliacao(metrica, arq, now):
     
    # Abrir arquivo de avaliacoes objetivas
    objectscores = open('metricas' + str(now) + '.txt', 'r')
    ct = objectscores.read()
    a = ct.split('\n')                 
    c = a[:-1]
    objectscores.close()    
    # ===================================

    # Abrir aquivo de avaliacoes subjetivas
    subjectscores = open(arq, 'r')                          
    ct1 = subjectscores.read()          
    aa = ct1.split('\n')              
    cc = aa[:-1]        
    subjectscores.close()
    # ===================================

    # === Inicializacao das variaveis ===
    x = np.array([])
    y = np.array([])
    x0 = np.array([])
    y0 = np.array([])
    l = np.array([])
    total_pontos = 0
    ref = ''
    ab = ''
    # ===================================
    
    for i in xrange(0, len(cc)):
    # ======= METRICA X DMOS ========
        w = aa[i].split()
        for line in cc:
            if w[0] in line:
                cond  = line.split()
                if cond[1]==w[0]:
                    w1=cond
        w = aa[i].split()
        ww = w[2::]
        ww1 = w1[2::]

        total_pontos = len(ww) + total_pontos

        b = c[i].split(';')                         
        psnr = b[metrica]

        h = np.asarray([e for e in ww if (e != str(''))])
        h1 = np.asarray([e for e in ww1 if (e != str(''))])
        

        v2 = np.mean(h1.astype(np.float))
        v1 = np.mean(h.astype(np.float))
        v=v2-v1
        if (psnr !=str('inf')):
            x0 = np.insert(x0,0,psnr)         
            y0 = np.insert(y0,0,v)           
        r = float(psnr)                     
    u = np.median(y0)
    s = np.std(y0)
    
    for j in xrange(0, len(y0)):
        if (u - 2 * s < y0[j] < u + 2 * s):
            y = np.insert(y,0,y0[j])
            x = np.insert(x,0,x0[j])
            
    # ===================================


    # =========== Regressoes  ============
    a, b = mapeamentos.regressaoLinear(x,y)     
    t, p = mapeamentos.levenberg(x,y,metrica)   
    linear = '(' + str(a) + ')*x + (' + str(b) + ')'
    logistica = '(' + str(p[0])+ ') * (' + str(0.5) + '-' + str(1) + '/(exp(' + str(p[1]) + '*(x-(' + str(p[2])+ '))))) + (' + str(p[3])+ ') * x+(' + str(p[4]) + ')'
    # ====================================

    
    # ===== Avaliacao das metricas  ======
    x_sort = np.sort(x)
    out_total, up, down = mapeamentos.outlier(x,y,v, t)
    razao_outliers = float(out_total)/len(x)

    yd = p[0]*(0.5 - 1/(1+np.exp(p[1]*(x -p[2])))) + p[3]*x +p[4]
    coeficiente_spearman = mapeamentos.spearman(yd,y)    
    coeficiente_pearson = mapeamentos.pearson(yd,y)      
    coeficiente_anova_F = mapeamentos.anova(yd,y)
    # ====================================
    
    return razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up
