# metricas.py
# Object scores file
#
# Luana Goncalves, Leonardo Brito
# 02.nov.2017

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import biblioteca

"""
Creates a file with objective evaluation metrics using  MSE, PSNR, SNR, SSIM,
UQI, PBVIF, MSSIM and NQM metrics.
"""


def metricas (orig, test, now, arquivo):
    """
    Creates a file with objective evaluation metrics using  MSE, PSNR, SNR,
    SSIM, UQI, PBVIF, MSSIM and NQM metrics.

    metricas(orig,test, now, arquivo)

    Parameters
    ----------
    orig      : original images directory.
    test      : test images directory.
    arquivo   : subjective assessment file -format: original image + ' ' + test image + ' ' + subjective scores.
    now       : date and time from now = datetime.now()- format: now = '  ' +now[0:13] + '-' +now[14:16]+'-'+now[17:19]

    """
    
    subjectscores = open(arquivo, 'r')
    ct = subjectscores.read()
    aa = ct.split('\n')                
    cc = aa[:-1]                                       
    subjectscores.close()
    

    
    for i in xrange(0, len(cc)):

        a = aa[i].split()
        dirOrig = orig
        imag = a[0]
        e =  dirOrig+'/'+imag
        ref = scipy.misc.imread(e, mode='L')
        
        dirTest = test
        imag = a[1]
        ee = dirTest + '/'+imag
        teste = scipy.misc.imread(ee, mode='L')
       
        psrn = biblioteca.psnr(ref, teste)
        
        mse  = biblioteca.mse(ref, teste)

        uqi = biblioteca.uqi(ref, teste)

        snr = biblioteca.snr(ref, teste)

        pbvif = biblioteca.pbvif(ref, teste)

        nqm = biblioteca.nqm(ref, teste)

        rmse = biblioteca.rmse(ref, teste)

        ssim = biblioteca.msim(ref, teste)
        
        f = open('metricas' + str(now) + '.txt','a')
        f.write(imag + ';' + str(psrn) + ';' + str(mse) + ';' + str(ssim) + ';' + str(uqi) + ';' + str(snr) + ';' + str(pbvif) + ';' + str(nqm)+ ';' + str(rmse) + '\n' )
        f.close()
      

