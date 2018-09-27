# main_aqv.py
# Main File 
#
# Luana Goncalves, Leonardo de Brito
# 02.jun.2017

import numpy as np
import metricas as p
import avaliacao
import graficos
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
now = str(now)
now = now[0:19]
now = ' ' +now[0:13] + '-' +now[14:16]+'-'+now[17:19]

def aqv(orig, teste, arquivo, psnr, mse, msim, uqi, snr, pbvif, nqm, rmse, lin, log, pearson, spearman, out, anova, plot):
    f = open('Avaliacao de Desempenho de Metricas de Qualidade Visual' + str(now)+ '.txt', 'a')
    a = p.metricas(orig, teste, now, arquivo)
    if psnr:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(1, arquivo, now)
        f.write('PSNR' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')  
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('PSNR')
            plt.grid()
            plt.show()
        
    if mse:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(2, arquivo, now)
        f.write('MSE' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('MSE')
            plt.grid()
            plt.show()
        
    if msim:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(3, arquivo, now)
        f.write('MSIM' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('MSIM')
            plt.grid()
            plt.show()
        
    if uqi:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(4, arquivo, now)
        f.write('UQI' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('UQI')
            plt.grid()
            plt.show()        
    if snr:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(5, arquivo, now)
        f.write('SNR' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('SNR')
            plt.grid()
            plt.show()
        
    if pbvif:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(6, arquivo,now)
        f.write('PVIF' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('PBVIF')
            plt.grid()
            plt.show()
        
    if nqm:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(7, arquivo,now)
        f.write('NQM' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('NQM')
            plt.grid()
            plt.show()
        
    if rmse:
        razao_outliers, coeficiente_spearman, coeficiente_pearson, coeficiente_anova_F, logistica, linear, a, b, t, x, y, down,up = avaliacao.avaliacao(8, arquivo,now)
        f.write('RMSE' + '\n')
        if out:
            f.write('Razao de outliers: ' + str(razao_outliers)+ '\n')
        if spearman:
            f.write('Coeficiente de Spearman: ' + str(coeficiente_spearman) + '\n')
        if pearson:
            f.write('Coeficiente de Pearson: ' + str(coeficiente_pearson) + '\n')
        if anova:
            f.write('F-score da Analise de Variancia: ' + str(coeficiente_anova_F) + '\n')
        if plot:
            graficos.grafico(x, y, down, up)
            if log:
                f.write('Funcao Losgistica:' + logistica + '\n')
                graficos.grafico_levemberg(x, t)
            if lin:
                graficos.grafico_linear(a, b, x)
                f.write('Regressao Linear:' + linear + '\n')
            plt.xlabel('RMSE')
            plt.grid()
            plt.show()
        
    f.close()
