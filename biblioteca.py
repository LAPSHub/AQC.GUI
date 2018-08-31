# biblioteca.py
# Object scores file
#
# Luana Goncalves, Leonardo Brito
# 02.nov.2017

import scipy.ndimage
import matplotlib.pyplot as plt
from numpy.ma.core import exp
from scipy.constants.constants import pi
from scipy import signal,misc, ndimage
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.ndimage.filters import gaussian_filter as __gaussian_filter
from scipy.ndimage.filters import convolve as __convolve
from scipy.ndimage.filters import correlate as __correlate
from scipy.fftpack import fftshift as __fftshift


def mse(reference, query):
    
    (ref, que) = (reference.astype('double'), query.astype('double'))
    diff = ref - que
    square = (diff ** 2)
    mean = square.mean()
    return mean


def rmse(reference, query):
    msev = mse(reference, query)
    return np.sqrt(msev)


def psnr(reference, query, normal=(2**int(24) - 1)):
    
    normalization = float(normal)
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(normalization * normalization / msev)
    else:
        value = float("inf")
    return value


def snr(reference, query):
    
    signal_value = (reference.astype('double') ** 2).mean()
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(signal_value / msev)
    else:
        value = float("inf")
    return value


def ssim(reference, query):


    def __get_kernels():
        k1, k2, l = (0.01, 0.03, 255.0)
        kern1, kern2 = map(lambda x: (x * l) ** 2, (k1, k2))
        return kern1, kern2

    def __get_mus(i1, i2):
        mu1, mu2 = map(lambda x: __gaussian_filter(x, 1.5), (i1, i2))
        m1m1, m2m2, m1m2 = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
        return m1m1, m2m2, m1m2

    def __get_sigmas(i1, i2, delta1, delta2, delta12):
        f1 = __gaussian_filter(i1 * i1, 1.5) - delta1
        f2 = __gaussian_filter(i2 * i2, 1.5) - delta2
        f12 = __gaussian_filter(i1 * i2, 1.5) - delta12
        return f1, f2, f12

    def __get_positive_ssimap(C1, C2, m1m2, mu11, mu22, s12, s1s1, s2s2):
        num = (2 * m1m2 + C1) * (2 * s12 + C2)
        den = (mu11 + mu22 + C1) * (s1s1 + s2s2 + C2)
        return num / den

    def __get_negative_ssimap(C1, C2, m1m2, m11, m22, s12, s1s1, s2s2):
        (num1, num2) = (2.0 * m1m2 + C1, 2.0 * s12 + C2)
        (den1, den2) = (m11 + m22 + C1, s1s1 + s2s2 + C2)
        ssim_map = np.ones(img1.shape)
        indx = (den1 * den2 > 0)
        ssim_map[indx] = (num1[indx] * num2[indx]) / (den1[indx] * den2[indx])
        indx = np.bitwise_and(den1 != 0, den2 == 0)
        ssim_map[indx] = num1[indx] / den1[indx]
        return ssim_map

    (img1, img2) = (reference.astype('double'), query.astype('double'))
    (m1m1, m2m2, m1m2) = __get_mus(img1, img2)
    (s1, s2, s12) = __get_sigmas(img1, img2, m1m1, m2m2, m1m2)
    (C1, C2) = __get_kernels()
    if C1 > 0 and C2 > 0:
        ssim_map = __get_positive_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
    else:
        ssim_map = __get_negative_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
    ssim_value = ssim_map.mean()
    return ssim_value


def uqi(reference, query):

    def __conv(x):
        window = np.ones((BLOCK_SIZE, BLOCK_SIZE))
        if len(x.shape) < 3:
            return __convolve(x, window)
        else:
            channels = x.shape[2]
            f = [__convolve(x[:, :, c], window) for c in xrange(channels)]
            return np.array(f)

    def __get_filtered(im1, im2, BLOCK_SIZE):
        (im1im1, im2im2, im1im2) = (im1 * im1, im2 * im2, im1 * im2)
        (b1, b2, b3, b4, b5) = map(__conv, (im1, im2, im1im1, im2im2, im1im2))
        (b6, b7) = (b1 * b2, b1 * b1 + b2 * b2)
        return (b1, b2, b3, b4, b5, b6, b7)

    def __get_quality_map(b1, b2, b3, b4, b5, b6, b7, BLOCK_SIZE):
        N = BLOCK_SIZE * BLOCK_SIZE
        numerator = 4.0 * (N * b5 - b6) * b6
        denominator1 = N * (b3 + b4) - b7
        denominator = denominator1 * b7
        index = np.bitwise_and(denominator1 == 0, b7 != 0)
        quality_map = np.ones(denominator.shape)
        quality_map[index] = 2.0 * b6[index] / b7[index]
        index = (denominator != 0)
        quality_map[index] = numerator[index] / denominator[index]
        return quality_map[index]

    BLOCK_SIZE = 8
    (img1, img2) = (reference.astype('double'), query.astype('double'))
    (b1, b2, b3, b4, b5, b6, b7) = __get_filtered(img1, img2, BLOCK_SIZE)
    quality_map = __get_quality_map(b1, b2, b3, b4, b5, b6, b7, BLOCK_SIZE)
    value = quality_map.mean()
    return value


def pbvif(reference, query):

    def __get_sigma(win, ref, dist, mu1_sq, mu2_sq, mu1_mu2):
        sigma1_sq = __filter2(win, ref * ref) - mu1_sq
        sigma2_sq = __filter2(win, dist * dist) - mu2_sq
        sigma12 = __filter2(win, ref * dist) - mu1_mu2
        (sigma1_sq[sigma1_sq < 0], sigma2_sq[sigma2_sq < 0]) = (0.0, 0.0)
        return (sigma2_sq, sigma12, sigma1_sq)

    def __get_normalized(s1s1, s2s2, s1s2):
        g = s1s2 / (s1s1 + 1e-10)
        sv_sq = s2s2 - g * s1s2
        g[s1s1 < 1e-10] = 0
        sv_sq[s1s1 < 1e-10] = s2s2[s1s1 < 1e-10]
        s1s1[s1s1 < 1e-10] = 0
        g[s2s2 < 1e-10] = 0
        sv_sq[s2s2 < 1e-10] = 0
        sv_sq[g < 0] = s2s2[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10
        return (g, sv_sq)

    def __get_num(s1s1, sv_sq, sigma_nsq, g):
        normg = (g ** 2) * s1s1 / (sv_sq + sigma_nsq)
        snr = np.log10(1.0 + normg).sum()
        return snr

    def __get_den(s1s1, sigma_nsq):
        snr = np.log10(1.0 + s1s1 / sigma_nsq)
        return snr.sum()

    def __get_num_den_level(ref, dist, scale):
        sig = 2.0
        N = (2.0 ** (4 - scale + 1.0)) + 1.0
        win = __get_gaussian_kernel(N, N / 5.0)
        if scale > 1:
            ref = __filter2(win, ref)
            dist = __filter2(win, dist)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
        (mu1, mu2) = (__filter2(win, ref), __filter2(win, dist))
        (m1m1, m2m2, m1m2) = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
        (s2s2, s1s2, s1s1) = __get_sigma(win, ref, dist, m1m1, m2m2, m1m2)
        (g, svsv) = __get_normalized(s1s1, s2s2, s1s2)
        (num, den) = (__get_num(s1s1, svsv, sig, g), __get_den(s1s1, sig))
        return (num, den)

    (ref, dist) = (reference.astype('double'), query.astype('double'))
    zipped = map(lambda x: __get_num_den_level(ref, dist, x), xrange(1, 5))
    (nums, dens) = zip(*zipped)
    value = sum(nums) / sum(dens)
    return value


def mssim(reference, query):

    def __get_filt_kern():
        n = [131, -199, -101, 962, 932, 962, -101, -199, 131]
        d = [3463, 8344, 913, 2549, 1093, 2549, 913, 8344, 3463]
        num = np.matrix(n).T
        den = np.matrix(d).T
        lod = num.astype('double') / den.astype('double')
        lpf = np.dot(lod, lod.T)
        return lpf / lpf.sum()

    def __get_ssim(img1, img2, K):
        comp_ssim = __ssim_modified(img1, img2, K)[1]
        return (comp_ssim[1], comp_ssim[2])

    def __get_MVR(img1, img2, K, nlevs):
        (ssim_v, ssim_r) = (np.zeros((nlevs, 1)), np.zeros((nlevs, 1)))
        (ssim_v[0], ssim_r[0]) = __get_ssim(img1, img2, K)
        filt_kern = __get_filt_kern()
        for s in xrange(nlevs - 1):
            (img1, img2) = map(lambda x: __filter2(filt_kern, x), (img1, img2))
            (img1, img2) = (img1[::2, ::2], img2[::2, ::2])
            comp_ssim = __ssim_modified(img1, img2, K)[1]
            ssim_m = comp_ssim[0]
            ssim_v[s + 1] = comp_ssim[1]
            ssim_r[s + 1] = comp_ssim[2]
        return (ssim_m, ssim_v, ssim_r)

    def __calc_mssim_mvr(img1, img2):
        (K, weights) = ((0.01, 0.03), (0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
        (alpha, beta, lvl) = (0.1333, np.matrix(weights).T, len(weights))
        (ssim_m, ssim_v, ssim_r) = __get_MVR(img1, img2, K, lvl)
        m = ssim_m ** alpha
        v = (ssim_v ** beta).prod()
        r = (ssim_r ** beta).prod()
        return (m, v, r)

    (ref, quer) = (reference.astype('double'), query.astype('double'))
    ssim_mvr = np.matrix(__calc_mssim_mvr(ref, quer))
    value = ssim_mvr.prod()
    return value


def __filter2(B, X, shape='nearest'):
    B2 = np.rot90(np.rot90(B))
    if len(X.shape) < 3:
        return __correlate(X, B2, mode=shape)
    else:
        channels = X.shape[2]
        f = [__correlate(X[:, :, c], B2, mode=shape) for c in xrange(channels)]
        return np.array(f)


def __get_gaussian_kernel(N=15, sigma=1.5):
    (H, W) = ((N - 1) / 2, (N - 1) / 2)
    std = sigma
    (y, x) = np.mgrid[-H:H + 1, -W:W + 1]
    arg = -(x * x + y * y) / (2.0 * std * std)
    h = np.exp(arg)
    index = h < np.finfo(float).eps * h.max(0)
    h[index] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def __ssim_modified(reference, query, K):
    def __get_kern(K):
        L = 255
        kern = map(lambda x: (x * L) ** 2, K)
        return (kern[0], kern[1])

    def __get_filtering_window():
        window = __get_gaussian_kernel(11, 1.5)
        return window / window.sum()

    def __get_mus(img1, img2, window):
        (mu1, mu2) = map(lambda x: __filter2(window, x), (img1, img2))
        (m1m1, m2m2, m1m2) = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
        return (mu1, mu2, m1m1, m2m2, m1m2)

    def __get_sigmas(img1, img2, window, m1m1, m2m2, m1m2):
        s1s1 = __filter2(window, img1 * img1) - m1m1
        s2s2 = __filter2(window, img2 * img2) - m2m2
        s12 = __filter2(window, img1 * img2) - m1m2
        (s1, s2) = map(np.sqrt, (np.abs(s1s1), np.abs(s2s2)))
        return (s1s1, s2s2, s1, s2, s12)

    def __MVR_pos_kern(m, kern, s, s_square):
        (m11, m22, m12) = m
        (k1, k2) = kern
        (s1, s2) = s
        (s1s1, s2s2, s12) = s_square
        M = (2.0 * m12 + k1) / (m11 + m22 + k1)
        V = (2.0 * s1 * s2 + k2) / (s1s1 + s2s2 + k2)
        R = (s12 + k2 / 2.0) / (s1 * s2 + k2 / 2.0)
        return (M, V, R)

    def __MVR_neg_kern(m, s, s_square):
        def __calcM(mu1, m11, m22, m12):
            ssim_ln = 2.0 * m12
            ssim_ld = m11 + m22
            index_l = ssim_ld > 0
            M = np.ones(mu1.shape)
            M[index_l] = ssim_ln[index_l] / ssim_ld[index_l]
            return M

        def __calcV(mu1, s1, s2, s11, s22):
            ssim_cn = 2.0 * s1 * s2
            ssim_cd = s11 + s22
            V = np.ones(mu1.shape)
            index_c = ssim_cd > 0
            V[index_c] = ssim_cn[index_c] / ssim_cd[index_c]
            return V

        def __calcR(mu1, s1, s2, s12):
            (ssim_sn, ssim_sd) = (s12, s1 * s2)
            R = np.ones(mu1.shape)
            (index1, index2) = (s1 > 0, s2 > 0)
            index_s1 = index1 * index2 > 0
            R[index_s1] = ssim_sn[index_s1] / ssim_sd[index_s1]
            index_s2 = index1 * np.logical_not(index2) > 0
            R[index_s2] = 0.0
            return R

        (mu1, mu2, m11, m22, m12) = m
        (s1, s2) = s
        (s11, s22, s12) = s_square
        M = __calcM(mu1, m11, m22, m12)
        V = __calcV(mu1, s1, s2, s11, s22)
        R = __calcR(mu1, s1, s2, s12)
        return (M, V, R)

    def __get_composition_vector(img1, img2):
        filt = __get_filtering_window()
        (mu1, mu2, m11, m22, m12) = __get_mus(img1, img2, filt)
        (s11, s22, s1, s2, s12) = __get_sigmas(img1, img2, filt, m11, m22, m12)
        (kern1, kern2) = __get_kern(K)
        if kern1 > 0 and kern2 > 0:
            (m, kern, s) = ((m11, m22, m12), (kern1, kern2), (s1, s2))
            s_square = (s11, s22, s12)
            (M, V, R) = __MVR_pos_kern(m, kern, s, s_square)
        else:
            (m, s) = ((mu1, mu2, m11, m22, m12), (s1, s2))
            s_square = (s11, s22, s12)
            (M, V, R) = __MVR_neg_kern(m, s, s_square)
        return (M, V, R)

    def __get_ssim_map(M, V, R):
        ssim_map = M * V * R
        return ssim_map

    def __get_ssim_from_composition_vector(M, V, R):
        ssim_map = __get_ssim_map(M, V, R)
        ssim = ssim_map.mean()
        return ssim

    (img1, img2) = reference.astype('double'), query.astype('double')
    (M, V, R) = __get_composition_vector(img1, img2)
    composite_mean_vector = (M.mean(), V.mean(), R.mean())
    ssim = __get_ssim_from_composition_vector(M, V, R)
    return (ssim, composite_mean_vector)


def __convert_to_luminance(x):
    return np.dot(x[..., :3], [0.299, 0.587, 0.144]).astype('double')


def nqm(reference, query):


    def __ctf(f_r):
        """ Bandpass Contrast Threshold Function for RGB"""
        (gamma, alpha) = (0.0192 + 0.114 * f_r, (0.114 * f_r) ** 1.1)
        beta = np.exp(-alpha)
        num = 520.0 * gamma * beta
        return 1.0 / num

    def _get_masked(c, ci, a, ai, i):
        (H, W) = c.shape
        (c, ci, ct) = (c.flatten(1), ci.flatten(1), __ctf(i))
        ci[abs(ci) > 1.0] = 1.0
        T = ct * (0.86 * ((c / ct) - 1.0) + 0.3)
        (ai, a, a1) = (ai.flatten(1), a.flatten(1), (abs(ci - c) - T) < 0.0)
        ai[a1] = a[a1]
        return ai.reshape(H, W)

    def __get_thresh(x, T, z, trans=True):
        (H, W) = x.shape
        if trans:
            (x, z) = (x.flatten(1).T, z.flatten())
        else:
            (x, z) = (x.flatten(1), z.flatten(1))
        z[abs(x) < T] = 0.0
        return z.reshape(H, W)

    def __decompose_cos_log_filter(w1, w2, phase=np.pi):
        return 0.5 * (1 + np.cos(np.pi * np.log2(w1 + w2) - phase))

    def __get_w(r):
        w = [(r + 2) * ((r + 2 <= 4) * (r + 2 >= 1))]
        w += [r * ((r <= 4) * (r >= 1))]
        w += [r * ((r >= 2) * (r <= 8))]
        w += [r * ((r >= 4) * (r <= 16))]
        w += [r * ((r >= 8) * (r <= 32))]
        w += [r * ((r >= 16) * (r <= 64))]
        return w

    def __get_u(r):
        u = [4 * (np.logical_not((r + 2 <= 4) * (r + 2 >= 1)))]
        u += [4 * (np.logical_not((r <= 4) * (r >= 1)))]
        u += [0.5 * (np.logical_not((r >= 2) * (r <= 8)))]
        u += [4 * (np.logical_not((r >= 4) * (r <= 16)))]
        u += [0.5 * (np.logical_not((r >= 8) * (r <= 32)))]
        u += [4 * (np.logical_not((r >= 16) * (r <= 64)))]
        return u

    def __get_G(r):
        (w, u) = (__get_w(r), __get_u(r))
        phase = [np.pi, np.pi, 0.0, np.pi, 0.0, np.pi]
        dclf = __decompose_cos_log_filter
        return [dclf(w[i], u[i], phase[i]) for i in xrange(len(phase))]

    def __compute_fft_plane_shifted(ref, query):
        (x, y) = ref.shape
        (xplane, yplane) = np.mgrid[-y / 2:y / 2, -x / 2:x / 2]
        plane = (xplane + 1.0j * yplane)
        r = abs(plane)
        G = __get_G(r)
        Gshifted = map(__fftshift, G)
        return [Gs.T for Gs in Gshifted]

    def __get_c(a, l_0):
        c = [a[0] / l_0]
        c += [a[1] / (l_0 + a[0])]
        c += [a[2] / (l_0 + a[0] + a[1])]
        c += [a[3] / (l_0 + a[0] + a[1] + a[2])]
        c += [a[4] / (l_0 + a[0] + a[1] + a[2] + a[3])]
        return c

    def __get_ci(ai, li_0):
        ci = [ai[0] / (li_0)]
        ci += [ai[1] / (li_0 + ai[0])]
        ci += [ai[2] / (li_0 + ai[0] + ai[1])]
        ci += [ai[3] / (li_0 + ai[0] + ai[1] + ai[2])]
        ci += [ai[4] / (li_0 + ai[0] + ai[1] + ai[2] + ai[3])]
        return ci

    def __compute_contrast_images(a, ai, l, li):
        ci = __get_ci(ai, li)
        c = __get_c(a, l)
        return (c, ci)

    def __get_detection_thresholds():
        viewing_angle = (1.0 / 3.5) * (180.0 / np.pi)
        rotations = [2.0, 4.0, 8.0, 16.0, 32.0]
        return map(lambda x: __ctf(x / viewing_angle), rotations)

    def __get_account_for_supra_threshold_effects(c, ci, a, ai):
        r = xrange(len(a))
        return [_get_masked(c[i], ci[i], a[i], ai[i], i + 1) for i in r]

    def __apply_detection_thresholds(c, ci, d, a, ai):
        A = [__get_thresh(c[i], d[i], a[i], False) for i in xrange(len(a))]
        AI = [__get_thresh(ci[i], d[i], ai[i], True) for i in xrange(len(a))]
        return (A, AI)

    def __reconstruct_images(A, AI):
        return map(lambda x: np.add.reduce(x), (A, AI))

    def __compute_quality(imref, imquery):
        return snr(imref, imquery)

    def __get_ref_basis(ref_fft, query_fft, GS):
        (L_0, LI_0) = map(lambda x: GS[0] * x, (ref_fft, query_fft))
        (l_0, li_0) = map(lambda x: np.real(np.fft.ifft2(x)), (L_0, LI_0))
        return (l_0, li_0)

    def __compute_inverse_convolution(convolved_fft, GS):
        convolved = [GS[i] * convolved_fft for i in xrange(1, len(GS))]
        return map(lambda x: np.real(np.fft.ifft2(x)), convolved)

    def __correlate_in_fourier_domain(ref, query):
        (ref_fft, query_fft) = map(lambda x: np.fft.fft2(x), (ref, query))
        GS = __compute_fft_plane_shifted(ref, query)
        (l_0, li_0) = __get_ref_basis(ref_fft, query_fft, GS)
        a = __compute_inverse_convolution(ref_fft, GS)
        ai = __compute_inverse_convolution(query_fft, GS)
        return (a, ai, l_0, li_0)

    def __get_correlated_images(ref, query):
        (a, ai, l_0, li_0) = __correlate_in_fourier_domain(ref, query)
        (c, ci) = __compute_contrast_images(a, ai, l_0, li_0)
        d = __get_detection_thresholds()
        ai = __get_account_for_supra_threshold_effects(c, ci, a, ai)
        return __apply_detection_thresholds(c, ci, d, a, ai)

    if not len(reference.shape) < 3:
        reference = __convert_to_luminance(reference)
        query = __convert_to_luminance(query)
    (A, AI) = __get_correlated_images(reference, query)
    (y1, y2) = __reconstruct_images(A, AI)
    y = __compute_quality(y1, y2)
    return y


def wsnr(reference, query):


    def __genetate_meshgrid(x, y):
        f = lambda u: u / 2 + 0.5 - 1
        (H, W) = map(f, (x, y))
        return (H, W)

    def __create_complex_planes(x, y):
        (H, W) = __genetate_meshgrid(x, y)
        (xplane, yplane) = np.mgrid[-H:H + 1, -W:W + 1]
        return (xplane, yplane)

    def __get_evaluated_contrast_sensivity(plane):
        w = 0.7
        angle = np.angle(plane)
        return ((1.0 - w) / 2.0) * np.cos(4.0 * angle) + (1.0 + w) / 2.0

    def __get_radial_frequency(x, y):
        (xplane, yplane) = __create_complex_planes(x, y)
        nfreq = 60
        plane = (xplane + 1.0j * yplane) / x * 2.0 * nfreq
        s = __get_evaluated_contrast_sensivity(plane)
        radfreq = abs(plane) / s
        return radfreq

    def __generate_CSF(radfreq):
        a = -((0.114 * radfreq) ** 1.1)
        csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(a)
        f = radfreq < 7.8909
        csf[f] = 0.9809
        return csf

    def __weighted_fft_domain(ref, quer, csf):
        err = ref.astype('double') - quer.astype('double')
        err_wt = __fftshift(np.fft.fft2(err)) * csf
        im = np.fft.fft2(ref)
        return (err, err_wt, im)

    def __get_weighted_error_power(err_wt):
        return (err_wt * np.conj(err_wt)).sum()

    def __get_signal_power(im):
        return (im * np.conj(im)).sum()

    def __get_ratio(mss, mse):
        if mse != 0:
            ratio = 10.0 * np.log10(mss / mse)
        else:
            ratio = float("inf")
        return np.real(ratio)

    if not len(reference.shape) < 3:
        reference = __convert_to_luminance(reference)
        query = __convert_to_luminance(query)
    size = reference.shape
    (x, y) = (size[0], size[1])
    radfreq = __get_radial_frequency(x, y)
    csf = __generate_CSF(radfreq)
    (err, err_wt, im) = __weighted_fft_domain(reference, query, csf)
    mse = __get_weighted_error_power(err_wt)
    mss = __get_signal_power(im)
    ratio = __get_ratio(mss, mse)
    return ratio

def msim(img_mat_1, img_mat_2):

    # Valores default devem ser fornecidos
    std =3
    L = 6*std +1 # comprimento do filtro
    window = signal.gaussian(L, std) # vetor Gaussiano unidimensional
    #x = misc.lena()                  # imagem de entrada

    # kernel 2d (usando fft)
    w = window.reshape((L,1))
    wt = np.transpose(w)
    h = w * wt
    soma = np.sum(h)
    h = h/soma
    
    #print(h.shape)

    #convertere para float
    img_mat_1=img_mat_1.astype(np.float)
    img_mat_2=img_mat_2.astype(np.float)

    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,h)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,h)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,h)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,h)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,h)

    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;

    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    #print ssim_map
    index=np.average(ssim_map)
    return index
