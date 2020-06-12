'''
Assignment 6 [1]_ solved in Python

References
----------
.. [1] http://people.eecs.berkeley.edu/~mlustig/CS/CS_ex.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
import pywt
from phantominator import shepp_logan
from scipy.io import loadmat
from tqdm import trange
from skimage.metrics import normalized_root_mse as compare_nrmse

def fft(x, axes=(-1,)):
    fac = 1/np.sqrt(np.prod([x.shape[a] for a in axes]))
    return fac*np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
        x, axes=axes), axes=axes), axes=axes)

def ifft(x, axes=(-1,)):
    fac = np.sqrt(np.prod([x.shape[a] for a in axes]))
    return fac*np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(
        x, axes=axes), axes=axes), axes=axes)

if __name__ == '__main__':

    disp = True # generate all plots

    # 1) Sparse Signals and Denoising
    # 1a) Sparse Signals

    # Create signal with k nonzero coefficients
    N, k = 128, 5
    x = np.zeros(N)
    x[np.random.choice(N, size=k, replace=False)] = np.arange(1, k+1)/k

    # Add noise
    mu, sig = 0, .05
    n = np.random.normal(mu, sig, size=N)
    y = x + n

    # Plot with various lambda
    lambdas = np.array([.01, .05, .1, .2])
    xhat = y/(1 + lambdas[:, None])
    if disp:
        plt.plot(xhat.T)
        plt.title('Tikhonov regularization (not sparse)')
        plt.show()

    # b) Sparse Signals and the l1 Norm
    xhat = pywt.threshold(y, value=lambdas[:, None], mode='soft')
    if disp:
        lineObjects = plt.plot(xhat.T)
        plt.title('l1 regularization (sparse)')
        plt.legend(iter(lineObjects), ['λ=%g' % l for l in lambdas])
        plt.show()

    # c) Random Frequency Domain Sampling and Aliasing
    X = fft(x)
    Xu = np.zeros_like(X)
    Xu[::4] = X[::4]
    xu = ifft(Xu)*4
    if disp:
        plt.plot(np.abs(xu), label='recon')
        plt.plot(np.abs(x), '--', label='true')
        plt.title('|xu|')
        plt.legend()
        plt.show()

    k = 32
    Xr = np.zeros(N, dtype='complex')
    idx = np.random.choice(N, size=k, replace=False)
    Xr[idx] = X[idx]
    xr = ifft(Xr)*4
    if disp:
        plt.plot(np.abs(xr), label='recon')
        plt.plot(np.abs(x), '--', label='true')
        plt.legend()
        plt.show()

    # d) Reconstruction from Randomly Sampled Frequency Domain Data
    def pocs1d(Y, lambdas, niter=300):
        lambdas = np.array(lambdas)
        X = np.tile(Y.copy()[None, :], (len(lambdas), 1))
        for ii in range(niter):
            xst = pywt.threshold(ifft(X), value=lambdas[:, None], mode='soft')
            X = fft(xst)
            X[:, idx] = Y[idx]
        return X

    # Make a sparse signal in time domain
    Y = np.zeros_like(y, dtype='complex')
    Y[idx] = fft(y)[idx]
    X = pocs1d(Y, lambdas)
    if disp:
        lineObjects = plt.plot(np.abs(ifft(X)).T)
        plt.legend(iter(lineObjects), ['λ=%g' % l for l in lambdas])
        plt.title('IST Recon (random undersampling)')
        plt.show()

    # repeat for regular undersampling
    Y = np.zeros_like(y, dtype='complex')
    Y[::4] = fft(y)[::4]
    X = pocs1d(Y, lambdas)
    if disp:
        lineObjects = plt.plot(np.abs(ifft(X)).T)
        plt.legend(iter(lineObjects), ['λ=%g' % l for l in lambdas])
        plt.title('IST Recon (regular undersampling)')
        plt.show()

    # 2. Sparsity of Medical Images
    #wvlt = 'db1'
    wvlt = 'haar'
    def WVLT(x, wavelet=wvlt, level=1):
        '''Forward wavelet transform.'''
        return pywt.coeffs_to_array(
            pywt.wavedecn(x, wavelet=wavelet, level=level))
    def iWVLT(x, coeff_slices, wavelet=wvlt):
        '''Inverse wavelet transform.'''
        return pywt.waverecn(
            pywt.array_to_coeffs(x, coeff_slices), wavelet)

    N = 256
    im = shepp_logan(N)
    im_W, sl = WVLT(im)
    recon = iWVLT(im_W, sl)

    if disp:
        ax = plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.title('Phantom')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(1, 3, 2)
        plt.imshow(im_W, cmap='gray')
        plt.title('WVLT coeffs')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(1, 3, 3)
        plt.imshow(recon, cmap='gray')
        plt.title('Recon')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    # Use brain data
    data = loadmat('brain.mat')
    # keys: ['__header__', '__version__', '__globals__', 'im', 'pdf_unif', 'mask_unif', 'pdf_vardens', 'mask_vardens']
    brain = data['im']

    im_W, sl = WVLT(brain)
    if disp:
        plt.imshow(np.abs(im_W))
        plt.title('WVLT Tx of brain')
        plt.show()

    fs = np.array([.2, .125, .1, .05, .025])
    m = np.sort(np.abs(im_W).flatten())[::-1]
    idx = np.floor(len(m)*fs).astype(int)
    thresh = m[idx]

    im_W_th = im_W[..., None]*(np.abs(im_W)[..., None] > thresh)
    if disp:
        for ii, f in enumerate(fs):
            ax = plt.subplot(1, len(fs), ii+1)
            plt.imshow(np.abs(im_W_th[..., ii]))
            plt.title('%g%%' % (100*f))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    # Do the recons
    recons = np.empty_like(im_W_th)
    for ii in range(len(fs)):
        recons[..., ii] = iWVLT(im_W_th[..., ii], sl)

    # find dynamic range for difference images
    diffs = np.abs(recons - brain[..., None])
    vmin, vmax = np.min(diffs), np.max(diffs)

    if disp:
        for ii, f in enumerate(fs):
            # Show recon
            ax = plt.subplot(2, len(fs), ii+1)
            plt.imshow(np.abs(recons[..., ii]))
            plt.title('Recon %g%%' % (100*f))
            ax.set_xticks([])
            ax.set_yticks([])

            # Show diff image
            ax = plt.subplot(2, len(fs), ii+1+len(fs))
            plt.imshow(diffs[..., ii], vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    # 3) Compressed Sensing Reconsruction
    # a) Non-Uniform Random Sampling
    mask_unif = data['mask_unif']
    pdf_unif = data['pdf_unif']
    M = fft(brain, axes=(0, 1))
    Mu = M*mask_unif/pdf_unif
    recon = ifft(Mu, axes=(0, 1))

    if disp:
        ax = plt.subplot(2, 2, 1)
        plt.imshow(np.abs(recon))
        plt.title('Recons')
        plt.ylabel('Uniform mask')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 2, 2)
        plt.imshow(np.abs(brain - recon))
        plt.title('diff')
        ax.set_xticks([])
        ax.set_yticks([])

    # Repeat for the variable density mask
    mask_vardens = data['mask_vardens']
    pdf_vardens = data['pdf_vardens']
    M = fft(brain, axes=(0, 1))
    Mv = M*mask_vardens/pdf_vardens
    recon = ifft(Mv, axes=(0, 1))

    if disp:
        ax = plt.subplot(2, 2, 3)
        plt.imshow(np.abs(recon))
        plt.ylabel('Variable density mask')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 2, 4)
        plt.imshow(np.abs(brain - recon))
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    # b) Reconstruction from Random Sampled k-Space Data
    def pocs2d(Y, mask, lamda, mu=.8, niter=20, axes=(0, 1)):
        X = Y.copy()
        for ii in trange(niter):
            x = ifft(X, axes=axes)
            W, sl = WVLT(x)
            Wst = pywt.threshold(W, value=lamda, mode='soft')
            xst = iWVLT(Wst, sl)
            X = fft(xst, axes=axes)
            X[mask] = Y[mask]
            lamda *= mu
        return X

    # These recons need some parameter tuning:
    # Uniform:
    Y = Mu.copy()
    X = pocs2d(Mu, mask_unif, lamda=thresh[0]/3, mu=.98, niter=30)

    # Comment above out and uncomment below for variable density
    #Y = Mv.copy()
    #X = pocs2d(Mv, mask_vardens, lamda=thresh[0]/5, niter=10)
    recon = ifft(X, axes=(0, 1))

    if disp:
        ax = plt.subplot(2, 3, 1)
        plt.imshow(np.abs(brain))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 3, 2)
        plt.imshow(np.abs(recon))
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(2, 3, 5)
        plt.imshow(np.abs(brain - recon), vmin=0, vmax=vmax)
        plt.xlabel(compare_nrmse(np.abs(brain), np.abs(recon)))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 3, 3)
        recon = ifft(Y, axes=(0, 1))
        plt.imshow(np.abs(recon))
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(2, 3, 6)
        plt.imshow(np.abs(brain - recon), vmin=0, vmax=vmax)
        plt.xlabel(str(compare_nrmse(np.abs(brain), np.abs(recon))))
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()
