#https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
import numpy as np

__all__ = ['complex_cepstrum', 'real_cepstrum', 'inverse_complex_cepstrum',
           'minimum_phase']

def complex_cepstrum(x, n=None):
    """Compute the complex cepstrum of a real sequence.

    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.

    The complex cepstrum is given by

    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}

    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    See Also
    --------
    real_cepstrum: Compute the real cepstrum.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.

    Examples
    --------
    In the following example we use the cepstrum to determine the fundamental
    frequency of a set of harmonics. There is a distinct peak at the quefrency
    corresponding to the fundamental frequency. To be more precise, the peak
    corresponds to the spacing between the harmonics.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import complex_cepstrum

    >>> duration = 5.0
    >>> fs = 8000.0
    >>> samples = int(fs*duration)
    >>> t = np.arange(samples) / fs

    >>> fundamental = 100.0
    >>> harmonics = np.arange(1, 30) * fundamental
    >>> signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
    >>> ceps, _ = complex_cepstrum(signal)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(t, signal)
    >>> ax0.set_xlabel('time in seconds')
    >>> ax0.set_xlim(0.0, 0.05)
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t, ceps)
    >>> ax1.set_xlabel('quefrency in seconds')
    >>> ax1.set_xlim(0.005, 0.015)
    >>> ax1.set_ylim(-5., +10.)

    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum
    .. [2] M.P. Norton and D.G. Karczub, D.G.,
           "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
    .. [3] B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
           "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
           Autocovariance, Cross-Cepstrum and Saphe Cracking".
           Proceedings of the Symposium on Time Series Analysis
           Chapter 15, 209-243. New York: Wiley, 1963.

    """
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples+1)//2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
        unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
        return unwrapped, ndelay
    print("in complex len x: ", len(x))
    spectrum = np.fft.fft(x, n=n)
    print("in complex len spectrum: ", len(spectrum))
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j*unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay, log_spectrum


def real_cepstrum(x, n=None):
    """Compute the real cepstrum of a real sequence.

    x : ndarray
        Real sequence to compute real cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Returns
    -------
    ceps: ndarray
        The real cepstrum.

    The real cepstrum is given by

    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left|F{x[n]}\\right|}\\right}

    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform. Note that
    contrary to the complex cepstrum the magnitude is taken of the spectrum.


    See Also
    --------
    complex_cepstrum: Compute the complex cepstrum of a real sequence.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.

    Examples
    --------
    >>> from scipy.signal import real_cepstrum


    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum

    """
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps
