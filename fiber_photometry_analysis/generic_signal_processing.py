import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fiber_photometry_analysis.exceptions import FiberPhotometryGenericSignalProcessingError, \
    FiberPhotometryGenericSignalProcessingValueError


def down_sample_signal(source, factor):  # WARNING: any 1D signal
    """Downsample the data using a certain factor.

    Args :  source (arr) = The input signal
            factor (int) = The factor of down sampling

    Returns : sink = The downsampled signal
    """

    if source.ndim != 1:
        msg = "down_sample_signal() only accepts 1 dimension arrays. Got '{}'".format(source.ndim)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)

    sink = np.mean(source.reshape(-1, factor), axis=1)
    return sink


def smooth_signal(source, window_len=10, window_type='flat'):  # WARNING: any 1D signal
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args :  source (arr) = the input signal
            window_len (int) = the dimension of the smoothing window; should be an odd integer
            window_type (str) = the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns : sink = the smoothed signal

    Code taken from (https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html)
    """

    if source.ndim != 1:
        msg = "smooth only accepts 1 dimension arrays. Got '{}'".format(source.ndim)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)

    if source.size < window_len:
        msg = "Input vector size ({}) needs to be bigger than window size ({}).".format(source.size, window_len)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)

    if window_len < 3:
        return source

    window_types = ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')
    if window_type not in window_types:
        msg = "Window type '{}' is not recognised. Accepted types are '{}'".format(window_type, window_types)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)

    s = np.r_[source[window_len-1:0:-1], source, source[-2:-window_len-1:-1]]

    if window_type == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window_type + '(window_len)')

    sink = np.convolve(w/w.sum(), s, mode='valid')
    return sink


def baseline_asymmetric_least_squares_smoothing(source, l, p, n_iter=10):  # WARNING: any 1D signal
    """Algorithm using Asymmetric Least Squares Smoothing to determine the baseline
    of the signal. Code inspired by the paper from : P. Eilers and H. Boelens in 2005
    (https://www.researchgate.net/publication/228961729_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing)

    Args :      source (np.array) = The signal for which the baseline has to be estimated
                l (int) = smoothness (10^2 ≤ l ≤ 10^11)
                p (float) = asymmetry (0.001 ≤ p ≤ 0.1)
                n_iter (int) = number of iterations

    Returns :   sink (np.array) = The centered moving average of the input signal
    """
    L = len(source)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = l * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    for i in range(n_iter):
        W.setdiag(w)
        Z = W + D
        sink = spsolve(Z, w*source)
        w = p * (source > sink) + (1-p) * (source < sink)

    return sink


def crop_signal(signal, window):  # WARNING: any 1D signal
    """Small routine to trim the begining and the end of a recording to filter
    artifacts.

    Args :      signal (arr) = the signal
                window (float) = the time to crop before and after the recording (time * sampling_rate of the signal)

    Returns :   sink (arr) = The cropped signal
    """

    return signal if window == 0 else signal[int(window):-int(window)]