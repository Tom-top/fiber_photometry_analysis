import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags

from fiber_photometry_analysis.exceptions import FiberPhotometryGenericSignalProcessingValueError


def validate_shape(source, expected_shape, func_name=''):
    if source.ndim != expected_shape:
        msg = "{}() only accepts {} dimension arrays. Got '{}' dimensions"\
            .format(func_name, expected_shape, source.ndim)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)


def validate_size(source, window_len):
    if source.size < window_len:  # TODO: extract validator
        msg = "Input vector size ({}) needs to be bigger than window size ({})."\
            .format(source.size, window_len)
        raise FiberPhotometryGenericSignalProcessingValueError(msg)


def validate_dict(key, src_dict, key_name='Window type'):
    if key not in src_dict.keys():
        msg = "{} '{}' is not recognised. Accepted types are '{}'" \
            .format(key_name, key, src_dict.keys())
        raise FiberPhotometryGenericSignalProcessingValueError(msg)


def down_sample_signal(source, factor):
    """Downsample the data using a certain factor.

    Args :  source (arr) = The input signal
            factor (int) = The factor of down sampling

    Returns : sink = The downsampled signal
    """

    validate_shape(source, 1, 'down_sample_signal')
    sink = np.mean(source.reshape(-1, factor), axis=1)
    return sink


def smooth_signal(source, window_len=10, window_type='flat'):
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

    validate_shape(source, 1, 'smooth_signal')
    validate_size(source, window_len)

    if window_len < 3:
        return source

    windows = {
        'flat': np.ones(window_len, 'd'),  # moving avg
        'hanning': np.hanning(window_len),
        'hamming': np.hamming(window_len),
        'bartlett': np.bartlett(window_len),
        'blackman': np.blackman(window_len)
    }
    validate_dict(window_type, windows)

    s = np.r_[source[window_len-1:0:-1], source, source[-2:-window_len-1:-1]]
    w = windows[window_type]
    sink = np.convolve(w / w.sum(), s, mode='valid')
    return sink


def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=5e4, porder=1, itermax=50):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def baseline_asymmetric_least_squares_smoothing(source, l, p, n_iter=10):
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


def crop_signal(signal, sampling_rate, crop_start=0, crop_end=None):
    """Small routine to trim the begining and the end of a recording to filter
    artifacts.

    Args :      signal (arr) = the signal
                window (float) = the time to crop before and after the recording (time * sampling_rate of the signal)

    Returns :   sink (arr) = The cropped signal
    """

    crop_start = int(crop_start)*sampling_rate
    crop_end = -int(crop_end)*sampling_rate if crop_end is not None else crop_end

    return signal if crop_start == 0 and crop_end == 0 else signal[crop_start:crop_end]

def round_to_closest_ten(x):
    return int(round(x / 10.0)) * 10

def generate_new_x(sampling_rate, duration):
    return np.arange(0, duration-1, 1/sampling_rate)

def interpolate_signal(signal_x, signal_y, new_x):
    spl = interp1d(signal_x, signal_y)
    sink = spl(new_x)
    return sink