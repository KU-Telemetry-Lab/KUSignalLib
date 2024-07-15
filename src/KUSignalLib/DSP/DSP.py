import numpy as np
import math
from scipy import interpolate as intp, signal as sig
import matplotlib.pyplot as plt

def direct_form_2(b, a, x):
    """
    Direct Form II IIR filter implementation.
    
    Note: This implementation assumes that the coefficients 'b' and 'a' are ordered such that b[0] and a[0] correspond 
    to the highest-order coefficient, and b[-1] and a[-1] correspond to the lowest-order coefficient.

    :param b: Numerator coefficients (numpy array)
    :param a: Denominator coefficients (numpy array)
    :param x: Input signal
    :return: Output signal
    """
    # Initialize the delay lines
    n = len(b)
    m = len(a)
    if n > m:
        max_len = n
        a = np.concatenate((a, np.zeros(n - m)))
    else:
        max_len = m
        b = np.concatenate((b, np.zeros(m - n)))
    denominator = a.copy()
    denominator[1:] = -denominator[1:]  # flip sign of denominator coefficients
    denominator[0] = 0  # zero out current p(0) value for multiply, will add this coeff. back in for new x[n] term
    x = np.concatenate((x, np.zeros(max_len - 1)))  # zero pad x
    y = np.zeros(len(x), dtype=complex)
    delay_line = np.zeros(max_len, dtype=complex)
    for i, value in enumerate(x):
        y[i] = np.dot(b, delay_line)  # df2 right side
        tmp = np.dot(denominator, delay_line)  # df2 left side
        delay_line[1:] = delay_line[:-1]  # shift delay line
        delay_line[0] = value * a[0] + tmp  # new value is x[n] * a[0] + sum of left side
    
    return y[1:]

def interpolate(x, n, mode="linear"):
    """
    Perform interpolation on an upsampled signal.

    :param x: Input signal (already upsampled with zeros).
    :param n: Upsampled factor.
    :param mode: Interpolation type. Modes = "linear", "quadratic".
    :return: Interpolated signal.
    """
    nonzero_indices = np.arange(0, len(x), n)
    nonzero_values = x[nonzero_indices]
    interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind=mode, fill_value='extrapolate')
    new_indices = np.arange(len(x))
    interpolated_signal = interpolation_function(new_indices)
    return interpolated_signal

def upsample(x, L, offset=0, interpolate_flag=True):
    """
    Discrete signal upsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Upsample factor.
    :param offset: Int type. Offset size for input array.
    :param interpolate: Boolean type. Flag indicating whether to perform interpolation.
    :return: Numpy array type. Upsampled signal.
    """
    x_upsampled = [0] * offset  # Initialize with offset zeros
    if interpolate_flag:
        x_upsampled.extend(interpolate(x, L))
    else:
        for i, sample in enumerate(x):
            x_upsampled.append(sample)
            x_upsampled.extend([0] * (L - 1))
    return np.array(x_upsampled)

def downsample(x, l, offset=0):
    """
    Discrete signal downsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param l: Int type. Downsample factor.
    :param offset: Int type. Offset size for input array.
    :return: Numpy array type. Downsampled signal.
    """
    x_downsampled = [0+0j] * offset  # Initialize with offset zeros
    if l > len(x):
        raise ValueError("Downsample rate larger than signal size.")
    # Loop over the signal, downsampling by skipping every l elements
    for i in range(math.floor(len(x) / l)):
        x_downsampled.append(x[i * l])
    
    return np.array(x_downsampled)

def phase_difference(x, y):
    """
    Phase detector implementation.

    :param sample1: Complex number. First point.
    :param sample2: Complex number. Second point.
    :return: Float type. Phase difference between the two points within range [0, 2pi].
    """
    return (np.angle(x) - np.angle(y)) % (2 * np.pi)

# windowing look up table used in aid of FIR filterdesign flows
window_lut= {"rectangular": {"sidelobe amplitude": 10**(-13/10), 
                             "mainlobe width": 4*np.pi, 
                             "approximation error": 10**(-21/10)},
             "bartlett": {"sidelobe amplitude": 10**(-25/10), 
                          "mainlobe width": 8*np.pi, 
                          "approximation error": 10**(-25/10)},
             "hanning": {"sidelobe amplitude": 10**(-31/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-44/10)},
             "hamming": {"sidelobe amplitude": 10**(-41/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-53/10)},
             "blackman": {"sidelobe amplitude": 10**(-57/10), 
                          "mainlobe width": 12*np.pi, 
                          "approximation error": 10**(-74/10)}
            }

def apply_window(n, window_type):
    """
    Windowing function used in aid of FIR design flows.

    :param N: Window length (number of coefficients).
    :param window_type: Window type (see below).
    :return w_n: Numpy array type. Calculated window filter coefficients.
    """
    if window_type == "rectangular":
        w_n = np.array([1 for i in range(n)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (n - 1) / 2)) / (n - 1)) for i in range(n)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (n - 1))) for i in range(n)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (n - 1)) for i in range(n)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (n - 1)) + 0.08 * np.cos((4 * np.pi * i) / (n - 1)) for i in range(n)])
    else: #default to 'rectangular'
        w_n = np.array([1 for i in range(n)])
    return w_n

def fir_low_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR low pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation.
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital lowpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else fc / np.pi for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_high_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR high pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital highpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([-(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else 1 - (fc / np.pi) for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_band_pass(fc1, fc2, window=None, fs1=None, fp1=None, fp2=None, fs2=None, ks1=10**(-40/10), ks2=10**(-40/10)):
    """
    FIR band pass filter design.

    :param fc1: Digital cutoff frequency one.
    :param fc2: Digital cutoff frequency two.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp1: Passband digital frequency cutoff one.
    :param fs1: Stopband digital frequency cutoff one.
    :param fp2: Passband digital frequency cutoff two.
    :param fs2: Stopband digital frequency cutoff two.
    :param ks1: Stopband attenuation level one.
    :param ks2: Stopband attenuation level two.
    :return: Numpy array. Coefficients (numerator) of digital bandpass filter.
    """
    if fp1 is None or fs1 is None or fp2 is None or fs2 is None:
        fs1 = fc1 + (.125 * fc1)
        fp1 = fc1 - (.125 * fc1)
        fp2 = fc2 - (.125 * fc2)
        fs2 = fc2 + (.125 * fc2)
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < min(ks1, ks2)), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / min(np.abs(fs1 - fp1), np.abs(fs2 - fp2))) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([((np.sin(fc2 * (i - alpha))) / (np.pi * (i - alpha)) - (np.sin(fc1 * (i - alpha))) / (np.pi * (i - alpha))) if i != alpha else (fc2 / np.pi - fc1 / np.pi)  for i in range(n)]) # determining the filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn*w_n

def modulate_by_exponential(x, f_c, f_s, phase=0):
    """
    Modulates a signal by exponential carrier (cos(x) + jsin(x)).

    :param x: List or numpy array. Input signal to modulate.
    :param f_c: Float. Carrier frequency of the modulation.
    :param f_s: Float. Sampling frequency of the input signal.
    :return: List. Modulated signal.
    """
    y = []
    for i, value in enumerate(x):
        # Exponential modulation using complex exponential function
        modulation_factor = np.exp(-1j * 2 * np.pi * f_c * i / f_s + phase)
        y.append(value * modulation_factor)
    return np.array(y)

def plot_complex_points(data, referencePoints = None):
    """
    Plot complex points on a 2D plane.

    :param data: List or numpy array. Complex points to plot.
    """
    plt.plot([point.real for point in data], [point.imag for point in data], 'ro')
    if referencePoints is not None:
        plt.plot([point.real for point in referencePoints], [point.imag for point in referencePoints], 'b+')
    plt.show()

def convolve(x, h, mode='full'):
    """
    Convolution between two sequences. Can return full or same lengths.

    :param x: List or numpy array. Input sequence one.
    :param h: List or numpy array. Input sequence two.
    :param mode: String. Specifies return sequence length.
    :return: Numpy array. Resulting convolution output.
    """
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    h_padded = np.pad(h, (0, N - len(h)), mode='constant')
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    y = np.fft.ifft(X * H)

    if mode == 'same':
        start = (len(h) - 1) // 2
        end = start + len(x)
        y = y[start:end]
    return y

def cross_correlation(signal, sequence):
    """
    Cross-correlation between two signals.

    :param signal: List or numpy array. Input signal.
    :param sequence: List or numpy array. Input sequence should be the sequence 
     your searching for, shold be shoter then signal.

    :return: Numpy array. Cross-correlation output.
    """
    output = []
    signal = np.pad(signal, (len(sequence), len(sequence)), 'constant', constant_values=(0, 0))
    for i in range(len(signal)-len(sequence)-1):
        output.append(np.dot(signal[i:i + len(sequence)], sequence))
    return output