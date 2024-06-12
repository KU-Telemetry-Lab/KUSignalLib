import numpy as np
import math
from scipy import interpolate as intp


def DirectForm2(b, a, x):
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
        maxLen = n
        a = np.concatenate((a, np.zeros(n - m)))
    else:
        maxLen = m
        b = np.concatenate((b, np.zeros(m - n)))
    denominator = a.copy()
    denominator[1:] = -denominator[1:] #flip sign of denominator coefficients
    denominator[0] = 0 #zero out curent p(0) value for multiply, will add this coeff. back in for new x[n] term
    x = np.concatenate((x, np.zeros(maxLen - 1))) #zero pad x
    y = np.zeros(len(x))
    delayLine = np.zeros(maxLen)
    for i in range(len(x)):
        y[i] = np.dot(b, delayLine) #df2 right side
        tmp = np.dot(denominator, delayLine) #df2 left side
        delayLine[1:] = delayLine[:-1] #shift delay line
        delayLine[0] = x[i]*a[0] + tmp #new value is x[n] * a[0] + sum of left side
    return y[1:]


def Interpolate(x, n, mode="linear"):
    """
    Perform interpolation on an upsampled signal.

    :param x: Input signal.
    :param n: Upsampled factor (signal is already upsampled)
    :param mode: Interpolation type. Modes = "linear", "quadratic"
    :return: Interpolated signal.
    """
    nonzero_indices = np.arange(0, len(x)*n, n) # Generate indices for upsampled signal
    interpolation_function = intp.interp1d(nonzero_indices, np.array(x), kind=mode, fill_value='extrapolate') # create interpolation function
    interpolated_signal = interpolation_function(np.arange(len(x)*n)) # interpolate the signal
    return interpolated_signal


def Upsample(x, L, offset=0, interpolate=True):
    """
    Discrete signal upsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Upsample factor.
    :param offset: Int type. Offset size for input array.
    :param interpolate: Boolean type. Flag indicating whether to perform interpolation. True = interpolate. False = don't interpolate.
    :return: Numpy array type. Upsampled signal.
    """
    x_upsampled = [0 for i in range(offset)]  # Initialize a list to store the upsampled signal (add offset if needed)
    if interpolate:
        x_upsampled = Interpolate(x, L, mode="linear")
    else:
        for i in range(len(x)):  # Iterate over each element in the input signal
            x_upsampled += [x[i]] + list(np.zeros(L-1, dtype=type(x[0])))  # Add the current element and L zeros after each element
    return x_upsampled


def Downsample(x, L, offset=0):
    """
    Discrete signal downsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Downsampled factor.
    :param offset: Int type. Offset size for input array.
    :return: Numpy array type. Downsampled signal.
    """
    x_downsampled = [0 for i in range(offset)]  # Initialize an empty list to store the downsampled signal (add offset if needed)
    if L > len(x):  # Check if the downsample rate is larger than the signal size
        raise ValueError("Downsample rate larger than signal size.")
    # Loop over the signal, downsampling by skipping every L elements
    for i in range(math.floor(len(x) // L)):
        x_downsampled.append(x[i*L])
    return x_downsampled  # Return the downsampled signal


def FIRLowPass(cutoff_frequency, window=None, **kwargs):
    """
    FIR low pass filter design.

    Generic design parameters.

    :param cutoff_frequency: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).

    Detailed design parameters (optional).

    :param passband_cutoff: Passband digital frequency cutoff.
    :param stopband_cutoff: Stopband digital frequency cutoff.
    :param passband_attenuation: Passband attenuation level.
    :param stopband_attenuation: Stopband attenuation level.

    :return: Numpy array type. Coefficients (numerator) of digital lowpass filter.
    """
    window_lut = {
            "rectangular": {"sidelobe amplitude": 10**(-13/10), "mainlobe width": 4*np.pi, "approximation error": 10**(-21/10)},
            "bartlett": {"sidelobe amplitude": 10**(-25/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-25/10)},
            "hanning": {"sidelobe amplitude": 10**(-31/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-44/10)},
            "hamming": {"sidelobe amplitude": 10**(-41/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-53/10)},
            "blackman": {"sidelobe amplitude": 10**(-57/10), "mainlobe width": 12*np.pi, "approximation error": 10**(-74/10)}}
    
    if 'passband_cutoff' in kwargs and 'stopband_cutoff' in kwargs:
        wp = kwargs['passband_cutoff']
        ws = kwargs['stopband_cutoff']
        if 'passband_attenuation' in kwargs and 'stopband_attenuation' in kwargs:
            kp = kwargs['passband_attenuation']
            ks = kwargs['stopband_attenuation']
        else: # using standard attenuation levels
            kp = 10**(-3/10)
            ks = 10**(-40/10)
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else: # using standard transition width
        wp = cutoff_frequency - (.125 * cutoff_frequency)
        ws = cutoff_frequency + (.125 * cutoff_frequency)
        kp = 10**(-3/10)
        ks = 10**(-40/10)
        if window == None:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        else:
            window_type=window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(ws - wp))
    wc = (wp + ws) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([(np.sin(wc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else wc / np.pi for i in range(N)])

    if window_type == "rectangular":
        w_n = np.array([1 for i in range(N)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (N - 1) / 2)) / (N - 1)) for i in range(N)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (N - 1))) for i in range(N)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (N - 1)) for i in range(N)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (N - 1)) + 0.08 * np.cos((4 * np.pi * i) / (N - 1)) for i in range(N)])

    return h_dn*w_n


def FIRHighPass(cutoff_frequency, window=None, **kwargs):
    """
    FIR high pass filter design.

    Generic design parameters.

    :param cutoff_frequency: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).

    Detailed design parameters (optional).

    :param stopband_cutoff: Stopband digital frequency cutoff.
    :param passband_cutoff: Passband digital frequency cutoff.
    :param stopband_attenuation: Stopband attenuation level.
    :param passband_attenuation: Passband attenuation level.

    :return: Numpy array type. Coefficients (numerator) of digital highpass filter.
    """
    window_lut = {
            "rectangular": {"sidelobe amplitude": 10**(-13/10), "mainlobe width": 4*np.pi, "approximation error": 10**(-21/10)},
            "bartlett": {"sidelobe amplitude": 10**(-25/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-25/10)},
            "hanning": {"sidelobe amplitude": 10**(-31/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-44/10)},
            "hamming": {"sidelobe amplitude": 10**(-41/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-53/10)},
            "blackman": {"sidelobe amplitude": 10**(-57/10), "mainlobe width": 12*np.pi, "approximation error": 10**(-74/10)}}
    
    if 'stopband_cutoff' in kwargs and 'passband_cutoff' in kwargs:
        ws = kwargs['stopband_cutoff']
        wp = kwargs['passband_cutoff']
        if 'stopband_attenuation' in kwargs and 'passband_attenuation' in kwargs:
            ks = kwargs['stopband_attenuation']
            kp = kwargs['passband_attenuation']
        else: # using standard attenuation levels
            ks = 10**(-40/10)
            kp = 10**(-3/10)
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else: # using standard transition width
        ws = cutoff_frequency - (.125 * cutoff_frequency)
        wp = cutoff_frequency + (.125 * cutoff_frequency)
        ks = 10**(-40/10)
        kp = 10**(-3/10)
        if window == None:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        else:
            window_type=window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(ws - wp))
    wc = (wp + ws) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([-(np.sin(wc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else 1 - (wc / np.pi) for i in range(N)])

    if window_type == "rectangular":
        w_n = np.array([1 for i in range(N)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (N - 1) / 2)) / (N - 1)) for i in range(N)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (N - 1))) for i in range(N)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (N - 1)) for i in range(N)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (N - 1)) + 0.08 * np.cos((4 * np.pi * i) / (N - 1)) for i in range(N)])

    return h_dn*w_n