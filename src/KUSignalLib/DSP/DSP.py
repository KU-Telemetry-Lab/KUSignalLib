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

def phaseDetector(sample1, sample2, Kp):
    """
    Phase detector implementation.

    :param sample1: Complex number. First point.
    :param sample2: Complex number. Second point.
    :param Kp: Float type. Proportional gain, should be less than one.
    :return: Float type. Phase difference between the two points.
    """
    return (np.angle(sample2) - np.angle(sample1))*Kp

def DDS(v, f_c, f_old, fs):
    pass
    