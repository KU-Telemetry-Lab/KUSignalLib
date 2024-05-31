import numpy as np
import math
from scipy import interpolate as intp

def IIRDirectForm2(b, a, x):
    """
    Direct Form II IIR filter implementation.

    :param b: Numerator coefficients.
    :param a: Denominator coefficients.
    :param x: Input signal.
    :return: Output signal.
    """
    # Initialize the delay lines
    n = len(b)
    m = len(a)
    maxLen = max(n, m)
    denominator = a.copy()
    denominator[1:] = -denominator[1:] #flip sign of denominator coefficients
    denominator[0] = 0 #zero out curent p(0) value for multiply, will add this coeff. back in for new x[n] term
    x = np.concatenate((x, np.zeros(maxLen - 1))) #zero pad x
    y = np.zeros(len(x))
    delayLine = np.zeros(maxLen)
    delayLine[0] = x[0]
    for i in range(len(x)):
        y[i] = np.dot(b, delayLine) #df2 right side
        tmp = np.dot(denominator, delayLine) #df2 left side
        delayLine[1:] = delayLine[:-1] #shift delay line
        delayLine[0] = x[i]*a[0] + tmp #new value is x[n] * a[0] + sum of left side
    return y

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


def Upsample(x, L, interpolate=True):
    """
    Discrete signal upsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Upsample factor.
    :param interpolate: Boolean type. Flag indicating whether to perform interpolation. True = interpolate. False = don't interpolate.
    :return: Numpy array type. Upsampled signal.
    """
    x_upsampled = []  # Initialize a list to store the upsampled signal
    if interpolate:
        x_upsampled = Interpolate(x, L, mode="linear")
    else:
        for i in range(len(x)):  # Iterate over each element in the input signal
            x_upsampled += [x[i]] + list(np.zeros(L-1, dtype=type(x[0])))  # Add the current element and L zeros after each element
    return x_upsampled


def Downsample(x, L):
    """
    Discrete signal downsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Downsampled factor.
    :return: Numpy array type. Downsampled signal.
    """
    x_downsampled = []  # Initialize an empty list to store the downsampled signal
    if L > len(x):  # Check if the downsample rate is larger than the signal size
        raise ValueError("Downsample rate larger than signal size.")
    # Loop over the signal, downsampling by skipping every L elements
    for i in range(math.floor(len(x) // L)):
        x_downsampled.append(x[i*L])
    return x_downsampled  # Return the downsampled signal