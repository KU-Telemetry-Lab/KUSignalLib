import numpy as np

def IIRDirectForm2(b, a, x):
    """
    DO NOT TRUST THIS 99.9% sure its wrong
    Direct Form II IIR filter implementation
    :param b: numerator coefficients
    :param a: denominator coefficients
    :param x: input signal
    :return: output signal
    """
    # Initialize the delay lines
    n = len(b)
    m = len(a)
    x = np.concatenate((np.zeros(n - 1), x))
    y = np.zeros(len(x))
    delayLine = np.zeros(n)
    delayLine[0] = x[0]
    for i in range(1, len(x)):
        y[i] = np.dot(b, delayLine) - np.dot(a[1:], y[i - 1])
        delayLine[1:] = delayLine[:-1]
        delayLine[0] = x[i]
    return y