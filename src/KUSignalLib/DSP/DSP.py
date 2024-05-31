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
    ldfkgn;ldkng
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