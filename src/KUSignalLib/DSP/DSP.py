import numpy as np

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
    return y
