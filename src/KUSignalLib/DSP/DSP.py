import numpy as np
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
    :param mode: Interpolation type -> "linear", "quadratic"
    :return: Interpolated signal.
    """
    nonzero_indices = np.arange(0, len(x)*n, n) # Generate indices for upsampled signal
    interpolation_function = intp.interp1d(nonzero_indices, np.array(x), kind=mode, fill_value='extrapolate') # create interpolation function
    interpolated_signal = interpolation_function(np.arange(len(x)*n)) # interpolate the signal
    return interpolated_signal


def Upsample(x, n, dim=1, interpolate=True, axis=0):
    """
    Discrete signal upsample implementation.

    :param x: Input signal.
    :param n: Upsample factor.
    :param dim: Dimension of input signal. Default is 1.
    :param interpolate: Flag indicating whether to perform interpolation. Default is True.
    :param axis: Axis of x that is upsampled. Default is 0.
    :return: Upsampled signal.
    """
    x_upsampled = []  # Initialize a list to store the upsampled signal

    if dim == 1:  # If the input signal is 1D
        if interpolate:
            x_upsampled = Interpolate(x, n, mode="linear")
        else:
            for i in range(len(x)):  # Iterate over each element in the input signal
                x_upsampled += [x[i]] + list(np.zeros(n-1, dtype=type(x[0])))  # Add the current element and n zeros after each element
        return x_upsampled

    elif dim == 2:  # If the input signal is 2D
        if axis == 0:  # If axis input is 0, interpolate over rows
            if interpolate:
                for i in range(len(x)):
                    x_upsampled += [list(Interpolate(x[i], n, mode="linear"))]
            else:
                for i in range(len(x)):  # Iterate over each row in the input signal
                    x_i_upsampled = []  # Initialize a list to store the upsampled row
                    for j in range(len(x[i])):  # Iterate over each element in the current row
                        x_i_upsampled += [x[i][j]] + list(np.zeros(n-1, dtype=type(x[i][0]))) # Add the current element and n zeros after each element
                    x_upsampled.append(x_i_upsampled)  # Add the upsampled row to the upsampled signal
            return x_upsampled

        elif axis == 1:  # If axis input is 1, interpolate over columns
            if interpolate:
                pass
            else:
                for i in range(len(x)):
                    for j in range(n-1):
                        x_upsampled += [x[i]]
                        for k in range(n-1):
                            x_upsampled += [list(np.zeros(len(x[i]), dtype=type(x[i][0])))]
            return x_upsampled

        else:
            raise ValueError("Invalid axis input.")  # Raise an error indicating invalid axis input
    else:  # If the input dimension is not 1 or 2
        raise ValueError("Invalid dimension input.")  # Raise an error indicating invalid dimension input


def Downsample(x, n, dim=1, axis=0):
    """
    Discrete signal downsample implementation.

    :param x: Input signal.
    :param n: Downsampled factor.
    :param dim: Dimension of input signal. Default is 1.
    :param axis: Axis of x that is downsampled. Default is 0.
    :return: Downsampled signal.
    """

    x_downsampled = []  # Initialize an empty list to store the downsampled signal
    
    if dim == 1:  # If the dimension is 1D
        if n > len(x):  # Check if the downsample rate is larger than the signal size
            raise ValueError("Downsample rate larger than signal size.")
        
        # Loop over the signal, downsampling by skipping every n elements
        for i in range(len(x) // n):
            x_downsampled.append(x[i*n])
        
        return x_downsampled  # Return the downsampled signal
    
    elif dim == 2:  # If the dimension is 2D
        if axis == 0:  # If downsampling over rows
            # Loop over each row in the signal
            for i in range(len(x)):
                x_i_downsampled = []
                # Downsampling each row by skipping every n elements
                for j in range(len(x[i]) // n):
                    x_i_downsampled.append(x[i][j*n])
                # Append the downsamples row to the downsampled signal list
                x_downsampled.append(x_i_downsampled)
            return x_downsampled  # Return the downsampled signal
        
        elif axis == 1:  # If downsampling over columns
            # Loop over the signal, downsampling by skipping every n rows
            for i in range(0, len(x) // n, n):
                x_downsampled.append(x[i])
            return x_downsampled  # Return the downsampled signal
        
    else:  # If the input dimension is not 1 or 2
        raise ValueError("Invalid dimension input.")