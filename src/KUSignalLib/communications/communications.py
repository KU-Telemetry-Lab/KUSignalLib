import math
import matplotlib.axes
import numpy as np
import sys
from scipy import interpolate as intp

def bin_to_char(x):
    """
    Converts a binary array into 7 bit ascii equivalents.

    :param x: List or numpy array type. Input binary signal.
    :return: String containing concatenated ascii characters.
    """
    segmented_arrays = [x[i:i+7] for i in range(0, len(x), 7)]

    bin_chars = []

    for segment in segmented_arrays:
        binary_string = ''.join(str(bit) for bit in segment)
        decimal_value = int(binary_string, 2)
        ascii_char = chr(decimal_value)
        bin_chars.append(ascii_char)

    return ''.join(bin_chars)

def nearest_neighbor(x, constellation = None, binary = True):
    """
    Find the nearest neighbor in a given constellation.

    :param x: Complex number or array of complex numbers. Point(s) to find the nearest neighbor for.
    :param constellation: 2D numpy array containing point-value pairs. List of complex numbers 
           representing the constellation point and its binary value. defaults to BPAM/BPSK
    :return: List of binary values corresponding to the nearest neighbors in the constellation.
    """
    if constellation is None:
        constellation =  [[complex(1+0j), 0b1], [complex(-1+0j), 0b0]]
    output = []
    for input_value in x:
        smallest_distance = float('inf')
        value = None
        for point in constellation:
            distance = np.abs(input_value - point[0])
            if distance < smallest_distance:
                smallest_distance = distance
                if binary:
                    value = point[1]
                else:
                    value = point[0]
        output.append(value)
    return output

def half_sine_pulse(mag, length):
    """
    Generates a half sine pulse.

    :param mag: Magnitude of pulse.
    :param length: Length of pulse.
    :return: List. Half sine pulse.
    """
    inc = math.pi/(length-1)
    temp = []
    for i in range(length):
        temp.append(math.sin(i*inc)*mag)
    return temp

def srrc(alpha, m, length):
    """
    Generates a square root raised cosine pulse.

    :param alpha: Roll-off or excess factor.
    :param m: Number of symbols per symbol.
    :param length: Length of pulse. Should be k*m+1 where k is an integer.
    :return: List. Square root raised cosine pulse.
    """
    pulse = []
    for n in range(length):
        n_prime = n - np.floor(length/2)
        if n_prime == 0:
            n_prime = sys.float_info.min  # Handle case when n_prime is zero
        if alpha != 0:
            if np.abs(n_prime) == m/(4*alpha):
                n_prime += 0.1e-12
        num = np.sin(np.pi*((1-alpha)*n_prime/m)) + (4*alpha*n_prime/m)*np.cos(np.pi*((1+alpha)*n_prime/m))
        den = (np.pi*n_prime/m)*(1-(4*alpha*n_prime/m)**2)*np.sqrt(m)
        if den == 0:
            pulse.append(1.0)  # Handle division by zero case
        else:
            pulse.append(num/den)
    return pulse

def bin_to_symbol(sequence, constellation = None):
    """

    :param sequence: List or numpy array type. Input binary signal must be an integer multiple of symbol size.
    :param constellation: 2D numpy array containing point-value pairs. List of complex numbers
    :
    """
    if constellation is None:
        constellation = [[complex(1+0j), 0b1], [complex(-1+0j), 0b0]]
    symbol_size = max([len(bin(sub_array[1])) for sub_array in constellation])-2
    # print(symbol_size)
    values = []
    for i in range(int(len(sequence)/symbol_size)):
        bin_array = sequence[i*symbol_size:(i+1)*symbol_size]
        bin_string = ''.join(str(bit) for bit in bin_array)
        decimal_value = int(bin_string, 2)
        for point in constellation:
            if point[1] == decimal_value:
                values.append(point) 
    return values

def differential_encoder_bin(x):
    """
    Differential encoder assumes 0 indicates change, designed for BPSK.

    :param x: List or numpy array type. Input binary signal.
    :return: List. Differential encoded binary signal.
    """
    x.insert(0, 0)
    output = [x[0]]
    for i in range(1, len(x)):
        output.append(int(not(x[i]^output[i-1])))
    output.remove(0)
    return output

def differential_decoder_bin(x):
    """
    Differential encoder assumes 0 indicates change, designed for BPSK.

    :param x: List or numpy array type. Input binary signal.
    :return: List. Differential encoded binary signal.
    """
    output = [1]
    for i in range(0, len(x)):
        output.append((x[i])^(not (x[i-1])))
    output.remove(0)
    return output

def differential_decoder(x, LUT, allowedError = np.pi/12):
    """
    Differential encoder assumes 0 indicates change, designed for BPSK.

    :param x: List or numpy array type. Input signal.
    :return: List. Differential encoded signal.
    """
    output = [0b10]
    for i in range(1, len(x)):
        phaseDiff = np.angle(x[i])-np.angle(x[i-1])
        if phaseDiff > np.pi:
            phaseDiff -= 2*np.pi
        elif phaseDiff <= -np.pi:
            phaseDiff += 2*np.pi
        for j in LUT:
            if abs(phaseDiff-j[0]) < allowedError:
                output.append(j[1])
    return output

def clock_offset(signal, sample_rate, offset_fraction):
    """
    Simulates clock offset due to mismatched synchronization or skew. Input is
    usually after upsample and match filtering.

    :param signal: List or numpy array type. Input signal with no clock offset.
    :param sample_rate: Int type. Sample rate of input signal.
    :param offset_fraction: Float type. Offset fraction to be normalized by sample duration.
    :return: Numpy array type. Clock offset / shifted version of input signal.
    """
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction

    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset 
    x_shifted = interpolator(t_shifted)
    return x_shifted
