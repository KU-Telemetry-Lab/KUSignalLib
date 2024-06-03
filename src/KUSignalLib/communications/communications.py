import numpy as np
bpsk = [[complex(1+0j), 0b1], [complex(-1+0j), 0b0]]

def bin_to_char(x):
    """
    Converts a binary array into 7 bit ascii equivalents.

    :param x: List or numpy array type. Input binary signal.
    :return: String containting concatenated ascii characters.
    """
    segmented_arrays = [x[i:i+7] for i in range(0, len(x), 7)]

    bin_chars = []

    for segment in segmented_arrays:
        binary_string = ''.join(str(bit) for bit in segment)
        decimal_value = int(binary_string, 2)
        ascii_char = chr(decimal_value)
        bin_chars.append(ascii_char)

    return ''.join(bin_chars)

def nearest_neighbor(x, constellation=bpsk):
    """
    Find the nearest neighbor in a given constellation.

    :param x: Complex number. Point to find the nearest neighbor for.
    :param constellation: 2d numpy array type containing point-value pairs. List of complex numbers representing the constellation point and its binary value default is bpsk.
    :return: Complex number. Nearest neighbor in the constellation.
    """
    output = []
    for input in x:
        smallest_distance = float('inf')
        binary_value = None
        for point in constellation:
            distance = np.abs(input - point[0])
            if distance < smallest_distance:
                smallest_distance = distance
                binary_value = point[1]
        output.append(binary_value)
    return output
