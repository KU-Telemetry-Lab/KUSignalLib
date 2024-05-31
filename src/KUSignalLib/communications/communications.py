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

    # just a test
    