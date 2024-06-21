


def fir_low_pass(fc, window=None, kp=10**(-3/10), ks=10**(-40/10)):
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
        wp = fc - (.125 * fc)
        ws = fc + (.125 * fc)
        kp = 10**(-3/10)
        ks = 10**(-40/10)
        if window is None:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        else:
            window_type = window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(ws - wp))
    wc = (wp + ws) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([(np.sin(wc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else wc / np.pi for i in range(N)])
    w_n = apply_window(N, window_type)

    return h_dn*w_n