import numpy as np
import math
import matplotlib.pyplot as plt


def FIRLowPass(wp, ws, window_type=, kp=10**(-3/10), ks=10**(-60/10)):
    """
    FIR low pass filter design.

    :param wp: Passband digital frequency cutoff.
    :param ws: Stopband digital frequency cutoff.
    :param kp: Passband attenuation level.
    :param ks: Stopband attenuation level.
    """
    window_lut = {
        "rectangular": {"sidelobe amplitude": 10**(-13/10), "mainlobe width": 4*np.pi, "approximation error": 10**(-21/10)},
        "bartlett": {"sidelobe amplitude": 10**(-25/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-25/10)},
        "hanning": {"sidelobe amplitude": 10**(-31/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-44/10)},
        "hamming": {"sidelobe amplitude": 10**(-41/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-53/10)},
        "blackman": {"sidelobe amplitude": 10**(-57/10), "mainlobe width": 12*np.pi, "approximation error": 10**(-74/10)}
    }
    # selecting window type to meet sidelobe attenuation requirement
    try:
        window_type = max((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks),key=lambda k: window_lut[k]["sidelobe amplitude"])
    except:
        window_type = "blackman"

    # calculating filter shifted and truncated filter parameters
    N = math.ceil(window_lut[window_type]["mainlobe width"] / ws-wp)
    wc = (wp+wp) / 2
    alpha = (N-1)/2

    # determining delayed filter and window coefficients
    h_dn = np.array([(np.sin(wc*(i-alpha)))/(np.pi*(i - alpha)) if i != alpha else wc/np.pi for i in range(N)])

    if window_type == "rectangular":
        w_n = np.array([1 for i in range(N)])
    elif window_type == "bartlett":
        w_n = np.array([(1-(2*np.abs(i-(N-1)/2))/(N-1)) for i in range(N)])
    elif window_type == "hanning":
        w_n = np.array([.5*(1-np.cos((2*np.pi*i)/(N-1))) for i in range(N)])
    elif window_type == "hamming":
        w_n = np.array([.54-.46*np.cos((2*np.pi*i)/(N-1)) for i in range(N)])
    elif window_type == "blackman":
        w_n = np.array([.42-.5*np.cos((2*np.pi*i)/(N-1))+.08*np.cos((4*np.pi*i)/(N-1)) for i in range(N)])

    return h_dn, w_n


wp = 0.3 * np.pi
ws = 0.4 * np.pi

h_dn, w_n = FIRLowPass(wp, ws)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.stem(h_dn)
plt.title('h_dn')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(w_n)
plt.title('w_n')
plt.grid(True)

plt.tight_layout()
plt.show()
