import numpy as np

class PLL():
    '''
    This class is used to simulate a PLL discretely.
    Components can be called individually or as a whole depending on user need.
    Use as an object and initialize variables in init if you want full functionality.
    '''
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, loop_bandwidth = None, damping_factor = None, kp=1, k0=1, k1=0, k2=0, wstart=1, thetaStart=0, fs=1):
        '''
        :param loop_bandwidth: Float type. Loop bandwidth. if specified with damping factor, will compute loop filter gains.
        :param dampingFactor: Float type. Damping factor. if specified with loop bandwidth, will compute loop filter gains.
        :param kp: Float type. Proportional gain.
        :param k0: Float type. DDS gain.
        :param k1: Float type. Loop filter gain feed-forward.
        :param k2: Float type. Loop filter gain feed-back.
        :param wstart: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency.
        Initialize the PLL object for repeated use, if left blank the object will be initialized with default values.
        '''
        if loop_bandwidth == None and damping_factor == None:
            self.Kp = kp
            self.K0 = k0
            self.K1 = k1
            self.K2 = k2
        else:
            self.compute_loop_constants(loop_bandwidth, damping_factor, 1/fs, k0)
        self.w0 = wstart
        self.phase = thetaStart
        self.sigOut = np.exp(1j * thetaStart)
        self.fs = fs
        
    def compute_loop_constants(self, loopBandwidth, dampingFactor, T, K0=1):
        """
        :param loopBandwidth: Float type. Loop bandwidth.
        :param dampingFactor: Float type. Damping factor.
        :param T: Float type. this can be your sampleling peiod(i.e. 1/fs), or in communication systems it
        can be your symbol time / N (where N is bits sample per symbol) for a higher bandwidth design.
        Compute the loop filter gains based on the loop bandwidth and damping factor.
        """
        theta_n = loopBandwidth*T/(dampingFactor + 1/(4*dampingFactor))
        K0_Kp = (4*theta_n)/(1+2*dampingFactor*theta_n+theta_n**2)
        self.K1 = dampingFactor
        self.K2 = theta_n
        self.K0 = K0
        self.Kp = K0_Kp/K0


    def insert_new_sample(self, incomingSignal, n, internalSignal=None):
        """
        :param incomingSignal: Complex number. The current sample of the received signal.
        :param internalSignal: Complex number. The current signal your LO is at. Will use default from constructor if left blank.
        :param n: Int type. The current sample index, used to insert a new sample of the received signal and LO.
        If using as an object, this is the index of the only function you need to call to achieve PLL functionality.
        """
        if internalSignal is None:
            internalSignal = np.exp(1j * (2 * np.pi * (self.w0 / self.fs) * n + self.phase))
        phaseError = self.phase_detector(internalSignal, incomingSignal)
        V_t = self.loop_filter(phaseError)
        pointOut = self.DDS(n, V_t)
        return pointOut

    def phase_detector(self, sample1, sample2, Kp=None): # for BPSK take sign of real and mult. my imag
        """
        Phase detector implementation.

        :param sample1: Complex number. First point.
        :param sample2: Complex number. Second point.
        :param Kp: Float type. Proportional gain, should be less than one.
        :return: Float type. Phase difference between the two points.
        """
        if Kp is None:
            Kp = self.Kp
        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle * Kp

    def loop_filter(self, phaseError, K1=None, K2=None):
        """
        Loop filter implementation.
        :param phaseError: Float type. Phase error.
        :param K1: Float type. Loop filter gain according to Fig C.2.6.
        :param K2: Float type. Loop filter gain according to Fig C.2.6.
        """
        if K1 is None:
            K1 = self.K1
        if K2 is None:
            K2 = self.K2
        LFK2 = K2 * phaseError + self.LFK2prev
        output = K1 * phaseError + LFK2
        self.LFK2prev = LFK2
        return output

    def DDS(self, n, v, k0=None, w0=None, fs=None):
        """
        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :param k0: Float type. DDS gain.
        :param w0: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency.
        DDS implementation.
        """
        if k0 is None:
            k0 = self.K0
        if w0 is None:
            w0 = self.w0
        if fs is None:
            fs = self.fs
        self.phase += v * k0
        self.sigOut = np.exp(1j * (2 * np.pi * (w0 / fs) * n + self.phase))
        return self.sigOut
    
    def get_current_phase(self):
        return self.phase
