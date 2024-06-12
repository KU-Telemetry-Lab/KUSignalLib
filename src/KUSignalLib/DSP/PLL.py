import numpy as np

class PLL():
    '''
    This class is used to simulate a PLL discreetly.
    components can be called individually or as a whole depending on user need.
    Use as an object and initialize variable's in init if you want full functionality.
    '''
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, kp=1, k0 = 1, k1=0, k2=0, wstart = 1, thetaStart = 0, fs = 1):
        '''
        :param kp: Float type. Proportional gain.
        :param k0: Float type. DDS gain.
        :param k1: Float type. Loop filter gain feed-forward.
        :param k2: Float type. Loop filter gain feed-back.
        :param wstart: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency.
        Initialize the PLL object for repeated use, if left blank the object will be initialized with default values.
        '''
        self.Kp = kp
        self.K0 = k0
        self.K1 = k1
        self.K2 = k2
        self.w0 = wstart
        self.phase = thetaStart
        self.sigOut = np.exp(1j*(thetaStart))
        self.fs = fs

    def InsertNewSample(self, incomingSignal, n, internalSignal = None):
        """
        :param incomingSignal: Complex number. The current sample of the received signal.
        :param internalSignal: Complex number. the current signal you LO is at. will use default from constructor if left blank
        :param n: Int type. The current sample index, used to
        insert a new sample of the received received signal an LO. if using ass object, this is the index of the
        only function you need to call to achieve PLL functionality.
        """
        if internalSignal is None:
            internalSignal = np.exp(1j*(2*np.pi*(self.w0/self.fs)*n + self.phase))
        phaseError = self.phaseDetector(internalSignal, incomingSignal )
        V_t = self.loopFilter(phaseError)
        pointOut = self.DDS(n, V_t)
        return pointOut

    def PhaseDetector(self, sample1, sample2, Kp = None):
        """
        Phase detector implementation.

        :param sample1: Complex number. First point.
        :param sample2: Complex number. Second point.
        :param Kp: Float type. Proportional gain, should be less than one.
        :return: Float type. Phase difference between the two points.
        """
        if Kp is None:
            Kp = self.Kp
        angel = np.angle(sample2) - np.angle(sample1)
        if angel > np.pi:
            angel -= 2*np.pi
        elif angel < -np.pi:
            angel += 2*np.pi
        return angel*Kp

    def LoopFilter(self, phaseError, K1 = None, K2 = None):
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
        LFK2 = K2*phaseError + self.LFK2prev
        output = K1*phaseError + LFK2
        self.LFK2prev = LFK2
        return output
        
    def DDS(self, n, v, k0 = None, w0 = None, fs = None):
        """
        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :param k0: Float type. DDS gain.
        :param w0: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency, set to .
        DDS implementation.
        """
        if k0 is None:
            k0 = self.K0
        if w0 is None:
            w0 = self.w0
        if fs is None:
            fs = self.fs
        self.phase += v*k0
        self.sigOut = np.exp(1j*(2*np.pi*(w0/self.fs)*n + self.phase))
        return self.sigOut
    
    def GetCurrentPhase(self):
        return self.phase