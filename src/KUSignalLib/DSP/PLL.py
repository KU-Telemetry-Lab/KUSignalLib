import numpy as np

class PLL():
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, kp=1, k0 = 1, k1=0, k2=0, wstart = 1, fs = 1):
        self.Kp = kp
        self.K0 = k0
        self.K1 = k1
        self.K2 = k2
        self.w0 = wstart
        self.fs = fs

    def insertNewSample(self, lockingSignal, internalSignal, n):
        """
        insert a new sample of the received signal we are trying to lock to.
        """
        phaseError = self.phaseDetector(lockingSignal, internalSignal)
        V_t = self.loopFilter(phaseError)
        pointOut = self.DDS(n, V_t)

    def phaseDetector(self, sample1, sample2, Kp = None):
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

    def loopFilter(self, phaseError, K1 = None, K2 = None):
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
        
    def DDS(self, n, v, k0 = None, w0 = None):
        """
        DDS implementation.
        """
        if k0 is None:
            k0 = self.K0
        # currentPhase = self.phase
        # self.phase += v*k0
        # arg = 2*np.pi*n*self.w0 + currentPhase
        # self.sigOut = np.exp(1j*arg)
        # return self.sigOut
        self.phase += v*k0
        arg = 2*np.pi*n*self.w0 + self.phase
        self.sigOut = np.exp(1j*arg)
        return self.sigOut
    
    def getCurentPhase(self):
        return self.phase