import numpy as np



class SCS2:
    def __init__(self, loop_bandwidth = None, damping_factor = None, kp=1, k1=0, k2=0, fs=1, sampsPerSym=1 , gain = 1):
        # where index 0 is the real and index 1 is the imaginary
        self.delay2 = [[0, 0], [0,0]]
        self.delay1 = [[0, 0, 0], [0,0,0]]
        self.interpolatedPoints = [[0,0,0], [0,0,0]]
        self.delta_e = 0
        self.delta_e_prev = 0
        self.LFK2prev = 0
        self.strobe = False
        self.gain = gain
        if loop_bandwidth == None and damping_factor == None:
            self.Kp = kp
            self.K1 = k1
            self.K2 = k2
        else:
            self.compute_loop_constants(loop_bandwidth, damping_factor, 1/fs, sampsPerSym, kp)

    def compute_loop_constants(self, loopBandwidth, dampingFactor, T, sampsPerSym, kp):
        """
        :param loopBandwidth: Float type. Loop bandwidth.
        :param dampingFactor: Float type. Damping factor.
        :param T: Float type. this can be your sampleling peiod(i.e. 1/fs), or in communication systems it
        can be your symbol time / N (where N is bits sample per symbol) for a higher bandwidth design.
        Compute the loop filter gains based on the loop bandwidth and damping factor.
        """
        theta_n = (loopBandwidth*T/sampsPerSym)/(dampingFactor + 1/(4*dampingFactor))
        factor = (4*theta_n)/(1+2*dampingFactor*theta_n+theta_n**2)
        self.K1 = dampingFactor * factor/kp
        self.K2 = theta_n * factor/kp
        self.Kp = kp

    def insert_new_sample(self, input):
        interpR = self.farrow_interpolator(np.real(input), row = 0)# give the interpolated point for n-2 sample
        interpI = self.farrow_interpolator(np.imag(input), row = 1)
        error = self.ELTED()
        filtered_error = self.loop_filter(error)
        
        self.strobe = not self.strobe #mod 1 counter
        if self.strobe:
            self.delta_e = self.delta_e_prev
        print(self.delta_e, error)
        self.delta_e_prev = filtered_error*self.gain
        self.interpolatedPoints[0].pop()
        self.interpolatedPoints[0].append(interpR)
        self.interpolatedPoints[1].pop()
        self.interpolatedPoints[1].append(interpI)
        return np.complex128(interpR, interpI)

        

    def farrow_interpolator(self, input, row = 0):
        tmp = self.delta_e
        # self.delta_e = 0
        
        d1next = -0.5*input
        d2next = input
        v2 = -d1next + self.delay1[row][0] + self.delay1[row][1] - self.delay1[row][2]
        v1 = d1next - self.delay1[row][0] + self.delay2[row][0] + self.delay1[row][1] + self.delay1[row][2]
        v0 = self.delay2[row][1]
        output = -(((v2*self.delta_e)+v1)*self.delta_e + v0)
        self.delay1[row].pop()
        self.delay2[row].pop()
        self.delay1[row].append(d1next)
        self.delay2[row].append(d2next)

        self.delta_e = tmp
        
        return output

    def ELTED(self):
        out = 0
        if self.strobe:
            real_est = (self.interpolatedPoints[0][2] - self.interpolatedPoints[0][0]) * (-1 if self.interpolatedPoints[0][1] < 0 else 1)
            imag_est = (self.interpolatedPoints[1][2] - self.interpolatedPoints[1][0]) * (-1 if self.interpolatedPoints[1][1] < 0 else 1)
            # out = (real_est + imag_est)/2
            out = real_est
        return out
    
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
    