import numpy as np
from scipy import interpolate as intp

class SCS():
    def __init__(self, input_signal_complex, samples_per_symbol, loop_bandwidth=None, damping_factor=None, k0=1, kp=1, k1=0, k2=0, upsample_rate=10):
        '''
        Initialize the SCS (Symbol Clock Syncrhonization) subsystem class.

        :param input_signal_complex: Complex numpy array type. Input clock skewed sampled complex signal.
        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock on to the timing error (similar to PLL).
        :param kp: Float type. TBD.
        :param k0: Float type. TBD.
        :param k1: Float type. Loop filter coefficient one.
        :param k2: Float type. Loop filter coefficient two.
        :param upsample_rate: Int type. Upsample rate of timing error correction interpolation.
        '''
        self.input_signal_real = np.real(input_signal_complex)
        self.input_signal_imag = np.imag(input_signal_complex)
        self.samples_per_symbol = samples_per_symbol
        self.upsample_rate = upsample_rate

        self.k2_prev = 0
        if loop_bandwidth == None and damping_factor == None:
            self.kp = kp
            self.k0 = k0
            self.k1 = k1
            self.k2 = k2
        else:
            self.compute_loop_constants(loop_bandwidth, damping_factor, k0, kp)
        
        self.adjusted_symbol_block_real = [0, 0, 0]
        self.adjusted_symbol_block_imag = [0, 0, 0]
        self.timing_error_record = []
        self.loop_filter_record = []
        
        self.scs_output = np.zeros(len(input_signal_complex), dtype=complex)
        
    def compute_loop_constants(self, loop_bandwidth, damping_factor, k0, kp):
        """
        Compute the loop filter gains based on the loop bandwidth and damping factor.

        :param loop_bandwidth: Float type. Loop bandwidth of control loop.
        :param damping_factor: Float type. Damping factor of control loop.
        :param samples_per_symbol: Int type. Nukber of samples per signal.
        :param k0: Float type. TBD.
        :param kp: Float type. TBD.
        """
        theta_n = (loop_bandwidth*(1/self.samples_per_symbol)/self.samples_per_symbol)/(damping_factor + 1/(4*damping_factor))
        factor = (-4*theta_n)/(1+2*damping_factor*theta_n+theta_n**2)
        self.k1 = damping_factor * factor/kp
        self.k2 = theta_n * factor/kp
        self.k0 = k0
        self.kp = kp

    def get_timing_error(self):
        """
        Get the recorded timing errors.
        
        :return: List type. Recorded timing errors.
        """
        return self.timing_error_record

    def get_loop_filter_record(self):
        """
        Get the recorded loop filter outputs.
        
        :return: List type. Recorded loop filter outputs.
        """
        return self.loop_filter_record
    
    def interpolate(self, input_signal):
        """
        Perform interpolation on an upsampled signal.

        :param input_signal: Numpy array type. Input signal (already upsampled with zeros).
        :param upsample_rate: Int type. Upsample rate of interpolation.
        :return: Numpy array type. Interpolated signal.
        """
        nonzero_indices = np.arange(0, len(input_signal), self.upsample_rate)
        nonzero_values = input_signal[nonzero_indices]
        interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind="linear", fill_value='extrapolate')
        new_indices = np.arange(len(input_signal))
        interpolated_signal = interpolation_function(new_indices)
        return interpolated_signal

    def upsample(self, input_signal):
        """
        Discrete signal upsample implementation.

        :param input_signal: List or Numpy array type. Input signal.
        :return: Numpy array type. Upsampled signal.
        """
        input_signal_upsampled = []
        for i, sample in enumerate(input_signal):
            input_signal_upsampled.append(float(sample))
            input_signal_upsampled.extend([0] * (self.upsample_rate - 1))

        input_signal_upsampled = self.interpolate(np.array(input_signal_upsampled))
        return np.array(input_signal_upsampled) 

    def loop_filter(self, timing_error):
        """
        Loop filter implementation.
        
        :param timing_error: Float type. The current timing error.
        :return: Float type. The output of the loop filter.
        """
        k2 = self.k2 * timing_error + self.k2_prev
        output = self.k1 * timing_error + k2
        self.k2_prev = k2
        self.loop_filter_record.append(output)
        return output

    def early_late_ted(self):
        """
        Early-late Timing Error Detector (TED) implementation.
        
        :return: Float type. The calculated timing error.
        """
        timing_error = (self.adjusted_symbol_block_real[1] * 
                        (self.adjusted_symbol_block_real[2] - self.adjusted_symbol_block_real[0]) + 
                        self.adjusted_symbol_block_imag[1] * 
                        (self.adjusted_symbol_block_imag[2] - self.adjusted_symbol_block_imag[0]))
        self.timing_error_record.append(timing_error)
        if timing_error >= 1:
            timing_error = 0.99
        return timing_error

    def runner(self):
        '''
        Loop through all symbols in received array and correct clock skew offsets.

        :return: Numpy array type. Synchronized signal.
        '''
        counter = self.samples_per_symbol # start at sample time
        for i in range(len(self.input_signal_real)):
            if counter == self.samples_per_symbol:
                if i == 0: # edge condition (start)
                    symbol_block_real = np.concatenate((np.zeros(1), self.input_signal_real[:2])) # [early, on-time, late]
                    symbol_block_imag = np.concatenate((np.zeros(1), self.input_signal_imag[:2]))
                elif i == len(self.input_signal_real) - 1: # edge condition (end)
                    symbol_block_real = np.concatenate((self.input_signal_real[-2:], np.zeros(1))) # [early, on-time, late]
                    symbol_block_imag = np.concatenate((self.input_signal_imag[-2:], np.zeros(1)))
                else:
                    symbol_block_real = self.input_signal_real[i-1:i+2] # [early, on-time, late]
                    symbol_block_imag = self.input_signal_imag[i-1:i+2]

                timing_error = self.early_late_ted()
                loop_filter_output = self.loop_filter(timing_error)

                symbol_block_real_interpolated = self.upsample(symbol_block_real)
                symbol_block_imag_interpolated = self.upsample(symbol_block_imag)

                on_time_index = self.upsample_rate + int(timing_error * self.upsample_rate) # on-time in upsampled block

                self.scs_output[i] = symbol_block_real_interpolated[on_time_index] + 1j * symbol_block_imag_interpolated[on_time_index]

                self.adjusted_symbol_block_real = [
                    float(symbol_block_real_interpolated[on_time_index - self.upsample_rate]),
                    float(symbol_block_real_interpolated[on_time_index]),
                    float(symbol_block_real_interpolated[on_time_index + self.upsample_rate])
                ]

                self.adjusted_symbol_block_imag = [
                    float(symbol_block_imag_interpolated[on_time_index - self.upsample_rate]),
                    float(symbol_block_imag_interpolated[on_time_index]),
                    float(symbol_block_imag_interpolated[on_time_index + self.upsample_rate])
                ]
                
                counter = 0 # reset counter
            counter += 1
        return self.scs_output