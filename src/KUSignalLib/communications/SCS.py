import numpy as np
from scipy import interpolate as intp

class SCS():
    def __init__(self, samples_per_symbol, loop_bandwidth=None, damping_factor=None, k0=1, kp=1, k1=0, k2=0, upsample_rate=10):
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
        
        self.adjusted_symbol_block = np.zeros(3, dtype=complex)
        self.timing_error_record = []
        self.loop_filter_record = []
        self.counter = samples_per_symbol
        
        self.scs_output_record = []
        
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

    def get_timing_error_record(self):
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

    def get_scs_output_record(self):
        """
        Get the recorded SCS outputs.
        
        :return: List type. Recorded SCS outputs.
        """
        return self.scs_output_record
    
    def interpolate(self, symbol_block_upsampled):
        """
        Perform interpolation on an upsampled signal.

        :param input_signal: Numpy array type. Input signal (already upsampled with zeros).
        :param upsample_rate: Int type. Upsample rate of interpolation.
        :return: Numpy array type. Interpolated signal.
        """
        nonzero_indices = np.arange(0, len(symbol_block_upsampled), self.upsample_rate)
        nonzero_values = symbol_block_upsampled[nonzero_indices]
        interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind="linear", fill_value='extrapolate')
        new_indices = np.arange(len(symbol_block_upsampled))
        interpolated_signal = interpolation_function(new_indices)
        return interpolated_signal

    def upsample(self, symbol_block, interpolate=True):
        """
        Discrete signal upsample implementation.

        :param symbol_block: List or Numpy array type. Input signal.
        :return: Numpy array type. Upsampled signal.
        """
        symbol_block_upsampled = np.zeros(len(symbol_block) * self.upsample_rate, dtype=complex)
        for i, sample in enumerate(symbol_block):
            symbol_block_upsampled[i * self.upsample_rate] = sample
        if interpolate:
            symbol_block_upsampled = self.interpolate(symbol_block_upsampled)
        return symbol_block_upsampled


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
        timing_error = (np.real(self.adjusted_symbol_block[1]) * (np.real(self.adjusted_symbol_block[2]) - np.real(self.adjusted_symbol_block[0])) 
                        + np.imag(self.adjusted_symbol_block[1]) * (np.imag(self.adjusted_symbol_block[2]) - np.imag(self.adjusted_symbol_block[0])))
        self.timing_error_record.append(timing_error)
        if timing_error >= 1:
            timing_error = 0.99
        return timing_error

    def insert_new_sample(self, complex_early_sample, complex_on_time_sample, complex_late_sample):
        # sample by sample implementation
        if self.counter == self.samples_per_symbol:
            symbol_block = np.array([complex_early_sample, complex_on_time_sample, complex_late_sample])

            timing_error = self.early_late_ted()
            loop_filter_output = self.loop_filter(timing_error)

            symbol_block_interpolated = self.upsample(symbol_block, interpolate=True)

            on_time_index = self.upsample_rate + int(timing_error * self.upsample_rate)

            self.adjusted_symbol_block = np.array([
                symbol_block_interpolated[on_time_index - self.upsample_rate],
                symbol_block_interpolated[on_time_index],
                symbol_block_interpolated[on_time_index + self.upsample_rate]
            ], dtype=complex)

            self.counter = 0
            self.scs_output_record.append(symbol_block_interpolated[on_time_index])
        else:
            self.counter += 1