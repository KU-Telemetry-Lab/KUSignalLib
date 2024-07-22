import numpy as np
from scipy import interpolate as intp
import matplotlib.pyplot as plt

class SCS:
    def __init__(self, samples_per_symbol, loop_bandwidth=None, damping_factor=None, k0=1, kp=1, k1=0, k2=0, upsample_rate=10):
        '''
        Initialize the SCS (Symbol Clock Synchronization) subsystem class.

        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock-on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock-on to the timing error (similar to PLL).
        :param kp: Float type. Proportional gain.
        :param k0: Float type. Loop gain.
        :param k1: Float type. Loop filter coefficient one.
        :param k2: Float type. Loop filter coefficient two.
        :param upsample_rate: Int type. Upsample rate of timing error correction interpolation.
        '''
        self.samples_per_symbol = samples_per_symbol
        self.upsample_rate = upsample_rate

        self.k2_prev = 0
        if loop_bandwidth is None and damping_factor is None:
            self.kp = kp
            self.k0 = k0
            self.k1 = k1
            self.k2 = k2
        else:
            self.compute_loop_constants(loop_bandwidth, damping_factor, k0, kp)
        
        self.adjusted_symbol_block = np.zeros(3, dtype=complex)
        self.timing_error_record = []
        self.loop_filter_record = []
        self.counter = samples_per_symbol - 1
        
        self.scs_output_record = []
        
    def compute_loop_constants(self, loop_bandwidth: float, damping_factor: float, k0: float, kp: float):
        """
        Compute the loop filter gains based on the loop bandwidth and damping factor.

        :param loop_bandwidth: Float type. Loop bandwidth of control loop.
        :param damping_factor: Float type. Damping factor of control loop.
        :param k0: Float type. Loop gain.
        :param kp: Float type. Proportional gain.
        """
        theta_n = (loop_bandwidth * (1 / self.samples_per_symbol) / self.samples_per_symbol) / (damping_factor + 1 / (4 * damping_factor))
        factor = (-4 * theta_n) / (1 + 2 * damping_factor * theta_n + theta_n**2)
        self.k1 = damping_factor * factor / kp
        self.k2 = theta_n * factor / kp
        self.k0 = k0
        self.kp = kp

    def get_timing_error_record(self):
        """
        Get the recorded timing errors.
        
        :return: Numpy array type. Recorded timing errors.
        """
        return np.array(self.timing_error_record)

    def get_loop_filter_record(self):
        """
        Get the recorded loop filter outputs.
        
        :return: Numpy array type. Recorded loop filter outputs.
        """
        return np.array(self.loop_filter_record)

    def get_scs_output_record(self):
        """
        Get the recorded SCS outputs.
        
        :return: Numpy array type. Recorded SCS outputs.
        """
        return np.array(self.scs_output_record)

    def interpolate(self, symbol_block, mode='cubic'):
        """
        Discrete signal upsample implementation.

        :param symbol_block: List or Numpy array type. Input signal.
        :param mode: String type. Interpolation mode ('linear' or 'cubic').
        :return: Numpy array type. Upsampled signal.
        """
        if mode == "linear":
            symbol_block_upsampled = np.zeros(len(symbol_block) * self.upsample_rate, dtype=complex)
            for i, sample in enumerate(symbol_block):
                symbol_block_upsampled[i * self.upsample_rate] = sample
            nonzero_indices = np.arange(0, len(symbol_block_upsampled), self.upsample_rate)
            nonzero_values = symbol_block_upsampled[nonzero_indices]
            new_indices = np.arange(len(symbol_block_upsampled))
            interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind="linear", fill_value='extrapolate')
            symbol_block_interpolated = interpolation_function(new_indices)
        elif mode == "cubic":
            symbol_block = np.append(symbol_block, 0)
            interpolation_function = intp.CubicSpline(np.arange(0, len(symbol_block)), symbol_block)
            symbol_block_interpolated = interpolation_function(np.linspace(0, len(symbol_block)-1, num=(len(symbol_block)-1) * self.upsample_rate))
        else:
            symbol_block_interpolated = symbol_block_upsampled
        return symbol_block_interpolated

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
        timing_error = (np.real(self.adjusted_symbol_block[1]) * (np.real(self.adjusted_symbol_block[2]) - np.real(self.adjusted_symbol_block[0])))
        timing_error = timing_error * (1/self.samples_per_symbol)
        self.timing_error_record.append(timing_error)
        return timing_error

    def insert_new_sample(self, complex_early_sample, complex_on_time_sample, complex_late_sample):
        """
        Insert new samples for processing.

        :param complex_early_sample: Complex type. Early sample.
        :param complex_on_time_sample: Complex type. On-time sample.
        :param complex_late_sample: Complex type. Late sample.
        """
        if self.counter == self.samples_per_symbol - 1:
            symbol_block = np.array([complex_early_sample, complex_on_time_sample, complex_late_sample])

            symbol_block_interpolated = self.interpolate(symbol_block, mode='linear')
            timing_error = self.early_late_ted()
            loop_filter_output = self.loop_filter(timing_error)
        
            adjusted_on_time_index = self.upsample_rate + int(loop_filter_output * self.upsample_rate)

            self.adjusted_symbol_block = np.array([
                symbol_block_interpolated[adjusted_on_time_index - self.upsample_rate],
                symbol_block_interpolated[adjusted_on_time_index],
                symbol_block_interpolated[adjusted_on_time_index + self.upsample_rate]
            ], dtype=complex)

            self.counter = 0
            self.scs_output_record.append(symbol_block_interpolated[adjusted_on_time_index])
        else:
            self.counter += 1
