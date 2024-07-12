import numpy as np
from scipy import interpolate as intp

class SCS():
    def __init__(self, input_signal_complex, samples_per_symbol, interpolation_factor):
        """
        Initialize the SCS object for symbol synchronization.
        
        :param input_signal_complex: Complex array. The input signal to be processed.
        :param samples_per_symbol: Int. Number of samples per symbol.
        :param interpolation_factor: Int. Interpolation factor for upsampling.
        """
        self.input_signal_real = np.real(input_signal_complex)
        self.input_signal_imag = np.imag(input_signal_complex)
        self.samples_per_symbol = samples_per_symbol
        self.counter = samples_per_symbol  # Start at sample
        self.timing_error = 0
        self.interpolation_factor = interpolation_factor

        self.loop_bandwidth = 0.2 * samples_per_symbol
        self.damping_factor = 1 / np.sqrt(2)
        theta_n = self.loop_bandwidth * (1 / self.samples_per_symbol) / (self.damping_factor + 1 / (4 * self.damping_factor))
        self.K1 = self.damping_factor
        self.K2 = theta_n
        self.LFK2Prev = 0

        self.adjusted_symbol_block_real = [0, 0, 0]
        self.adjusted_symbol_block_imag = [0, 0, 0]

        self.timing_error_record = []
        self.scs_output_real = []
        self.scs_output_imag = []

    def interpolate(self, x, n, mode="linear"):
        """
        Perform interpolation on an upsampled signal.

        :param x: Input signal (already upsampled with zeros).
        :param n: Upsampled factor.
        :param mode: Interpolation type. Modes = "linear", "quadratic".
        :return: Interpolated signal.
        """
        nonzero_indices = np.arange(0, len(x), n)
        nonzero_values = x[nonzero_indices]
        interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind=mode, fill_value='extrapolate')
        new_indices = np.arange(len(x))
        interpolated_signal = interpolation_function(new_indices)
        return interpolated_signal

    def upsample(self, x, L, offset=0, interpolate_flag=True):
        """
        Discrete signal upsample implementation.

        :param x: List or Numpy array type. Input signal.
        :param L: Int type. Upsample factor.
        :param offset: Int type. Offset size for input array.
        :param interpolate: Boolean type. Flag indicating whether to perform interpolation.
        :return: Numpy array type. Upsampled signal.
        """
        x_upsampled = [0] * offset  # Initialize with offset zeros
        for i, sample in enumerate(x):
            x_upsampled.append(float(sample))
            x_upsampled.extend([0] * (L - 1))
        if interpolate_flag:
            x_upsampled = self.interpolate(np.array(x_upsampled), self.interpolation_factor, mode="linear")

        return np.array(x_upsampled)

    def loop_filter(self, timing_error):
        """
        Loop filter implementation.
        
        :param timing_error: Float. The current timing error.
        :return: Float. The output of the loop filter.
        """
        LFK2 = self.K2 * timing_error + self.LFK2Prev
        lf_output = self.K1 * timing_error + LFK2
        self.LFK2Prev = LFK2
        return lf_output

    def get_timing_error(self):
        """
        Get the recorded timing errors.
        
        :return: List of floats. Recorded timing errors.
        """
        return self.timing_error_record

    def early_late_ted(self):
        """
        Early-late Timing Error Detector (TED) implementation.
        
        :return: Float. The calculated timing error.
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
        """
        Main function to process the input signal for symbol synchronization.
        
        :return: Complex array. The synchronized signal.
        """
        for i in range(len(self.input_signal_real)):
            if self.counter == self.samples_per_symbol:
                # Splice symbol block
                if i == 0:
                    symbol_block_real = np.concatenate((np.zeros(1), self.input_signal_real[i:i+2]))  # Early, on time, late
                    symbol_block_imag = np.concatenate((np.zeros(1), self.input_signal_imag[i:i+2]))
                else:
                    symbol_block_real = self.input_signal_real[i-1:i+2]
                    symbol_block_imag = self.input_signal_imag[i-1:i+2]
            
                self.timing_error = self.early_late_ted()
                loop_filter_output = self.loop_filter(self.timing_error)

                symbol_block_real_interpolated = self.upsample(symbol_block_real, self.interpolation_factor, interpolate_flag=True)
                symbol_block_imag_interpolated = self.upsample(symbol_block_imag, self.interpolation_factor, interpolate_flag=True)

                # print(symbol_block_real_interpolated)

                on_time_index = self.interpolation_factor + int(self.timing_error * self.interpolation_factor)

                self.scs_output_real.append(symbol_block_real_interpolated[on_time_index])
                self.scs_output_imag.append(symbol_block_imag_interpolated[on_time_index])

                self.adjusted_symbol_block_real = [
                    float(symbol_block_real_interpolated[on_time_index - self.interpolation_factor]),
                    float(symbol_block_real_interpolated[on_time_index]),
                    float(symbol_block_real_interpolated[on_time_index + self.interpolation_factor])
                ]

                self.adjusted_symbol_block_imag = [
                    float(symbol_block_imag_interpolated[on_time_index - self.interpolation_factor]),
                    float(symbol_block_imag_interpolated[on_time_index]),
                    float(symbol_block_imag_interpolated[on_time_index + self.interpolation_factor])
                ]

                self.counter = 0  # Reset counter
            self.counter += 1  # Increment counter
        return np.array(self.scs_output_real) + 1j * np.array(self.scs_output_imag)


