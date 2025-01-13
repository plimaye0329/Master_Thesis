import sys
from scipy.interpolate import interp1d
import psrchive
import numpy as np
from scipy.stats import kurtosis
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import argparse
import subprocess

# Define a Gaussian function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Define a function to fit multiple Gaussians
def multi_gaussian(x, *params):
    n = len(params) // 3
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(n):
        amp = params[i * 3]
        mean = params[i * 3 + 1]
        stddev = params[i * 3 + 2]
        result += gaussian(x, amp, mean, stddev)
    return result

# Function to calculate FWHM (Full Width at Half Maximum) and intercepts
def calculate_fwhm(amp, mean, stddev):
    left_intercept = mean - stddev * np.sqrt(2 * np.log(2))
    right_intercept = mean + stddev * np.sqrt(2 * np.log(2))
    fwhm = right_intercept - left_intercept
    return fwhm, left_intercept, right_intercept

# Function to perform iterative Gaussian fitting
def iterative_gaussian_fitting(time_series, x, chi_square_threshold=0.01, min_stddev=0.5, sn_threshold=7):
    fitted_params = []
    current_residual = time_series
    num_gaussians = 0
    chi_square = float('inf')

    noise_level = np.std(time_series[0:50])

    while chi_square > chi_square_threshold:
        num_gaussians += 1

        smoothed_residual = gaussian_filter1d(current_residual, sigma=1)
        peaks, properties = find_peaks(smoothed_residual, height=sn_threshold)
        if len(peaks) == 0:
            break

        peak_idx = peaks[np.argmax(smoothed_residual[peaks])]
        guess_amplitude = smoothed_residual[peak_idx]
        guess_mean = x[peak_idx]
        guess_stddev = 1.0

        initial_guess = fitted_params + [guess_amplitude, guess_mean, max(guess_stddev, min_stddev)]
        try:
            popt, _ = curve_fit(multi_gaussian, x, time_series, p0=initial_guess)
        except RuntimeError:
            break

        fitted_signal = multi_gaussian(x, *popt)
        residual = time_series - fitted_signal
        chi_square = np.sum((residual / time_series) ** 2)

        fitted_params = popt.tolist()
        current_residual = residual

    fwhm_values = []
    intercepts = []
    if len(fitted_params) % 3 == 0:
        for i in range(len(fitted_params) // 3):
            amp = fitted_params[i * 3]
            mean = fitted_params[i * 3 + 1]
            stddev = fitted_params[i * 3 + 2]
            fwhm, left, right = calculate_fwhm(amp, mean, stddev)
            fwhm_values.append((fwhm, left, right))
            intercepts.append((left, right))

    leftmost_intercept = min([intercept[0] for intercept in intercepts], default=None)
    rightmost_intercept = max([intercept[1] for intercept in intercepts], default=None)
    overall_width = rightmost_intercept - leftmost_intercept if leftmost_intercept and rightmost_intercept else None

    return {
        "fitted_params": fitted_params,
        "fwhm_values": fwhm_values,
        "overall_width": overall_width,
        "leftmost_intercept": leftmost_intercept,
        "rightmost_intercept": rightmost_intercept
    }

# Function to extract MJD value from header command output
def extract_mjd_from_header(header_cmd):
    result = subprocess.run(header_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if "Time stamp of first sample" in line:
            mjd_value = line.split(":")[1].strip()
            return float(mjd_value)
    return None  # Return None if MJD is not found
def load_psrchive(fname, nchan, dm):
    archive = psrchive.Archive_load(fname)
    archive.fscrunch_to_nchan(nchan)
    archive.bscrunch_to_nbin(2048)
    archive.pscrunch()
    archive.remove_baseline()
    archive.set_dispersion_measure(dm)
    archive.dedisperse()
    weights = archive.get_weights().squeeze()

    waterfall = np.ma.masked_array(archive.get_data().squeeze())
    waterfall[weights == 0] = np.ma.masked
    f_channels = np.array([
        archive.get_first_Integration().get_centre_frequency(i) \
        for i in range(archive.get_nchan())])
    t_res = archive.get_first_Integration().get_duration() \
        / archive.get_nbin()

    if archive.get_bandwidth() < 0:
        waterfall = np.flipud(waterfall)
        f_channels = f_channels[::-1]

    return waterfall, f_channels, t_res, weights

def main():
    parser = argparse.ArgumentParser(description='Generate a burst plot with Gaussian fitting from a PSRCHIVE file.')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the PSRCHIVE file')
    parser.add_argument('-n', '--nchan', type=int, required=True, help='Number of frequency channels to scrunch to')
    parser.add_argument('-d', '--dm', type=float, required=True, help='Dispersion measure to set')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output PNG filename')

    args = parser.parse_args()

    # Extract MJD from the header
    header_cmd = ["header", args.filename]  # Assuming the header command with the filename is the correct one
    mjd_value = extract_mjd_from_header(header_cmd) # Start MJD of the burst archive (important reference for measuring the burst MJD)
    if mjd_value is not None:
        print(f"Extracted MJD value: {mjd_value}")
    else:
        print("MJD value could not be extracted.")

    waterfall, f_channels, t_res, weights = load_psrchive(args.filename, args.nchan, args.dm)
    mask = np.count_nonzero(weights == 0)
    block_size = (1, 1)
    waterfall_reduced = block_reduce(waterfall, block_size=block_size, func=np.mean)
    t_res_reduced = t_res * block_size[1]
    f_channels_reduced = f_channels[::block_size[0]]

    time_series = np.average(waterfall_reduced[~np.isnan(waterfall_reduced[:, 0])], axis=0)
    time = np.arange(waterfall_reduced.shape[1])  # * t_res_reduced

    results = iterative_gaussian_fitting(time_series, time, sn_threshold=np.std(time_series[0:50] * 3))
    leftmost_intercept = int(results["leftmost_intercept"])
    rightmost_intercept = int(results["rightmost_intercept"])

    # Calculating Isotropic Burst Energy:
    BW = 650 * 1e6 - (mask * 1e6)  # Bandwidth of UBB Band3 (in Hz)
    print('bandwidth is', BW * 1e-6, 'MHz')
    l = 1.882e27  # Distance to the source (in cm) 610Mpc
    z = 0.13  # Source redshift

    fluence = (np.sum(time_series[leftmost_intercept:rightmost_intercept]) * t_res_reduced) / 1000  # Fluence in Jy-s
    energy = (fluence * BW * 10e-23 * 4 * np.pi * l ** 2) / (1 + z)  # Isotropic Energy (in ergs)
    print('Burst isotropic energy is', energy, 'ergs')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), gridspec_kw={'height_ratios': [1, 3]})
    ax1.plot(time, time_series, color='black', label='Time Series')

    if results["fitted_params"]:
        fitted_y = multi_gaussian(time, *results["fitted_params"])
        peak_flux = np.max(fitted_y)  # Saved in units of mJy
        print(f"Peak Y-Value of the Gaussian Fit: {peak_flux:.3f}")
        ax1.plot(time, multi_gaussian(time, *results["fitted_params"]), color='red', label='Gaussian Fit')
        if results["leftmost_intercept"] and results["rightmost_intercept"]:
            ax1.axvline(results["leftmost_intercept"], color='red', linestyle='dotted', label=f"Width: {results['overall_width'] * t_res_reduced * 1e3:.2f} ms")
            ax1.axvline(results["rightmost_intercept"], color='red', linestyle='dotted')

    ax1.set_ylabel('Flux (mJy)')
    ax1.legend(loc='upper right')

    im = ax2.imshow(waterfall_reduced, aspect='auto', cmap='plasma', origin='lower',
                    extent=[time[0], time[-1], f_channels_reduced.min(), f_channels_reduced.max()],
                    vmin=np.nanpercentile(waterfall_reduced, 5), vmax=np.nanpercentile(waterfall_reduced, 95))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (MHz)')
    plt.tight_layout()

    plt.savefig(args.output)
    plt.show()

    burst_width = results['overall_width'] * t_res_reduced * 1e3  # Saved in units of milliseconds

    burst_mjd_val = mjd_value + (0.5/86400)
    burst_mjd_val = np.array(burst_mjd_val)


    peak_flux = np.array(peak_flux)
    burst_width = np.array(burst_width)
    energy = np.array(energy)

    burst_params = np.array([[burst_mjd_val, peak_flux, burst_width, energy]])

    filepath = 'burst_properties.txt'
    np.savetxt(filepath, burst_params, fmt=['%.8f','%d', '%d', '%.4e'], delimiter='\t')

if __name__ == "__main__":
    main()


