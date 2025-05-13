import os
from scipy.ndimage import gaussian_filter1d
#from kneed import KneeLocator
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
from matplotlib.backends.backend_pdf import PdfPages

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
            popt, pcov = curve_fit(multi_gaussian, x, time_series, p0=initial_guess)
        except RuntimeError:
            break

        fitted_signal = multi_gaussian(x, *popt)
        residual = time_series - fitted_signal
        chi_square = np.sum((residual / time_series) ** 2)

        fitted_params = popt.tolist()
        fitted_cov = pcov
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

    uncertainties = None
    if 'fitted_cov' in locals() and fitted_cov is not None:
        uncertainties = np.sqrt(np.diag(fitted_cov))

    # Extract the MJD of each Gaussian component
    gaussian_mjds = [fitted_params[i * 3 + 1] for i in range(len(fitted_params) // 3)]

    return {
        "fitted_params": fitted_params,
        "fwhm_values": fwhm_values,
        "overall_width": overall_width,
        "leftmost_intercept": leftmost_intercept,
        "rightmost_intercept": rightmost_intercept,
        "uncertainties": uncertainties,
        "gaussian_mjds": gaussian_mjds  # Add this to the return values
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


def group_non_overlapping_gaussians(params, threshold=5):
    """
    Group Gaussian components into non-overlapping bursts based on their means and sigmas.
    :param params: Flattened list of Gaussian parameters [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    :param threshold: Minimum separation in sigma units to consider components as non-overlapping
    :return: List of lists, where each inner list contains indices of one group
    """
    if not params:
        return []

    groups = []
    current_group = [0]
    prev_mu = params[1]
    prev_sigma = params[2]

    for i in range(3, len(params), 3):
        curr_mu = params[i + 1]
        curr_sigma = params[i + 2]

        # If current mean is within threshold * sigma of previous, consider it overlapping
        if abs(curr_mu - prev_mu) < threshold * (prev_sigma + curr_sigma) / 2:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

        prev_mu = curr_mu
        prev_sigma = curr_sigma

    groups.append(current_group)
    return groups





def process_files(files, nchan, dm, output_prefix):
    results_summary = []
    gaussian_mjd_list = []

    pdf_pages = PdfPages(f"{output_prefix}_plots.pdf")
    error_log = open(f"{output_prefix}_error.log", "w")

    for filename in files:
        burst_code = filename.split('_')[-1].split('.')[0]
        header_cmd = ["header", filename]
        mjd_value = extract_mjd_from_header(header_cmd)

        if mjd_value is None:
            print(f"MJD value could not be extracted for {filename}.")
            error_log.write(f"{filename}: MJD extraction failed\n")
            continue

        try:
            waterfall, f_channels, t_res, weights = load_psrchive(filename, nchan, dm)
        except Exception as e:
            print(f"Error loading data for {filename}: {e}")
            error_log.write(f"{filename}: Data loading failed - {e}\n")
            continue

        block_size = (1, 1)
        waterfall_reduced = block_reduce(waterfall, block_size=block_size, func=np.mean)
        t_res_reduced = t_res * block_size[1]
        f_channels_reduced = f_channels[::block_size[0]]

        time_series = np.average(waterfall_reduced[~np.isnan(waterfall_reduced[:, 0])], axis=0)
        time = np.arange(waterfall_reduced.shape[1])

        zoom_width = 200
        center_time_index = len(time) // 2
        zoom_start = max(0, center_time_index - zoom_width)
        zoom_end = min(len(time), center_time_index + zoom_width)

        zoomed_time_series = time_series[zoom_start:zoom_end]
        zoomed_time = time[zoom_start:zoom_end]

        try:
            results = iterative_gaussian_fitting(zoomed_time_series, zoomed_time, sn_threshold=np.std(zoomed_time_series[0:50]) * 3)
        except Exception as e:
            print(f"Error fitting Gaussian model for {filename}: {e}")
            error_log.write(f"{filename}: Gaussian fitting failed - {e}\n")
            continue

        try:
            leftmost_intercept = int(results["leftmost_intercept"])
            rightmost_intercept = int(results["rightmost_intercept"])
        except (TypeError, ValueError) as e:
            print(f"Warning: Unable to extract valid intercepts for {filename}. Skipping intercept-based calculations.")
            error_log.write(f"{filename}: Intercept extraction failed - {e}\n")
            continue

        grouped_indices = group_non_overlapping_gaussians(results["fitted_params"])

        for group_id, indices in enumerate(grouped_indices):
            sub_params = []
            for idx in indices:
                sub_params.extend(results["fitted_params"][idx:idx + 3])

            try:
                sub_results = {
                    "fitted_params": sub_params,
                    "leftmost_intercept": leftmost_intercept,
                    "rightmost_intercept": rightmost_intercept,
                    "overall_width": results.get("overall_width", None)
                }
            except Exception as e:
                error_log.write(f"{filename}: Sub-burst intercept extraction failed - {e}\n")
                continue

            gaussian_mjds_for_burst = []
            for i in range(0, len(sub_params), 3):
                mean = sub_params[i + 1]
                gaussian_mjd = (mean * t_res_reduced / 86400) + mjd_value
                gaussian_mjds_for_burst.append(gaussian_mjd)

            gaussian_mjd_list.append(gaussian_mjds_for_burst)

            BW = f_channels_reduced.max() - f_channels_reduced.min()
            l = 1.882e27  # cm
            z = 0.13

            
            # Calculate local time bounds for this group
            group_means = [sub_params[i + 1] for i in range(0, len(sub_params), 3)]
            group_sigmas = [sub_params[i + 2] for i in range(0, len(sub_params), 3)]
            group_min = int(max(0, np.floor(min(group_means) - 3 * max(group_sigmas))))
            group_max = int(min(len(time_series), np.ceil(max(group_means) + 3 * max(group_sigmas))))

# Fluence localized to this group
            if group_max > group_min:
                fluence = (np.sum(time_series[group_min:group_max]) * t_res_reduced) / 1000  # Jy * ms = Jy-s
            else:
                fluence = np.nan

# Width = weighted mean sigma
            if group_sigmas:
                burst_width = np.average(group_sigmas) * t_res_reduced * 1e3  # ms
            else:
                burst_width = np.nan

            burst_width_uncertainty = burst_width * 0.1 if not np.isnan(burst_width) else np.nan
            burst_mjd_val = mjd_value + (0.5 / 86400)
            peak_flux = np.max(multi_gaussian(time, *sub_params)) if sub_params else np.nan

            results_summary.append([burst_mjd_val, peak_flux, fluence, burst_width, f"{burst_code}_g{group_id}"])

            np.savez(f"{output_prefix}_{burst_code}_g{group_id}.npz", dynamic_spectrum=waterfall_reduced,
                     f_channels=f_channels_reduced, t_res=t_res_reduced)

            left_idx = max(0, int(leftmost_intercept))
            right_idx = min(len(time), int(rightmost_intercept))
            freq_response = np.nanmean(waterfall_reduced[:, left_idx:right_idx], axis=1)
            freq_response[weights == 0] = np.nan
            valid_idx = ~np.isnan(freq_response)
            filtered_freqs = f_channels_reduced[valid_idx]
            filtered_response = freq_response[valid_idx]

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), gridspec_kw={'height_ratios': [1, 3]})
            ax1.plot(zoomed_time, zoomed_time_series, color='black', label='Time Series')
            # Mark burst width (intercepts)
            intercept_start_time = time[group_min]
            intercept_end_time = time[group_max]

            ax1.axvline(x=intercept_start_time, color='blue', linestyle='--', linewidth=1.5, label='Burst Start')
            ax1.axvline(x=intercept_end_time, color='blue', linestyle='--', linewidth=1.5, label='Burst End')

            im = ax2.imshow(waterfall_reduced[:, zoom_start:zoom_end], aspect='auto', cmap='plasma', origin='lower',
                            extent=[time[zoom_start], time[zoom_end], f_channels_reduced.min(), f_channels_reduced.max()],
                            vmin=np.nanpercentile(waterfall_reduced, 5), vmax=np.nanpercentile(waterfall_reduced, 95))
            ax2.set_xlabel('Time (samples)')
            ax2.set_ylabel('Frequency (MHz)')
            fig.suptitle(f"{output_prefix}_{burst_code}_g{group_id}.npz", fontsize=14)
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

    pdf_pages.close()
    error_log.close()

    max_gaussian_components = max(len(mjds) for mjds in gaussian_mjd_list)
    padded_gaussian_mjds = [mjds + [np.nan] * (max_gaussian_components - len(mjds)) for mjds in gaussian_mjd_list]
    gaussian_mjd_array = np.array(padded_gaussian_mjds, dtype=float)
    np.savetxt(f"{output_prefix}_gaussian_mjds.txt", gaussian_mjd_array, fmt='%.8f', delimiter='\t')

    burst_properties = np.array(results_summary, dtype=object)
    header = 'Burst_MJD\tPeak_Flux(mJy)\tFluence(Jy-s)\tWidth(ms)\tBurst_Code'
    np.savetxt(f"{output_prefix}_burst_properties.txt", burst_properties, fmt=['%.8f', '%4f', '%.8f', '%.4f', '%s'], header=header, delimiter='\t')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple PSRCHIVE files for burst analysis.')
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True, help='List of PSRCHIVE files')
    parser.add_argument('-n', '--nchan', type=int, required=True, help='Number of frequency channels to scrunch to')
    parser.add_argument('-d', '--dm', type=float, required=True, help='Dispersion measure to set')
    parser.add_argument('-o', '--output_prefix', type=str, required=True, help='Output prefix for results')

    args = parser.parse_args()
    process_files(args.files, args.nchan, args.dm, args.output_prefix)
