import os
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator
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







def process_files(files, nchan, dm, output_prefix):
    results_summary = []
    gaussian_mjd_list = []  # Store the MJD for each Gaussian component for each burst

    pdf_pages = PdfPages(f"{output_prefix}_plots.pdf")
    error_log = open(f"{output_prefix}_error.log", "w")  # Open a log file to record skipped files

    for filename in files:
        burst_code = filename.split('_')[-1].split('.')[0]  # Extract burst code

        # Extract MJD from the header (this is calculated per file)
        header_cmd = ["header", filename]
        mjd_value = extract_mjd_from_header(header_cmd)
        if mjd_value is None:
            print(f"MJD value could not be extracted for {filename}.")
            error_log.write(f"{filename}: MJD extraction failed\n")  # Log the skipped file
            continue

        try:
            waterfall, f_channels, t_res, weights = load_psrchive(filename, nchan, dm)
        except Exception as e:
            print(f"Error loading data for {filename}: {e}")
            error_log.write(f"{filename}: Data loading failed - {e}\n")  # Log the skipped file
            continue

        mask = np.count_nonzero(weights == 0)
        block_size = (1, 1)
        waterfall_reduced = block_reduce(waterfall, block_size=block_size, func=np.mean)
        #mean_per_row = np.mean(waterfall_reduced, axis=1, keepdims=True)
        #std_per_row = np.std(waterfall_reduced, axis=1, keepdims=True)
        #std_per_row[std_per_row == 0] = 1  # Avoid division by zero

        #waterfall_reduced = (waterfall_reduced - mean_per_row) / std_per_row


        t_res_reduced = t_res * block_size[1]
        f_channels_reduced = f_channels[::block_size[0]]

        time_series = np.average(waterfall_reduced[~np.isnan(waterfall_reduced[:, 0])], axis=0)
        time = np.arange(waterfall_reduced.shape[1])

        # Define the zoom window around the center of the time axis
        zoom_width = 200  # Define how many samples to zoom around the center (fixed range)
        center_time_index = len(time) // 2  # Find the center of the time axis
        zoom_start = max(0, center_time_index - zoom_width)
        zoom_end = min(len(time), center_time_index + zoom_width)

        # Only apply Gaussian fitting to the zoomed portion
        zoomed_time_series = time_series[zoom_start:zoom_end]
        zoomed_time = time[zoom_start:zoom_end]

        try:
            results = iterative_gaussian_fitting(zoomed_time_series, zoomed_time, sn_threshold=np.std(zoomed_time_series[0:50] * 3))
        except Exception as e:
            print(f"Error fitting Gaussian model for {filename}: {e}")
            error_log.write(f"{filename}: Gaussian fitting failed - {e}\n")  # Log the skipped file
            continue

        # Attempt to extract intercepts and handle errors gracefully
        try:
            leftmost_intercept = int(results["leftmost_intercept"])
            rightmost_intercept = int(results["rightmost_intercept"])
        except (TypeError, ValueError) as e:
            print(f"Warning: Unable to extract valid intercepts for {filename}. Skipping intercept-based calculations.")
            leftmost_intercept = None
            rightmost_intercept = None
            error_log.write(f"{filename}: Intercept extraction failed - {e}\n")  # Log the skipped file
            continue

        # Store MJD of Gaussian component peaks for each burst
        gaussian_mjds_for_burst = []
        for i in range(0, len(results["fitted_params"]), 3):  # Loop through Gaussian components
            mean = results["fitted_params"][i + 1]  # Extract the mean (time center) of each Gaussian

            # Convert the mean (in samples) to MJD using the current mjd_value of the file
            gaussian_mjd = (mean * t_res_reduced / 86400) + mjd_value
            gaussian_mjds_for_burst.append(gaussian_mjd)

        gaussian_mjd_list.append(gaussian_mjds_for_burst)
        BW =f_channels_reduced.max() -  f_channels_reduced.min()
        print(t_res_reduced)
        # Calculate isotropic burst energy
        l = 1.882e27 # Source Distance in centimeters
        z = 0.13 # Source redshift
        # Check if intercepts are available before calculating fluence
        if leftmost_intercept is not None and rightmost_intercept is not None:
            fluence = (np.sum(time_series[leftmost_intercept:rightmost_intercept]) * t_res_reduced) / 1000
        else:
            fluence = np.nan
        #fluence_uncertainty = fluence * 0.1  # 10% uncertainty as an example
        energy = (fluence * BW * 10e-23 * 4 * np.pi * l ** 2) / (1 + z) if not np.isnan(fluence) else np.nan
        #energy_uncertainty = (fluence_uncertainty * BW * 10e-23 * 4 * np.pi * l ** 2) / (1 + z) if not np.isnan(fluence_uncertainty) else np.nan

        # Save burst parameters, check for 'overall_width' before using it
        try:
            burst_width = results['overall_width'] * t_res_reduced * 1e3 if results.get('overall_width') is not None else np.nan
        except TypeError:
            burst_width = np.nan  # Handle the case where 'overall_width' is None
        burst_width_uncertainty = burst_width * 0.1 if not np.isnan(burst_width) else np.nan
        burst_mjd_val = mjd_value + (0.5 / 86400)
        peak_flux = np.max(multi_gaussian(time, *results["fitted_params"])) if results["fitted_params"] else np.nan

        # Append results for saving
        #results_summary.append([burst_mjd_val, peak_flux, fluence, burst_width, energy, burst_code])

        # Save the dynamic spectrum
        np.savez(f"{output_prefix}_{burst_code}.npz", dynamic_spectrum=waterfall_reduced, f_channels=f_channels_reduced, t_res=t_res_reduced)
        


        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        # Compute intensity as a function of frequency (average over time)
        
            
        left_idx = max(0, int(leftmost_intercept))
        right_idx = min(len(time), int(rightmost_intercept))

    # Average over time *only* within the burst width window
        freq_response = np.nanmean(waterfall_reduced[:, left_idx:right_idx], axis=1)

# Replace zero-weighted channels with NaN
        freq_response[weights == 0] = np.nan

# Mask out NaN values
        valid_idx = ~np.isnan(freq_response)
        filtered_freqs = f_channels_reduced[valid_idx]
        filtered_response = freq_response[valid_idx]

# Debugging output
        print(f"Total channels: {len(weights)}")
        print(f"Non-zero channels: {np.count_nonzero(weights)}")
        print(f"Valid channels after NaN filtering: {len(filtered_freqs)}")
        print(f"Valid channels before  NaN filtering: {len(f_channels_reduced)}")





        # Step 1: Interpolate over NaN values to handle misleading flat regions
        # Step 1: Interpolate over NaN values to handle misleading flat regions
        nan_mask = np.isnan(filtered_response)
        valid_indices = np.where(~nan_mask)[0]
        filtered_response[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_indices, filtered_response[valid_indices])

# Step 2: Calculate the Cumulative Distribution Function (CDF)
        cumulative_response = np.cumsum(filtered_response)
        cumulative_response /= cumulative_response[-1]  # Normalize the CDF to [0, 1]
        cumulative_response = gaussian_filter1d(cumulative_response, sigma=5)  # Gaussian 1D smoothing

# Step 3: Compute the first derivative of the CDF to detect rises and flattening
        cdf_derivative = np.gradient(cumulative_response, filtered_freqs)

# Step 4 (Modified): Identify frequency indices where CDF equals 0.5
        cdf_midpoint_index = np.argmin(np.abs(cumulative_response - 0.5))

# Determine lower bound
        lower_region = cdf_derivative[:cdf_midpoint_index]
        if np.all(lower_region > 0):
            rise_index = 0  # Set to minimum frequency if always increasing
        else:
            rise_index = np.where(lower_region <= 0)[0][-1] + 1  # Last point before non-positive slope

# Determine upper bound
        upper_region = cdf_derivative[cdf_midpoint_index:]
        if np.all(upper_region > 0):
            drop_index = len(filtered_freqs) - 1  # Set to max frequency if always increasing
        else:
            drop_index = np.where(upper_region <= 0)[0][0] + cdf_midpoint_index  # First point where slope stops increasing

# Step 6: Calculate the frequency bounds and bandwidth
        lower_bound_freq = filtered_freqs[rise_index]
        upper_bound_freq = filtered_freqs[drop_index]
        bandwidth = upper_bound_freq - lower_bound_freq

        fluence = fluence / (np.sqrt(bandwidth/BW))
        energy = energy*(bandwidth/BW)

        results_summary.append([burst_mjd_val, peak_flux, fluence, burst_width, energy, lower_bound_freq, upper_bound_freq, bandwidth, burst_code,])
        


# Step 7: Plot the CDF with the updated bounds
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_freqs, cumulative_response, label='CDF of Intensity Response', color='b')
        plt.axvline(lower_bound_freq, color='r', linestyle='--', label=f'Lower Bound: {lower_bound_freq:.3f} MHz (Rise)')
        plt.axvline(upper_bound_freq, color='g', linestyle='--', label=f'Upper Bound: {upper_bound_freq:.3f} MHz (Drop/Flat)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Normalized Cumulative Intensity')
        #plt.title('CDF of Burst Intensity with Corrected Bandwidth Markers')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig('cdf_response.png',dpi=300,bbox_inches='tight')

# Step 8: Print calculated bandwidth
        print(f"Lower Bound Frequency (Significant Rise): {lower_bound_freq:.3f} MHz")
        print(f"Upper Bound Frequency (Significant Drop/Flat): {upper_bound_freq:.3f} MHz")
        print(f"Estimated Burst Bandwidth: {bandwidth:.3f} MHz")






# Plot frequency response and Gaussian fit
        #ax.scatter(filtered_freqs, filtered_response, color='red', label='Freq Response')  # Show all channels (including NaNs)

        # Generate and save the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), gridspec_kw={'height_ratios': [1, 3]})
        #freq_response = np.nanmean(waterfall_reduced[:,burst_width], axis=1)  # Average along time axis

        # Time series plot (zoomed in)
        ax1.plot(zoomed_time, zoomed_time_series, color='black', label='Time Series')



        # Dynamic spectrum plot (zoomed in)
        im = ax2.imshow(waterfall_reduced[:, zoom_start:zoom_end], aspect='auto', cmap='plasma', origin='lower',
                        extent=[time[zoom_start], time[zoom_end], f_channels_reduced.min(), f_channels_reduced.max()],
                        vmin=np.nanpercentile(waterfall_reduced, 5), vmax=np.nanpercentile(waterfall_reduced, 95))
        ax2.axhline(y = upper_bound_freq,linewidth=3.0,color='white',label='Frequency Coverage')
        ax2.axhline(y = lower_bound_freq,linewidth=3.0,color='white')

        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Frequency (MHz)')
        

        # Set the title with the .npz filename
        npz_filename = f"{output_prefix}_{burst_code}.npz"
        fig.suptitle(npz_filename, fontsize=14)

        plt.tight_layout()

        # Save the plot to the PDF
        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()
    error_log.close()  # Close the error log file when done

    # After processing all files, pad and save Gaussian MJDs
    # Calculate the maximum number of components across all bursts
    max_gaussian_components = max(len(mjds) for mjds in gaussian_mjd_list)

    # Pad the Gaussian MJDs for each burst to match the maximum number of components
    padded_gaussian_mjds = []
    for mjds in gaussian_mjd_list:
        padded_mjds = mjds + [np.nan] * (max_gaussian_components - len(mjds))  # Pad with NaN
        padded_gaussian_mjds.append(padded_mjds)

    # Convert to numpy array
    gaussian_mjd_array = np.array(padded_gaussian_mjds, dtype=float)

    # Save to text file
    np.savetxt(f"{output_prefix}_gaussian_mjds.txt", gaussian_mjd_array, fmt='%.8f', delimiter='\t')

    # Save burst properties to a text file (with NaNs where calculations failed)
    burst_properties = np.array(results_summary, dtype=float)  # Ensure all elements are numeric (float)
    header = 'Burst_MJD\tPeak_Flux(mJy)\tFluence(Jy-s)\tWidth(ms)\tEnergy(ergs)\tLower_Frequency(MHz)\tUpper_Frequency(MHz)\tOverall_Bandwidth(MHz)\tBurst_Code'
    np.savetxt(f"{output_prefix}_burst_properties.txt", burst_properties, fmt=['%.8f', '%4f','%.8f', '%.4f','%.4e', '%.4f', '%.4f','%.4f', '%s'], header=header, delimiter='\t')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple PSRCHIVE files for burst analysis.')
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True, help='List of PSRCHIVE files')
    parser.add_argument('-n', '--nchan', type=int, required=True, help='Number of frequency channels to scrunch to')
    parser.add_argument('-d', '--dm', type=float, required=True, help='Dispersion measure to set')
    parser.add_argument('-o', '--output_prefix', type=str, required=True, help='Output prefix for results')

    args = parser.parse_args()
    process_files(args.files, args.nchan, args.dm, args.output_prefix)

