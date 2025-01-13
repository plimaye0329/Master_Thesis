import os
import subprocess
import warnings
import numpy as np
import fitsio as fio
import argparse

class UBBPlotter:
    def __init__(self, filename, burst_mjd, fluxcal_file,dm):
        self.fname = filename
        self.burst_mjd = burst_mjd
        self.fluxcal = fluxcal_file
        self.primary = fio.read_header(self.fname, ext=0)
        self.cfreq = self.primary['OBSFREQ'] # Center Frequency of the Band (MHz)
        self.bw = self.primary['OBSBW'] # Bandwidth of the Band (MHz)
        self.topfreq = self.bw/2 + self.cfreq # Top Frequency of the Band (MHz)
        self.dm = dm
        imjd = self.primary['STT_IMJD']
        smjd = self.primary['STT_SMJD']
        soffs = self.primary['STT_OFFS']
        self.tmjd = imjd + (smjd / 86_400.) + (soffs / 86_400.)
        #self.toa = (self.burst_mjd - self.tmjd) * 86400
        self.delay_seconds = 4.1487416 * 10**6 * ((1 / self.cfreq**2) - (1 / self.topfreq**2)) / 1000 * dm
        self.toa = ((burst_mjd + self.delay_seconds/86400) - self.tmjd)*86400 - 0.5
        self.cepoch = self.tmjd + self.toa/86400
        print(f"Time of Arrival (TOA): {self.toa}")



    def run_dspsr_command(self, dm, bins, output_filename):
        # Construct the dspsr command
        command1 = [
            'dspsr',
            '-S', str(self.toa),
            '-T', '1.0',
            '-c', '1.0',
            '--scloffs',
            '-b', str(bins),
            '-D', str(dm),
            '-cepoch', str(self.cepoch),
            '-O', output_filename,
            self.fname
        ]

        print(f"Running dspsr with command: {' '.join(command1)}")
        result1 = subprocess.run(command1, capture_output=True, text=True)

        if result1.returncode != 0:
            print("dspsr command failed:")
            print(result1.stderr)
            return None

        print("dspsr command executed successfully.")
        output_ar_file = f"{output_filename}.ar"

        if not os.path.exists(output_ar_file):
            print(f"Error: dspsr did not create the expected .ar file: {output_ar_file}")
            return None

        # Construct and run the paz command
        paz_command = ['paz', '-L', '-m', output_ar_file]
        print(f"Running paz with command: {' '.join(paz_command)}")
        result_paz = subprocess.run(paz_command, capture_output=True, text=True)

        if result_paz.returncode != 0:
            print("paz command failed:")
            print(result_paz.stderr)
            return None

        print("paz command executed successfully.")
        paz_output_ar_file = output_ar_file  # Overwrite with the same file after paz modifies it

        if not os.path.exists(paz_output_ar_file):
            print(f"Error: paz did not modify the expected .ar file: {paz_output_ar_file}")
            return None

        # Construct and run the pac command
        command2 = ['pac', '-d', self.fluxcal, paz_output_ar_file]
        print(f"Running pac with command: {' '.join(command2)}")
        result2 = subprocess.run(command2, capture_output=True, text=True)

        if result2.returncode != 0:
            print("pac command failed:")
            print(result2.stderr)
            return None

        #print("pac command executed successfully.")
        #return paz_output_ar_file






def main():
    parser = argparse.ArgumentParser(description="Run dspsr and pac commands to process pulsar data.")
    parser.add_argument('-f', '--filename', type=str, required=True, help='Input FITS filename')
    parser.add_argument('-d', '--fluxcal_file', type=str, required=True, help='Input flux calibrator database.txt file')
    parser.add_argument('-m', '--mjd', type=float, required=True, help='Burst MJD')
    parser.add_argument('-D', '--dm', type=float, required=True, help='Dispersion Measure (DM)')
    parser.add_argument('-b', '--bins', type=int, required=True, help='Number of phase bins')
    parser.add_argument('-O', '--output', type=str, required=True, help='Output filename')

    args = parser.parse_args()

    plotter = UBBPlotter(args.filename, args.mjd, args.fluxcal_file,args.dm)
    output_calib_file = plotter.run_dspsr_command(args.dm, args.bins, args.output)

    if output_calib_file:
        print(f"Calibrated file created: {output_calib_file}")
    else:
        print("Calibration process failed.")

if __name__ == "__main__":
    main()
