import subprocess
import argparse

def read_columns_from_file(file_path):
    """Read identifiers (second column) and MJDs (third column) from the text file."""
    identifiers_and_mjds = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:  # Ensure there are at least three columns
                    identifier = parts[1]  # Second column
                    try:
                        mjd = float(parts[2])  # Convert MJD (third column) to float
                        identifiers_and_mjds.append((identifier, mjd))
                    except ValueError:
                        print(f"Warning: Could not convert MJD '{parts[2]}' to float. Skipping line.")
                else:
                    print(f"Warning: Line does not have enough columns: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return identifiers_and_mjds

def run_singleband_archive(filename, fluxcal_file, identifier, mjd, dm, bins, output_base):
    """Run the Singleband_Archive.py script for a given MJD with identifier-based output."""
    output_file = f"{output_base}_{identifier}"  # Use the identifier for the output filename
    command = [
        "python3", "Singleband_Archive.py",
        "-f", filename,
        "-d", fluxcal_file,
        "-m", str(mjd),
        "-D", str(dm),
        "-b", str(bins),
        "-O", output_file
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Success: Output for identifier {identifier} created at {output_file}.")
    else:
        print(f"Error running Singleband_Archive.py for identifier {identifier}:")
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Wrapper script to run Singleband_Archive.py for multiple identifiers and MJDs.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Input FITS filename')
    parser.add_argument('-d', '--fluxcal_file', type=str, required=True, help='Input flux calibrator database.txt file')
    parser.add_argument('-D', '--dm', type=float, required=True, help='Dispersion Measure (DM)')
    parser.add_argument('-b', '--bins', type=int, required=True, help='Number of phase bins')
    parser.add_argument('-o', '--output_base', type=str, required=True, help='Base name for output files')

    args = parser.parse_args()

    # Read identifiers and MJDs from the text file
    identifiers_and_mjds = read_columns_from_file(args.input_file)
    if not identifiers_and_mjds:
        print("Error: No valid identifiers or MJD values found in the input file.")
        return

    # Run the script for each identifier and MJD
    for identifier, mjd in identifiers_and_mjds:
        run_singleband_archive(args.filename, args.fluxcal_file, identifier, mjd, args.dm, args.bins, args.output_base)

if __name__ == "__main__":
    main()
