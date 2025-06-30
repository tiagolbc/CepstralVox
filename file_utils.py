# file_utils.py
import csv
import os

def save_csv(results, filename):
    """
    Saves a list of result dicts (from batch or single analysis) to CSV.
    If results contains per-file data, exports all; if only one, exports just the value.
    """
    if not results:
        return
    # Check if single result or list of dicts
    is_batch = isinstance(results, list)
    if not is_batch:
        # Wrap single result for uniformity
        results = [results]
    fieldnames = ['filename', 'cpp', 'region_start', 'region_end']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {
                'filename': res.get('filename', ''),
                'cpp': res.get('cpp', ''),
                'region_start': f"{res.get('region', (None,))[0]:.3f}" if res.get('region') else '',
                'region_end': f"{res.get('region', (None, None))[1]:.3f}" if res.get('region') else ''
            }
            writer.writerow(row)

def get_wav_files_in_folder(folder_path):
    """
    Returns a sorted list of full paths to .wav files in a folder.
    """
    files = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".wav"):
            files.append(os.path.join(folder_path, fname))
    return files
