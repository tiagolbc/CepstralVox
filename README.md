# CepstralVox

CepstralVox
CepstralVox is a free, open-source, cross-platform tool for cepstral and voice analysis.
It provides user-friendly batch and interactive analysis of CPP and CPPS directly from WAV files, replicating Praat’s acoustic algorithms, and is designed for both research and clinical settings.

Features
Accurate extraction of CPP and CPPS (Cepstral Peak Prominence, Praat-style)

Visual, interactive spectrogram with pitch overlay (fundamental frequency curve from Praat)

Automatic batch processing of multiple audio files

Region of interest (ROI) selection for focused analysis

Praat-compatible: Uses Praat’s algorithms for maximal reproducibility

Export of results to CSV for statistical analysis

Clean, intuitive GUI (Tkinter + Matplotlib)

Ready-to-use for clinical or research purposes

Installation
Requirements
Python 3.8+

Praat (must be installed and accessible as praat.exe or praat in your PATH)

Recommended: Anaconda

Python dependencies
Install all dependencies with:

bash
Copiar
Editar
pip install numpy matplotlib soundfile parselmouth pillow
Download
Download the latest release from GitHub Releases

Or clone with:

bash
Copiar
Editar
git clone https://github.com/tiagolbc/cepstralvox.git
cd cepstralvox
Usage
Launching the GUI
bash
Copiar
Editar
python main.py
Main Features
Open WAV File: Select a file for analysis

Select Analysis Type: Choose CPP or CPPS, and file type (sustained vowel or connected speech)

ROI Selection: Click and drag on the spectrogram to select a region for analysis

Run Analysis: Calculate and display results

Show Quefrency Plot: Visualize the quefrency spectrum with main peak and trend

Batch Process: Analyze multiple files at once

Export CSV: Save your results for further analysis

Batch Mode
Use the Batch Process button in the GUI to process all WAV files in a selected folder.

Results and quefrency plots are saved automatically.

Screenshot
<p align="center"> <img src="figures/gui.png" width="720"> </p>
How to Cite
If you use CepstralVox in scientific publications, please cite:

Cruz, Tiago Lima Bicalho. CepstralVox: A tool for cepstral and voice analysis. Zenodo. https://doi.org/10.5281/zenodo.9999999

Support and Contact
Instagram: @fonotechacademy

Email: fonotechacademy@gmail.com

For feature requests or bug reports, please open an issue on GitHub.

License
This project is licensed under the MIT License.

Acknowledgements
CepstralVox is inspired by classic Praat methods (Paul Boersma & David Weenink), and incorporates community feedback from voice researchers and clinicians.
