# spectrogram.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from parselmouth import SpectralAnalysisWindowShape


def _gaussian_window_factory(std_frac: float):
    """
    Gaussian window compatible with matplotlib Axes.specgram.
    """
    def _win(x_like: np.ndarray):
        N = int(np.asarray(x_like).shape[0])
        n = np.arange(N, dtype=float)
        mu = (N - 1) / 2.0
        sigma = max(1.0, std_frac * N)
        return np.exp(-0.5 * ((n - mu) / sigma) ** 2)

    return _win


def plot_praat_spectrogram(
    ax,
    file_path,
    max_freq=5000,
    fmin=50,
    fmax=1500,
    cmap="magma",
    dynamic_range_db=70,
    window_length=0.03,
    time_step=0.002,
    show_mean_f0=True,
):
    """
    Plots a high-resolution colored narrowband spectrogram with F0 overlay.

    Spectrogram:
    - 30 ms Gaussian window
    - 75% overlap
    - 70 dB dynamic range
    - high harmonic visibility

    F0:
    - Extracted with Praat/Parselmouth autocorrelation
    - Displayed as F0 (autocorrelation)
    """
    ax.clear()
    ax.set_facecolor("#07111f")

    snd = parselmouth.Sound(file_path)

    audio = np.asarray(snd.values[0], dtype=float)
    fs = float(snd.sampling_frequency)

    # High-definition narrowband spectrogram for clearer harmonic structure.
    # The window length controls the real acoustic resolution.
    # The FFT size controls visual/frequency interpolation.
    window_length = 0.03      # 40 ms: better harmonic definition for voice
    fft_size = 4096           # HD visual frequency resolution
    overlap_fraction = 0.80   # smoother time display

    real_window_samples = int(round(window_length * fs))
    NFFT = max(fft_size, real_window_samples)

    noverlap = int(overlap_fraction * NFFT)
    noverlap = min(max(0, noverlap), NFFT - 1)

    window = _gaussian_window_factory(std_frac=(1.0 / 6.0))

    Pxx, freqs, bins, im = ax.specgram(
        audio,
        NFFT=NFFT,
        Fs=fs,
        noverlap=noverlap,
        window=window,
        mode="psd",
        cmap=cmap,
        scale_by_freq=True,
        sides="default",
        detrend=None,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        S_db = 10.0 * np.log10(np.maximum(Pxx, 1e-12))

    vmax = float(np.nanmax(S_db))
    vmin = vmax - float(dynamic_range_db)
    im.set_clim(vmin=vmin, vmax=vmax)

    duration = snd.get_total_duration()

    ax.set_xlim(0.0, duration)
    ax.set_ylim(0, max_freq)

    ax.set_xlabel("Time (s)", color="#d7dde8")
    ax.set_ylabel("Frequency (Hz)", color="#d7dde8")
    ax.set_title("Spectrogram & F0 (autocorrelation)", color="#e8edf7", fontsize=11, fontweight="bold")

    ax.tick_params(axis="both", colors="#b8c0cc")

    for spine in ax.spines.values():
        spine.set_color("#3a4658")

    ax.grid(True, alpha=0.12)

    # F0 extraction using Praat/Parselmouth autocorrelation
    pitch = snd.to_pitch(
        time_step=0.01,
        pitch_floor=fmin,
        pitch_ceiling=fmax,
    )

    times = np.array(pitch.xs())
    hz = np.array(pitch.selected_array["frequency"], dtype=float)

    valid = np.isfinite(hz) & (hz > 0)
    times_valid = times[valid]
    pitch_valid = hz[valid]

    pitch_mask = (pitch_valid >= fmin) & (pitch_valid <= fmax)

    if np.any(pitch_mask):
        x = times_valid[pitch_mask]
        y = pitch_valid[pitch_mask]

        ax.plot(
            x,
            y,
            color="#00c8ff",
            linewidth=2.0,
            label="F0 (autocorrelation)",
            zorder=5,
        )

        legend = ax.legend(
            loc="upper right",
            fontsize=9,
            frameon=True,
            facecolor="#0d1726",
            edgecolor="#263347",
        )

        for text in legend.get_texts():
            text.set_color("#e8edf7")

        if show_mean_f0:
            mean_f0 = np.nanmean(y)
            last_x = x[-1]
            last_y = y[-1]

            x_offset = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0])

            ax.text(
                last_x + x_offset,
                last_y,
                f"Mean F0 = {mean_f0:.1f} Hz",
                color="#dff7ff",
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
                bbox=dict(
                    facecolor="#0d1726",
                    edgecolor="#00c8ff",
                    boxstyle="round,pad=0.25",
                    alpha=0.88,
                ),
                zorder=6,
            )


def extract_f0_autocorrelation(
    file_path,
    fmin=50,
    fmax=1500,
    time_step=0.01,
):
    """
    Extracts F0 using Praat/Parselmouth autocorrelation.

    Returns
    -------
    times : np.ndarray
    f0 : np.ndarray
        F0 values in Hz. Unvoiced frames are returned as NaN.
    """
    snd = parselmouth.Sound(file_path)

    pitch = snd.to_pitch(
        time_step=time_step,
        pitch_floor=fmin,
        pitch_ceiling=fmax,
    )

    times = np.array(pitch.xs())
    f0 = np.array(pitch.selected_array["frequency"], dtype=float)

    f0[(f0 <= 0) | ~np.isfinite(f0)] = np.nan

    return times, f0


def get_f0_summary(
    file_path,
    fmin=50,
    fmax=1500,
    region=None,
    time_step=0.01,
):
    """
    Computes simple F0 summary values using autocorrelation.

    Parameters
    ----------
    file_path:
        Audio file path.
    region:
        Optional tuple: (start_time, end_time)

    Returns
    -------
    dict
        mean_f0, median_f0, std_f0, min_f0, max_f0, voiced_percent
    """
    times, f0 = extract_f0_autocorrelation(
        file_path,
        fmin=fmin,
        fmax=fmax,
        time_step=time_step,
    )

    if region is not None:
        start, end = region
        mask = (times >= start) & (times <= end)
        times = times[mask]
        f0 = f0[mask]

    voiced = np.isfinite(f0)

    if len(f0) == 0:
        voiced_percent = 0.0
    else:
        voiced_percent = 100.0 * np.sum(voiced) / len(f0)

    if not np.any(voiced):
        return {
            "mean_f0": None,
            "median_f0": None,
            "std_f0": None,
            "min_f0": None,
            "max_f0": None,
            "voiced_percent": voiced_percent,
        }

    valid_f0 = f0[voiced]

    return {
        "mean_f0": float(np.nanmean(valid_f0)),
        "median_f0": float(np.nanmedian(valid_f0)),
        "std_f0": float(np.nanstd(valid_f0)),
        "min_f0": float(np.nanmin(valid_f0)),
        "max_f0": float(np.nanmax(valid_f0)),
        "voiced_percent": float(voiced_percent),
    }