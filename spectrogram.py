# spectrogram.py

import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from parselmouth import SpectralAnalysisWindowShape

def plot_praat_spectrogram(ax, file_path, max_freq=5000, fmin=50, fmax=1500):
    """
    Plots a Praat-style spectrogram with the pitch curve (from Praat) always overlaid.
    """
    ax.clear()
    ax.set_facecolor("white")
    snd = parselmouth.Sound(file_path)
    spec = snd.to_spectrogram(
        window_length=0.03,
        maximum_frequency=max_freq,
        window_shape=SpectralAnalysisWindowShape.GAUSSIAN
    )
    S = spec.values
    db = 10 * np.log10(S + np.finfo(float).eps)
    max_db = db.max()
    db = np.clip(db, max_db - 70, max_db)
    extent = [spec.xmin, spec.xmax, spec.ymin, spec.ymax]
    ax.imshow(
        db,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap='Greys'
    )
    ax.set_xlim(spec.xmin, spec.xmax)
    ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Praat-style Spectrogram & F0 (PRAAT)")

    # ---- SEMPRE EXTRAI PITCH DO PRAAT E PLOTA ----
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)
    times = np.array(pitch.xs())
    hz = np.array(pitch.selected_array['frequency'])
    valid = ~np.isnan(hz)
    pitch_valid = hz[valid]
    times_valid = times[valid]
    pitch_mask = (pitch_valid >= fmin) & (pitch_valid <= fmax)
    if np.any(pitch_mask):
        ax.plot(
            times_valid[pitch_mask],
            pitch_valid[pitch_mask],
            color='#0071bc',
            linewidth=2,
            label=f"F0 (PRAAT)"
        )
        ax.legend(loc='upper right', fontsize=10)
        
        mean_f0 = np.nanmean(pitch_valid[pitch_mask])

        # Pega o último valor da curva de pitch visível (para alinhar com a linha azul!)
        last_x = times_valid[pitch_mask][-1]
        last_y = pitch_valid[pitch_mask][-1]

        # Plota o texto ao lado direito da linha, na mesma altura
        ax.text(
            last_x + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),  # ~7% além do último ponto
            last_y,
            f"Mean F₀ = {mean_f0:.1f} Hz",
            color='#1a6edb',
            fontsize=8,
            fontweight='bold',
            va='center', ha='left',
            bbox=dict(facecolor='white', edgecolor='#0071bc', boxstyle='round,pad=0.25', alpha=0.8)
        )

