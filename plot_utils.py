# plot_utils.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def style_quefrency_axes(ax):
    ax.set_facecolor("#111a28")
    ax.tick_params(colors="#b8c0cc")

    for spine in ax.spines.values():
        spine.set_color("#344155")

    ax.xaxis.label.set_color("#d7dde8")
    ax.yaxis.label.set_color("#d7dde8")
    ax.title.set_color("#e8edf7")


def plot_quefrency(ax, quefrency, spectrum, trend=None, label="Cepstrum", method="CPP", value=None):
    """
    Draws the quefrency spectrum on an existing Matplotlib axis.
    This works inside the PySide6 GUI and avoids plt.show() conflicts.
    """
    ax.clear()
    style_quefrency_axes(ax)

    q_ms = np.asarray(quefrency) * 1000.0
    s = np.asarray(spectrum)

    ax.plot(q_ms, s, color="#00c8ff", linewidth=1.8, label=label)

    if trend is not None:
        ax.plot(q_ms, trend, "--", color="#ff4fb3", linewidth=1.5, label="Trend line")

    mask = (q_ms >= 2) & (q_ms <= 12)

    if np.any(mask):
        x_roi = q_ms[mask]
        y_roi = s[mask]
        peak_idx = int(np.argmax(y_roi))

        q_peak = x_roi[peak_idx]
        y_peak = y_roi[peak_idx]
        f0_peak = 1.0 / (q_peak / 1000.0)

        ax.scatter([q_peak], [y_peak], color="#ff4b4b", s=52, zorder=5)

        ax.text(
            q_peak,
            y_peak + 4,
            f"{q_peak:.2f} ms | F0 {f0_peak:.1f} Hz",
            color="#ff6868",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

        if value is not None:
            ax.set_title(
                f"{method} = {value:.2f} dB | Quefrency Spectrum",
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.set_title(f"Quefrency Spectrum ({method})", fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Quefrency Spectrum ({method})", fontsize=12, fontweight="bold")

    ax.set_xlabel("Quefrency (ms)")
    ax.set_ylabel("Amplitude (dB)")

    leg = ax.legend(frameon=True, facecolor="#101826", edgecolor="#2a3548")
    for text in leg.get_texts():
        text.set_color("#e8edf7")

    ax.grid(alpha=0.15)