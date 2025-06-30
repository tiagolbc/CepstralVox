# plot_utils.py
import matplotlib.pyplot as plt

def plot_quefrency(ax, quefrency, spectrum, trend=None, label="Cepstrum"):
    ax.clear()
    ax.plot(quefrency * 1000, spectrum, label=label)  # quefrency in ms
    if trend is not None:
        ax.plot(quefrency * 1000, trend, "r--", label="Trend line")
    ax.set_xlabel("Quefrency (ms)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    ax.set_title("Quefrency Spectrum")
