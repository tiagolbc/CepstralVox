import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
import shutil

from cpp_analysis import extract_cpp, batch_extract_cpp
from file_utils import save_csv
from spectrogram import plot_praat_spectrogram

BG_COLOR = "#CDCDC1"
#b0afa6
#CDCDC1
FONT = ("Calibri", 13)
BTN_FONT = ("Calibri", 11)
ROI_COLOR = "red"
ROI_ALPHA = 0.25
APP_VERSION = "1.0.0"

def plot_quefrency_figure(res, method, save_path=None, show=False):
    q = np.array(res['quefrency']) * 1000  # ms
    s = res['spectrum']
    trend = res['trend']
    val = res.get('cpp', None)
    fig, ax = plt.subplots(figsize=(9, 3.6))  # maior altura p/ garantir espaço
    label1 = f"{method} Cepstrum" if method else "Cepstrum"
    label2 = f"{method} Trend" if method else "Trend"
    ax.plot(q, s, label=label1)
    if trend is not None:
        ax.plot(q, trend, '--', label=label2)
    q_peak = y_peak = f0_peak = None
    mask = (q >= 2) & (q <= 12)
    if np.any(mask):
        x_roi = q[mask]
        y_roi = s[mask]
        peak_idx = np.argmax(y_roi)
        q_peak = x_roi[peak_idx]
        y_peak = y_roi[peak_idx]
        f0_peak = 1.0 / (q_peak / 1000)
        ax.annotate(
            '', xy=(q_peak, y_peak), xytext=(q_peak, y_peak + 20),
            arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05, width=2, headwidth=8)
        )
        ax.text(q_peak, y_peak + 26, f"{q_peak:.2f} ms", color="red",
                fontsize=13, fontweight="bold", ha="center", va="bottom")
    ax.set_xlabel("Quefrency (ms)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"Quefrency Spectrum ({method})")
    ax.set_ylim(30, 110)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()
    # TITULO COMPLETO (Praat-like, nunca cortado)
    if val is not None and q_peak is not None and f0_peak is not None:
        title = (
            f"{method} = {val:.2f} dB (quefrency: {q_peak / 1000:.3f} s, "
            fr"$f_{{\it{{o}}}}$" f": {f0_peak:.2f} Hz)"
        )
        fig.suptitle(title, x=0.5, y=0.97, fontsize=13, ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.85])  # reserva 15% pro título!
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)

class CPPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cepstral Vox version 1.0.0")
        self.root.configure(bg=BG_COLOR)
        self.audio_path = None
        self.audio_data = None
        self.sr = None
        self.region = None
        self.analysis_result = None
        self.analysis_method = None
        self.results_type = None
        self.batch_results = []
        self.span = None
        self.roi_patch = None

        # Top controls
        top = tk.Frame(root, bg=BG_COLOR)
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=4)

        try:
            from PIL import Image, ImageTk
            logo_img = Image.open("logo.png").resize((200, 200))
            self.logo = ImageTk.PhotoImage(logo_img)
            tk.Label(top, image=self.logo, bg=BG_COLOR).pack(side=tk.LEFT, padx=(5, 16))
        except Exception:
            tk.Label(top, text="[Logo]", bg=BG_COLOR, font=FONT).pack(side=tk.LEFT, padx=(5, 16))

        btn_row = tk.Frame(top, bg=BG_COLOR)
        btn_row.pack(side=tk.LEFT, expand=True)

        self.open_btn = tk.Button(btn_row, text="Open WAV File", font=BTN_FONT, command=self.load_audio)
        self.open_btn.pack(side=tk.LEFT, padx=3, pady=2, ipadx=6, ipady=2)

        self.file_type_var = tk.StringVar(value="Sustained vowel")
        file_type_menu = ttk.Combobox(btn_row, textvariable=self.file_type_var,
            values=["Sustained vowel", "Connected speech"], state="readonly", width=18, font=BTN_FONT)
        file_type_menu.pack(side=tk.LEFT, padx=5)

        self.analysis_type_var = tk.StringVar(value="CPP")
        for label in ["CPP", "CPPS"]:
            tk.Radiobutton(btn_row, text=label, variable=self.analysis_type_var, value=label,
                bg=BG_COLOR, font=BTN_FONT).pack(side=tk.LEFT, padx=4)

        self.batch_btn = tk.Button(btn_row, text="Batch Process", font=BTN_FONT, command=self.batch_process)
        self.batch_btn.pack(side=tk.LEFT, padx=10, ipadx=5, ipady=2)

        self.loaded_file_label = tk.Label(top, text="No file loaded", fg="gray", bg=BG_COLOR, font=BTN_FONT)
        self.loaded_file_label.pack(side=tk.LEFT, padx=(28, 5))

        self.result_display = tk.Label(
            root, text="", font=("Calibri", 17, "bold"),
            fg="#2d832c", bg=BG_COLOR, anchor="center"
        )
        self.result_display.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 5))

        center = tk.Frame(root, relief=tk.SUNKEN, bg=BG_COLOR, borderwidth=1)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=7)
        self.fig, self.ax = plt.subplots(figsize=(9, 3.4))
        self.fig.patch.set_facecolor(BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(root, text="", fg="#14598d", bg=BG_COLOR, font=BTN_FONT)
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4))

        bot = tk.Frame(root, bg=BG_COLOR)
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        self.play_btn = tk.Button(bot, text="Play Audio", font=BTN_FONT, command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=4, ipadx=4)
        self.run_btn = tk.Button(bot, text="Run Analysis", font=BTN_FONT, command=self.run_analysis, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=6, ipadx=8)
        self.show_quef_btn = tk.Button(bot, text="Show Quefrency Plot", font=BTN_FONT, command=self.show_quefrency_plot, state=tk.DISABLED)
        self.show_quef_btn.pack(side=tk.LEFT, padx=6, ipadx=8)
        self.export_btn = tk.Button(bot, text="Export CSV", font=BTN_FONT, command=self.export_csv, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=6, ipadx=8)
        self.exit_btn = tk.Button(bot, text="Exit", font=BTN_FONT, command=self.force_exit)
        self.exit_btn.pack(side=tk.RIGHT, padx=4, ipadx=8)
        self.root.protocol("WM_DELETE_WINDOW", self.force_exit)
        self.about_btn = tk.Button(bot, text="About", font=BTN_FONT, command=self.show_about)
        self.about_btn.pack(side=tk.RIGHT, padx=12, ipadx=8)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.audio_path = file_path
            self.audio_data, self.sr = sf.read(file_path)
            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
            # ---- LIMPA O ROI PATCH E REGIÃO ----
            if self.roi_patch:
                try:
                    self.roi_patch.remove()
                except Exception:
                    pass
                self.roi_patch = None
            # Desconectar o SpanSelector antigo, se existir
            if self.span:
                try:
                    self.span.disconnect_events()
                except Exception:
                    pass
                self.span = None

            self.region = None
            self.region = None
            self.analysis_result = None
            self.analysis_method = None
            self.results_type = None
            self.result_display.config(text="")
            self.loaded_file_label.config(text=os.path.basename(file_path), fg="black")
            self.show_spectrogram()
            self.play_btn.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.DISABLED)
            self.show_quef_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Select ROI: click and drag on the spectrogram.")
        else:
            self.status_label.config(text="No file loaded.", fg="red")

    def show_spectrogram(self):
        self.ax.clear()
        plot_praat_spectrogram(self.ax, self.audio_path, max_freq=5000)
        self.ax.set_facecolor(BG_COLOR)
        self.fig.patch.set_facecolor(BG_COLOR)
        self.canvas.draw()
        if self.roi_patch:
            self.roi_patch.remove()
            self.roi_patch = None
        if self.span:
            self.span.disconnect_events()
        self.span = SpanSelector(
            self.ax, self.on_select, 'horizontal',
            useblit=True,
            props=dict(alpha=ROI_ALPHA, facecolor=ROI_COLOR),
            interactive=True
        )
        self.region = None

    def on_select(self, tmin, tmax):
        duration = self.audio_data.shape[0] / self.sr
        tmin = max(0, min(duration, tmin))
        tmax = max(0, min(duration, tmax))
        if tmax > tmin:
            if self.roi_patch:
                self.roi_patch.remove()
            self.roi_patch = self.ax.axvspan(tmin, tmax, color=ROI_COLOR, alpha=ROI_ALPHA)
            self.region = (tmin, tmax)
            self.status_label.config(text=f"Selected ROI: {tmin:.2f} - {tmax:.2f} s")
            self.canvas.draw()

    def run_analysis(self):
        if self.audio_path is None:
            messagebox.showerror("Error", "Load an audio file first.")
            return
        region = self.region
        method = self.analysis_type_var.get()
        self.analysis_method = method
        file_type = self.file_type_var.get()
        try:
            results = extract_cpp(self.audio_path, region=region, method=method, file_type=file_type)
            self.analysis_result = results
            self.results_type = file_type
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            return
        sroi = f"(ROI: {region[0]:.2f}-{region[1]:.2f}s)" if region else ""
        val = results.get('cpp', None)
        if val is None:
            self.result_display.config(text="Analysis failed: no valid CPP/CPPS value.")
            self.status_label.config(text="Analysis failed.", fg="red")
            return
        msg = f"{method} = {val:.2f} dB\n{sroi}"
        self.result_display.config(text=msg)
        self.status_label.config(text=f"{method} analysis complete.")
        self.export_btn.config(state=tk.NORMAL)
        self.show_quef_btn.config(state=tk.NORMAL)
        # Salva a figura automaticamente com todos os dados
        outpath = os.path.splitext(self.audio_path)[0] + f"_{method}_quefrency.png"
        plot_quefrency_figure(self.analysis_result, method, save_path=outpath, show=False)

    def show_quefrency_plot(self):
        # Só mostra o gráfico de quefrency, NÃO o espectrograma
        if not self.analysis_result or self.analysis_result.get('quefrency') is None:
            messagebox.showinfo("No Data", "No quefrency data available for plotting.")
            return
        plot_quefrency_figure(self.analysis_result, self.analysis_method, show=True)

    def save_quefrency_figure(self, res, method, save_path):
        plot_quefrency_figure(res, method, save_path=save_path, show=False)

    def batch_process(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        import tkinter.simpledialog
        method = tkinter.simpledialog.askstring("Batch Method", "Type 'CPP' or 'CPPS' for batch processing:",
                                                initialvalue="CPP").strip().upper()
        if method not in ["CPP", "CPPS"]:
            messagebox.showerror("Batch", "You must type CPP or CPPS.")
            return
        file_type = self.file_type_var.get()
        self.batch_method = method
        self.status_label.config(text="Batch processing, please wait...")
        self.root.update()
        try:
            batch_results = batch_extract_cpp(folder_path, method=method, file_type=file_type)
        except Exception as e:
            messagebox.showerror("Batch Error", str(e))
            return
        self.batch_results = batch_results
        num_ok = len([r for r in batch_results if 'cpp' in r])
        num_err = len([r for r in batch_results if 'error' in r])
        self.result_display.config(text=f"{method} batch: {num_ok} ok, {num_err} errors.", fg="#267022")
        self.status_label.config(text=f"Batch: {num_ok} files processed, {num_err} errors.")
        self.export_btn.config(state=tk.NORMAL)
        plot_dir = os.path.join(folder_path, "quefrency_plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for r in self.batch_results:
            if r.get("quefrency") is not None and r.get("spectrum") is not None:
                base = os.path.splitext(r.get("filename", "unnamed"))[0]
                save_path = os.path.join(plot_dir, f"{base}_{method}_quefrency.png")
                plot_quefrency_figure(r, method, save_path=save_path, show=False)

    def export_csv(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not save_path:
            return
        import csv
        if self.analysis_result and not self.batch_results:
            region = self.region or (None, None)
            row = {
                "filename": os.path.basename(self.audio_path),
                "file_type": self.results_type,
                "roi_start": f"{region[0]:.3f}" if region[0] is not None else "",
                "roi_end": f"{region[1]:.3f}" if region[1] is not None else "",
                self.analysis_method: f"{self.analysis_result['cpp']:.3f}" if self.analysis_result else "",
            }
            with open(save_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
        elif self.batch_results:
            batch_method = getattr(self, "batch_method", "CPP")
            value_column = batch_method.lower()
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "file_type", value_column, "roi_start", "roi_end"])
                for r in self.batch_results:
                    region = r.get("region", (None, None))
                    writer.writerow([
                        r.get("filename", ""),
                        self.file_type_var.get(),
                        f"{r.get('cpp', ''):.3f}" if "cpp" in r else "",
                        f"{region[0]:.3f}" if region and region[0] is not None else "",
                        f"{region[1]:.3f}" if region and region[1] is not None else ""
                    ])
        self.status_label.config(text="Results exported.", fg="green")

    def show_about(self):
        about_text = (
            f"**CepstralVox**  (Version {APP_VERSION})\n\n"
            "CepstralVox is a modern tool for acoustic voice analysis, specialized in cepstral measures "
            "such as CPP and CPPS. It delivers Praat-like accuracy with a simple interface.\n\n"
            "Features:\n"
            "- Extraction and visualization of CPP and CPPS\n"
            "- Visualization of quefrency peak and calculated fundamental frequency\n"
            "- Support for sustained vowels and connected speech\n"
            "- Batch processing\n\n"
            "For questions, collaborations, or bug reports:\n"
            "E-mail: fonotechacademy@gmail.com\n"
            "Instagram: @fonotechacademy\n"
            "Website: www.fonotechacademy.com\n\n"
            "Citation (if used for research):\n"
            "Cruz, Tiago Lima Bicalho. (2025). CepstralVox: A tool for cepstral and voice analysis (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15773397\n"
        )
        # Mostra About em janela de texto
        about_win = tk.Toplevel(self.root)
        about_win.title("About CepstralVox")
        about_win.configure(bg=BG_COLOR)
        about_win.resizable(False, False)
        text = tk.Text(about_win, wrap="word", font=("Calibri", 12), bg=BG_COLOR, relief="flat", bd=0, height=22,
                       width=65)
        text.insert("1.0", about_text)
        text.config(state="disabled")
        text.pack(padx=20, pady=16)
        tk.Button(about_win, text="Close", command=about_win.destroy, font=BTN_FONT).pack(pady=(0, 12))
        about_win.grab_set()

    def play_audio(self):
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(self.audio_data, self.sr)
        except Exception:
            messagebox.showinfo("Audio", "Install the package 'sounddevice' for playback (pip install sounddevice).")

    def force_exit(self):
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        temp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_praat")
        try:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"Erro ao remover temp_praat: {e}")
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = CPPApp(root)
    root.mainloop()
