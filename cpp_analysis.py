import parselmouth
import numpy as np
import os
import subprocess
import uuid
import soundfile as sf

def parse_praat_powercepstrum_txt(filepath):
    """Parseia arquivo Praat (PowerCepstrum short text) e retorna (x, y)"""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip() != ""]
    header_idx = None
    for i, l in enumerate(lines):
        if l.startswith("Object class"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"PowerCepstrum header not found in {filepath}!")
    xmin = float(lines[header_idx + 1])
    xmax = float(lines[header_idx + 2])
    nx = int(lines[header_idx + 3])
    dx = float(lines[header_idx + 4])
    x1 = float(lines[header_idx + 5])
    y = np.array([float(val) for val in lines[header_idx + 6 : header_idx + 6 + nx]])
    x = x1 + np.arange(nx) * dx
    return x, y

def extract_voiced_only(audio_path, min_f0=50, max_f0=500):
    """
    Extrai somente os segmentos vozeados do áudio (zera o unvoiced), usando Parselmouth.
    Retorna caminho do novo arquivo WAV apenas com partes vozeadas.
    """
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=min_f0, pitch_ceiling=max_f0)
    values = pitch.selected_array['frequency']
    times = pitch.xs()
    samples = snd.values[0]
    sr = snd.sampling_frequency
    voiced_mask = np.zeros_like(samples, dtype=bool)
    for t, f0 in zip(times, values):
        idx = int(t * sr)
        if 0 <= idx < len(samples) and f0 > 0 and not np.isnan(f0):
            # Marca como vozeado numa janela de 10 ms
            left = max(0, idx - int(0.005 * sr))
            right = min(len(samples), idx + int(0.005 * sr))
            voiced_mask[left:right] = True
    voiced_samples = samples * voiced_mask
    temp_wav = audio_path.replace(".wav", "_voicedonly.wav")
    sf.write(temp_wav, voiced_samples, int(sr))
    return temp_wav

def remove_pauses_with_parselmouth(audio_path, silence_threshold=-35):
    """
    Remove silêncios e pausas usando Parselmouth+Praat. Retorna caminho para novo WAV.
    """
    snd = parselmouth.Sound(audio_path)
    # Usa Trim silences do Praat
    trimmed = parselmouth.praat.call(
        snd, "Trim silences",
        0.08,  # minimum silent duration (s)
        0,  # only at start and end (0 = no, 1 = yes)
        100,  # trim threshold (dB)
        0,  # channel (0 = all)
        silence_threshold,  # silence threshold (dB)
        0.1,  # min sounding interval
        0.05,  # min silent interval
        "no",  # mid points
        "trimmed"  # name
    )

    # Pegue o primeiro elemento da lista (que é um Sound)
    if isinstance(trimmed, list):
        trimmed_sound = trimmed[0]
    else:
        trimmed_sound = trimmed

    temp_wav = audio_path.replace(".wav", "_nopause.wav")
    trimmed_sound.save(temp_wav, "WAV")
    return temp_wav


def preprocess_connected_speech(audio_path):
    """
    Pipeline: (1) Extrai vozeado -> (2) Remove pausas -> retorna novo wav
    """
    temp_voiced = extract_voiced_only(audio_path)
    temp_nopause = remove_pauses_with_parselmouth(temp_voiced)
    try: os.remove(temp_voiced)
    except Exception: pass
    return temp_nopause

def extract_cpp(audio_path, region=None, method="CPP", file_type="Sustained vowel", praat_path="praat.exe"):
    # ===== PRÉ-PROCESSAMENTO PARA CONNECTED SPEECH =====
    if file_type.lower().startswith("connected"):
        wav_path_for_praat = preprocess_connected_speech(audio_path)
    else:
        wav_path_for_praat = audio_path

    snd = parselmouth.Sound(wav_path_for_praat)
    duration = snd.get_total_duration()
    if region is not None:
        start, end = max(0, region[0]), min(duration, region[1])
        snd = snd.extract_part(from_time=start, to_time=end, preserve_times=False)
    else:
        start, end = 0, duration

    center_time = (start + end) / 2

    if method.upper() == "CPP":
        subtract_trend = "yes"
        time_avg_win = 0.001
        quef_avg_win = 0.00005
        trend_type = "Exponential decay"
    else:  # CPPS
        subtract_trend = "no"
        time_avg_win = 0.01
        quef_avg_win = 0.001
        trend_type = "Straight"

    temp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_praat")
    os.makedirs(temp_folder, exist_ok=True)
    file_id = uuid.uuid4().hex
    temp_wav_path = os.path.join(temp_folder, f"{file_id}.wav")
    temp_script_path = os.path.join(temp_folder, f"{file_id}.praat")
    output_file = temp_wav_path + ".output.txt"
    cepstrum_file = temp_wav_path + ".ceps.txt"
    snd.save(temp_wav_path, "WAV")

    script_content = f'''
Read from file: "{temp_wav_path}"
To PowerCepstrogram: 60, 0.002, 5000, 50
cpps = Get CPPS: "{subtract_trend}", {time_avg_win}, {quef_avg_win}, 60, 330, 0.05, "Parabolic", 0.001, 0, "{trend_type}", "Robust"
writeFileLine: "{output_file}", cpps
To PowerCepstrum (slice): {center_time}
Smooth: 0.0005, 1
Write to short text file: "{cepstrum_file}"
'''
    with open(temp_script_path, 'w', encoding='utf-8') as temp_script:
        temp_script.write(script_content)

    quefrency, spectrum, trend = None, None, None
    print("\n========== DEBUG PRAAT ==========")
    print("Temp wav:", temp_wav_path)
    print("Temp praat script:", temp_script_path)
    print("Output file:", output_file)
    print("Cepstrum file:", cepstrum_file)
    print("Slice time usado:", center_time)
    print("=================================")

    try:
        result = subprocess.run([praat_path, "--run", temp_script_path], capture_output=True, text=True)
        print("Praat stdout:", result.stdout)
        print("Praat stderr:", result.stderr)
        cpp_val = None
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                try:
                    cpp_val = float(f.read().strip())
                except Exception:
                    cpp_val = None
        else:
            print("Arquivo output não encontrado!")
            raise RuntimeError(f"Praat did not produce output: {result.stderr}")
        # Now load cepstrum
        if os.path.exists(cepstrum_file):
            try:
                quefrency, spectrum = parse_praat_powercepstrum_txt(cepstrum_file)
                if spectrum is not None and np.max(spectrum) > 200:
                    spectrum = 10 * np.log10(spectrum + 1e-10)
                if spectrum is not None and quefrency is not None:
                    trend = np.polyval(np.polyfit(quefrency, spectrum, 1), quefrency)
            except Exception as e:
                print(f"Erro ao ler o arquivo cepstrum: {cepstrum_file}")
                print(e)
                quefrency, spectrum, trend = None, None, None
        else:
            print("Arquivo cepstrum NÃO encontrado:", cepstrum_file)
            quefrency, spectrum, trend = None, None, None
    finally:
        # Uncomment to remove temp files after debug
        for f in [temp_wav_path, temp_script_path, output_file, cepstrum_file]:
            try: os.remove(f)
            except Exception: pass
        pass

    # Limpeza dos arquivos temporários de pré-processamento (opcional)
    if file_type.lower().startswith("connected"):
        try: os.remove(wav_path_for_praat)
        except Exception: pass

    return {
        "cpp": float(cpp_val) if cpp_val is not None else None,
        "quefrency": quefrency,
        "spectrum": spectrum,
        "trend": trend,
        "region": (start, end)
    }

def batch_extract_cpp(folder_path, method="CPP", file_type="Sustained vowel", praat_path="praat.exe", save_dir=None):
    results = []
    if save_dir is None:
        save_dir = folder_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".wav"):
            fpath = os.path.join(folder_path, fname)
            try:
                res = extract_cpp(fpath, region=None, method=method, file_type=file_type, praat_path=praat_path)
                res['filename'] = fname
                results.append(res)

            except Exception as e:
                print(f"Erro ao processar {fname}: {e}")
                results.append({"filename": fname, "error": str(e)})

    return results
