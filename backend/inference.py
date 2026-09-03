"""
inference.py

This wraps the exact inference logic your team already proved out at the
end of train_model.ipynb into a reusable function the web server can call.

It loads the three trained artifacts once at startup:
    model/mlp_model.pkl   - the trained MLPRegressor
    model/scaler_X.pkl    - scaler fit on noisy-magnitude training data
    model/scaler_y.pkl    - scaler fit on clean-magnitude training data

Then enhance_audio() takes the path to any noisy .wav file and returns
the path to a cleaned .wav file, using the same STFT -> predict -> ISTFT
pipeline from the notebook.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import joblib

# =========================
# SETTINGS
# (must match what extract_features.py / train_model.ipynb used)
# =========================
N_FFT = 512
HOP_LENGTH = 256
TARGET_SAMPLE_RATE = 16000

# Folder where the 3 .pkl files live. Override with the MODEL_DIR
# environment variable if you deploy them somewhere else.
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
)

_model = None
_scaler_X = None
_scaler_y = None


class ModelNotLoadedError(Exception):
    """Raised when the .pkl files aren't present in MODEL_DIR yet."""
    pass


def load_model():
    """
    Loads the model + scalers into memory once. Safe to call more than
    once - later calls just reuse what's already loaded.
    """
    global _model, _scaler_X, _scaler_y

    if _model is not None:
        return

    model_path = os.path.join(MODEL_DIR, "mlp_model.pkl")
    scaler_x_path = os.path.join(MODEL_DIR, "scaler_X.pkl")
    scaler_y_path = os.path.join(MODEL_DIR, "scaler_y.pkl")

    missing = [
        p for p in [model_path, scaler_x_path, scaler_y_path]
        if not os.path.exists(p)
    ]
    if missing:
        raise ModelNotLoadedError(
            "Missing model file(s): "
            + ", ".join(missing)
            + f". Place mlp_model.pkl, scaler_X.pkl and scaler_y.pkl in {MODEL_DIR}"
        )

    _model = joblib.load(model_path)
    _scaler_X = joblib.load(scaler_x_path)
    _scaler_y = joblib.load(scaler_y_path)


def is_model_loaded() -> bool:
    return _model is not None


def estimate_snr_db(y: np.ndarray, frame_length: int = 2048,
                     hop_length: int = 512, noise_percentile: float = 10) -> float:
    """
    Reference-free SNR estimate.

    Real SNR needs a known clean signal to compare against (that's what
    the team's offline test-set evaluation used). For audio a user
    uploads live on the website, there's no clean reference - so instead
    we estimate the noise floor from the quietest frames in the signal
    itself, and compare overall energy against that floor. This is a
    common blind estimation approach, not a lab-grade measurement -
    treat it as indicative, not exact.
    """
    if len(y) < frame_length:
        frame_length = max(256, len(y))
        hop_length = max(1, frame_length // 2)

    energies = []
    for i in range(0, max(1, len(y) - frame_length), hop_length):
        frame = y[i:i + frame_length]
        energies.append(np.mean(frame ** 2) + 1e-12)

    energies = np.array(energies)
    if len(energies) < 2:
        return 0.0

    sorted_e = np.sort(energies)
    k = max(1, int(len(sorted_e) * noise_percentile / 100))
    noise_power = np.mean(sorted_e[:k])
    signal_power = np.mean(energies)

    ratio = max(signal_power - noise_power, 1e-12) / noise_power
    return float(10 * np.log10(ratio))


def enhance_audio(input_path: str, output_path: str) -> dict:
    """
    Runs the full denoising pipeline on one audio file.

    input_path:  path to a noisy .wav (or any format soundfile/librosa can read)
    output_path: where to write the cleaned .wav

    Returns a dict:
        {
          "output_path": str,
          "original_snr_db": float,   # estimated, on the uploaded audio
          "enhanced_snr_db": float,   # estimated, on the cleaned audio
          "improvement_db": float,
        }
    """
    load_model()  # no-op if already loaded

    # -------------------------
    # Load audio
    # -------------------------
    noisy, sr = sf.read(input_path)
    noisy = np.asarray(noisy, dtype=np.float32)

    # Stereo -> mono, same as mix_noise.py did for training data
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)

    # Resample to the rate the model was trained on
    if sr != TARGET_SAMPLE_RATE:
        noisy = librosa.resample(
            noisy, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE
        )
        sr = TARGET_SAMPLE_RATE

    # -------------------------
    # STFT -> magnitude + phase
    # -------------------------
    noisy_stft = librosa.stft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    # (freq, time) -> (time, freq) to match training layout
    X = noisy_mag.T

    # -------------------------
    # Predict clean magnitude
    # -------------------------
    X_scaled = _scaler_X.transform(X)
    pred_scaled = _model.predict(X_scaled)
    pred_mag = _scaler_y.inverse_transform(pred_scaled)

    # Back to (freq, time), clip negatives (magnitude can't be negative)
    pred_mag = pred_mag.T
    pred_mag = np.maximum(pred_mag, 0)

    # -------------------------
    # Rebuild waveform: predicted magnitude + ORIGINAL noisy phase
    # -------------------------
    enhanced_stft = pred_mag * np.exp(1j * noisy_phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=HOP_LENGTH)

    sf.write(output_path, enhanced_audio, sr)

    original_snr = estimate_snr_db(noisy)
    enhanced_snr = estimate_snr_db(enhanced_audio)

    return {
        "output_path": output_path,
        "original_snr_db": round(original_snr, 2),
        "enhanced_snr_db": round(enhanced_snr, 2),
        "improvement_db": round(enhanced_snr - original_snr, 2),
    }
