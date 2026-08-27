import os
import numpy as np
import soundfile as sf
import librosa

# =========================
# PATHS
# =========================

BASE = r"C:\Users\padhi\Downloads\siH26052_data"

DATASET = "train_new"

CLEAN_DIR = os.path.join(BASE, DATASET, "clean")
NOISY_DIR = os.path.join(BASE, DATASET, "noisy")

OUTPUT_DIR = os.path.join(BASE, "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 256

MAX_FILES = 4000

# =========================
# GET FILES
# =========================

files = [
    f for f in os.listdir(CLEAN_DIR)
    if f.lower().endswith(".wav")
]

files.sort()

# Use only 4000 files
files = files[:MAX_FILES]

print("Files selected:", len(files))

# =========================
# STORE FEATURES
# =========================

X_list = []
y_list = []

# =========================
# PROCESS FILES
# =========================

for i, filename in enumerate(files):

    clean_path = os.path.join(
        CLEAN_DIR,
        filename
    )

    noisy_path = os.path.join(
        NOISY_DIR,
        filename
    )

    try:

        # -------------------------
        # Load audio
        # -------------------------

        clean, sr_clean = sf.read(clean_path)
        noisy, sr_noisy = sf.read(noisy_path)

        clean = clean.astype(np.float32)
        noisy = noisy.astype(np.float32)

        # -------------------------
        # STFT
        # -------------------------

        clean_stft = librosa.stft(
            clean,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        noisy_stft = librosa.stft(
            noisy,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # -------------------------
        # Magnitude
        # -------------------------

        clean_mag = np.abs(clean_stft)
        noisy_mag = np.abs(noisy_stft)

        # -------------------------
        # Convert:
        # frequency × time
        #
        # to:
        # time × frequency
        # -------------------------

        X = noisy_mag.T
        y = clean_mag.T

        X_list.append(X)
        y_list.append(y)

        # -------------------------
        # Progress
        # -------------------------

        if (i + 1) % 100 == 0:

            print(
                f"Processed {i + 1}/{len(files)}"
            )

    except Exception as e:

        print(
            "Error:",
            filename,
            e
        )

# =========================
# COMBINE
# =========================

print("\nCombining features...")

X_train = np.vstack(X_list)
y_train = np.vstack(y_list)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# =========================
# SAVE
# =========================

X_path = os.path.join(
    OUTPUT_DIR,
    "X_train.npy"
)

y_path = os.path.join(
    OUTPUT_DIR,
    "y_train.npy"
)

np.save(X_path, X_train)
np.save(y_path, y_train)

print("\nSaved:")
print(X_path)
print(y_path)

print("\nDONE!")
