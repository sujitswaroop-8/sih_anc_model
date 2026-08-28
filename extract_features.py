import os
import numpy as np
import soundfile as sf
import librosa

# =========================
# PATHS
# =========================

BASE = r"C:\Users\padhi\Downloads\siH26052_data"

OUTPUT_DIR = os.path.join(BASE, "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================

N_FFT = 512
HOP_LENGTH = 256

DATASETS = {
    "train_new": 4000,
    "dev_new": 500,
    "test_new": 500
}

# =========================
# FUNCTION
# =========================

def extract_dataset(dataset, max_files):

    clean_dir = os.path.join(
        BASE,
        dataset,
        "clean"
    )

    noisy_dir = os.path.join(
        BASE,
        dataset,
        "noisy"
    )

    files = [
        f for f in os.listdir(clean_dir)
        if f.lower().endswith(".wav")
    ]

    files.sort()

    files = files[:max_files]

    print("\n" + "=" * 50)
    print(dataset)
    print("=" * 50)

    print("Files selected:", len(files))

    X_list = []
    y_list = []

    for i, filename in enumerate(files):

        clean_path = os.path.join(
            clean_dir,
            filename
        )

        noisy_path = os.path.join(
            noisy_dir,
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
            # Time × Frequency
            # -------------------------

            X = noisy_mag.T
            y = clean_mag.T

            X_list.append(X)
            y_list.append(y)

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

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # =========================
    # SAVE
    # =========================

    prefix = dataset.replace(
        "_new",
        ""
    )

    X_path = os.path.join(
        OUTPUT_DIR,
        f"X_{prefix}.npy"
    )

    y_path = os.path.join(
        OUTPUT_DIR,
        f"y_{prefix}.npy"
    )

    np.save(X_path, X)
    np.save(y_path, y)

    print("Saved:", X_path)
    print("Saved:", y_path)


# =========================
# RUN
# =========================

for dataset, max_files in DATASETS.items():

    extract_dataset(
        dataset,
        max_files
    )

print("\nALL DATASETS PROCESSED!")





       





