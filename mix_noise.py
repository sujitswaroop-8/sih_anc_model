import os
import numpy as np
import soundfile as sf
import librosa

# =========================
# PATHS
# =========================

BASE = r"C:\Users\padhi\Downloads\speech_signals"
NOISE_DIR = r"C:\Users\padhi\Downloads\fan_noise"

OUTPUT_BASE = r"C:\Users\padhi\Downloads\siH26052_data"

# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000

SNR_LEVELS = [5, 10, 15]

DATASETS = {
    "train_new": 4000,
    "dev_new": 500,
    "test_new": 500
}

# =========================
# GET NOISE FILES
# =========================

noise_files = []

for root, dirs, files in os.walk(NOISE_DIR):

    for file in files:

        if file.lower().endswith((".wav", ".flac")):

            noise_files.append(
                os.path.join(root, file)
            )

print("Noise files found:", len(noise_files))


# =========================
# FUNCTION: MIX NOISE
# =========================

def add_noise(clean, noise, snr_db):

    # Make noise same length as speech
    if len(noise) < len(clean):

        repetitions = int(
            np.ceil(len(clean) / len(noise))
        )

        noise = np.tile(
            noise,
            repetitions
        )

    noise = noise[:len(clean)]

    # Signal power
    speech_power = np.mean(clean ** 2)

    # Noise power
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:

        return clean

    # Desired noise power
    desired_noise_power = (
        speech_power /
        (10 ** (snr_db / 10))
    )

    # Scale noise
    noise = noise * np.sqrt(
        desired_noise_power /
        noise_power
    )

    # Mix
    noisy = clean + noise

    # Prevent clipping
    max_value = np.max(
        np.abs(noisy)
    )

    if max_value > 1:

        noisy = noisy / max_value

    return noisy


# =========================
# PROCESS DATASETS
# =========================

for dataset, expected_count in DATASETS.items():

    input_dir = os.path.join(
        BASE,
        dataset
    )

    clean_output = os.path.join(
        OUTPUT_BASE,
        dataset,
        "clean"
    )

    noisy_output = os.path.join(
        OUTPUT_BASE,
        dataset,
        "noisy"
    )

    os.makedirs(
        clean_output,
        exist_ok=True
    )

    os.makedirs(
        noisy_output,
        exist_ok=True
    )

    # =========================
    # FIND SPEECH FILES
    # =========================

    speech_files = []

    for root, dirs, files in os.walk(
        input_dir
    ):

        for file in files:

            if file.lower().endswith(
                (".wav", ".flac")
            ):

                speech_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    speech_files.sort()

    # Limit number of clean speech files
    speech_files = speech_files[:expected_count]

    print("\n" + "=" * 50)
    print("Processing:", dataset)
    print("=" * 50)

    print(
        "Clean speech files:",
        len(speech_files)
    )

    print(
        "SNR levels:",
        SNR_LEVELS
    )

    # =========================
    # PROCESS SPEECH
    # =========================

    for i, speech_file in enumerate(
        speech_files
    ):

        try:

            # -------------------------
            # Load clean speech
            # -------------------------

            clean, sr = sf.read(
                speech_file
            )

            # Stereo → mono
            if clean.ndim > 1:

                clean = np.mean(
                    clean,
                    axis=1
                )

            # Resample
            if sr != SAMPLE_RATE:

                clean = librosa.resample(
                    clean.astype(np.float32),
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE
                )

            clean = clean.astype(
                np.float32
            )

            # -------------------------
            # Original filename
            # -------------------------

            filename = os.path.splitext(
                os.path.basename(
                    speech_file
                )
            )[0]

            # -------------------------
            # Save clean speech ONCE
            # -------------------------

            clean_path = os.path.join(
                clean_output,
                filename + ".wav"
            )

            sf.write(
                clean_path,
                clean,
                SAMPLE_RATE
            )

            # =========================
            # CREATE ALL 3 SNR VERSIONS
            # =========================

            for snr in SNR_LEVELS:

                # -------------------------
                # Select random noise
                # -------------------------

                noise_file = np.random.choice(
                    noise_files
                )

                noise, noise_sr = sf.read(
                    noise_file
                )

                # Stereo → mono
                if noise.ndim > 1:

                    noise = np.mean(
                        noise,
                        axis=1
                    )

                # Resample noise
                if noise_sr != SAMPLE_RATE:

                    noise = librosa.resample(
                        noise.astype(np.float32),
                        orig_sr=noise_sr,
                        target_sr=SAMPLE_RATE
                    )

                noise = noise.astype(
                    np.float32
                )

                # -------------------------
                # Add noise
                # -------------------------

                noisy = add_noise(
                    clean,
                    noise,
                    snr
                )

                # -------------------------
                # Save noisy speech
                # -------------------------

                noisy_filename = (
                    f"{filename}_{snr}dB.wav"
                )

                noisy_path = os.path.join(
                    noisy_output,
                    noisy_filename
                )

                sf.write(
                    noisy_path,
                    noisy,
                    SAMPLE_RATE
                )

            # -------------------------
            # Progress
            # -------------------------

            if (i + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{i + 1}/{len(speech_files)} "
                    f"speech files"
                )

        except Exception as e:

            print(
                "Error processing:",
                speech_file,
                e
            )


print("\nDONE!")
