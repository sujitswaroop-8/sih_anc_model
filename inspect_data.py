import os
import soundfile as sf
from collections import Counter

BASE = r"C:\Users\padhi\Downloads\speech_signals"

folders = ["train_new", "dev_new", "test_new"]

for folder in folders:

    path = os.path.join(BASE, folder)

    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith((".wav", ".flac")):
                files.append(os.path.join(root, filename))

    sample_rates = Counter()
    channels = Counter()
    durations = []

    for file in files:
        try:
            info = sf.info(file)

            sample_rates[info.samplerate] += 1
            channels[info.channels] += 1
            durations.append(info.duration)

        except Exception as e:
            print("Error:", file, e)

    print("\n" + "=" * 50)
    print(folder)
    print("=" * 50)

    print("Total files      :", len(files))
    print("Sample rates     :", dict(sample_rates))
    print("Channels         :", dict(channels))

    if durations:
        print("Min duration (s) :", round(min(durations), 2))
        print("Max duration (s) :", round(max(durations), 2))
        print("Average duration :", round(sum(durations) / len(durations), 2))
