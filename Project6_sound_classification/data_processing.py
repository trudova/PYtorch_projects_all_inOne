import os
import random

import soundfile as sf
import torch
from utils.ploting_audio import plot_specgram

wav_path = "./data/set"
wav_filenames = os.listdir(wav_path)
CLASSES = []

for f_name in wav_filenames:
    class_name = f_name.split("_")[0].lower()
    if (class_name not in CLASSES) and ("unlabelledtest" not in class_name):
        CLASSES.append(class_name)
print(CLASSES)
random.shuffle(wav_filenames)

for f_index, fname in enumerate(wav_filenames):
    class_name = fname.split("_")[0].lower()
    target_path = "train" if f_index < 180 else "test"
    class_path = f"{target_path}/{class_name}"
    file_path = f"{wav_path}/{fname}"

    f_basename_no_ext = os.path.splitext(os.path.basename(fname))[0]
    target_file_path = f"{class_path}/{f_basename_no_ext}.png"

    if class_name in CLASSES:
        os.makedirs(class_path, exist_ok=True)

        data, sr = sf.read(file_path, always_2d=True)
        waveform = torch.tensor(data.T, dtype=torch.float32)
        plot_specgram(waveform=waveform, sample_rate=sr, file_path=target_file_path)
