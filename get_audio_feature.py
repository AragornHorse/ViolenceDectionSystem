import librosa
from moviepy.editor import *
import glob
import numpy as np


max_t = 400
n_mels = 128


def wav_to_freq_array(pth):
    data, sr = librosa.load(pth, sr=None)
    data = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=1024, n_mels=n_mels)
    data = librosa.power_to_db(data)
    return data.T


def mp4_to_freq_array(pth, alignment=True):
    audio = AudioFileClip(pth)
    if audio is None:
        return None
    sr = audio.fps
    # print(audio.to_soundarray())
    try:
        audio = np.mean(audio.to_soundarray(), axis=1)
    except:
        return None
    if np.max(np.abs(audio)) < 0.03:
        return -1
    data = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, n_mels=n_mels)
    data = librosa.power_to_db(data).T
    if alignment:
        if data.shape[0] > max_t:
            data = data[:max_t, :]
        elif data.shape[0] < max_t:
            x_ = np.zeros([max_t, n_mels])
            x_[:data.shape[0], :] = data
            data = x_
    return data


if __name__ == '__main__':

    fight_list = glob.glob(r"D:\violence_sound\fight\*.wav")
    no_fight_list = glob.glob(r"D:\violence_sound\nonfight\*.wav")


    x = []
    y = []

    for pth in fight_list:
        x_ = np.zeros([max_t, n_mels])
        _ = wav_to_freq_array(pth)
        u = min([max_t, _.shape[0]])
        x_[:u] = _[:u]
        x.append(x_)
        y.append(1)

    for pth in no_fight_list:
        x_ = np.zeros([max_t, n_mels])
        _ = wav_to_freq_array(pth)
        while True:
            u = min([max_t, _.shape[0]])
            if u < 50:
                break
            x_[:u] = _[:u]
            x.append(x_)
            y.append(0)
            _ = _[u:]


    x = np.stack(x)
    y = np.array(y)

    print(x.shape, y.shape)

    np.save(r"./x.npy", x)
    np.save(r"./y.npy", y)
