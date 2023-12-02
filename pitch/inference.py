import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import librosa
import argparse
import numpy as np
#import crepe
from RMVPEF0Predictor import RMVPEF0Predictor
import soundfile

def save_csv_pitch(pitch, uv, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        pitch *= uv
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")  # csv for excel
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #pitch = compute_f0_sing(args.wav, device)
    predictor = RMVPEF0Predictor(hop_length=320, f0_min=50, f0_max=1100, device=device)
    audio, sampling_rate = soundfile.read(args.wav)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    pitch, uv = predictor.compute_f0_uv(audio)
    pitch = np.repeat(pitch, 2, -1)
    uv = np.repeat(uv, 2, -1)
    save_csv_pitch(pitch, uv, args.pit)
    #tmp = load_csv_pitch(args.pit)
    #save_csv_pitch(tmp, "tmp.csv")
