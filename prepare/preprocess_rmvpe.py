import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import librosa
import torch
import argparse
from tqdm import tqdm
from multiprocessing import set_start_method
from concurrent.futures import ProcessPoolExecutor, as_completed
from RMVPEF0Predictor import RMVPEF0Predictor

def compute_f0(filename, save, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))
    audio = audio + torch.randn_like(audio) * 0.001
    predictor = RMVPEF0Predictor(hop_length=160, f0_min=50, f0_max=1100, device=device)

    pitch, uv = predictor.compute_f0_uv(audio)

    pitch = abs(pitch) * uv
    pitch = torch.from_numpy(pitch).to(torch.float32)
    pitch = pitch.squeeze(0)
    np.save(save, pitch, allow_pickle=False)


def process_file(file, wavPath, spks, pitPath, device):
    if file.endswith(".wav"):
        file = file[:-4]
        compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit", device)

def process_files_with_process_pool(wavPath, spks, pitPath, device, process_num=None):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ProcessPoolExecutor(max_workers=process_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, spks, pitPath, device): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
      set_start_method('spawn')

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            process_files_with_process_pool(wavPath, spks, pitPath, device, process_num)
