"""Offline transcription of one six-axis NPZ recording using a trained checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .model import CNNBiLSTMCTC
from .text import Vocabulary

def transcribe(path: str|Path, checkpoint: str|Path, device: str = "cpu") -> str:
    saved = torch.load(checkpoint,map_location=device,weights_only=True)
    vocab = Vocabulary(saved["vocabulary"])
    model = CNNBiLSTMCTC(**saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval().to(device)
    with np.load(path, allow_pickle=False) as data:
        if "axes" not in data: raise ValueError("NPZ must contain six-channel axes")
        x = np.asarray(data["axes"],dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite IMU array [T,6]")
    mean = np.asarray(saved["normalizer_mean"])
    std = np.asarray(saved["normalizer_std"])
    # OnHW and ESP32 pen have different rates/units/axes until calibrated.
    batch = torch.from_numpy(((x-mean)/std).astype("float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        logp, lengths = model(batch,torch.tensor([len(x)]))
    return vocab.greedy(logp[:int(lengths[0]),0].argmax(-1).tolist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--recording",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    print(transcribe(args.recording,args.checkpoint,args.device))

if __name__=="__main__":
    main()
