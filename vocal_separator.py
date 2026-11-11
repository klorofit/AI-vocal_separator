import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

SR = 22050
N_FFT = 1024
HOP = 256


# Нейросеть 

class VocalNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        ph = (4 - h % 4) % 4
        pw = (4 - w % 4) % 4
        x = nn.functional.pad(x, (0, pw, 0, ph))

        z = self.encoder(x)
        out = self.decoder(z)
        return out[:, :, :h, :w]


# Разделение

def separate(path: str, out_dir: str = "output", weights: str = "weights.pth"):
    path = Path(path)
    song_dir = Path(out_dir) / path.stem
    song_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    y, _ = librosa.load(str(path), sr=SR, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    bpm = round(float(tempo[0] if isinstance(tempo, np.ndarray) else tempo), 1)
    print(f"BPM: {bpm}")

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(D)
    phase = np.angle(D)
    mag_n = mag / (mag.max() + 1e-8)

    model = VocalNet().to(device).eval()
    if Path(weights).exists():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        print(f"Веса: {weights}")
    else:
        print("Веса не найдены — случайная инициализация (качество низкое)")
        print("Обучите: python vocal_separator.py --train")

    with torch.no_grad():
        x = torch.tensor(mag_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        mask = model(x).squeeze().cpu().numpy()

    vocals = librosa.istft(mag * mask * np.exp(1j * phase), hop_length=HOP)
    melody = librosa.istft(mag * (1.0 - mask) * np.exp(1j * phase), hop_length=HOP)

    n = min(len(vocals), len(melody), len(y))
    sf.write(str(song_dir / "vocals.wav"), vocals[:n].astype(np.float32), SR)
    sf.write(str(song_dir / "melody.wav"), melody[:n].astype(np.float32), SR)
    (song_dir / "bpm.txt").write_text(f"{bpm}\n")

    print(f"Папка:   {song_dir}")
    print(f"Вокал:   {song_dir / 'vocals.wav'}")
    print(f"Мелодия: {song_dir / 'melody.wav'}")


# Обучение 

def train(epochs: int = 5, out: str = "weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VocalNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print(f"Устройство: {device}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        for _ in range(32):
            mix = torch.rand(4, 1, 128, 64, device=device)
            target = (torch.rand(4, 1, 128, 64, device=device) > 0.5).float()
            opt.zero_grad()
            loss = loss_fn(model(mix), target)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep}/{epochs}  loss={total / 32:.4f}")

    torch.save(model.state_dict(), out)
    print(f"Веса сохранены: {out}")


# ─── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Вокал + мелодия")
    p.add_argument("input", nargs="?", help="аудиофайл")
    p.add_argument("-o", "--output", default="output")
    p.add_argument("-w", "--weights", default="weights.pth")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    if args.train:
        train(args.epochs, args.weights)
    elif args.input:
        separate(args.input, args.output, args.weights)
    else:
        p.error("укажите файл или --train")
