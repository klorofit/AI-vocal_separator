# Vocal Separator

Всё в одном файле: `vocal_separator.py`

- U-Net
- Разделение на вокал + мелодию
- Расчёт BPM
- Папка с именем песни

---

## Установка

```bash
pip install -r requirements.txt
```

## Разделение

```bash
python vocal_separator.py song.mp3
```

Результат:

```
output/song/
├── vocals.wav
├── melody.wav
├── bpm.txt
└── info.json
```

## Обучение

```bash
python vocal_separator.py --train --epochs 10
```

```bash
python vocal_separator.py song.mp3 -w weights/vocal_unet.pth
```
