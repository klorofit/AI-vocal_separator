# Vocal Separator

Всё в одном файле: `vocal_separator.py`

- U-Net
- Разделение на вокал + мелодию
- Расчёт BPM
- Папка с именем песни

---

## Установка

```bash
pip install -r app.txt
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
