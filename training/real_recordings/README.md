# Real Recording Layout

Drop real WAV recordings here before retraining.

Supported classes
1. `speech`
2. `keyboard`
3. `dog_bark`
4. `traffic`
5. `siren`
6. `vacuum`
7. `music`

Accepted layouts
1. Flat per-class folders:

```text
training/real_recordings/
  speech/
    clip-001.wav
  siren/
    clip-001.wav
```

2. Pre-split folders:

```text
training/real_recordings/
  train/
    speech/
      clip-001.wav
  val/
    speech/
      clip-002.wav
  test/
    speech/
      clip-003.wav
```

Manifest build

```powershell
training\.venv\Scripts\python -m training.build_real_manifest --source-dir training\real_recordings --output-manifest training\real_recordings\manifest.jsonl
```

Training with real data

```powershell
training\.venv\Scripts\python -m training.train --manifest training\real_recordings\manifest.jsonl --output-dir training\artifacts\real-v1
```
