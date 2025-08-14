# Temporal posed and spontaneous gesture recognition from electromyography in the Rock-Paper-Scissors game
This repository contains code for recognizing **posed** and **spontaneous** self-gestures, as well as predicting **opponent gestures**, using EMG signals recorded during Rock-Paper-Scissors gameplay.

## Project Structure
- `opponent_gesture_prediction/`
- `self_gesture_recognition_posed/`
- `self_gesture_recognition_spontaneous/`
- `data/` (place dataset downloaded from [OSF](https://osf.io/fmrea))

## Getting Started
1. Clone this repo.
2. Download dataset from OSF and place it in the data/ folder.
3. Run preprocessing and training scripts.

## Requirements
- **AutoGluon** is supported on **Python 3.9 – 3.12** and is available on **Linux**, **macOS**, and **Windows**.
- Other dependencies can be found in the `requirements.txt` or specified within the individual project folders.

## Disk Space Requirements

Please ensure sufficient disk space before training:

- `self_gesture_recognition_posed`: ~130 GB for a full training run.
- `self_gesture_recognition_spontaneous`: ~37 GB for a single training run.
  - In the paper’s evaluation, training was repeated **five times**, requiring up to **185 GB**.
- `opponent_gesture_prediction` : ~1.3 **TB** for a full training run.

---
