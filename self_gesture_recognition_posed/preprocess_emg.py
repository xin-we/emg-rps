import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, ellip, detrend
import more_itertools
from pathlib import Path


# -------------------- Signal Processing Functions --------------------
def notch_filter(data, notch, width, fs):
    fa, fb = (notch - width) / (fs / 2), (notch + width) / (fs / 2)
    b, a = butter(4, [fa, fb], btype="stop")
    return filtfilt(b, a, data)


def elliptic_filter(data, flow, fhigh, fs):
    Wn = np.array([flow, fhigh]) * (2 / fs)
    b, a = ellip(4, 0.1, 40, Wn, btype="pass")
    return filtfilt(b, a, data)


def local_filter(data, notch, width, flow, fhigh, fs):
    signal = data
    for n, w in zip(np.atleast_1d(notch), np.atleast_1d(width)):
        signal = notch_filter(signal, n, w, fs)
    signal = elliptic_filter(signal, flow, fhigh, fs)
    return detrend(signal)


# -------------------- Feature Extraction Pipeline --------------------
def extract_features(data, EID, pid, fs, f_notch, width, flow, fhigh, window_size, step_size, invalid_trials):
    # Apply filtering to both channels
    ch1 = pd.Series(local_filter(data[f"P{pid}_CH1"] - data[f"P{pid}_CH1"].iloc[0], f_notch, width, flow, fhigh, fs))
    ch2 = pd.Series(local_filter(data[f"P{pid}_CH2"] - data[f"P{pid}_CH2"].iloc[0], f_notch, width, flow, fhigh, fs))

    # Annotations: 'S1' indicates stage 1 (calibration), 'i' indicates trial ID.
    # There are 15 trials in total, with 5 trials per gesture class.
    stage1_annos = []
    for i in range(1, 16):
        if EID == 4 and i in invalid_trials[pid-1]:
            continue
        stage1_annos.append(f"S1_{i}")

    annotations = []
    features_ch1, features_ch2, classes = [], [], []

    indices = []
    for anno in stage1_annos:
        indices.append(data[data.annotation == anno].index.tolist())
        if not indices:
            continue
    indices_list = np.concatenate(indices)

    # Normalization
    ch1[indices_list] = (ch1[indices_list] - ch1[indices_list].mean()) / ch1[indices_list].std()
    ch2[indices_list] = (ch2[indices_list] - ch2[indices_list].mean()) / ch2[indices_list].std()
    for i, anno in enumerate(stage1_annos):
        windowed_ch1 = list(more_itertools.windowed(ch1[indices[i]], window_size, step=step_size))
        windowed_ch2 = list(more_itertools.windowed(ch2[indices[i]], window_size, step=step_size))
        # Gesture type
        # 1: Rock
        # 2: Paper
        # 3: Scissors
        classes.append(data[f"P{pid}_action"][indices[i][:len(windowed_ch1)]])

        # Remove incomplete windows
        windowed_ch1 = [np.array(w) for w in windowed_ch1 if None not in w]
        windowed_ch2 = [np.array(w) for w in windowed_ch2 if None not in w]

        for w1, w2 in zip(windowed_ch1, windowed_ch2):
            rms1 = np.sqrt(np.mean(w1**2))
            rms2 = np.sqrt(np.mean(w2**2))
            features_ch1.append(list(w1) + [rms1])
            features_ch2.append(list(w2) + [rms2])
            annotations.append(anno)

    features_ch1, features_ch2 = pd.DataFrame(features_ch1), pd.DataFrame(features_ch2)
    # Assemble feature dataframe
    columns_name = []
    for i in range(2):
        for j in range(1, 52):
            columns_name.append(f"processed_data_{j}_ch{i+1}")
        columns_name.append(f"rms_ch{i+1}")
    df = pd.concat([features_ch1, features_ch2], axis=1)
    df.columns = columns_name
    classes = np.concatenate(classes)
    df["class"] = classes
    df["annotation"] = annotations
    return df


# -------------------- Main Entry Point --------------------

def main():
    # Sampling rate and filtering parameters
    fs = 512
    f_notch = np.array([60, 120, 180, 240])
    width = np.ones(len(f_notch)) * 3  # Notch bandwidth (Hz)
    flow, fhigh = 5, 250  # Bandpass limits (Hz)
    window_size = 51  # 100 ms at 512 Hz
    step_size = 1     # 1-sample stride

    input_dir = Path("../data")
    output_dir = Path("features")
    output_dir.mkdir(parents=True, exist_ok=True)

    invalid_trials = [[6], [6]] # for E4

    for EID in range(1, 13):  # EID: experimental session ID (1–12)
        file_path = input_dir / f"E{EID}_data.csv"
        if not file_path.exists():
            continue

        data = pd.read_csv(file_path)
        for pid in [1, 2]:  # Two participants per session
            if EID == 2 and pid == 1:   # Skip invalid data due to device problem
                continue
            df = extract_features(data, EID, pid, fs, f_notch, width, flow, fhigh, window_size, step_size, invalid_trials)
            df.to_parquet(output_dir / f"E{EID}_P{pid}.parquet", index=False)


if __name__ == "__main__":
    main()
