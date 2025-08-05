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


# -------------------- Feature Extraction --------------------

def extract_rms_features(signal, indices, window_size, step_size):
    signal_array = np.concatenate(signal.loc[indices].values)
    windows = list(more_itertools.windowed(signal_array, window_size, step=step_size))
    windows = [np.array(w) for w in windows if None not in w]

    features = []
    for window in windows:
        rms = np.sqrt(np.mean(window**2))
        features.append(list(window) + [rms])
    return features, len(windows)


def extract_participant_features(data, emg_signal, stage1_annos, stage2_annos,
                                  invalid_stage1, invalid_stage2,
                                  EID, participant_id, window_size, step_size):
    """Extract features and labels for a participant (P1 or P2) for stage 1 and stage 2."""

    pid = participant_id
    p_str = f"P{pid}"
    train_indices = []
    test_indices = []

    # Collect valid indices
    for i, anno in enumerate(stage1_annos):
        if EID == 4 and (i + 1) in invalid_stage1[pid-1]:
            continue
        train_indices.append(data[data.annotation == anno].index.tolist())

    for i, anno in enumerate(stage2_annos):
        if EID == 4 and (i + 1) in invalid_stage2[pid-1]:
            continue
        test_indices.append(data[data.annotation.str.startswith(anno) == True].index.tolist())

    concat_train = np.concatenate(train_indices)
    concat_test = np.concatenate(test_indices)

    # Initialize containers
    train_features_all = []
    test_features_all = []
    train_classes, test_classes= [], []
    train_annos, test_annos = [], []

    # Process each channel
    for cid in range(2):
        signal = emg_signal[pid-1][cid]
        train_signal = signal.loc[concat_train]
        test_signal = signal.loc[concat_test]
        mean, std = train_signal.mean(), train_signal.std()

        normalized_train = (train_signal - mean) / std
        normalized_test = (test_signal - mean) / std

        ch_train_features = []
        ch_test_features = []

        invalid_counter_s1 = 0
        for i, anno in enumerate(stage1_annos):
            if EID == 4 and (i + 1) in invalid_stage1[pid-1]:
                invalid_counter_s1 += 1
                continue
            indices = train_indices[i - invalid_counter_s1]
            features, n_win = extract_rms_features(normalized_train, indices, window_size, step_size)
            ch_train_features.extend(features)

            if cid == 0:
                train_classes.extend(data[f"{p_str}_action"].loc[indices[:n_win]])
                train_annos.extend([anno] * n_win)

        invalid_counter_s2 = 0
        for i, anno in enumerate(stage2_annos):
            if EID == 4 and (i + 1) in invalid_stage2[pid-1]:
                invalid_counter_s2 += 1
                continue
            indices = test_indices[i - invalid_counter_s2]
            features, n_win = extract_rms_features(normalized_test, indices, window_size, step_size)
            ch_test_features.extend(features)

            if cid == 0:
                test_classes.extend(data[f"{p_str}_action"].loc[indices[:n_win]])
                test_annos.extend(data.annotation.loc[indices[:n_win]])

        train_features_all.append(pd.DataFrame(ch_train_features).reset_index(drop=True))
        test_features_all.append(pd.DataFrame(ch_test_features).reset_index(drop=True))

    return train_features_all, train_classes, train_annos, test_features_all, test_classes, test_annos


# -------------------- Main Execution --------------------

def main():
    # Filtering and windowing config
    fs = 512
    f_notch = np.array([60, 120, 180, 240])
    width = np.ones(len(f_notch)) * 3
    flow, fhigh = 5, 250
    window_size = 51
    step_size = 1

    stage1_annos = [f"S1_{i}" for i in range(1, 16)]
    stage2_annos = [f"S2_{i}_" for i in range(1, 21)]

    # invalid trials in E4 for P1 and P2
    invalid_stage1 = [[6], [6]]
    invalid_stage2 = [[7], [7, 17, 18]]

    feature_dir = Path("features")

    for EID in range(1, 13): # EID: experimental session ID (1–12)
        data = pd.read_csv(f"../data/E{EID}_data.csv")

        # Preprocess each channel
        P1_EMG = [
            pd.DataFrame(local_filter(data["P1_CH1"] - data["P1_CH1"].iloc[0], f_notch, width, flow, fhigh, fs)),
            pd.DataFrame(local_filter(data["P1_CH2"] - data["P1_CH2"].iloc[0], f_notch, width, flow, fhigh, fs)),
        ]
        P2_EMG = [
            pd.DataFrame(local_filter(data["P2_CH1"] - data["P2_CH1"].iloc[0], f_notch, width, flow, fhigh, fs)),
            pd.DataFrame(local_filter(data["P2_CH2"] - data["P2_CH2"].iloc[0], f_notch, width, flow, fhigh, fs)),
        ]
        EMG_signals = [P1_EMG, P2_EMG]

        for pid in [1, 2]:
            if EID == 2 and pid == 1:   # Skip invalid data due to device problem
                continue
            train_features, train_classes, train_annos, test_features, test_classes, test_annos = extract_participant_features(
                data, EMG_signals, stage1_annos, stage2_annos,
                invalid_stage1, invalid_stage2, EID, pid, window_size, step_size
            )

            # Prepare output
            column_names = []
            for i in range(2):
                for j in range(1, window_size + 1):
                    column_names.append(f"processed_data_{j}_ch{i+1}")
                column_names.append(f"rms_ch{i+1}")
            column_names.extend(["class", "annotation"])

            train_df = pd.concat([train_features[0], train_features[1]], axis=1)
            train_df["class"] = train_classes
            train_df["annotation"] = train_annos
            train_df.columns = column_names
            train_df.to_csv(feature_dir / f"E{EID}_P{pid}_stage1.csv", index=False)

            test_df = pd.concat([test_features[0], test_features[1]], axis=1)
            test_df["class"] = test_classes
            test_df["annotation"] = test_annos
            test_df.columns = column_names
            test_df.to_csv(feature_dir / f"E{EID}_P{pid}_stage2.csv", index=False)


if __name__ == "__main__":
    main()
