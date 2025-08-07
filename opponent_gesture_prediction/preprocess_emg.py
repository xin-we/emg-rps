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

def extract_features(signal, indices, window_size, step_size):
    """Generate sliding window + RMS features."""
    data = signal.loc[indices].values
    windows = list(more_itertools.windowed(data, window_size, step=step_size))
    windows = [np.array(w) for w in windows if None not in w]

    feature_list = []
    for w in windows:
        rms = np.sqrt(np.mean(w ** 2))
        feature_list.append(list(w) + [rms])
    return pd.DataFrame(feature_list), len(windows)


def generate_per_trial_features(data, emg_signals, session_id, pid, trial_id, invalid_trials, window_size, step_size):
    """Generate train/test sets for one trial as test, rest as train."""
    participant = f"P{pid}"
    opponent_pid = 2 if pid == 1 else 1
    opponent = f"P{opponent_pid}"

    stage2_annos = []
    for i in range(1, 21):
        if session_id == 4 and i in invalid_trials[pid-1]:
            continue
        stage2_annos.append(f"S2_{i}_")

    test_anno = f"S2_{trial_id}_"
    train_annos = [a for a in stage2_annos if a != test_anno]

    train_features_ch, test_features_ch = [], []
    train_self, train_opp, train_annos_out = [], [], []
    test_self, test_opp, test_annos_out = [], [], []

    for ch in range(2):
        train_ch_feats, test_ch_feats = [], []

        train_indices = []
        for anno in train_annos:
            indices = data[data.annotation.str.startswith(anno) == True].index.tolist()
            if not indices:
                continue
            train_indices.append(indices)
        train_indices = np.concatenate(train_indices)

        test_indices = data[data.annotation.str.startswith(test_anno) == True].index.tolist()

        signal = emg_signals[pid-1][ch]
        train_mean, train_std = signal.loc[train_indices].mean(), signal.loc[train_indices].std()
        train_normalized = (signal.loc[train_indices] - train_mean) / train_std
        test_normalized = (signal.loc[test_indices] - train_mean) / train_std

        for anno in train_annos:
            indices = data[data.annotation.str.startswith(anno) == True].index.tolist()
            if not indices:
                continue

            label_self = data[f"{participant}_action"].iloc[indices[0]]
            label_opp = data[f"{opponent}_action"].iloc[indices[0]]

            feats, n_windows = extract_features(train_normalized, indices, window_size, step_size)
            train_ch_feats.append(feats)
            if ch == 0:
                train_self.extend([label_self] * n_windows)
                train_opp.extend([label_opp] * n_windows)
                train_annos_out.extend([anno] * n_windows)

        label_self = data[f"{participant}_action"].iloc[test_indices[0]]
        label_opp = data[f"{opponent}_action"].iloc[test_indices[0]]
        feats, n_windows = extract_features(test_normalized, test_indices, window_size, step_size)
        test_ch_feats.append(feats)
        if ch == 0:
            test_self.extend([label_self] * n_windows)
            test_opp.extend([label_opp] * n_windows)
            test_annos_out.extend([anno] * n_windows)

        train_features_ch.append(pd.concat(train_ch_feats, ignore_index=True))
        test_features_ch.append(pd.concat(test_ch_feats, ignore_index=True))

    # Combine both channels and add labels
    def build_df(ch_feats, opp, self_, annos):
        df = pd.concat([ch_feats[0], ch_feats[1]], axis=1)
        df.columns = [f"processed_data_{i+1}_ch1" for i in range(ch_feats[0].shape[1] - 1)] + ["rms_ch1"] + \
                     [f"processed_data_{i+1}_ch2" for i in range(ch_feats[1].shape[1] - 1)] + ["rms_ch2"]
        df["opponent"] = opp
        df["self"] = self_
        df["annotation"] = annos
        return df.reset_index(drop=True)

    train_df = build_df(train_features_ch, train_opp, train_self, train_annos_out)
    test_df = build_df(test_features_ch, test_opp, test_self, test_annos_out)

    return train_df, test_df


# -------------------- Main --------------------

def main():
    fs = 512
    f_notch = np.array([60, 120, 180, 240])
    width = np.ones(len(f_notch)) * 3
    flow, fhigh = 5, 250
    window_size = 51
    step_size = 1

    input_dir = Path("../data")
    output_dir = Path("features")
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_trials = [[7], [7, 17, 18]]

    for EID in range(1, 13):
        data_path = input_dir / f"E{EID}_data.csv"
        if not data_path.exists():
            continue

        data = pd.read_csv(data_path)

        # Filter EMG for each participant/channel
        P1_EMG = [
            pd.Series(local_filter(data["P1_CH1"] - data["P1_CH1"].iloc[0], f_notch, width, flow, fhigh, fs)),
            pd.Series(local_filter(data["P1_CH2"] - data["P1_CH2"].iloc[0], f_notch, width, flow, fhigh, fs))
        ]
        P2_EMG = [
            pd.Series(local_filter(data["P2_CH1"] - data["P2_CH1"].iloc[0], f_notch, width, flow, fhigh, fs)),
            pd.Series(local_filter(data["P2_CH2"] - data["P2_CH2"].iloc[0], f_notch, width, flow, fhigh, fs))
        ]
        emg_signals = [P1_EMG, P2_EMG]

        for pid in [1, 2]:
            if EID == 2 and pid == 1:   # Skip invalid data due to device problem
                continue
            for trial in range(1, 21):
                if EID == 4 and trial in invalid_trials[pid-1]:
                    continue
                train_df, test_df = generate_per_trial_features(
                    data, emg_signals, EID, pid, trial, invalid_trials, window_size, step_size
                )
                train_df.to_parquet(output_dir / f"E{EID}_P{pid}_train_{trial}.parquet", index=False)
                test_df.to_parquet(output_dir / f"E{EID}_P{pid}_test_{trial}.parquet", index=False)


if __name__ == "__main__":
    main()
