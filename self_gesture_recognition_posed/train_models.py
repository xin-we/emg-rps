import pandas as pd
import itertools
import random
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score
from pathlib import Path


def train_and_evaluate_model(df, label_col, model_root, EID, pid):
    rock = set(df[df[label_col] == 1].annotation)
    paper = set(df[df[label_col] == 2].annotation)
    scissor = set(df[df[label_col] == 3].annotation)

    all_combinations = list(itertools.product(rock, paper, scissor))
    sampled_combinations = random.sample(all_combinations, 5)

    for comb in sampled_combinations:
        test_labels = comb
        train_labels = set(df.annotation.unique()) - set(test_labels)

        test_trials = []
        for label in test_labels:
            test_trials.append(label.split('_')[-1])

        model_path  = str(model_root / f"E{EID}_P{pid}_{test_trials[0]}_{test_trials[1]}_{test_trials[2]}")

        train_data = df[df.annotation.isin(train_labels)]
        train = TabularDataset(train_data[["rms_ch1", "rms_ch2", label_col]])
        predictor = TabularPredictor(label=label_col, path=model_path, verbosity=0).fit(train)

        for label in test_labels:
            test_data = df[df.annotation == label]
            test = TabularDataset(test_data[["rms_ch1", "rms_ch2", label_col]])
            preds = predictor.predict(test.drop(columns=[label_col]))
            acc = accuracy_score(test[label_col], preds)
            print(f"E{EID}, P{pid}, {comb}, {label}, accuracy: {acc:.3f}")


def main():
    feature_dir = Path("features")
    model_root = Path("models")
    model_root.mkdir(parents=True, exist_ok=True)

    for EID in range(1, 2):
        for pid in [1, 2]:
            file_path = feature_dir / f"E{EID}_P{pid}.csv"
            if not file_path.exists():
                continue

            df = pd.read_csv(file_path)
            train_and_evaluate_model(df, label_col="class", model_root=model_root, EID=EID, pid=pid)


if __name__ == "__main__":
    main()
