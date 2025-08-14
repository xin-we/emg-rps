import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor, TabularDataset
import os

def train_and_evaluate(train_data, test_data, feature_cols, model_path, pred_path, trial, label="opponent"):
    train_dataset = TabularDataset(train_data[feature_cols])
    if os.path.exists(model_path):
        predictor = TabularPredictor.load(model_path)
    else:
        predictor = TabularPredictor(label=label, path=str(model_path), verbosity=0).fit(train_dataset)

    test_dataset = TabularDataset(test_data[feature_cols])

    y_pred = predictor.predict(test_dataset.drop(columns=[label]))
    y_pred.to_frame(name='prediction').to_parquet(pred_path)

    acc = accuracy_score(test_dataset[label], y_pred)
    print(f"trial {trial}, accuracy: {acc:.3f}")


if __name__ == "__main__":
    # Train on sessions 1–12 by default
    input_dir = Path("features")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path("predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for EID in range(1, 13):
        for pid in [1, 2]:
            print(f"E{EID}, P{pid}...")
            for trial in range(1, 21):
                train_path = input_dir / f"E{EID}_P{pid}_train_{trial}.parquet"
                test_path = input_dir / f"E{EID}_P{pid}_test_{trial}.parquet"
                if not train_path.exists() or not test_path.exists():
                    continue

                train_data = pd.read_parquet(train_path)
                test_data = pd.read_parquet(test_path)
                feature_cols = train_data.columns[:-2] # drop self and annotation
                print(feature_cols)


                label = "opponent"
                model_path = model_dir / f"E{EID}_P{pid}_{trial}"
                pred_path = pred_dir / f"E{EID}_P{pid}_{trial}.parquet"
                train_and_evaluate(train_data, test_data, feature_cols, model_path, pred_path, trial, label)
