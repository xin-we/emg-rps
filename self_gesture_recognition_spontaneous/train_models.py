import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import accuracy_score
from pathlib import Path


def train_and_evaluate(train_df, test_df, feature_cols, label, model_path):
    """
    Train AutoGluon classifier and evaluate performance.
    """
    train_df = TabularDataset(train_df[feature_cols])
    predictor = TabularPredictor(label=label, path=model_path, verbosity=0).fit(train_df)

    stage2_annos = [f"S2_{i}_" for i in range(1, 21)]
    for i, anno in enumerate(stage2_annos):
        test_trial_df = TabularDataset(test_df[test_df.annotation.str.startswith(anno) == True][feature_cols])
        if len(test_trial_df) > 0: # skip invalid trials
            y_true = test_trial_df[label]
            y_pred = predictor.predict(test_trial_df.drop(columns=[label]))

            acc = accuracy_score(y_true, y_pred)
            print(f"trial {i+1}, accuracy: {acc:.3f}")
    return


def main():
    feature_dir = Path("features")
    model_root = Path("models")
    model_root.mkdir(parents=True, exist_ok=True)

    for EID in range(1, 2):
        for pid in [1, 2]:
            train_path = feature_dir / f"E{EID}_P{pid}_stage1.csv"
            test_path = feature_dir / f"E{EID}_P{pid}_stage2.csv"
            if not train_path.exists() or not test_path.exists():
                    continue

            feature_cols = ["rms_ch1", "rms_ch2", "class"]
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(f"E{EID}, P{pid} model training and test...")
            train_and_evaluate(
                    train_df,
                    test_df,
                    feature_cols=feature_cols,
                    label="class",
                    model_path=str(model_root / f"E{EID}_P{pid}")
                )



if __name__ == "__main__":
    main()
