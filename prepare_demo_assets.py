from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "two_stage_activity_classifier_best.pkl"
SOURCE_DATA_PATH = DATA_DIR / "sensor_ADE.csv"
FUSION_SOURCE_PATH = DATA_DIR / "fusion_df.csv"

FEATURE_BANK_PATH = DATA_DIR / "feature_bank.csv"
RAW_WINDOW_BANK_PATH = DATA_DIR / "raw_window_bank.csv"
IMPORTANCE_PATH = DATA_DIR / "importance.csv"
IMPORTANCE_META_PATH = DATA_DIR / "importance_meta.json"

WINDOW_SIZE = 50
STEP_SIZE = 25
SKIP_HEAD = 80
IMPORTANCE_SAMPLE_SIZE = 300
RANDOM_STATE = 42

TARGET_ACTIVITIES = ["Walking", "Sitting", "Standing"]


class TwoStageActivityClassifier:
    def __init__(self, bundle: dict):
        self.stage1_model = bundle["stage1_model"]
        self.stage2_model = bundle["stage2_model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.classes_ = np.array(["Walking", "Sitting", "Standing"])

    def fit(self, X=None, y=None):
        return self

    def _class_index(self, model, target_value: int) -> int:
        classes = list(model.classes_)
        return classes.index(target_value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_ordered = X[self.feature_columns]
        stage1_pred = self.stage1_model.predict(X_ordered)
        final_pred = []
        for idx, stage1_value in enumerate(stage1_pred):
            if int(stage1_value) == 1:
                final_pred.append("Walking")
            else:
                stage2_value = int(self.stage2_model.predict(X_ordered.iloc[[idx]])[0])
                final_pred.append("Sitting" if stage2_value == 0 else "Standing")
        return np.array(final_pred)

    def score(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
        pred = self.predict(X)
        y_array = np.asarray(y)
        return float(np.mean(pred == y_array))


def load_bundle() -> dict:
    bundle = joblib.load(MODEL_PATH)
    config = bundle.get("config", {})
    if config.get("window_size") != WINDOW_SIZE:
        raise ValueError(f"window_size mismatch: {config.get('window_size')} != {WINDOW_SIZE}")
    if config.get("step_size") != STEP_SIZE:
        raise ValueError(f"step_size mismatch: {config.get('step_size')} != {STEP_SIZE}")
    if config.get("skip_head") != SKIP_HEAD:
        raise ValueError(f"skip_head mismatch: {config.get('skip_head')} != {SKIP_HEAD}")
    return bundle


def load_source_df() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DATA_PATH)
    df = df[
        (df["device"] == "watch")
        & (df["activity_name"].isin(TARGET_ACTIVITIES))
        & (df["sensor"].isin(["accel", "gyro"]))
    ].copy()
    df = df.sort_values(["subject_id", "activity_name", "sensor", "timestamp"]).reset_index(drop=True)
    return df


def build_sensor_raw_windows(source_df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    sensor_df = source_df[source_df["sensor"] == sensor_name].copy()
    rows: list[pd.DataFrame] = []

    for (subject_id, activity_name), group in sensor_df.groupby(["subject_id", "activity_name"], sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group = group.iloc[SKIP_HEAD:].reset_index(drop=True)

        window_seq = 0
        for start in range(0, len(group) - WINDOW_SIZE + 1, STEP_SIZE):
            window_df = group.iloc[start:start + WINDOW_SIZE].copy().reset_index(drop=True)
            window_df["subject_id"] = subject_id
            window_df["activity_name"] = activity_name
            window_df["window_seq"] = window_seq
            window_df["t"] = np.arange(WINDOW_SIZE, dtype=int)
            window_df[f"{sensor_name}_magnitude"] = np.sqrt(
                window_df["x"] ** 2 + window_df["y"] ** 2 + window_df["z"] ** 2
            )
            rows.append(
                window_df[
                    [
                        "subject_id",
                        "activity_name",
                        "window_seq",
                        "t",
                        "x",
                        "y",
                        "z",
                        f"{sensor_name}_magnitude",
                    ]
                ].rename(
                    columns={
                        "x": f"{sensor_name}_x",
                        "y": f"{sensor_name}_y",
                        "z": f"{sensor_name}_z",
                    }
                )
            )
            window_seq += 1

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_feature_bank(bundle: dict) -> pd.DataFrame:
    fusion_df = pd.read_csv(FUSION_SOURCE_PATH)
    ordered_cols = ["subject_id", "activity_name", "window_seq"] + list(bundle["feature_columns"])
    extra_cols = [col for col in fusion_df.columns if col not in ordered_cols]
    feature_bank = fusion_df[ordered_cols + extra_cols].copy()
    feature_bank.to_csv(FEATURE_BANK_PATH, index=False, encoding="utf-8-sig")
    return feature_bank


def build_raw_window_bank(feature_bank: pd.DataFrame) -> pd.DataFrame:
    source_df = load_source_df()
    accel_raw = build_sensor_raw_windows(source_df, "accel")
    gyro_raw = build_sensor_raw_windows(source_df, "gyro")

    raw_window_bank = accel_raw.merge(
        gyro_raw,
        on=["subject_id", "activity_name", "window_seq", "t"],
        how="inner",
    )

    valid_keys = feature_bank[["subject_id", "activity_name", "window_seq"]].drop_duplicates()
    raw_window_bank = raw_window_bank.merge(
        valid_keys,
        on=["subject_id", "activity_name", "window_seq"],
        how="inner",
    )

    ordered_cols = [
        "subject_id",
        "activity_name",
        "window_seq",
        "t",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_magnitude",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "gyro_magnitude",
    ]
    raw_window_bank = raw_window_bank[ordered_cols].sort_values(
        ["subject_id", "activity_name", "window_seq", "t"]
    ).reset_index(drop=True)
    raw_window_bank.to_csv(RAW_WINDOW_BANK_PATH, index=False, encoding="utf-8-sig")
    return raw_window_bank


def stratified_sample(feature_bank: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(feature_bank) <= sample_size:
        return feature_bank.copy()

    fractions = feature_bank["activity_name"].value_counts(normalize=True)
    pieces = []
    for activity_name, fraction in fractions.items():
        activity_df = feature_bank[feature_bank["activity_name"] == activity_name]
        n_samples = max(1, int(round(sample_size * fraction)))
        n_samples = min(n_samples, len(activity_df))
        pieces.append(activity_df.sample(n=n_samples, random_state=RANDOM_STATE))

    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled


def build_importance(feature_bank: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    estimator = TwoStageActivityClassifier(bundle)
    feature_columns = list(bundle["feature_columns"])
    sampled = stratified_sample(feature_bank, IMPORTANCE_SAMPLE_SIZE)
    X = sampled[feature_columns]
    y = sampled["activity_name"]

    result = permutation_importance(
        estimator=estimator,
        X=X,
        y=y,
        scoring="accuracy",
        n_repeats=1,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    meta = {
        "sample_size": int(len(sampled)),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "skip_head": SKIP_HEAD,
        "model_path": str(MODEL_PATH.name),
    }
    IMPORTANCE_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return importance_df


def main() -> None:
    bundle = load_bundle()
    feature_bank = build_feature_bank(bundle)
    raw_window_bank = build_raw_window_bank(feature_bank)
    importance_df = build_importance(feature_bank, bundle)

    print("데모 자산 생성 완료")
    print(f"- feature_bank : {FEATURE_BANK_PATH}")
    print(f"- raw_window_bank : {RAW_WINDOW_BANK_PATH}")
    print(f"- importance : {IMPORTANCE_PATH}")
    print(f"- feature_bank rows : {len(feature_bank):,}")
    print(f"- raw_window_bank rows : {len(raw_window_bank):,}")
    print("top5 importance:")
    print(importance_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
