from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVITY_LABELS = ["Walking", "Sitting", "Standing"]
ACTIVITY_META = {
    "Walking": {"label_kr": "걷기", "emoji": "🚶", "color": "#ff7a00"},
    "Standing": {"label_kr": "서기", "emoji": "🧍", "color": "#2a9d8f"},
    "Sitting": {"label_kr": "앉기", "emoji": "🪑", "color": "#457b9d"},
}


@dataclass
class DemoAssets:
    feature_bank: pd.DataFrame
    raw_window_bank: pd.DataFrame
    importance_df: pd.DataFrame
    class_profile_df: pd.DataFrame
    feature_columns: list[str]


class TwoStageActivityClassifier:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.stage1_model = bundle["stage1_model"]
        self.stage2_model = bundle["stage2_model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.classes_ = np.array(ACTIVITY_LABELS)

    def _proba_index(self, model, target_value: int) -> int:
        return list(model.classes_).index(target_value)

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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_ordered = X[self.feature_columns]
        stage1_prob = self.stage1_model.predict_proba(X_ordered)
        walk_idx = self._proba_index(self.stage1_model, 1)
        static_idx = self._proba_index(self.stage1_model, 0)

        stage2_prob = self.stage2_model.predict_proba(X_ordered)
        sit_idx = self._proba_index(self.stage2_model, 0)
        stand_idx = self._proba_index(self.stage2_model, 1)

        p_walking = stage1_prob[:, walk_idx]
        p_static = stage1_prob[:, static_idx]
        p_sitting = p_static * stage2_prob[:, sit_idx]
        p_standing = p_static * stage2_prob[:, stand_idx]
        return np.column_stack([p_walking, p_sitting, p_standing])


def load_model_bundle(model_path: Path) -> dict:
    return joblib.load(model_path)


def build_class_profiles(feature_bank: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    cols = ["activity_name"] + [col for col in top_features if col in feature_bank.columns]
    profile_df = feature_bank[cols].groupby("activity_name").mean(numeric_only=True)
    return profile_df.reset_index()


def load_demo_assets(
    feature_bank_path: Path,
    raw_window_bank_path: Path,
    importance_path: Path,
    feature_columns: list[str],
) -> DemoAssets:
    feature_bank = pd.read_csv(feature_bank_path)
    raw_window_bank = pd.read_csv(raw_window_bank_path)
    importance_df = pd.read_csv(importance_path)
    top_features = importance_df["feature"].head(5).tolist()
    class_profile_df = build_class_profiles(feature_bank, top_features)
    return DemoAssets(
        feature_bank=feature_bank,
        raw_window_bank=raw_window_bank,
        importance_df=importance_df,
        class_profile_df=class_profile_df,
        feature_columns=feature_columns,
    )


def make_window_key(row: pd.Series | dict) -> tuple[int, str, int]:
    return (int(row["subject_id"]), str(row["activity_name"]), int(row["window_seq"]))


def sample_activity_window(
    feature_bank: pd.DataFrame,
    raw_window_bank: pd.DataFrame,
    activity_name: str,
    used_keys: set[tuple[int, str, int]],
    random_state: int | None = None,
) -> dict | None:
    candidates = feature_bank[feature_bank["activity_name"] == activity_name].copy()
    if used_keys:
        candidate_keys = candidates.apply(make_window_key, axis=1)
        candidates = candidates[~candidate_keys.isin(used_keys)].copy()

    if candidates.empty:
        return None

    picked = candidates.sample(n=1, random_state=random_state).iloc[0]
    key = make_window_key(picked)
    raw_segment = raw_window_bank[
        (raw_window_bank["subject_id"] == key[0])
        & (raw_window_bank["activity_name"] == key[1])
        & (raw_window_bank["window_seq"] == key[2])
    ].copy()
    raw_segment = raw_segment.sort_values("t").reset_index(drop=True)

    return {
        "subject_id": key[0],
        "activity_name": key[1],
        "window_seq": key[2],
        "feature_row": picked.to_dict(),
        "raw_window": raw_segment.to_dict(orient="records"),
    }


def build_segment_prediction(classifier: TwoStageActivityClassifier, feature_row: dict, segment_index: int) -> dict:
    feature_df = pd.DataFrame([feature_row])
    X = feature_df[classifier.feature_columns]
    pred_label = str(classifier.predict(X)[0])
    proba = classifier.predict_proba(X)[0]
    proba_map = {label: float(proba[idx]) for idx, label in enumerate(classifier.classes_)}
    return {
        "segment_index": segment_index,
        "selected_label": feature_row["activity_name"],
        "predicted_label": pred_label,
        "confidence": float(np.max(proba)),
        "probabilities": proba_map,
        "feature_row": feature_row,
    }


def build_result_table(predictions: list[dict], segment_seconds: int) -> pd.DataFrame:
    rows = []
    for item in predictions:
        selected_label = item["selected_label"]
        predicted_label = item["predicted_label"]
        meta = ACTIVITY_META[predicted_label]
        rows.append(
            {
                "Segment": item["segment_index"] + 1,
                "Time Range": f"{item['segment_index'] * segment_seconds}~{(item['segment_index'] + 1) * segment_seconds}s",
                "Selected Class": ACTIVITY_META[selected_label]["label_kr"],
                "Predicted Class": ACTIVITY_META[predicted_label]["label_kr"],
                "Confidence": round(item["confidence"], 4),
                "Emoji": meta["emoji"],
            }
        )
    return pd.DataFrame(rows)


def build_timeline_df(sequence_items: list[dict], segment_seconds: int, sample_rate: int) -> pd.DataFrame:
    frames = []
    for idx, item in enumerate(sequence_items):
        raw_df = pd.DataFrame(item["raw_window"]).copy()
        raw_df["segment_index"] = idx
        raw_df["time_sec"] = idx * segment_seconds + raw_df["t"] / sample_rate
        frames.append(raw_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_sensor_timeline(timeline_df: pd.DataFrame, segment_count: int, segment_seconds: int):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    sensor_specs = [
        ("accel", axes[0], ["accel_x", "accel_y", "accel_z"], "Accelerometer XYZ"),
        ("gyro", axes[1], ["gyro_x", "gyro_y", "gyro_z"], "Gyroscope XYZ"),
    ]
    colors = ["#e76f51", "#2a9d8f", "#264653"]

    for _, axis, cols, title in sensor_specs:
        for col, color in zip(cols, colors):
            axis.plot(timeline_df["time_sec"], timeline_df[col], label=col, linewidth=1.6, color=color)
        for boundary in range(1, segment_count):
            axis.axvline(boundary * segment_seconds, color="#999999", linestyle="--", linewidth=1, alpha=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def summarize_final_prediction(predictions: list[dict]) -> dict:
    pred_labels = [item["predicted_label"] for item in predictions]
    counts = pd.Series(pred_labels).value_counts()
    representative_label = counts.index[0]
    representative_rows = [item for item in predictions if item["predicted_label"] == representative_label]
    representative = max(representative_rows, key=lambda item: item["confidence"])
    return {
        "label": representative_label,
        "emoji": ACTIVITY_META[representative_label]["emoji"],
        "label_kr": ACTIVITY_META[representative_label]["label_kr"],
        "vote_count": int(counts.iloc[0]),
        "representative_segment_index": representative["segment_index"],
        "representative_prediction": representative,
    }


def build_importance_view(
    importance_df: pd.DataFrame,
    feature_row: dict,
    class_profile_df: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, str]:
    top_df = importance_df.head(top_n).copy()
    top_df["Current Value"] = top_df["feature"].map(lambda col: round(float(feature_row.get(col, np.nan)), 5))

    predicted_label = feature_row.get("activity_name")
    class_row = class_profile_df[class_profile_df["activity_name"] == predicted_label]
    overall_dynamic = any("energy" in feat or "std" in feat or "magnitude" in feat for feat in top_df["feature"])

    if predicted_label == "Walking" and overall_dynamic:
        message = "현재 구간은 에너지·변동성 계열 값이 비교적 커서 걷기 패턴으로 해석되는 경향이 있습니다."
    elif predicted_label == "Standing":
        message = "현재 구간은 전반적으로 정적이며 흔들림이 낮아 서기 패턴으로 분류되는 경향이 있습니다."
    else:
        message = "현재 구간은 정적인 패턴이지만 서기보다 자세 변화가 적어 앉기 계열로 해석되는 경향이 있습니다."

    if not class_row.empty:
        for feature_name in top_df["feature"].head(2):
            mean_value = float(class_row.iloc[0][feature_name])
            current_value = float(feature_row.get(feature_name, np.nan))
            if np.isfinite(current_value) and current_value > mean_value:
                message += f" `{feature_name}` 값이 해당 클래스 평균보다 높은 편입니다."
                break

    return top_df[["feature", "importance", "Current Value"]], message
