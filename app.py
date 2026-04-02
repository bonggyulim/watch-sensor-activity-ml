from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from prepare_demo_assets import main as prepare_demo_assets_main
from utils import (
    ACTIVITY_META,
    DemoAssets,
    TwoStageActivityClassifier,
    build_importance_view,
    build_result_table,
    build_segment_prediction,
    build_timeline_df,
    load_demo_assets,
    load_model_bundle,
    plot_sensor_timeline,
    sample_activity_window,
    summarize_final_prediction,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "two_stage_activity_classifier_best.pkl"
FEATURE_BANK_PATH = BASE_DIR / "data" / "processed" / "feature_bank.csv"
RAW_WINDOW_BANK_PATH = BASE_DIR / "data" / "processed" / "raw_window_bank.csv"
IMPORTANCE_PATH = BASE_DIR / "data" / "processed" / "importance.csv"

WINDOW_SIZE = 50
STEP_SIZE = 25
SKIP_HEAD = 80
SEGMENT_SECONDS = 2
MAX_SEGMENTS = 5
SAMPLE_RATE = 25
PLAYBACK_STEPS = 10
PLAYBACK_DELAY = 0.08


st.set_page_config(page_title="활동 분류 데모", layout="wide")


def init_session_state() -> None:
    defaults = {
        "sequence_items": [],
        "used_keys": set(),
        "predictions": [],
        "result_table": None,
        "timeline_df": None,
        "final_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state() -> None:
    st.session_state.sequence_items = []
    st.session_state.used_keys = set()
    st.session_state.predictions = []
    st.session_state.result_table = None
    st.session_state.timeline_df = None
    st.session_state.final_summary = None


def ensure_demo_assets() -> None:
    required_paths = [FEATURE_BANK_PATH, RAW_WINDOW_BANK_PATH, IMPORTANCE_PATH]
    if all(path.exists() for path in required_paths):
        return
    prepare_demo_assets_main()


@st.cache_resource
def get_classifier() -> TwoStageActivityClassifier:
    bundle = load_model_bundle(MODEL_PATH)
    return TwoStageActivityClassifier(bundle)


@st.cache_data
def get_assets() -> DemoAssets:
    classifier = get_classifier()
    return load_demo_assets(
        feature_bank_path=FEATURE_BANK_PATH,
        raw_window_bank_path=RAW_WINDOW_BANK_PATH,
        importance_path=IMPORTANCE_PATH,
        feature_columns=classifier.feature_columns,
    )


def add_activity(activity_name: str) -> None:
    if len(st.session_state.sequence_items) >= MAX_SEGMENTS:
        return
    assets = get_assets()
    selected = sample_activity_window(
        feature_bank=assets.feature_bank,
        raw_window_bank=assets.raw_window_bank,
        activity_name=activity_name,
        used_keys=st.session_state.used_keys,
    )
    if selected is None:
        st.warning(f"{ACTIVITY_META[activity_name]['label_kr']} 후보 윈도우가 더 이상 없습니다.")
        return

    key = (selected["subject_id"], selected["activity_name"], selected["window_seq"])
    st.session_state.sequence_items.append(selected)
    st.session_state.used_keys.add(key)


def run_analysis() -> None:
    classifier = get_classifier()
    predictions = []

    status_placeholder = st.empty()
    progress_placeholder = st.progress(0.0)
    slot_placeholder = st.empty()

    total_slots = MAX_SEGMENTS * PLAYBACK_STEPS
    current_step = 0

    for idx, item in enumerate(st.session_state.sequence_items):
        prediction = build_segment_prediction(classifier, item["feature_row"], idx)
        predictions.append(prediction)

        meta = ACTIVITY_META[prediction["predicted_label"]]
        for sub_step in range(PLAYBACK_STEPS):
            current_step += 1
            elapsed_seconds = idx * SEGMENT_SECONDS + (sub_step + 1) * (SEGMENT_SECONDS / PLAYBACK_STEPS)
            progress_placeholder.progress(min(current_step / total_slots, 1.0))
            slot_placeholder.markdown(
                f"**진행 상태**  \n최대 10초 기준 `{elapsed_seconds:.1f} / {MAX_SEGMENTS * SEGMENT_SECONDS:.0f}초`"
            )
            status_placeholder.markdown(
                f"""
                <div style="padding:1.2rem 1rem;border-radius:18px;background:#f7f3eb;border:1px solid #e6dccd;text-align:center;">
                    <div style="font-size:3rem;line-height:1;">{meta['emoji']}</div>
                    <div style="font-size:1.4rem;font-weight:700;margin-top:0.4rem;">현재 예측: {meta['label_kr']}</div>
                    <div style="color:#5f5a54;">구간 {idx + 1} / {len(st.session_state.sequence_items)} · confidence {prediction['confidence']:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(PLAYBACK_DELAY)

    timeline_df = build_timeline_df(
        sequence_items=st.session_state.sequence_items,
        segment_seconds=SEGMENT_SECONDS,
        sample_rate=SAMPLE_RATE,
    )
    result_table = build_result_table(predictions, SEGMENT_SECONDS)
    final_summary = summarize_final_prediction(predictions)

    st.session_state.predictions = predictions
    st.session_state.result_table = result_table
    st.session_state.timeline_df = timeline_df
    st.session_state.final_summary = final_summary


def render_sequence_status() -> None:
    sequence_items = st.session_state.sequence_items
    st.subheader("시퀀스 구성")
    if not sequence_items:
        st.info("아직 선택된 구간이 없습니다. 걷기 / 서기 / 앉기 버튼으로 실제 윈도우를 추가하세요.")
        return

    parts = []
    for item in sequence_items:
        meta = ACTIVITY_META[item["activity_name"]]
        parts.append(f"{meta['label_kr']} {meta['emoji']}")
    st.markdown(" → ".join(parts))
    st.caption(
        f"현재 {len(sequence_items)}개 구간, 누적 {len(sequence_items) * SEGMENT_SECONDS}초 / 최대 {MAX_SEGMENTS * SEGMENT_SECONDS}초"
    )


def render_result_section() -> None:
    if st.session_state.result_table is None or st.session_state.timeline_df is None:
        return

    result_table = st.session_state.result_table.copy()
    timeline_df = st.session_state.timeline_df.copy()
    final_summary = st.session_state.final_summary
    assets = get_assets()

    st.subheader("분석 결과")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("선택 구간 수", len(st.session_state.sequence_items))
    with col2:
        st.metric("분석 길이", f"{len(st.session_state.sequence_items) * SEGMENT_SECONDS}초")
    with col3:
        st.metric("윈도우 크기", f"{WINDOW_SIZE} samples")

    st.markdown("### 구간별 요약")
    display_table = result_table.copy()
    confidence_col = "Confidence" if "Confidence" in display_table.columns else "confidence"
    if confidence_col in display_table.columns:
        display_table[confidence_col] = display_table[confidence_col].map(lambda x: f"{x:.2%}")
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("### 시계열 시각화")
    fig = plot_sensor_timeline(
        timeline_df=timeline_df,
        segment_count=len(st.session_state.sequence_items),
        segment_seconds=SEGMENT_SECONDS,
    )
    st.pyplot(fig, use_container_width=True)

    st.markdown("### 최종 결과 카드")
    # 전체 대표 결과는 구간별 예측 결과의 다수결로 계산한다.
    st.markdown(
        f"""
        <div style="padding:1.2rem 1rem;border-radius:18px;background:#fff7e8;border:1px solid #efd7a7;">
            <div style="font-size:2.6rem;line-height:1;">{final_summary['emoji']}</div>
            <div style="font-size:1.35rem;font-weight:700;margin-top:0.4rem;">대표 결과: {final_summary['label_kr']}</div>
            <div style="color:#6b5f4f;">다수결 {final_summary['vote_count']}표 · 대표 구간 {final_summary['representative_segment_index'] + 1}번</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    representative = final_summary["representative_prediction"]
    importance_feature_row = representative["feature_row"].copy()
    importance_feature_row["activity_name"] = representative["predicted_label"]
    importance_df, interpretation = build_importance_view(
        importance_df=assets.importance_df,
        feature_row=importance_feature_row,
        class_profile_df=assets.class_profile_df,
        top_n=5,
    )
    importance_df["importance"] = importance_df["importance"].map(lambda x: round(float(x), 5))

    st.markdown("### 중요 특성 TOP 5")
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    st.caption(interpretation)


ensure_demo_assets()
init_session_state()

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f4ee 0%, #fffdf8 100%); }
    h1, h2, h3 { color: #2a2926; }
    .demo-note { color:#5f5a54; font-size:0.96rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("활동 분류 Streamlit 데모")
st.markdown(
    '<div class="demo-note">실행 방법: <code>streamlit run app.py</code> · 모델 경로와 데모 자산 경로는 파일 상단 상수에서 관리합니다.</div>',
    unsafe_allow_html=True,
)

controls = st.columns([1, 1, 1, 1, 1])
with controls[0]:
    st.button("걷기 추가", use_container_width=True, on_click=add_activity, args=("Walking",))
with controls[1]:
    st.button("서기 추가", use_container_width=True, on_click=add_activity, args=("Standing",))
with controls[2]:
    st.button("앉기 추가", use_container_width=True, on_click=add_activity, args=("Sitting",))
with controls[3]:
    if st.button("초기화", use_container_width=True):
        reset_session_state()
        st.rerun()
with controls[4]:
    start_clicked = st.button("시작", use_container_width=True)

render_sequence_status()

if start_clicked:
    if not st.session_state.sequence_items:
        st.warning("최소 1개 이상의 구간을 추가한 뒤 시작해 주세요.")
    else:
        run_analysis()

render_result_section()
