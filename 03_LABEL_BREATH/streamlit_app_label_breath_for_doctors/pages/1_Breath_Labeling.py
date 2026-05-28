import math

import plotly.graph_objects as go
import streamlit as st

from auth import require_shared_password
from utils import (
    ANNOTATOR_COLORS,
    ANNOTATORS,
    BREATH_LABELS,
    append_label,
    build_breath_candidates,
    extract_patient_id,
    extract_version_tag,
    get_labeled_item_ids,
    get_latest_comments_map,
    get_latest_labels_map,
    list_detected_versions,
    load_labels,
    load_patient_outputs,
    make_label_row,
    scan_patient_dirs,
    undo_last_label,
)

TS_PER_SECOND = 1000
WIDE_MARGIN_SEC = 30
ZOOM_MARGIN_SEC = 8


def _state_key(annotator: str, patient_dir: str, anomaly_only: bool, unlabeled_only: bool) -> str:
    return f"breath_offset::{annotator}::{patient_dir}::{int(anomaly_only)}::{int(unlabeled_only)}"


def _inject_label_button_colors() -> None:
    st.markdown(
        """
        <style>
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(1) button {
            background-color: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(2) button {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: 1px solid #1d4ed8 !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(3) button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(4) button {
            background-color: #7c3aed !important;
            color: #ffffff !important;
            border: 1px solid #6d28d9 !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(5) button {
            background-color: #14b8a6 !important;
            color: #ffffff !important;
            border: 1px solid #0f766e !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(6) button {
            background-color: #ec4899 !important;
            color: #ffffff !important;
            border: 1px solid #db2777 !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(7) button {
            background-color: #f59e0b !important;
            color: #111827 !important;
            border: 1px solid #d97706 !important;
        }
        .st-key-breath_label_actions div[data-testid="stColumn"]:nth-of-type(8) button {
            background-color: #6b7280 !important;
            color: #ffffff !important;
            border: 1px solid #4b5563 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_breath(signal_df, current, margin_sec: int):
    start_ts = float(current["start_ts"])
    end_ts = float(current["end_ts"])
    center_ts = (start_ts + end_ts) / 2.0
    half = margin_sec * TS_PER_SECOND
    window_start = center_ts - half
    window_end = center_ts + half

    wdf = signal_df[(signal_df["timestamp"] >= window_start) & (signal_df["timestamp"] <= window_end)].copy()
    if wdf.empty:
        return go.Figure()

    wdf["time_sec_abs"] = wdf["timestamp"] / TS_PER_SECOND
    start_sec = start_ts / TS_PER_SECOND
    end_sec = end_ts / TS_PER_SECOND

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wdf["time_sec_abs"],
            y=wdf["edi_raw"],
            mode="lines",
            name="edi_raw",
            line=dict(color="#0f172a", width=1.8),
        )
    )

    peaks = wdf[wdf["merged_peak_mask"] == True].copy()  # noqa: E712
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["time_sec_abs"],
                y=peaks["edi_raw"],
                mode="markers",
                name="merged_peak",
                marker=dict(color="#f97316", size=12, symbol="triangle-up"),
            )
        )

    target_mask = (wdf["timestamp"] >= start_ts) & (wdf["timestamp"] <= end_ts)
    target_df = wdf[target_mask].copy()
    other_df = wdf[(wdf["breath_mask"] == True) & (~target_mask)].copy()  # noqa: E712

    if not other_df.empty:
        for _, breath_chunk in other_df.groupby("breath_id", dropna=True):
            if breath_chunk.empty:
                continue
            fig.add_vrect(
                x0=float(breath_chunk["timestamp"].min()) / TS_PER_SECOND,
                x1=float(breath_chunk["timestamp"].max()) / TS_PER_SECOND,
                fillcolor="#cbd5e1",
                opacity=0.12,
                layer="below",
                line_width=0,
            )

    fig.add_vrect(
        x0=start_sec,
        x1=end_sec,
        fillcolor="#ef4444",
        opacity=0.24,
        layer="below",
        line_width=0,
    )
    fig.add_vline(x=start_sec, line_width=2, line_dash="dash", line_color="#dc2626")
    fig.add_vline(x=end_sec, line_width=2, line_dash="dash", line_color="#dc2626")

    if not target_df.empty:
        fig.add_trace(
            go.Scatter(
                x=target_df["time_sec_abs"],
                y=target_df["edi_raw"],
                mode="lines",
                name="target_breath",
                line=dict(color="rgba(220, 38, 38, 0.45)", width=4),
            )
        )

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="time (sec, absolute)",
        yaxis_title="edi",
        xaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.18)", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.12)", gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


st.set_page_config(page_title="Breath labeling", layout="wide")
require_shared_password("Breath labeling")
st.title("Breath labeling")
st.caption(
    "`AA/BB_patient_NN_clustered_breaths.pkl`의 breath를 보고 "
    "`Phasic`, `Sigh`, `Apnea`, `Hiccup`, `Tonic Burst`, `Crying`, `NeedSplit`, `NotSure` 라벨을 저장합니다."
)
st.info(
    "간단 가이드: "
    "`Phasic` - 정상 위상 호흡 | "
    "`Sigh` - 한숨 | "
    "`Apnea` - 무호흡 | "
    "`Hiccup` - 딸국질 | "
    "`Tonic Burst` - 긴장성 버스트 | "
    "`Crying` - 울음 | "
    "`NeedSplit` - 여러 조건이 겹쳐져 있는 호흡 | "
    "`NotSure` - 선택지에 없음. Comment 작성 요함."
)

with st.sidebar:
    st.subheader("설정")
    annotator = st.selectbox("Annotator", options=ANNOTATORS, index=ANNOTATORS.index("Test"))
    _color = ANNOTATOR_COLORS.get(annotator, "#4b5563")
    st.markdown(
        f"<div style='background:{_color};color:#fff;padding:6px 12px;"
        f"border-radius:6px;font-weight:bold;font-size:1.05em;margin:4px 0 8px 0'>"
        f"현재 Annotator: {annotator}</div>",
        unsafe_allow_html=True,
    )
    detected_versions = list_detected_versions()
    version_options = ["ALL"] + detected_versions if detected_versions else ["ALL"]
    selected_version = st.selectbox(
        "Data folder",
        options=version_options,
        index=1 if len(version_options) > 1 else 0,
    )

    patient_dirs = scan_patient_dirs(version_tag=selected_version)
    if not patient_dirs:
        st.warning("breath detect 결과 폴더가 없습니다.")
        st.stop()

    patient_dir = st.selectbox("Patient", options=patient_dirs)
    anomaly_only = st.checkbox("Show anomaly breaths only", value=True)
    unlabeled_only = st.checkbox("Show unlabeled only", value=False)

    if st.button("Undo last label"):
        try:
            ok = undo_last_label(annotator, extract_version_tag(patient_dir))
            if ok:
                st.success("마지막 라벨 1개를 제거했습니다.")
            else:
                st.warning("제거할 라벨이 없습니다.")
        except Exception as e:
            st.error(f"Undo 실패: {e}")

patient_id = extract_patient_id(patient_dir)
version_tag = extract_version_tag(patient_dir)

try:
    signal_df, breath_df, run_meta_df = load_patient_outputs(patient_dir)
except Exception as e:
    st.error(f"환자 출력 로딩 실패: {e}")
    st.stop()

try:
    labels_df = load_labels(annotator, version_tag)
except Exception as e:
    st.error(f"라벨 파일 로딩 실패: {e}")
    st.stop()

all_candidates = build_breath_candidates(breath_df, patient_id, anomaly_only=anomaly_only)
labeled_ids = get_labeled_item_ids(labels_df, patient_id, "breath_label")
latest_labels = get_latest_labels_map(labels_df, patient_id, "breath_label")
latest_comments = get_latest_comments_map(labels_df, patient_id, "breath_label")

candidates = all_candidates
if unlabeled_only:
    candidates = [c for c in candidates if c["item_id"] not in labeled_ids]

total = len(candidates)
if total == 0:
    st.info("현재 조건에 맞는 breath가 없습니다.")
    st.stop()

all_candidate_ids = {c["item_id"] for c in all_candidates}
labeled_count = len(all_candidate_ids & labeled_ids)
remaining_count = max(0, len(all_candidate_ids) - labeled_count)
progress = 0.0 if len(all_candidate_ids) == 0 else labeled_count / len(all_candidate_ids)

with st.sidebar:
    st.markdown("---")
    st.subheader("Progress")
    st.write(f"Labeled: {labeled_count}")
    st.write(f"Remaining: {remaining_count}")
    st.progress(progress)

    if not run_meta_df.empty:
        st.markdown("---")
        st.subheader("Run meta")
        st.dataframe(run_meta_df, width="stretch")

offset_key = _state_key(annotator, patient_dir, anomaly_only, unlabeled_only)
if offset_key not in st.session_state:
    first_unlabeled_idx = next(
        (i for i, c in enumerate(candidates) if c["item_id"] not in labeled_ids),
        0,
    )
    st.session_state[offset_key] = first_unlabeled_idx

if st.session_state[offset_key] >= len(candidates):
    st.session_state[offset_key] = max(0, len(candidates) - 1)

current_idx = int(st.session_state[offset_key])
current = candidates[current_idx]
current_label = latest_labels.get(current["item_id"])

with st.sidebar:
    st.markdown("---")
    st.subheader("Current Item")
    st.write(f"Index: {current_idx + 1}/{len(candidates)}")
    st.write(f"Patient: {patient_id}")
    st.write(f"Breath ID: {current['breath_id']}")

comment_item_key = f"breath_comment_item::{annotator}::{patient_dir}"
comment_input_key = f"breath_comment_input::{annotator}::{patient_dir}"
if st.session_state.get(comment_item_key) != current["item_id"]:
    st.session_state[comment_item_key] = current["item_id"]
    st.session_state[comment_input_key] = latest_comments.get(current["item_id"], "")

st.markdown(
    f"<span style='background:{ANNOTATOR_COLORS.get(annotator, '#4b5563')};color:#fff;padding:3px 10px;"
    f"border-radius:4px;font-weight:bold'>{annotator}</span>&nbsp;&nbsp;"
    f"<span style='color:#6b7280;font-size:0.9em'>"
    f"patient: {patient_id} | breath_id: {current['breath_id']} | "
    f"original_breath_id: {current['original_breath_id']} | item_id: {current['item_id']}</span>",
    unsafe_allow_html=True,
)
st.write(f"Index: {current_idx + 1}/{len(candidates)}")

top1, top2 = st.columns(2)
top1.metric("Abnormal", "Yes" if current["is_abnormal"] else "No")
top2.metric("Major cluster", "Yes" if current["major_cluster"] else "No")

info1, info2, info3 = st.columns(3)
info1.metric("Duration (sec)", "-" if math.isnan(current["duration_sec"]) else f"{current['duration_sec']:.2f}")
info2.metric("Anomaly score", "-" if math.isnan(current["anomaly_score"]) else f"{current['anomaly_score']:.5f}")
info3.metric("Breath label", current["breath_label"])

if current.get("is_split"):
    st.info(
        f"이 항목은 split된 breath입니다. split_index={current.get('split_index')} | "
        f"group={current.get('split_group_id')} | source={current.get('original_item_id')}"
    )
    if current.get("split_comment"):
        st.caption(f"Split note: {current['split_comment']}")

if current_label:
    st.info(f"현재 항목 기존 선택: {current_label}")
else:
    st.warning("현재 항목 기존 선택: 없음")

st.markdown("**Wide View**")
st.plotly_chart(_plot_breath(signal_df, current, WIDE_MARGIN_SEC), width="stretch")
st.markdown("**Zoom View**")
st.plotly_chart(_plot_breath(signal_df, current, ZOOM_MARGIN_SEC), width="stretch")

st.text_input(
    "Comment",
    key=comment_input_key,
    placeholder="선택 이유나 메모가 있으면 남겨주세요. 메모를 쓰고 Enter를 꼭 눌러야 저장됩니다.",
)


def _save_and_advance(label: str) -> None:
    append_label(
        annotator,
        make_label_row(
            annotator=annotator,
            patient_id=patient_id,
            label_type="breath_label",
            item_id=current["item_id"],
            label=label,
            comment=st.session_state.get(comment_input_key, "").strip(),
            start_ts=current["start_ts"],
            end_ts=current["end_ts"],
        ),
        version_tag=version_tag,
    )
    st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
    st.rerun()


_inject_label_button_colors()
with st.container(key="breath_label_actions"):
    label_columns = st.columns(len(BREATH_LABELS))
    for col, label_name in zip(label_columns, BREATH_LABELS):
        if col.button(label_name, width="stretch"):
            try:
                _save_and_advance(label_name)
            except Exception as e:
                st.error(f"저장 실패: {e}")

nav1, nav2, nav3 = st.columns(3)
if nav1.button("Previous", width="stretch", disabled=current_idx <= 0):
    st.session_state[offset_key] = max(0, current_idx - 1)
    st.rerun()
if nav2.button("Next", width="stretch", disabled=current_idx >= len(candidates) - 1):
    st.session_state[offset_key] = min(len(candidates) - 1, current_idx + 1)
    st.rerun()
if nav3.button("Go to first unlabeled", width="stretch"):
    first_unlabeled_idx = next(
        (i for i, c in enumerate(candidates) if c["item_id"] not in labeled_ids),
        0,
    )
    st.session_state[offset_key] = first_unlabeled_idx
    st.rerun()
