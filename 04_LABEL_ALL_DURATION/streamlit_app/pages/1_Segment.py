import plotly.graph_objects as go
import streamlit as st

from utils import (
    ANNOTATORS,
    SEGMENT_REVIEW_LABELS,
    append_label,
    build_segment_candidates,
    extract_patient_id,
    extract_version_tag,
    get_labeled_item_ids,
    get_latest_comments_map,
    get_latest_labels_map,
    load_labels,
    load_patient_outputs,
    list_detected_versions,
    make_label_row,
    scan_patient_dirs,
    undo_last_label,
)

TS_PER_SECOND = 1000
WIDE_MARGIN_SEC = 30
ZOOM_MARGIN_SEC = 8


def _state_key(annotator: str, patient_dir: str, segment_type_filter: str, unlabeled_only: bool) -> str:
    return f"segment_offset::{annotator}::{patient_dir}::{segment_type_filter}::{int(unlabeled_only)}"


def _inject_label_button_colors() -> None:
    st.markdown(
        """
        <style>
        .st-key-segment_label_actions div[data-testid="stColumn"]:nth-of-type(1) button {
            background-color: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
        }
        .st-key-segment_label_actions div[data-testid="stColumn"]:nth-of-type(2) button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        .st-key-segment_label_actions div[data-testid="stColumn"]:nth-of-type(3) button {
            background-color: #f59e0b !important;
            color: #111827 !important;
            border: 1px solid #d97706 !important;
        }
        .st-key-segment_label_actions div[data-testid="stColumn"]:nth-of-type(4) button {
            background-color: #64748b !important;
            color: #ffffff !important;
            border: 1px solid #475569 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_segment(sample_df, current, margin_sec: int):
    start_ts = float(current["start_ts"])
    end_ts = float(current["end_ts"])
    center_ts = (start_ts + end_ts) / 2.0
    half = margin_sec * TS_PER_SECOND
    window_start = center_ts - half
    window_end = center_ts + half

    wdf = sample_df[(sample_df["timestamp"] >= window_start) & (sample_df["timestamp"] <= window_end)].copy()
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
            line=dict(color="#94a3b8", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wdf["time_sec_abs"],
            y=wdf["edi_smooth_for_detection"],
            mode="lines",
            name="edi_smooth",
            line=dict(color="#059669", width=2),
        )
    )

    peaks = wdf[wdf["merged_peak_mask"] == True].copy()  # noqa: E712
    if not peaks.empty:
        peaks["y_marker"] = peaks["edi_smooth_for_detection"]
        fig.add_trace(
            go.Scatter(
                x=peaks["time_sec_abs"],
                y=peaks["y_marker"],
                mode="markers",
                name="merged_peak",
                marker=dict(color="#f97316", size=8, symbol="triangle-up"),
            )
        )

    fig.add_vrect(
        x0=start_sec,
        x1=end_sec,
        fillcolor="#ef4444",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    fig.add_vline(x=start_sec, line_width=2, line_dash="dash", line_color="#dc2626")
    fig.add_vline(x=end_sec, line_width=2, line_dash="dash", line_color="#dc2626")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="time (sec, absolute)",
        yaxis_title="edi",
        xaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.18)", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.12)", gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


st.set_page_config(page_title="Segment viewer", layout="wide")
st.title("Segment viewer / labeling")
st.caption("모델이 만든 세그먼트를 확인하고 최종 라벨을 `breath / apnea / not_known / NotSure`로 저장합니다.")

with st.sidebar:
    st.subheader("설정")
    annotator = st.selectbox("Annotator", options=ANNOTATORS, index=ANNOTATORS.index("Test"))
    detected_versions = list_detected_versions()
    version_options = ["ALL"] + detected_versions if detected_versions else ["ALL"]
    selected_version = st.selectbox(
        "Data folder",
        options=version_options,
        index=1 if len(version_options) > 1 else 0,
    )
    patient_dirs = scan_patient_dirs(version_tag=selected_version)
    if not patient_dirs:
        st.warning("세그먼트 결과 폴더가 없습니다. 먼저 notebook 분석 결과를 생성해주세요.")
        st.stop()

    patient_dir = st.selectbox("Patient", options=patient_dirs)
    segment_type_filter = st.selectbox("Predicted type filter", options=["ALL", "breath", "apnea", "not_known"])
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
    sample_df, segment_df, run_meta_df = load_patient_outputs(patient_dir)
except Exception as e:
    st.error(f"세그먼트 결과 로딩 실패: {e}")
    st.stop()

try:
    labels_df = load_labels(annotator, version_tag)
except Exception as e:
    st.error(f"라벨 파일 로딩 실패: {e}")
    st.stop()

candidates = build_segment_candidates(segment_df, patient_id, segment_type_filter=segment_type_filter)
labeled_ids = get_labeled_item_ids(labels_df, patient_id, "segment")
latest_labels = get_latest_labels_map(labels_df, patient_id, "segment")
latest_comments = get_latest_comments_map(labels_df, patient_id, "segment")

if unlabeled_only:
    candidates = [c for c in candidates if c["item_id"] not in labeled_ids]

total = len(candidates)
if total == 0:
    st.info("현재 조건에 맞는 세그먼트가 없습니다.")
    st.stop()

all_candidate_ids = {c["item_id"] for c in build_segment_candidates(segment_df, patient_id, segment_type_filter=segment_type_filter)}
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

offset_key = _state_key(annotator, patient_dir, segment_type_filter, unlabeled_only)
if offset_key not in st.session_state:
    st.session_state[offset_key] = 0

if st.session_state[offset_key] >= len(candidates):
    st.session_state[offset_key] = max(0, len(candidates) - 1)

current_idx = int(st.session_state[offset_key])
current = candidates[current_idx]
current_label = latest_labels.get(current["item_id"])

comment_item_key = f"segment_comment_item::{annotator}::{patient_dir}"
comment_input_key = f"segment_comment_input::{annotator}::{patient_dir}"
if st.session_state.get(comment_item_key) != current["item_id"]:
    st.session_state[comment_item_key] = current["item_id"]
    st.session_state[comment_input_key] = latest_comments.get(current["item_id"], "")

st.caption(f"patient: {patient_id} | segment_id: {current['segment_id']} | item_id: {current['item_id']}")
st.write(f"Index: {current_idx + 1}/{len(candidates)}")

info1, info2, info3, info4 = st.columns(4)
info1.metric("Predicted", current["predicted_label"])
info2.metric("Duration (sec)", f"{current['duration_sec']:.2f}")
info3.metric("Samples", int(current["end_idx"] - current["start_idx"] + 1))
info4.metric("Merged peaks", int(current["n_peaks"]))

if current_label:
    st.info(f"현재 항목 기존 선택: {current_label}")
else:
    st.warning("현재 항목 기존 선택: 없음")

segment_row = segment_df[segment_df["segment_id"].astype(str) == current["segment_id"]].head(1)
if not segment_row.empty:
    seg_series = segment_row.iloc[0]
    st.write(
        f"Segment label: `{seg_series['segment_label']}` | "
        f"time: {float(seg_series['start_time_sec_from_start']):.2f}s - {float(seg_series['end_time_sec_from_start']):.2f}s"
    )

st.markdown("**Wide View**")
st.plotly_chart(_plot_segment(sample_df, current, WIDE_MARGIN_SEC), width="stretch")
st.markdown("**Zoom View**")
st.plotly_chart(_plot_segment(sample_df, current, ZOOM_MARGIN_SEC), width="stretch")


def _save_comment_immediately():
    try:
        new_comment = st.session_state.get(comment_input_key, "").strip()
        old_comment = latest_comments.get(current["item_id"], "").strip()
        if not new_comment or new_comment == old_comment:
            return
        label_for_comment = current_label if current_label in SEGMENT_REVIEW_LABELS else "NotSure"
        append_label(
            annotator,
            make_label_row(
                annotator,
                patient_id,
                current["item_id"],
                label_for_comment,
                current["predicted_label"],
                new_comment,
                start_ts=current["start_ts"],
                end_ts=current["end_ts"],
            ),
            version_tag=version_tag,
        )
    except Exception as e:
        st.error(f"코멘트 저장 실패: {e}")


comment_text = st.text_input(
    "comment",
    max_chars=200,
    key=comment_input_key,
    label_visibility="collapsed",
    on_change=_save_comment_immediately,
)
st.caption("코멘트 입력 후 Enter를 누르면 즉시 기록됩니다.")

_inject_label_button_colors()
with st.container(key="segment_label_actions"):
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("breath", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    current["item_id"],
                    "breath",
                    current["predicted_label"],
                    comment_text.strip() or latest_comments.get(current["item_id"], ""),
                    start_ts=current["start_ts"],
                    end_ts=current["end_ts"],
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col2.button("apnea", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    current["item_id"],
                    "apnea",
                    current["predicted_label"],
                    comment_text.strip() or latest_comments.get(current["item_id"], ""),
                    start_ts=current["start_ts"],
                    end_ts=current["end_ts"],
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col3.button("not_known", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    current["item_id"],
                    "not_known",
                    current["predicted_label"],
                    comment_text.strip() or latest_comments.get(current["item_id"], ""),
                    start_ts=current["start_ts"],
                    end_ts=current["end_ts"],
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col4.button("Not Sure", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    current["item_id"],
                    "NotSure",
                    current["predicted_label"],
                    comment_text.strip() or latest_comments.get(current["item_id"], ""),
                    start_ts=current["start_ts"],
                    end_ts=current["end_ts"],
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

if not labels_df.empty:
    history_base = labels_df[
        (labels_df["patient_id"] == patient_id)
        & (labels_df["type"] == "segment")
        & (labels_df["item_id"] == current["item_id"])
    ].copy()
    if not history_base.empty:
        st.markdown("---")
        st.subheader("라벨 기록")
        show_cols = [c for c in ["timestamp", "label", "predicted_label", "comment", "start_ts", "end_ts"] if c in history_base.columns]
        st.dataframe(history_base[show_cols].tail(10).iloc[::-1], width="stretch")

nav1, nav2, nav3 = st.columns(3)
if nav1.button("Prev", width="stretch", disabled=current_idx <= 0):
    st.session_state[offset_key] = max(0, current_idx - 1)
    st.rerun()
if nav2.button("Next", width="stretch", disabled=current_idx >= len(candidates) - 1):
    st.session_state[offset_key] = min(len(candidates) - 1, current_idx + 1)
    st.rerun()
if nav3.button("Skip", width="stretch"):
    st.session_state[offset_key] = min(len(candidates) - 1, current_idx + 1)
    st.rerun()
