import math
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from utils import (
    ANNOTATORS,
    build_breath_candidates,
    extract_patient_id,
    extract_version_tag,
    list_detected_versions,
    load_filter_reviews,
    load_patient_outputs,
    save_filter_review,
    save_remove_breath,
    save_split_breath_segments,
    scan_patient_dirs,
)

TS_PER_SECOND = 1000
WIDE_MARGIN_SEC = 30
ZOOM_MARGIN_SEC = 8
MATERIALIZE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "materialize_split_breaths.py"


def _state_key(annotator: str, patient_dir: str, abnormal_only: bool, unsplit_only: bool) -> str:
    return f"breath_split_offset::{annotator}::{patient_dir}::{int(abnormal_only)}::{int(unsplit_only)}"


def _is_candidate_complete(candidate, review_df, annotator: str) -> bool:
    if bool(candidate.get("is_split")):
        return True
    if review_df.empty:
        return False

    annotator_mask = review_df["annotator"].fillna("").astype(str) == str(annotator)
    original_id = str(candidate["original_breath_id"])
    breath_id = str(candidate["breath_id"])

    keep_done = not review_df[
        annotator_mask
        & (review_df["target_original_breath_id"].fillna("").astype(str) == original_id)
        & (review_df["decision"].fillna("").astype(str) == "keep")
    ].empty
    if keep_done:
        return True

    remove_done = not review_df[
        annotator_mask
        & (review_df["target_breath_id"].fillna("").astype(str) == breath_id)
        & (review_df["decision"].fillna("").astype(str) == "remove")
    ].empty
    return remove_done


def _plot_breath(signal_df, current, margin_sec: int, split_boundaries_ts=None):
    split_boundaries_ts = split_boundaries_ts or []
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
                marker=dict(color="#f97316", size=7, symbol="triangle-up"),
            )
        )

    other_df = wdf[(wdf["breath_mask"] == True) & ~((wdf["timestamp"] >= start_ts) & (wdf["timestamp"] <= end_ts))].copy()  # noqa: E712
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
        opacity=0.22,
        layer="below",
        line_width=0,
    )
    fig.add_vline(x=start_sec, line_width=2, line_dash="dash", line_color="#dc2626")
    fig.add_vline(x=end_sec, line_width=2, line_dash="dash", line_color="#dc2626")

    target_df = wdf[(wdf["timestamp"] >= start_ts) & (wdf["timestamp"] <= end_ts)].copy()
    if not target_df.empty:
        fig.add_trace(
            go.Scatter(
                x=target_df["time_sec_abs"],
                y=target_df["edi_raw"],
                mode="lines",
                name="target_breath",
                line=dict(color="rgba(220, 38, 38, 0.55)", width=4),
            )
        )

    for boundary_ts in split_boundaries_ts:
        fig.add_vline(
            x=float(boundary_ts) / TS_PER_SECOND,
            line_width=2,
            line_dash="dot",
            line_color="#2563eb",
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


def _suggest_boundary_indices(signal_df, current, split_count: int):
    start_idx = int(current["start_idx"])
    end_idx = int(current["end_idx"])
    needed = max(0, split_count - 1)
    if needed == 0:
        return []

    seg = signal_df[(signal_df["sample_idx"] >= start_idx) & (signal_df["sample_idx"] <= end_idx)].copy()
    if seg.empty:
        return []

    peak_rows = seg[seg["merged_peak_mask"] == True].copy()  # noqa: E712
    candidate_boundaries = []
    if len(peak_rows) >= 2:
        peak_idx_list = peak_rows["sample_idx"].dropna().astype(int).tolist()
        for left_peak, right_peak in zip(peak_idx_list[:-1], peak_idx_list[1:]):
            between = seg[(seg["sample_idx"] >= left_peak) & (seg["sample_idx"] <= right_peak)].copy()
            if between.empty:
                continue
            min_row = between.loc[between["edi_raw"].astype(float).idxmin()]
            candidate_boundaries.append(int(min_row["sample_idx"]))

    if len(candidate_boundaries) < needed:
        span = max(1, end_idx - start_idx)
        fallback = [
            start_idx + max(1, min(span - 1, round(span * i / split_count)))
            for i in range(1, split_count)
        ]
        candidate_boundaries.extend(fallback)

    cleaned = []
    for value in candidate_boundaries:
        value = max(start_idx + 1, min(end_idx - 1, int(value)))
        if value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= needed:
            break
    return cleaned[:needed]


def _lookup_timestamp(signal_df, sample_idx: int, fallback_ts: float) -> float:
    match = signal_df.loc[signal_df["sample_idx"] == int(sample_idx), "timestamp"]
    if match.empty:
        return float(fallback_ts)
    return float(match.iloc[0])


def _build_segments(signal_df, current, boundary_indices):
    start_idx = int(current["start_idx"])
    end_idx = int(current["end_idx"])
    sorted_boundaries = sorted(set(int(x) for x in boundary_indices))
    prev_start = start_idx
    segments = []
    for boundary_idx in sorted_boundaries:
        segments.append(
            {
                "start_idx": prev_start,
                "end_idx": boundary_idx,
                "start_ts": _lookup_timestamp(signal_df, prev_start, current["start_ts"]),
                "end_ts": _lookup_timestamp(signal_df, boundary_idx, current["end_ts"]),
            }
        )
        prev_start = boundary_idx
    segments.append(
        {
            "start_idx": prev_start,
            "end_idx": end_idx,
            "start_ts": _lookup_timestamp(signal_df, prev_start, current["start_ts"]),
            "end_ts": _lookup_timestamp(signal_df, end_idx, current["end_ts"]),
        }
    )
    return segments


st.set_page_config(page_title="Split Breath", layout="wide")
st.title("Split Breath")
st.caption("`AA_patient_NN_clustered_breaths.pkl`를 보고 필요 시 split 후 `BB_patient_NN_clustered_breaths.pkl`로 저장합니다.")
st.info("원본 AA 파일은 유지하고, split 결과는 BB 파일로 저장됩니다. 의사용 앱은 BB가 있으면 BB를 우선 읽습니다.")
st.markdown(
    """
    <style>
    .st-key-next_action button {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #15803d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("설정")
    annotator = st.selectbox("Annotator", options=ANNOTATORS, index=ANNOTATORS.index("Test"))
    detected_versions = list_detected_versions()
    version_options = ["ALL"] + detected_versions if detected_versions else ["ALL"]
    selected_version = st.selectbox("Data folder", options=version_options, index=1 if len(version_options) > 1 else 0)

    patient_dirs = scan_patient_dirs(version_tag=selected_version)
    if not patient_dirs:
        st.warning("breath detect 결과 폴더가 없습니다.")
        st.stop()

    patient_dir = st.selectbox("Patient", options=patient_dirs)
    abnormal_only = st.checkbox("Show AE_abnormal only", value=True)
    unsplit_only = st.checkbox("Show unsplit only", value=False)

patient_id = extract_patient_id(patient_dir)
version_tag = extract_version_tag(patient_dir)

try:
    signal_df, breath_df, run_meta_df = load_patient_outputs(patient_dir)
except Exception as e:
    st.error(f"환자 출력 로딩 실패: {e}")
    st.stop()

if "sample_idx" not in signal_df.columns or signal_df["sample_idx"].isna().all():
    st.error("Split_Breath 기능에는 signal CSV의 `sample_idx` 컬럼이 필요합니다.")
    st.stop()

all_candidates = build_breath_candidates(breath_df, patient_id, anomaly_only=abnormal_only)
candidates = all_candidates
if unsplit_only:
    candidates = [c for c in candidates if not c["is_split"]]

if not candidates:
    st.info("현재 조건에 맞는 breath가 없습니다.")
    st.stop()

offset_key = _state_key(annotator, patient_dir, abnormal_only, unsplit_only)
if offset_key not in st.session_state:
    first_unsplit_idx = next((i for i, c in enumerate(candidates) if not c["is_split"]), 0)
    st.session_state[offset_key] = first_unsplit_idx
if st.session_state[offset_key] >= len(candidates):
    st.session_state[offset_key] = max(0, len(candidates) - 1)

current_idx = int(st.session_state[offset_key])
current = candidates[current_idx]
review_df = load_filter_reviews(version_tag, patient_id=patient_id)
current_keep_reviewed = False
current_removed = False
if not review_df.empty:
    current_keep_reviewed = not review_df[
        (review_df["annotator"].fillna("").astype(str) == str(annotator))
        & (review_df["target_original_breath_id"].fillna("").astype(str) == str(current["original_breath_id"]))
        & (review_df["decision"].fillna("").astype(str) == "keep")
    ].empty
    current_removed = not review_df[
        (review_df["annotator"].fillna("").astype(str) == str(annotator))
        & (review_df["target_breath_id"].fillna("").astype(str) == str(current["breath_id"]))
        & (review_df["decision"].fillna("").astype(str) == "remove")
    ].empty
current_completed = bool(current["is_split"] or current_keep_reviewed or current_removed)
if current["is_split"]:
    current_status = "Split saved"
elif current_removed:
    current_status = "Removed"
elif current_keep_reviewed:
    current_status = "Kept as-is"
else:
    current_status = "Not processed"

with st.sidebar:
    st.markdown("---")
    st.subheader("Current Item")
    st.write(f"Index: {current_idx + 1}/{len(candidates)}")
    st.write(f"Patient: {patient_id}")
    st.write(f"Original Breath ID: {current['original_breath_id']}")
    st.write(f"Status: {current_status}")
    complete_bg = "#dcfce7" if current_completed else "#f3f4f6"
    complete_fg = "#166534" if current_completed else "#374151"
    complete_label = "Complete" if current_completed else "Incomplete"
    st.markdown(
        (
            "<div style='margin-top:0.35rem;'>"
            f"<span style='display:inline-block; padding:0.35rem 0.65rem; border-radius:9999px; "
            f"background:{complete_bg}; color:{complete_fg}; font-weight:700; font-size:0.92rem;'>"
            f"{complete_label}</span></div>"
        ),
        unsafe_allow_html=True,
    )

split_count_key = f"split_count::{patient_dir}::{current['item_id']}"
if split_count_key not in st.session_state:
    st.session_state[split_count_key] = 0
boundary_count = max(0, int(st.session_state[split_count_key]))
split_count = boundary_count + 1

default_boundaries = _suggest_boundary_indices(signal_df, current, split_count) if boundary_count > 0 else []
boundary_indices = []
start_idx = int(current["start_idx"])
end_idx = int(current["end_idx"])

st.caption(f"patient: {patient_id} | original_breath_id: {current['original_breath_id']} | item_id: {current['item_id']}")
st.write(f"Index: {current_idx + 1}/{len(candidates)}")

meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Duration (sec)", "-" if math.isnan(current["duration_sec"]) else f"{current['duration_sec']:.2f}")
meta2.metric("AE_abnormal", "Yes" if current["is_abnormal"] else "No")
meta3.metric("Major Cluster", "Yes" if current["major_cluster"] else "No")
meta4.metric("Current status", current_status)

status_badge_bg = "#dcfce7" if current_completed else "#fef3c7"
status_badge_fg = "#166534" if current_completed else "#92400e"
status_badge_label = "Complete" if current_completed else "Need review"
st.markdown(
    (
        f"<div style='margin:0.35rem 0 0.8rem 0;'>"
        f"<span style='display:inline-block; padding:0.45rem 0.8rem; border-radius:9999px; "
        f"background:{status_badge_bg}; color:{status_badge_fg}; font-weight:700; font-size:0.95rem;'>"
        f"{status_badge_label}</span></div>"
    ),
    unsafe_allow_html=True,
)

if current["is_split"]:
    st.info(
        f"이 항목은 이미 split된 breath입니다. split_index={current.get('split_index')} | "
        f"group={current.get('split_group_id')} | source={current.get('original_item_id')}"
    )
    if current.get("split_comment"):
        st.caption(f"Split note: {current['split_comment']}")
else:
    if current_removed:
        st.info("이 항목은 최종 breath 목록에서 제외하기로 처리되었습니다.")
    elif current_keep_reviewed:
        st.info("이 항목은 split 없이 유지하기로 처리되었습니다.")
    else:
        st.warning("현재 저장된 split/유지/제외 처리 기록이 없습니다.")

st.markdown("**Split Boundaries**")
split_cfg1, split_cfg2 = st.columns(2)
if split_cfg1.button("Add Split", width="stretch", disabled=boundary_count >= 7):
    st.session_state[split_count_key] = boundary_count + 1
    st.rerun()
if split_cfg2.button("Reset Split Preview", width="stretch", disabled=boundary_count == 0):
    st.session_state[split_count_key] = 0
    st.rerun()

if boundary_count == 0:
    st.caption("현재 split preview가 없습니다. 필요하면 `Add Split`로 경계선을 추가하세요.")

for idx in range(boundary_count):
    key = f"split_boundary::{patient_dir}::{current['item_id']}::{split_count}::{idx}"
    if key not in st.session_state:
        if idx < len(default_boundaries):
            st.session_state[key] = int(default_boundaries[idx])
        else:
            span = end_idx - start_idx
            st.session_state[key] = start_idx + max(1, min(span - 1, round(span * (idx + 1) / split_count)))
    boundary_value = st.slider(
        f"Boundary {idx + 1} sample_idx",
        min_value=start_idx + 1,
        max_value=end_idx - 1,
        value=int(st.session_state[key]),
        key=key,
    )
    boundary_indices.append(int(boundary_value))

proposed_segments = _build_segments(signal_df, current, boundary_indices)
boundary_ts = [segment["end_ts"] for segment in proposed_segments[:-1]]

st.markdown("**Wide View**")
st.plotly_chart(
    _plot_breath(signal_df, current, WIDE_MARGIN_SEC, split_boundaries_ts=boundary_ts),
    width="stretch",
)
st.markdown("**Zoom View**")
st.plotly_chart(
    _plot_breath(signal_df, current, ZOOM_MARGIN_SEC, split_boundaries_ts=boundary_ts),
    width="stretch",
)

preview_rows = []
for idx, seg in enumerate(proposed_segments, start=1):
    preview_rows.append(
        {
            "segment": idx,
            "start_idx": seg["start_idx"],
            "end_idx": seg["end_idx"],
            "start_sec": round(float(seg["start_ts"]) / TS_PER_SECOND, 3),
            "end_sec": round(float(seg["end_ts"]) / TS_PER_SECOND, 3),
            "duration_sec": round((float(seg["end_ts"]) - float(seg["start_ts"])) / TS_PER_SECOND, 3),
        }
    )
st.dataframe(preview_rows, width="stretch")

comment_key = f"split_comment::{patient_dir}::{current['item_id']}"
if comment_key not in st.session_state:
    st.session_state[comment_key] = current.get("split_comment", "")
st.text_input("Comment", key=comment_key, placeholder="예: normal+apnea mixed, 3 segments로 재분할")


def _save_split():
    if not boundary_indices:
        st.error("먼저 `Add Split`로 split 경계선을 추가해주세요.")
        return
    sorted_boundaries = sorted(boundary_indices)
    if len(sorted_boundaries) != len(set(sorted_boundaries)):
        st.error("Boundary 값이 서로 달라야 합니다.")
        return
    if any(x <= start_idx or x >= end_idx for x in sorted_boundaries):
        st.error("Boundary는 breath 내부에 있어야 합니다.")
        return

    segments = _build_segments(signal_df, current, sorted_boundaries)
    if any(seg["end_idx"] <= seg["start_idx"] for seg in segments):
        st.error("분할 결과에 길이가 0 이하인 segment가 있습니다.")
        return

    out_path = save_split_breath_segments(
        patient_dir=patient_dir,
        breath_df=breath_df,
        target_breath_id=str(current["breath_id"]),
        segments=segments,
        annotator=annotator,
        comment=st.session_state.get(comment_key, "").strip(),
    )
    st.success(f"{len(segments)}개 segment로 split을 저장했습니다: {out_path.name}")
    st.session_state[split_count_key] = 0
    st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
    st.rerun()


def _materialize_bb():
    cmd = [sys.executable, str(MATERIALIZE_SCRIPT), "--patient-dir", patient_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        st.error(f"BB 재생성 실패: {detail}")
        return
    message = (result.stdout or "").strip()
    st.success("BB 파일을 재생성했습니다.")
    if message:
        st.caption(message)
    st.rerun()


def _mark_keep_and_advance():
    save_filter_review(
        patient_dir=patient_dir,
        annotator=annotator,
        target_breath_id=str(current["breath_id"]),
        target_original_breath_id=str(current["original_breath_id"]),
        decision="keep",
        comment=st.session_state.get(comment_key, "").strip(),
    )
    st.session_state[offset_key] = min(len(candidates) - 1, current_idx + 1)
    st.rerun()


def _mark_remove_and_advance():
    save_remove_breath(
        patient_dir=patient_dir,
        breath_df=breath_df,
        annotator=annotator,
        target_breath_id=str(current["breath_id"]),
        comment=st.session_state.get(comment_key, "").strip(),
    )
    st.session_state[offset_key] = min(len(candidates) - 1, current_idx + 1)
    st.rerun()


def _go_to_next_incomplete():
    next_idx = next(
        (
            i
            for i in range(current_idx + 1, len(candidates))
            if not _is_candidate_complete(candidates[i], review_df, annotator)
        ),
        None,
    )
    if next_idx is None:
        st.info("다음 incomplete breath가 없습니다.")
        return
    st.session_state[offset_key] = next_idx
    st.rerun()


nav_row1, nav_row2 = st.columns(2)
if nav_row1.button("Previous", width="stretch", disabled=current_idx <= 0):
    st.session_state[offset_key] = max(0, current_idx - 1)
    st.rerun()
if nav_row2.button("Next", width="stretch", disabled=current_idx >= len(candidates) - 1, key="next_action"):
    _mark_keep_and_advance()

action1, action2, action3 = st.columns(3)
if action1.button("Save Split_Breath", type="primary", width="stretch"):
    _save_split()
if action2.button("Remove Breath", width="stretch"):
    _mark_remove_and_advance()
action3.empty()

st.markdown("**Finalize Output**")
st.caption("저장된 split/peak 기록을 기준으로 현재 환자의 `BB_patient_NN_clustered_breaths.pkl`를 다시 생성합니다.")
if st.button("Rebuild BB_patient pickle", width="stretch"):
    _materialize_bb()

nav1, nav2, nav3 = st.columns(3)
if nav1.button("Go to first unsplit", width="stretch"):
    first_unsplit_idx = next((i for i, c in enumerate(candidates) if not c["is_split"]), 0)
    st.session_state[offset_key] = first_unsplit_idx
    st.rerun()
nav2.empty()
if nav3.button("Go to next incomplete", width="stretch"):
    _go_to_next_incomplete()
