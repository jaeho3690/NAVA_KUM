import plotly.graph_objects as go
import streamlit as st

from utils import (
    ANNOTATORS,
    append_label,
    build_peak_candidates,
    extract_patient_id,
    extract_version_tag,
    get_labeled_item_ids,
    get_latest_labels_map,
    load_labels,
    load_xlsx,
    list_detected_versions,
    make_label_row,
    scan_xlsx_files,
    undo_last_label,
)

TS_PER_SECOND = 1000  # timestamp unit is millisecond
WIDE_WINDOW_SEC = 60
ZOOM_WINDOW_SEC = 20


def _state_key(annotator: str, patient_file: str) -> str:
    return f"peak_offset::{annotator}::{patient_file}"


def _inject_label_button_colors() -> None:
    st.markdown(
        """
        <style>
        .st-key-peak_label_actions div[data-testid="stColumn"]:nth-of-type(1) button {
            background-color: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
        }
        .st-key-peak_label_actions div[data-testid="stColumn"]:nth-of-type(2) button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        .st-key-peak_label_actions div[data-testid="stColumn"]:nth-of-type(3) button {
            background-color: #f59e0b !important;
            color: #111827 !important;
            border: 1px solid #d97706 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_peak(data_df, peak_ts, window_sec: int):
    half = int((window_sec * TS_PER_SECOND) / 2)
    start = peak_ts - half
    end = peak_ts + half
    wdf = data_df[(data_df["timestamp"] >= start) & (data_df["timestamp"] <= end)].copy()
    wdf["time_sec_abs"] = wdf["timestamp"] / TS_PER_SECOND

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wdf["time_sec_abs"],
            y=wdf["edi"],
            mode="lines",
            name="edi",
            line=dict(color="#1f77b4", width=2),
        )
    )

    y_min = float(wdf["edi"].min()) if not wdf.empty else 0.0
    y_max = float(wdf["edi"].max()) if not wdf.empty else 1.0
    y_range = max(y_max - y_min, 0.1)
    pad = y_range * 0.08

    detected_all = wdf[wdf["detected_peak"] == True].copy()  # noqa: E712
    if not detected_all.empty:
        detected_all["y_pad"] = detected_all["edi"] + pad
        fig.add_trace(
            go.Scatter(
                x=detected_all["time_sec_abs"],
                y=detected_all["y_pad"],
                mode="markers",
                name="detected_peak (model)",
                marker=dict(color="#ff7f0e", size=7, symbol="triangle-up"),
            )
        )

    gt = wdf[wdf["gt_peak"] == True]  # noqa: E712
    if not gt.empty:
        fig.add_trace(
            go.Scatter(
                x=gt["time_sec_abs"],
                y=gt["edi"],
                mode="markers",
                name="gt_peak",
                marker=dict(color="#2ca02c", size=7, symbol="circle"),
            )
        )

    fig.add_vline(x=peak_ts / TS_PER_SECOND, line_width=2, line_dash="dash", line_color="#d62728")
    if not wdf.empty:
        y_min = float(wdf["edi"].min())
        y_max = float(wdf["edi"].max())
        fig.update_yaxes(range=[y_min, y_max + 2.0])
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="time (sec, absolute)",
        yaxis_title="edi",
        xaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.18)", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.12)", gridwidth=1),
    )
    return fig


st.set_page_config(page_title="Peak 검증", layout="wide")
st.title("Peak 검증")
st.caption("대상: detected_peak 중 gt_peak와 1.5초(1500ms) 이내 매칭이 없는 항목만 표시")

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
    xlsx_files = scan_xlsx_files(version_tag=selected_version)
    if not xlsx_files:
        st.warning(f"선택한 폴더({selected_version})에 XLSX 파일이 없습니다.")
        st.stop()
    patient_file = st.selectbox("Patient file", options=xlsx_files)

    if st.button("Undo last label"):
        try:
            ok = undo_last_label(annotator, extract_version_tag(patient_file))
            if ok:
                st.success("마지막 라벨 1개를 제거했습니다.")
            else:
                st.warning("제거할 라벨이 없습니다.")
        except Exception as e:
            st.error(f"Undo 실패: {e}")

patient_id = extract_patient_id(patient_file)
version_tag = extract_version_tag(patient_file)

try:
    data_df, params_df = load_xlsx(patient_file)
except Exception as e:
    st.error(f"XLSX 로딩 실패: {e}")
    st.stop()

candidates = build_peak_candidates(data_df, patient_id, gt_match_tolerance_ms=1500)

try:
    labels_df = load_labels(annotator, version_tag)
except Exception as e:
    st.error(f"라벨 파일 로딩 실패: {e}")
    st.stop()

labeled_ids = get_labeled_item_ids(labels_df, patient_id, "peak")
latest_labels = get_latest_labels_map(labels_df, patient_id, "peak")

labeled_count = len(set(c["item_id"] for c in candidates) & labeled_ids)
remaining_count = max(0, len(candidates) - labeled_count)
total = labeled_count + remaining_count
progress = 0.0 if total == 0 else labeled_count / total

with st.sidebar:
    st.markdown("---")
    st.subheader("Progress (peak)")
    st.write(f"Labeled (unique item_id): {labeled_count}")
    st.write(f"Remaining: {remaining_count}")
    st.progress(progress)

    if not params_df.empty and {"parameter", "value"}.issubset(params_df.columns):
        st.markdown("---")
        st.subheader("Params (preview)")
        params_preview = params_df[["parameter", "value"]].head(10).copy()
        params_preview["parameter"] = params_preview["parameter"].astype(str)
        params_preview["value"] = params_preview["value"].astype(str)
        st.dataframe(params_preview, width="stretch")

if total == 0:
    st.info("이 환자에서 peak 후보가 없습니다.")
    st.stop()

offset_key = _state_key(annotator, patient_file)
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
current_ts = int(current["timestamp"])
current_label = latest_labels.get(current["item_id"])

st.caption(f"patient: {patient_id} | item_id: {current['item_id']}")
st.write(f"Index: {current_idx + 1}/{len(candidates)}")
if current_label:
    st.info(f"현재 항목 기존 선택: {current_label}")
else:
    st.warning("현재 항목 기존 선택: 없음")

st.markdown("**Wide View (~60s)**")
st.plotly_chart(_plot_peak(data_df, current_ts, WIDE_WINDOW_SEC), width="stretch")
st.markdown("**Zoom View (~20s)**")
st.plotly_chart(_plot_peak(data_df, current_ts, ZOOM_WINDOW_SEC), width="stretch")

_inject_label_button_colors()
with st.container(key="peak_label_actions"):
    col1, col2, col3 = st.columns(3)

    if col1.button("O (Correct)", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(annotator, patient_id, "peak", current["item_id"], "O"),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col2.button("X (Incorrect)", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(annotator, patient_id, "peak", current["item_id"], "X"),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col3.button("Not Sure", width="stretch"):
        try:
            append_label(
                annotator,
                make_label_row(annotator, patient_id, "peak", current["item_id"], "NotSure"),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

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
