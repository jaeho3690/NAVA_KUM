import streamlit as st
import plotly.graph_objects as go

from utils import (
    ANNOTATORS,
    append_label,
    build_apnea_segments,
    clamp_interval,
    extract_patient_id,
    extract_version_tag,
    get_labeled_item_ids,
    get_latest_apnea_adjustments_map,
    get_latest_comments_map,
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
    return f"apnea_offset::{annotator}::{patient_file}"


def _adjust_key(annotator: str, patient_file: str, item_id: str) -> str:
    return f"apnea_adjust::{annotator}::{patient_file}::{item_id}"


def _inject_label_button_colors() -> None:
    st.markdown(
        """
        <style>
        .st-key-apnea_label_actions div[data-testid="stColumn"]:nth-of-type(1) button {
            background-color: #16a34a !important;
            color: #ffffff !important;
            border: 1px solid #15803d !important;
        }
        .st-key-apnea_label_actions div[data-testid="stColumn"]:nth-of-type(2) button {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
        }
        .st-key-apnea_label_actions div[data-testid="stColumn"]:nth-of-type(3) button {
            background-color: #f59e0b !important;
            color: #111827 !important;
            border: 1px solid #d97706 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_apnea(
    data_df,
    original_start_ts,
    original_end_ts,
    adjusted_start_ts,
    adjusted_end_ts,
    window_sec: int,
):
    center_ts = int((original_start_ts + original_end_ts) / 2)
    half = int((window_sec * TS_PER_SECOND) / 2)
    start = center_ts - half
    end = center_ts + half
    wdf = data_df[(data_df["timestamp"] >= start) & (data_df["timestamp"] <= end)].copy()
    wdf["time_sec_abs"] = wdf["timestamp"] / TS_PER_SECOND
    original_start_sec = original_start_ts / TS_PER_SECOND
    original_end_sec = original_end_ts / TS_PER_SECOND
    adjusted_start_sec = adjusted_start_ts / TS_PER_SECOND
    adjusted_end_sec = adjusted_end_ts / TS_PER_SECOND

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

    fig.add_vrect(
        x0=original_start_sec,
        x1=original_end_sec,
        fillcolor="red",
        opacity=0.15,
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=adjusted_start_sec,
        x1=adjusted_end_sec,
        fillcolor="red",
        opacity=0.35,
        layer="below",
        line_width=0,
    )
    fig.add_vline(x=original_start_sec, line_width=1, line_dash="dot", line_color="#f87171")
    fig.add_vline(x=original_end_sec, line_width=1, line_dash="dot", line_color="#f87171")
    fig.add_vline(x=adjusted_start_sec, line_width=2, line_dash="dash", line_color="#d62728")
    fig.add_vline(x=adjusted_end_sec, line_width=2, line_dash="dash", line_color="#d62728")
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


st.set_page_config(page_title="Apnea 검증", layout="wide")
st.title("Apnea region 검증")

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

candidates = build_apnea_segments(data_df, patient_id)

try:
    labels_df = load_labels(annotator, version_tag)
except Exception as e:
    st.error(f"라벨 파일 로딩 실패: {e}")
    st.stop()

labeled_ids = get_labeled_item_ids(labels_df, patient_id, "apnea")
latest_labels = get_latest_labels_map(labels_df, patient_id, "apnea")
latest_comments = get_latest_comments_map(labels_df, patient_id, "apnea")
latest_adjustments = get_latest_apnea_adjustments_map(labels_df, patient_id)

labeled_count = len(set(c["item_id"] for c in candidates) & labeled_ids)
remaining_count = max(0, len(candidates) - labeled_count)
total = labeled_count + remaining_count
progress = 0.0 if total == 0 else labeled_count / total

with st.sidebar:
    st.markdown("---")
    st.subheader("Progress (apnea)")
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
    st.info("이 환자에서 apnea 후보가 없습니다.")
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
start_ts = int(current["start_ts"])
end_ts = int(current["end_ts"])
current_label = latest_labels.get(current["item_id"])

data_ts_min = int(data_df["timestamp"].min())
data_ts_max = int(data_df["timestamp"].max())
adjust_state_key = _adjust_key(annotator, patient_file, current["item_id"])
if adjust_state_key not in st.session_state:
    latest_adjustment = latest_adjustments.get(current["item_id"])
    if latest_adjustment is not None:
        adjusted_start_ts, adjusted_end_ts, _ = latest_adjustment
        adjusted_start_ts, adjusted_end_ts = clamp_interval(
            adjusted_start_ts, adjusted_end_ts, data_ts_min, data_ts_max
        )
    else:
        adjusted_start_ts, adjusted_end_ts = start_ts, end_ts
    st.session_state[adjust_state_key] = (int(round(adjusted_start_ts)), int(round(adjusted_end_ts)))

adj_start_ts_state, adj_end_ts_state = st.session_state[adjust_state_key]
adj_start_ts_state, adj_end_ts_state = clamp_interval(
    adj_start_ts_state, adj_end_ts_state, data_ts_min, data_ts_max
)

window_half = int((WIDE_WINDOW_SEC * TS_PER_SECOND) / 2)
window_min = max(data_ts_min, int((start_ts + end_ts) / 2) - window_half)
window_max = min(data_ts_max, int((start_ts + end_ts) / 2) + window_half)
if window_min >= window_max:
    window_min, window_max = data_ts_min, data_ts_max
slider_min_sec = window_min / TS_PER_SECOND
slider_max_sec = window_max / TS_PER_SECOND

adj_start_ts_state, adj_end_ts_state = clamp_interval(adj_start_ts_state, adj_end_ts_state, window_min, window_max)
slider_key = f"{adjust_state_key}::slider"
if slider_key not in st.session_state:
    st.session_state[slider_key] = (
        float(adj_start_ts_state / TS_PER_SECOND),
        float(adj_end_ts_state / TS_PER_SECOND),
    )
adjusted_start_ts = int(adj_start_ts_state)
adjusted_end_ts = int(adj_end_ts_state)

st.caption(f"patient: {patient_id} | item_id: {current['item_id']}")
st.write(f"Index: {current_idx + 1}/{len(candidates)}")
if current_label:
    st.info(f"현재 항목 기존 선택: {current_label}")
else:
    st.warning("현재 항목 기존 선택: 없음")
st.markdown("**Wide View (~60s)**")
st.plotly_chart(
    _plot_apnea(
        data_df,
        start_ts,
        end_ts,
        adjusted_start_ts,
        adjusted_end_ts,
        WIDE_WINDOW_SEC,
    ),
    width="stretch",
)
adjusted_start_sec, adjusted_end_sec = st.slider(
    "Adjust apnea interval (sec, absolute)",
    min_value=float(slider_min_sec),
    max_value=float(slider_max_sec),
    step=0.1,
    key=slider_key,
)
adjusted_start_ts = int(round(adjusted_start_sec * TS_PER_SECOND))
adjusted_end_ts = int(round(adjusted_end_sec * TS_PER_SECOND))
adjusted_start_ts, adjusted_end_ts = clamp_interval(
    adjusted_start_ts, adjusted_end_ts, window_min, window_max
)
if adjusted_start_ts == adjusted_end_ts:
    if adjusted_end_ts < window_max:
        adjusted_end_ts += 1
    elif adjusted_start_ts > window_min:
        adjusted_start_ts -= 1
st.session_state[adjust_state_key] = (int(adjusted_start_ts), int(adjusted_end_ts))
st.markdown("**Zoom View (~20s)**")
st.plotly_chart(
    _plot_apnea(
        data_df,
        start_ts,
        end_ts,
        adjusted_start_ts,
        adjusted_end_ts,
        ZOOM_WINDOW_SEC,
    ),
    width="stretch",
)

comment_item_key = f"apnea_comment_item::{annotator}::{patient_file}"
comment_input_key = f"apnea_comment_input::{annotator}::{patient_file}"
if st.session_state.get(comment_item_key) != current["item_id"]:
    st.session_state[comment_item_key] = current["item_id"]
    st.session_state[comment_input_key] = latest_comments.get(current["item_id"], "")

annotator_color_map = {
    "이주영": "#0ea5e9",
    "이지선": "#22c55e",
    "조한나": "#f59e0b",
    "Test": "#ef4444",
}
annotator_color = annotator_color_map.get(annotator, "#2563eb")
st.markdown(
    f"**Optional comment (<span style='color:{annotator_color}'>{annotator}</span> 의사 선생님 코멘트)**",
    unsafe_allow_html=True,
)

def _save_comment_immediately():
    try:
        new_comment = st.session_state.get(comment_input_key, "").strip()
        old_comment = latest_comments.get(current["item_id"], "").strip()
        if not new_comment or new_comment == old_comment:
            return
        label_for_comment = current_label if current_label in {"O", "X", "NotSure"} else "NotSure"
        append_label(
            annotator,
            make_label_row(
                annotator,
                patient_id,
                "apnea",
                current["item_id"],
                label_for_comment,
                new_comment,
                start_ts=adjusted_start_ts,
                end_ts=adjusted_end_ts,
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
with st.container(key="apnea_label_actions"):
    col1, col2, col3 = st.columns(3)

    if col1.button("O (Correct)", width="stretch"):
        try:
            save_comment = comment_text.strip() or latest_comments.get(current["item_id"], "")
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    "apnea",
                    current["item_id"],
                    "O",
                    save_comment,
                    start_ts=adjusted_start_ts,
                    end_ts=adjusted_end_ts,
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col2.button("X (Incorrect)", width="stretch"):
        try:
            save_comment = comment_text.strip() or latest_comments.get(current["item_id"], "")
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    "apnea",
                    current["item_id"],
                    "X",
                    save_comment,
                    start_ts=adjusted_start_ts,
                    end_ts=adjusted_end_ts,
                ),
                version_tag=version_tag,
            )
            st.session_state[offset_key] = min(current_idx + 1, len(candidates) - 1)
            st.rerun()
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if col3.button("Not Sure", width="stretch"):
        try:
            save_comment = comment_text.strip() or latest_comments.get(current["item_id"], "")
            append_label(
                annotator,
                make_label_row(
                    annotator,
                    patient_id,
                    "apnea",
                    current["item_id"],
                    "NotSure",
                    save_comment,
                    start_ts=adjusted_start_ts,
                    end_ts=adjusted_end_ts,
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
        & (labels_df["type"] == "apnea")
        & (labels_df["item_id"] == current["item_id"])
    ].copy()
    if "comment" in history_base.columns:
        history_base["comment"] = history_base["comment"].fillna("").astype(str).str.strip()
        history_base = history_base[history_base["comment"] != ""]
    else:
        history_base = history_base.iloc[0:0]

    if not history_base.empty:
        st.markdown("---")
        st.subheader("코멘트 기록")
        st.dataframe(
            history_base[["timestamp", "comment"]].tail(10).iloc[::-1],
            width="stretch",
        )

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
