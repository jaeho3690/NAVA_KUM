from datetime import datetime

import streamlit as st

from utils import ANNOTATORS, build_patient_status_snapshot, list_detected_versions


st.set_page_config(page_title="NAVA All Duration Labeling", layout="wide")

st.title("NAVA All Duration Segment Labeling")
st.write("Edi 전체 duration을 세그먼트 단위로 검토하고 `breath / apnea / not_known` 라벨을 저장합니다.")
st.page_link("pages/1_Segment.py", label="Segment viewer / labeling")

st.markdown("---")
st.subheader("환자 완료 현황 (수동 스냅샷)")
status_annotator = st.selectbox(
    "Annotator",
    options=ANNOTATORS,
    index=ANNOTATORS.index("Test"),
    key="status_annotator",
)
detected_versions = list_detected_versions()
status_version_options = ["ALL"] + detected_versions if detected_versions else ["ALL"]
status_version = st.selectbox(
    "Data folder",
    options=status_version_options,
    index=1 if len(status_version_options) > 1 else 0,
    key="status_version",
)
status_filter = st.selectbox(
    "Segment type",
    options=["ALL", "breath", "apnea", "not_known"],
    index=0,
    key="status_filter",
)

if st.button("완료 현황 생성", type="primary"):
    with st.spinner("환자별 완료 상태를 계산하는 중..."):
        snapshot_df = build_patient_status_snapshot(
            status_annotator,
            version_tag=status_version,
            segment_type_filter=status_filter,
        )
        st.session_state["status_snapshot_df"] = snapshot_df
        st.session_state["status_snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["status_snapshot_annotator"] = status_annotator
        st.session_state["status_snapshot_version"] = status_version
        st.session_state["status_snapshot_filter"] = status_filter

if "status_snapshot_df" in st.session_state:
    snapshot_df = st.session_state["status_snapshot_df"]
    snapshot_time = st.session_state.get("status_snapshot_time", "-")
    snapshot_ann = st.session_state.get("status_snapshot_annotator", status_annotator)
    snapshot_ver = st.session_state.get("status_snapshot_version", status_version)
    snapshot_filter = st.session_state.get("status_snapshot_filter", status_filter)
    done_count = int((snapshot_df["status"] == "DONE").sum()) if not snapshot_df.empty else 0
    total_count = int(len(snapshot_df))
    st.caption(
        f"Snapshot: {snapshot_time} | Annotator: {snapshot_ann} | Data folder: {snapshot_ver} | "
        f"Segment type: {snapshot_filter} | DONE {done_count}/{total_count}"
    )
    st.dataframe(snapshot_df, width="stretch")
