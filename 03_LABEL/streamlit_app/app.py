import streamlit as st
from datetime import datetime

from utils import ANNOTATORS, build_patient_status_snapshot, list_detected_versions

st.set_page_config(page_title="NAVA Labeling", layout="wide")

st.title("NAVA Simple Labeling App")
st.markdown(
    "<h2 style='color:#dc2626; margin-bottom:0.4rem;'>먼저 Annotator를 선택해주세요.</h2>",
    unsafe_allow_html=True,
)
st.write("왼쪽 사이드바 또는 아래 페이지에서 라벨링을 시작하세요.")
st.page_link("pages/1_Peak.py", label="Peak 검증 페이지")
st.page_link("pages/2_Apnea.py", label="Apnea region 검증 페이지")
st.page_link("pages/3_Sigh.py", label="Sigh 검증 페이지")

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

if st.button("완료 현황 생성", type="primary"):
    with st.spinner("환자별 완료 상태를 계산하는 중..."):
        snapshot_df = build_patient_status_snapshot(
            status_annotator,
            version_tag=status_version,
        )
        st.session_state["status_snapshot_df"] = snapshot_df
        st.session_state["status_snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["status_snapshot_annotator"] = status_annotator
        st.session_state["status_snapshot_version"] = status_version

if "status_snapshot_df" in st.session_state:
    snapshot_df = st.session_state["status_snapshot_df"]
    snapshot_time = st.session_state.get("status_snapshot_time", "-")
    snapshot_ann = st.session_state.get("status_snapshot_annotator", status_annotator)
    snapshot_ver = st.session_state.get("status_snapshot_version", status_version)
    if not snapshot_df.empty:
        done_count = int(
            (
                snapshot_df["peak_status"].str.contains("DONE")
                & snapshot_df["apnea_status"].str.contains("DONE")
                & snapshot_df["sigh_status"].str.contains("DONE")
            ).sum()
        )
    else:
        done_count = 0
    total_count = int(len(snapshot_df))
    st.caption(
        f"Snapshot: {snapshot_time} | Annotator: {snapshot_ann} | Data folder: {snapshot_ver} | DONE {done_count}/{total_count}"
    )
    st.dataframe(snapshot_df, width="stretch")
