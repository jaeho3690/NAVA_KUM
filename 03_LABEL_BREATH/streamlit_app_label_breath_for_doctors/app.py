import streamlit as st

from utils import ANNOTATORS, build_patient_status_snapshot, list_detected_versions

st.set_page_config(page_title="NAVA Breath Labeling For Doctors", layout="wide")

st.title("NAVA Breath Labeling App For Doctors")
st.markdown(
    "<h2 style='color:#dc2626; margin-bottom:0.4rem;'>먼저 Annotator를 선택해주세요.</h2>",
    unsafe_allow_html=True,
)
st.write("왼쪽 사이드바 또는 아래 페이지에서 split 반영된 breath 라벨링을 시작하세요.")
st.page_link("pages/1_Breath_Labeling.py", label="Breath labeling 페이지")

st.markdown("---")
st.subheader("환자 완료 현황")

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
        snapshot_df = build_patient_status_snapshot(status_annotator, version_tag=status_version)
        st.session_state["breath_status_snapshot_df"] = snapshot_df

if "breath_status_snapshot_df" in st.session_state:
    snapshot_df = st.session_state["breath_status_snapshot_df"]
    done_count = int((snapshot_df["remaining"] == 0).sum()) if not snapshot_df.empty else 0
    total_count = int(len(snapshot_df))
    st.caption(
        f"Annotator: {status_annotator} | Data folder: {status_version} | DONE {done_count}/{total_count}"
    )
    st.dataframe(snapshot_df, width="stretch")
