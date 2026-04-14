import streamlit as st
import plotly.express as px

from utils import ANNOTATORS, build_filtering_status_snapshot, list_detected_versions

st.set_page_config(page_title="NAVA Breath Filtering", layout="wide")

st.title("NAVA Breath Filtering App")
st.markdown(
    "<h2 style='color:#b45309; margin-bottom:0.4rem;'>의사 전달 전 breath를 다시 쪼개는 전처리 앱입니다.</h2>",
    unsafe_allow_html=True,
)
st.write("왼쪽 사이드바 또는 아래 페이지에서 `Split_Breath` 작업을 시작하세요.")
st.page_link("pages/1_Split_Breath.py", label="Split_Breath 페이지")

st.markdown("---")
st.subheader("사용 원칙")
st.write("원본 `AA_patient_NN_clustered_breaths.pkl`는 유지하고, split 결과는 `BB_patient_NN_clustered_breaths.pkl`로 저장합니다.")
st.write("저장된 BB 파일은 의사용 라벨링 앱에서 자동으로 반영됩니다.")
st.caption(f"사용 가능한 annotator: {', '.join(ANNOTATORS)}")
st.caption(f"Data folders: {', '.join(list_detected_versions()[:5])}" if list_detected_versions() else "Data folder 없음")

st.markdown("---")
st.subheader("환자별 Split 진행 현황")

detected_versions = list_detected_versions()
status_version_options = ["ALL"] + detected_versions if detected_versions else ["ALL"]
status_annotator = st.selectbox(
    "Annotator",
    options=["ALL"] + ANNOTATORS,
    index=0,
    key="filter_status_annotator",
)
status_version = st.selectbox(
    "Data folder",
    options=status_version_options,
    index=1 if len(status_version_options) > 1 else 0,
    key="filter_status_version",
)

if st.button("진행 현황 생성", type="primary"):
    with st.spinner("환자별 split 진행률을 계산하는 중..."):
        snapshot_df = build_filtering_status_snapshot(
            annotator=status_annotator,
            version_tag=status_version,
        )
        st.session_state["filter_status_snapshot_df"] = snapshot_df

if "filter_status_snapshot_df" in st.session_state:
    snapshot_df = st.session_state["filter_status_snapshot_df"]
    done_count = int((snapshot_df["remaining"] == 0).sum()) if not snapshot_df.empty else 0
    total_count = int(len(snapshot_df))
    total_candidates = int(snapshot_df["total_candidates"].sum()) if not snapshot_df.empty else 0
    total_final_breaths = int(snapshot_df["final_breath_count"].sum()) if not snapshot_df.empty else 0
    total_split_saved = int(snapshot_df["split_saved"].sum()) if not snapshot_df.empty else 0
    total_kept = int(snapshot_df["kept_as_is"].sum()) if not snapshot_df.empty else 0
    total_removed = int(snapshot_df["removed"].sum()) if not snapshot_df.empty else 0
    total_processed = int(snapshot_df["processed"].sum()) if not snapshot_df.empty else 0
    progress_ratio = 0.0 if total_candidates == 0 else total_processed / total_candidates

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Patients Done", f"{done_count}/{total_count}")
    top2.metric("Processed", total_processed)
    top3.metric("Final Breaths", total_final_breaths)
    top4.metric("Overall Progress", f"{progress_ratio * 100:.1f}%")
    st.caption(
        f"Annotator: {status_annotator} | Data folder: {status_version} | split, keep, remove 처리와 final breath 수를 함께 보여줍니다."
    )

    chart_df = snapshot_df.copy()
    if not chart_df.empty:
        chart_df["patient_label"] = chart_df["version"].astype(str) + "/P" + chart_df["patient_id"].astype(str)
        fig = px.bar(
            chart_df,
            x="patient_label",
            y=["split_saved", "kept_as_is", "removed", "remaining"],
            barmode="stack",
            color_discrete_map={
                "split_saved": "#16a34a",
                "kept_as_is": "#2563eb",
                "removed": "#dc2626",
                "remaining": "#e5e7eb",
            },
            labels={"value": "count", "patient_label": "patient", "variable": "status"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(snapshot_df, width="stretch")
