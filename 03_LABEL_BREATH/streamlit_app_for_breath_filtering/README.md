# Breath Filtering Streamlit App

## 실행
```bash
cd <repo>/03_LABEL_BREATH
python -m venv .venv
source .venv/bin/activate
python -m pip install -r streamlit_app_for_breath_filtering/requirements.txt
streamlit run streamlit_app_for_breath_filtering/app.py
```

의존성은 `03_LABEL_BREATH/.venv`와 `streamlit_app_for_breath_filtering/requirements.txt` 기준으로 관리할 수 있습니다.
앱은 현재 파일 위치를 기준으로 `03_LABEL_BREATH` 내부 데이터 폴더를 찾도록 되어 있습니다.

## 목적
- 의사에게 전달하기 전, 혼합 breath를 여러 segment로 다시 나눕니다.
- 원본 `AA_patient_NN_clustered_breaths.pkl`는 수정하지 않습니다.
- split 결과는 `BB_patient_NN_clustered_breaths.pkl`로 저장하고, 의사용 라벨링 앱에서 이를 자동으로 읽습니다.

## 입력 경로
- `notebooks/outputs/03_breath_detect/<version>/patient_<id>/`
- 필수 파일
  - `AA_patient_<id>_clustered_breaths.pkl`
  - `patient_<id>_edi_filtered_signal.csv`

## 출력 경로
- `notebooks/outputs/03_breath_detect/<version>/patient_<id>/BB_patient_<id>_clustered_breaths.pkl`

## 저장 개념
- split 대상 breath row 1개를 여러 row로 치환합니다.
- 각 새 row는 추적 가능한 `breath_id`를 새로 받습니다.
- `AE_abnormal`, `cluster_label`, `major_cluster` 같은 메타 정보는 기존 row에서 이어집니다.

## 필요한 파이썬 패키지
- `streamlit`
- `plotly`
- `pandas`
- `numpy`
- `pyarrow`

`pyarrow`는 action log와 캐시 parquet 입출력에 필요합니다.
