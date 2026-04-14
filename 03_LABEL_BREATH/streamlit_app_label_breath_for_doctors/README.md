# Breath Labeling Streamlit App For Doctors

## 실행
```bash
cd /home/jhkim/NAVA/03_LABEL_BREATH/streamlit_app_label_breath_for_doctors
streamlit run app.py
```

Conda 환경은 `NAVA`를 사용하세요.

## 입력 경로
- `/home/jhkim/NAVA/03_LABEL_BREATH/notebooks/outputs/03_breath_detect/<version>/patient_<id>/`
- 필수 파일
  - `BB_patient_<id>_clustered_breaths.pkl` 우선, 없으면 `AA_patient_<id>_clustered_breaths.pkl`
  - `patient_<id>_edi_filtered_signal.csv`
  - `patient_<id>_run_meta.csv` (optional)

## 출력 경로
- `/home/jhkim/NAVA/03_LABEL_BREATH/stored_results/04_breath_labels/<version>/labels_<annotator>.parquet`

## 페이지 구성
- `pages/1_Breath_Labeling.py`: `BB`가 있으면 split 결과가 반영된 breath를, 없으면 `AA` breath를 보고 `Normal` / `Sigh` / `Apnea` / `Hiccup` / `NotSure` 라벨 저장

## 라벨 저장 스키마
- `label_id`: uuid4
- `timestamp`: ISO 시각 문자열
- `annotator`
- `patient_id`
- `type`: `breath_label`
- `item_id`: `patient_id|breath|breath_id`
- `label`: `Normal` | `Sigh` | `Apnea` | `Hiccup` | `NotSure`
- `comment`
- `start_ts`
- `end_ts`
