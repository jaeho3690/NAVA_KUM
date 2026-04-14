# Breath Labeling Streamlit App For Doctors

## 실행
```bash
cd <repo>/03_LABEL_BREATH
streamlit run streamlit_app_label_breath_for_doctors/app.py
```

Conda 환경은 `NAVA`를 사용하세요.

## 로그인
- 앱 진입 시 공용 비밀번호 입력이 필요합니다.
- 기본 비밀번호는 코드에 설정되어 있으며, 필요하면 환경 변수 `NAVA_DOCTOR_LABEL_PASSWORD`로 덮어쓸 수 있습니다.

## 입력 경로
- `notebooks/outputs/03_breath_detect/<version>/patient_<id>/`
- 필수 파일
  - `BB_patient_<id>_clustered_breaths.pkl` 우선, 없으면 `AA_patient_<id>_clustered_breaths.pkl`
  - `patient_<id>_edi_filtered_signal.csv`
  - `patient_<id>_run_meta.csv` (optional)

## 출력 경로
- `stored_results/04_breath_labels/<version>/labels_<annotator>.parquet`

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
