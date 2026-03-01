# Simple Streamlit Labeling App

## 실행
```bash
cd /home/jhkim/NAVA/03_LABEL/streamlit_app
streamlit run app.py
```

Conda 환경은 `NAVA`를 사용하세요.

## 입력 경로
- `/home/jhkim/NAVA/03_LABEL/stored_results/00_detected`
- 파일 형식: `movingwinddetected_patient_XX.xlsx`
- 시트: `data`, `params`

## 출력 경로
- `/home/jhkim/NAVA/03_LABEL/stored_results/01_labeled`
- annotator별 parquet 파일 생성:
  - `labels_이주영.parquet`
  - `labels_이지선.parquet`
  - `labels_조한나.parquet`
  - `labels_Test.parquet`

## 페이지 구성
- `pages/1_Peak.py`: detected peak 후보 검증
- `pages/2_Apnea.py`: apnea 연속구간 검증
- `pages/3_Sigh.py`: detected sigh 후보 검증

## 라벨 저장 스키마
- `label_id`: uuid4
- `timestamp`: ISO 시각 문자열
- `annotator`
- `patient_id`
- `type` (`peak` | `apnea` | `sigh`)
- `item_id`
- `label` (`O` | `X` | `NotSure`)
