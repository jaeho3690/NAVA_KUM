# NAVA Streamlit Labeling App - 전체 요약

## 1. 프로젝트 개요

**목적**: NAVA(Neurally Adjusted Ventilatory Assist) 데이터에서 모델이 자동 탐지한 Peak, Apnea, Sigh 후보를 의료진이 검증하고 라벨링하는 웹 기반 앱

**기술 스택**: Streamlit, Pandas, Plotly, Parquet

**실행 방법**:
```bash
cd /home/jhkim/NAVA/03_LABEL/streamlit_app
streamlit run app.py
```

---

## 2. 디렉토리 구조

```
streamlit_app/
├── app.py              # 메인 페이지 (대시보드, Annotator 선택, 완료 현황)
├── utils.py            # 공통 유틸리티 함수 (데이터 로드, 라벨 저장, 후보 생성)
├── README.md           # 실행 가이드
├── .cache_parquet/     # XLSX → Parquet 캐시 (자동 생성)
└── pages/
    ├── 1_Peak.py       # Peak 검증 페이지
    ├── 2_Apnea.py      # Apnea region 검증 페이지
    └── 3_Sigh.py       # Sigh 검증 페이지
```

---

## 3. 데이터 경로

### 입력 (Input)
- **경로**: `/home/jhkim/NAVA/03_LABEL/stored_results/00_detected/`
- **파일명 형식**: `movingwinddetected_patient_XX.xlsx` (또는 버전별 폴더 내)
- **시트**:
  - `data`: 타임스탬프, edi 신호, 탐지 결과 (detected_peak, detected_apnea, detected_sigh), GT 라벨 (gt_peak)
  - `params`: 탐지 파라미터 정보

### 출력 (Output)
- **경로**: `/home/jhkim/NAVA/03_LABEL/stored_results/01_labeled/`
- **파일명 형식**: `labels_{annotator}.parquet` (또는 버전별 폴더 내)
- **스키마**:
  ```python
  {
      "label_id": "uuid4",
      "timestamp": "ISO 시각 문자열",
      "annotator": "이주영 | 이지선 | 조한나 | 김재호 | 김시현 | 오창준 | Test",
      "patient_id": "patient_XX",
      "type": "peak | apnea | sigh",
      "item_id": "patient_XX|{type}|{timestamp/range}",
      "label": "O | X | NotSure",
      "comment": "optional text"
  }
  ```

---

## 4. 주요 파일 설명

### 4.1 app.py (메인 페이지)

**역할**: 
- Annotator 선택
- 3개 라벨링 페이지로의 네비게이션
- 환자별 완료 현황 스냅샷 생성 및 표시

**주요 기능**:
- `ANNOTATORS` 리스트에서 사용자 선택
- 버전별 데이터 폴더 선택 (ALL, 날짜별 버전, legacy)
- "완료 현황 생성" 버튼 → `build_patient_status_snapshot()` 호출
- 모든 환자 파일에 대해 peak/apnea/sigh 라벨링 진행률 집계 및 표시

**UI 구성**:
- 페이지 링크 (Peak, Apnea, Sigh)
- Annotator & Data folder 선택
- 환자별 진행 상태 DataFrame (✅ DONE, 🟡 IN_PROGRESS, 🔴 NOT_STARTED)

---

### 4.2 utils.py (핵심 유틸리티)

#### 상수 및 경로
```python
BASE_DIR = Path("/home/jhkim/NAVA/03_LABEL")
DETECTED_DIR = BASE_DIR / "stored_results" / "00_detected"
LABELED_DIR = BASE_DIR / "stored_results" / "01_labeled"
CACHE_DIR = BASE_DIR / "streamlit_app" / ".cache_parquet"

ANNOTATORS = ["이주영", "이지선", "조한나", "김재호", "김시현", "오창준", "Test"]
LABEL_SCHEMA = ["label_id", "timestamp", "annotator", "patient_id", "type", "item_id", "label", "comment"]
```

#### 주요 함수

**1. 파일 탐색 및 버전 관리**
- `list_detected_versions()`: 00_detected 내 버전 폴더 목록 반환 (날짜순 정렬)
- `scan_xlsx_files(version_tag)`: 특정 버전(또는 ALL)의 모든 XLSX 파일 경로 반환
- `extract_patient_id(patient_file)`: 파일명에서 patient_id 추출
- `extract_version_tag(patient_file)`: 파일 경로에서 버전 태그 추출

**2. 데이터 로딩**
- `load_xlsx(patient_file)`: XLSX 파일 로드 (캐싱 지원)
  - Parquet 캐시 활용 (MD5 해시 기반)
  - `data` 시트 → edi 신호 + 탐지 결과
  - `params` 시트 → 파라미터 정보
  - 필수 컬럼 검증: `timestamp`, `edi`, `gt_peak`, `detected_peak`, `detected_apnea` (또는 `apnea`)
- `normalize_bool_columns(data_df)`: 불린 컬럼 정규화 (True/False 통일)

**3. 후보 생성 (Candidate Building)**
- `build_peak_candidates(data_df, patient_id, gt_match_tolerance_ms=1500)`:
  - detected_peak 중 gt_peak와 1.5초 이내 매칭이 없는 항목만 추출
  - item_id: `{patient_id}|peak|{timestamp}`
  
- `build_apnea_segments(data_df, patient_id)`:
  - detected_apnea=True 연속 구간을 세그먼트로 변환
  - item_id: `{patient_id}|apnea|{start}-{end}`
  
- `build_sigh_candidates(data_df, patient_id)`:
  - detected_sigh=True 지점 추출
  - item_id: `{patient_id}|sigh|{timestamp}`

**4. 라벨 저장 및 관리**
- `_labels_path(annotator, version_tag)`: 라벨 파일 경로 생성
- `load_labels(annotator, version_tag)`: Parquet 라벨 파일 로드 (없으면 빈 DataFrame)
- `append_label(annotator, row, version_tag)`: 새 라벨 추가 (atomic write with temp file)
- `undo_last_label(annotator, version_tag)`: 마지막 라벨 1개 제거
- `make_label_row()`: 라벨 row dict 생성 (uuid4, timestamp, annotator, patient_id, type, item_id, label, comment)

**5. 라벨 조회**
- `get_labeled_item_ids(labels_df, patient_id, label_type)`: 특정 환자/타입의 라벨링된 item_id 집합 반환
- `get_latest_labels_map(labels_df, patient_id, label_type)`: item_id별 최신 라벨 맵 (O/X/NotSure)
- `get_latest_comments_map(labels_df, patient_id, label_type)`: item_id별 최신 코멘트 맵

**6. 진행률 집계**
- `build_patient_status_snapshot(annotator, version_tag)`: 모든 환자 파일에 대해 peak/apnea/sigh 라벨링 진행률 계산

---

### 4.3 pages/1_Peak.py (Peak 검증)

**목적**: 모델이 탐지한 Peak 중 GT와 매칭되지 않은 False Positive 후보를 검증

**UI 구성**:
- **사이드바**:
  - Annotator 선택
  - Data folder (버전) 선택
  - Patient file 선택
  - Undo last label 버튼
  - 진행률 표시 (Labeled / Remaining)
  - 파라미터 preview
  
- **메인 영역**:
  - 현재 후보 정보 (patient_id, item_id, index)
  - 기존 라벨 표시 (있으면 info, 없으면 warning)
  - **Wide View (60초)**: 넓은 시간 범위의 edi 신호 + detected_peak + gt_peak
  - **Zoom View (20초)**: 좁은 시간 범위의 상세 뷰
  - 라벨링 버튼: **O (Correct)**, **X (Incorrect)**, **Not Sure**
  - 네비게이션: Prev, Next, Skip

**시각화** (`_plot_peak`):
- Plotly 그래프: edi 신호 (파란 선)
- detected_peak (주황 삼각형)
- gt_peak (초록 원)
- 현재 후보 timestamp (빨간 수직선)

**상태 관리**:
- `st.session_state[f"peak_offset::{annotator}::{patient_file}"]`: 현재 인덱스
- 첫 방문 시 → 라벨링 안 된 첫 번째 후보로 이동
- 라벨 선택 시 → 다음 후보로 자동 이동 (`st.rerun()`)

**라벨 저장 로직**:
- O/X/NotSure 버튼 클릭 → `append_label()` 호출
- 버전별 parquet 파일에 append
- 즉시 다음 후보로 이동

---

### 4.4 pages/2_Apnea.py (Apnea 검증)

**목적**: detected_apnea 연속 구간(segment)의 타당성을 검증

**UI 구성**: Peak과 유사하지만 추가 기능 포함
- **사이드바**: Peak과 동일
- **메인 영역**:
  - Wide/Zoom View: 세그먼트 시작~끝 범위를 빨간 사각형으로 표시
  - **Optional comment**: Apnea만 코멘트 입력 가능
    - text_input with `on_change` → 입력 즉시 저장
    - Annotator별 색상 표시
  - 라벨링 버튼: O/X/NotSure (코멘트도 함께 저장)
  - 네비게이션: Prev, Next, Skip

**시각화** (`_plot_apnea`):
- edi 신호 + detected_peak + gt_peak
- Apnea 구간: 빨간 반투명 사각형 (`fig.add_vrect`)
- 시작/끝 지점: 빨간 수직선 2개

**코멘트 저장 로직**:
- `_save_comment_immediately()`: text_input의 on_change 콜백
- 코멘트만 변경 시 → 기존 라벨(O/X/NotSure) 유지하거나 NotSure로 저장
- O/X/NotSure 버튼 클릭 시 → 코멘트도 함께 저장

**상태 관리**:
- `apnea_offset::{annotator}::{patient_file}`: 현재 인덱스
- `apnea_comment_item::{annotator}::{patient_file}`: 현재 item_id (코멘트 동기화용)
- `apnea_comment_input::{annotator}::{patient_file}`: 코멘트 입력 텍스트

---

### 4.5 pages/3_Sigh.py (Sigh 검증)

**목적**: detected_sigh 후보 지점의 타당성을 검증

**UI 구성**:
- Peak과 거의 동일
- 차이점: detected_sigh 마커 추가 (빨간 X)

**시각화** (`_plot_sigh`):
- edi 신호 + detected_peak + detected_sigh + gt_peak
- detected_sigh: 빨간 X 마커
- 현재 후보 timestamp: 빨간 수직선

**나머지**: Peak과 동일한 로직 (상태 관리, 라벨 저장, 네비게이션)

---

## 5. 핵심 데이터 흐름

```
1. XLSX 로딩 (cache 활용)
   ↓
2. 후보 생성 (build_*_candidates)
   - Peak: gt_peak와 매칭 안 된 detected_peak
   - Apnea: detected_apnea 연속 구간
   - Sigh: detected_sigh 지점
   ↓
3. 기존 라벨 로딩 (load_labels)
   ↓
4. 라벨링된 item_id 필터링
   ↓
5. 미라벨링 후보 순회 (Prev/Next)
   ↓
6. 사용자 선택 (O/X/NotSure)
   ↓
7. Parquet 파일에 append (atomic write)
   ↓
8. 다음 후보로 이동 (st.rerun)
```

---

## 6. 중요한 설계 특징

### 6.1 버전 관리
- `00_detected/` 내 날짜별 폴더 (예: `20260228/`, `20260301/`)
- `legacy`: 폴더 없이 직접 XLSX 파일
- `ALL`: 모든 버전의 파일 통합 표시
- 라벨 파일도 버전별로 분리 저장 (`01_labeled/{version}/labels_{annotator}.parquet`)

### 6.2 Atomic Write
- Parquet 파일 저장 시 temp 파일 생성 후 `os.replace()`로 원자적 교체
- 동시 접근 시 데이터 손실 방지

### 6.3 Caching
- XLSX → Parquet 캐시 (`.cache_parquet/`)
- MD5 해시 기반 캐시 키
- mtime 비교로 stale cache 감지
- `@lru_cache` 활용

### 6.4 Progress Tracking
- `get_labeled_item_ids()`: 라벨링된 item_id 집합
- Labeled count vs Total count
- 사이드바 progress bar
- 첫 미라벨링 항목으로 자동 이동

### 6.5 Label Versioning
- 동일 item_id에 여러 번 라벨링 가능 (타임스탬프 순서)
- `get_latest_labels_map()`: 최신 라벨만 조회
- Undo: 마지막 row 제거

### 6.6 UI Customization
- CSS injection으로 버튼 색상 변경 (O=초록, X=빨강, NotSure=주황)
- Annotator별 코멘트 색상 구분

---

## 7. 주요 알고리즘

### 7.1 Peak Candidate Filtering (build_peak_candidates)
```python
1. detected_peak=True인 모든 timestamp 추출
2. 각 timestamp에 대해:
   - gt_peak 중 ±1500ms 이내에 매칭되는 것이 있는지 확인
   - 없으면 → 후보에 추가 (False Positive)
   - 있으면 → 스킵 (True Positive)
```

### 7.2 Apnea Segment 추출 (build_apnea_segments)
```python
1. timestamp 순서대로 순회
2. detected_apnea=True 시작 → run_start 기록
3. detected_apnea=True 계속 → run_end 업데이트
4. detected_apnea=False 전환 → segment 생성 (start~end)
5. 마지막까지 True → 마지막 segment 생성
```

### 7.3 GT Matching (tolerance-based)
```python
def _has_gt_match_within_tolerance(detected_ts, gt_timestamps_sorted, tolerance_ms):
    # Binary search로 가장 가까운 gt_timestamp 찾기
    idx = np.searchsorted(gt_timestamps_sorted, detected_ts)
    # idx, idx-1 중 가장 가까운 거리 계산
    # tolerance 이내면 매칭 성공
```

---

## 8. 사용 시나리오

### 시나리오 1: 새로운 Annotator 시작
1. app.py에서 Annotator 선택 (예: "이주영")
2. pages/1_Peak.py 이동
3. Data folder 선택 (예: "20260301")
4. Patient file 선택 (예: "movingwinddetected_patient_03.xlsx")
5. 첫 번째 미라벨링 peak 후보 자동 표시
6. Wide/Zoom View 확인
7. O/X/NotSure 선택 → 자동 저장 & 다음 이동
8. 모든 peak 완료 후 pages/2_Apnea.py, pages/3_Sigh.py 진행

### 시나리오 2: 진행 중인 작업 재개
1. Annotator & Patient file 선택
2. 자동으로 첫 번째 미라벨링 항목으로 이동
3. 이전 라벨은 "현재 항목 기존 선택" 섹션에 표시

### 시나리오 3: 실수로 잘못 라벨링
1. "Undo last label" 버튼 클릭
2. 마지막 1개 라벨 제거
3. 해당 항목 다시 라벨링

### 시나리오 4: 완료 현황 확인
1. app.py에서 Annotator 선택
2. "완료 현황 생성" 버튼 클릭
3. 모든 환자의 peak/apnea/sigh 진행률 테이블 확인

---

## 9. 확장 가능성

### 추가 가능한 기능
- 다중 Annotator 간 라벨 비교 (Inter-annotator Agreement)
- 라벨 수정 기능 (현재는 Undo만 가능)
- 라벨 export (CSV, JSON)
- 통계 대시보드 (정확도, 소요 시간 등)
- 키보드 단축키 (1=O, 2=X, 3=NotSure)

### 성능 최적화
- 대용량 XLSX 파일 → Parquet 사전 변환
- 멀티 프로세싱으로 여러 파일 병렬 로딩
- 진행률 집계 캐싱

---

## 10. 의존성 (주요 라이브러리)

```python
streamlit          # UI 프레임워크
pandas             # 데이터 처리
plotly             # 그래프 시각화
numpy              # 수치 연산
openpyxl           # XLSX 파일 읽기
pyarrow            # Parquet 파일 읽기/쓰기
```

---

## 11. 코드 스타일 특징

- Type hints 적극 활용 (`List[Dict]`, `Tuple[pd.DataFrame, pd.DataFrame]`)
- `@lru_cache` 데코레이터로 함수 결과 캐싱
- `st.session_state`로 페이지 상태 관리
- CSS injection으로 커스텀 스타일 적용
- Atomic file write 패턴 (temp file + os.replace)
- Boolean 컬럼 정규화 (문자열 → bool 변환)
- Item ID 구조: `{patient_id}|{type}|{timestamp/range}` (파싱 가능)

---

## 12. 알려진 제약사항

1. **동시 접근**: 여러 Annotator가 동일 파일 동시 편집 시 마지막 저장만 유효 (현재는 Annotator별 파일 분리로 우회)
2. **라벨 수정**: 기존 라벨 수정 UI 없음 (Undo 후 재라벨링 필요)
3. **버전 전환**: 버전 변경 시 session_state 초기화 안 됨 (수동 새로고침 필요)
4. **대용량 파일**: 매우 큰 XLSX 파일은 첫 로딩 시 느릴 수 있음 (캐시 후 빠름)

---

## 13. 다른 LLM에게 전달 시 핵심 포인트

이 Streamlit 앱은 **의료 신호 데이터(edi)의 이벤트 탐지 결과를 의료진이 검증하는 라벨링 도구**입니다. 주요 특징:

1. **3가지 이벤트 타입** (Peak, Apnea, Sigh) 각각 별도 페이지
2. **False Positive 중심 검증**: GT와 매칭 안 된 탐지 결과만 표시
3. **Annotator별 라벨 관리**: 각 의료진마다 독립적인 라벨 파일
4. **버전 관리**: 날짜별 데이터 버전 분리
5. **Parquet 기반 저장**: 빠른 읽기/쓰기, append 패턴
6. **Plotly 시각화**: 넓은 뷰와 좁은 뷰로 맥락 파악
7. **진행률 추적**: 실시간 라벨링 진행 상황 표시
8. **Undo 기능**: 마지막 라벨 제거 가능

**데이터 스키마**는 간단하지만 명확합니다:
- 입력: `timestamp`, `edi`, `gt_peak`, `detected_peak`, `detected_apnea`, `detected_sigh`
- 출력: `label_id`, `timestamp`, `annotator`, `patient_id`, `type`, `item_id`, `label`, `comment`

**코드 구조**가 명확하게 분리되어 있어 확장/수정이 용이합니다:
- `utils.py`: 순수 데이터 처리 로직 (재사용 가능)
- `pages/*.py`: UI 로직 (거의 동일한 패턴 반복)

이 정보를 바탕으로 유사한 라벨링 도구를 구축하거나, 기능을 추가하거나, 다른 프레임워크로 포팅할 수 있습니다.
