# MotionSense Streamlit Demo

기존에 학습한 활동 분류 모델과 실제 윈도우 센서 데이터를 사용해 만든 Streamlit 데모 앱입니다.  
사용자는 `걷기`, `서기`, `앉기` 버튼으로 2초 구간을 최대 5개까지 쌓아 시퀀스를 만들고, `시작` 버튼으로 각 구간의 예측 결과를 확인할 수 있습니다.

## 프로토타이핑 URL

- URL
https://watch-sensor-activity-ml.streamlit.app/

- 보고서
https://drive.google.com/file/d/1E1by3cUGXdX8hplLD4plwCnmvY81s9ix/view?usp=drive_link

## 개요

- 실제 데이터 기반 랜덤 샘플 사용
- `window_size=50`, `step_size=25`, `skip_head=80` 유지
- 저장된 2단 분류 모델을 앱 내부에서 3클래스(`Walking`, `Sitting`, `Standing`) 출력으로 래핑
- 선택한 실제 raw xyz 데이터를 이어붙여 전체 시계열 표시
- 글로벌 중요 특성 TOP 5와 현재 대표 구간의 특성값 표시


## Machine Learning
 **Scikit-learn**
2단 분류 구조
   - Stage 1 (SVM): `Walking` vs `Static`
   - Stage 2 (SVM): `Sitting` vs `Standing`


## 주요 기능

- `걷기 / 서기 / 앉기` 버튼으로 실제 노이즈 제거 윈도우 추가
- 최대 5개 구간, 총 10초 시퀀스 구성
- 구간별 예측 클래스와 confidence 표시
- 진행 바와 현재 분석 중인 구간 재생 연출
- accel XYZ / gyro XYZ 시계열 시각화
- 최종 대표 결과 카드 제공
- permutation importance 기반 중요 특성 표시

## 설치 방법

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행 방법

### 1. 데모 자산 생성

`feature_bank.csv`, `raw_window_bank.csv`, `importance.csv`는 Git에 포함하지 않고 실행 환경에서 생성합니다.

```powershell
.\.venv\Scripts\python.exe prepare_demo_assets.py
```

생성되는 파일:

- `data/processed/feature_bank.csv`
- `data/processed/raw_window_bank.csv`
- `data/processed/importance.csv`
- `data/processed/importance_meta.json`

### 2. Streamlit 앱 실행

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

브라우저에서 열리면 다음 순서로 사용하면 됩니다.

1. `걷기`, `서기`, `앉기` 버튼으로 구간 추가
2. 최대 5개까지 시퀀스 구성
3. `시작` 버튼 클릭
4. 구간별 예측, 전체 시계열, 중요 특성 확인

## 데이터 처리 규칙

- source 데이터는 `sensor_ADE.csv`를 사용
- `sensor=accel / gyro`를 분리해 처리
- `subject_id + activity_name` 기준 정렬
- `window_size=50`, `step_size=25`로 윈도우 생성
- `skip_head=80` 적용
- 최종 앱에서는 `feature_bank`와 `raw_window_bank`를
  - `subject_id`
  - `activity_name`
  - `window_seq`
  기준으로 연결

## 모델 동작 방식

저장된 모델은 2단 구조입니다.

- Stage 1
  - `Walking` vs `Static`
- Stage 2
  - `Sitting` vs `Standing`

앱 내부에서는 이를 최종 3클래스 결과로 변환합니다.

- `Walking`
- `Sitting`
- `Standing`

또한 모델 입력 시 `feature_columns` 순서를 저장된 번들 기준으로 그대로 사용해 학습 시점과 동일한 입력 순서를 유지합니다.

