# Watch-sensor-activity-ml

## 개요
학습한 활동 분류 모델과 실제 워치 센서 데이터를 사용해 만든 Streamlit 데모 앱입니다.  
사용자는 `걷기`, `서기`, `앉기` 버튼으로 2초 구간을 최대 5개까지 쌓아 시퀀스를 만들고, `시작` 버튼으로 각 구간의 예측 결과를 확인할 수 있습니다.

## 프로토타이핑 URL
- URL
https://watch-sensor-activity-ml.streamlit.app/

- 보고서
https://drive.google.com/file/d/1C39ooYD9BeqYCaCFAqEzeRHcUmsBtI2I/view?usp=sharing

## 주요 기능
- `걷기 / 서기 / 앉기` 버튼으로 실제 노이즈 제거 윈도우 추가
- 최대 5개 구간, 총 10초 시퀀스 구성
- 구간별 예측 클래스와 confidence 표시
- 진행 바와 현재 분석 중인 구간 재생 연출
- accel XYZ / gyro XYZ 시계열 시각화
- 최종 대표 결과 카드 제공
- permutation importance 기반 중요 특성 표시
  
## 데이터셋
- Source: UCI WISDM Dataset
- Sensors: smartwatch accel / gyro
- Activities used: Walking, Sitting, Standing

## Machine Learning
 **Scikit-learn**
2단 분류 구조
   - Stage 1 (SVM): `Dynamic` vs `Static`
   - Stage 2 (SVM): `Sitting` vs `Standing`

## 최종 성능
- Accuracy: **0.880**
- Macro F1-score: **0.880**
- Weighted F1-score: **0.881**
- Final setting: `window_size=50`, `step_size=25`, `skip_head=80`


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

### 2. Streamlit 앱 실행

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

또한 모델 입력 시 `feature_columns` 순서를 저장된 번들 기준으로 그대로 사용해 학습 시점과 동일한 입력 순서를 유지합니다.

