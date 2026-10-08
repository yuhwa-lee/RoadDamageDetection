# Ryu 논문 기반 포트홀 박스 탐지

딥러닝, 사전학습 모델, 학습형 분류기를 사용하지 않는 영상처리 구현입니다.
이미지 입력만으로 탐지하며, 정답 JSON은 탐지 완료 후 평가에만 사용합니다.

**초기 전체 평가의 IoU 0.5 기준 TP는 0개로, 현재 검출 성능은 부족합니다.** 실행 가능한 연구용 기준 구현이며, 성능 결과와 실패 분석은 [EVALUATION.md](EVALUATION.md)에 정리했습니다.

참고: Ryu, Kim & Kim (2015), **Image-Based Pothole Detection System for ITS Service and Road Management System**.
https://doi.org/10.1155/2015/968361

## 실행

PowerShell에서 `C:\Users\yuhwa\cv2026\box_ver`로 이동한 뒤 실행합니다.
프로젝트의 기존 `.venv`를 사용하므로 별도 설치는 필요 없습니다.

```powershell
# 전체 데이터 평가 (기존 ROI 사용)
..\.venv\Scripts\python.exe run.py

# 빠른 실행 점검: 파일명순 첫 20장, 전체 성능을 대표하지 않음
..\.venv\Scripts\python.exe run.py --limit 20 --workers 2

# 단일 사진: 정답 없이 탐지
..\.venv\Scripts\python.exe run.py --image RDD2020_train/train/img/China_MotorBike_000553.jpg

# ROI를 외부에서 처리한 영상 또는 ROI 영향 분석용
..\.venv\Scripts\python.exe run.py --roi none

# 국소 밝기 차이로 후보를 추출하는 RDD 대응 확장
..\.venv\Scripts\python.exe run.py --segmentation local

# 자동 테스트
..\.venv\Scripts\python.exe -m unittest test_detector -v
```

`--out 새폴더경로`로 저장 위치를 지정할 수 있습니다. 기존 폴더 덮어쓰기는 거부합니다.
`--workers 4`는 동시 실행 수, `--previews 12`는 시각화할 이미지 수입니다.
필요 라이브러리는 `numpy`, `opencv-python`뿐입니다.

## 파일 구성

- `ryu_detector.py`: 후보 추출, 형태 필터, 주변 비교, 박스 중복 제거.
- `run.py`: 이미지 입출력, 기존 ROI 연결, 원본 좌표 복원, JSON 정답 읽기, 박스 평가.
- `test_detector.py`: IoU/매칭/특징 계산 및 합성 입력 검증.
- `RDD2020_train/`: 사용자가 넣은 원본 데이터. 수정하지 않습니다.
- `outputs/실행시각/`: 각 실행의 결과. 실행마다 별도 폴더를 만듭니다.

## 알고리즘

1. 가로세로 비율을 유지해 긴 변을 최대 640으로 축소합니다.
2. 기존 `../src/image_ROI_Resize.py`의 `define_roi`를 호출합니다. 팀원 담당 ROI 코드를 수정하지 않습니다. 빈 ROI이면 검출도 빈 결과입니다.
3. 회색조 영상의 도로 영역 히스토그램에서 어두운 후보를 이진화합니다.
4. 3×3 median, 5×5 closing으로 후보의 작은 잡음과 틈을 정리합니다.
5. 면적, compactness(면적/둘레²), 장단축비, solidity로 후보를 필터링합니다.
6. convex hull 면적이 원래 후보의 1.8배 이내일 때 형태를 보정합니다.
7. 후보 주변 12px 고리에서 도로 마스크 밖과 다른 어두운 후보를 제외해 비교 영역을 만듭니다.
8. 내부 밝기 표준편차, 주변과 평균 밝기 차이, 밝기 OHI, Sobel OHI로 판정합니다.
9. 중복 박스를 NMS로 제거하고, 원본 사진 좌표의 `[x1,y1,x2,y2]`로 복원합니다.

아래 중 하나면 최종 판정에서 제외합니다.

- 내부 밝기 표준편차 < 8
- 주변 평균 밝기 − 내부 평균 밝기 < 6
- 밝기 OHI ≥ 0.87 **그리고** Sobel OHI ≥ 0.85

OHI가 클수록 두 영역 분포가 유사합니다. 코드의 OHI는 256-bin 히스토그램을 합이 1이 되도록 정규화하고, 빈도순으로 정렬한 뒤 bin별 최솟값을 합합니다. 밝기 bin 위치를 잃으므로 밝기 차이 조건과 함께 사용합니다. Sobel magnitude는 공통 고정 스케일(/4, 0~255 clipping)을 사용합니다.

## 논문과의 차이 및 한계

**원저자 구현의 정확한 재현이 아닌 논문 기반 독립 구현입니다.**
후보 추출 → 형태 보정 → 주변 특징 비교라는 구조를 참고했습니다.

- HST는 어두운 히스토그램 꼬리에 triangle 기준을 적용하는 근사로 구현했습니다. 어두운 꼬리를 정의할 수 없으면 Otsu를 사용합니다. 논문 MHST 전체를 재현하지 않았습니다.
- `--segmentation local`은 HST 초기 결과를 확인한 뒤 추가한 확장입니다. 31/61/121px 주변 평균보다 `max(8, 0.8×주변 표준편차)` 이상 어두운 부분의 합집합을 후보로 사용합니다. 이 단계는 논문 HST가 아니며, 뒤의 형태·주변 비교 단계는 동일합니다. 기본값 `hst`와 구분해 결과를 해석해야 합니다.
- OHI는 위에 명시한 정렬 히스토그램 교집합으로 구현했습니다. 논문 판정식 전체의 동일 구현을 주장하지 않습니다.
- 작은 RDD 객체를 고려해 최소 면적 32px, median 3×3, closing 5×5를 사용합니다. 논문 설정을 그대로 옮기지 않았습니다.
- 주변 고리 크기, hull 성장 제한, 종횡비·solidity 조건, NMS, ROI는 본 프로젝트의 구현 선택입니다.
- HST 초기 설정은 평가 전 정했으며, local 확장은 초기 평가의 실패 양상을 보고 추가했습니다. 따라서 전체 결과는 개발용 평가입니다. 작은 포트홀, 밝은 포트홀, 큰 그림자, ROI 바깥 손상에는 취약할 수 있습니다.
- ROI는 기존 모듈을 읽어 사용하므로 팀원이 그 모듈을 변경하면 결과가 달라집니다.
- 데이터 폴더명만으로 공식 RDD2020 전체 데이터와 같다고 주장하지 않습니다.

## 결과 읽기

- `summary.json`: 전체 지표, 설정, 빈 정답 파일 제외 시 지표.
- `REPORT.md`: 주변 비교 적용 전후 요약표.
- `predictions.jsonl`: 이미지별 GT/예측 박스, 각 후보 특징과 탈락 사유, ROI 비율, 처리 시간.
- `manifest.json`: 사용 이미지와 정답 파일의 SHA-256. 평가 데이터 확인용.
- `run_metadata.json`: 이후 실행의 라이브러리 버전, 설정, 소스 SHA-256. 최초 비교 실행들은 이 기록 기능 추가 전에 실행했습니다.
- `previews/*_result.jpg`: **초록=정답, 빨강=최종 예측**. 겹칠 때 빨강이 위에 표시됩니다.
- `previews/*_roi.jpg`: 축소 영상 좌표계의 도로 마스크.
- `previews/*_candidates.jpg`: 최종 판정 전 이진 후보 영상.

단일 이미지 실행은 `predictions.json`과 시각화를 만들며 평가 지표는 만들지 않습니다.

### 평가 정의

- Supervisely JSON의 `classTitle == pothole`, `geometryType == rectangle`만 정답입니다. 다른 손상 클래스는 포트홀 정답에 포함하지 않습니다.
- box 면적은 `(x2-x1)*(y2-y1)`인 연속 좌표 해석입니다. 끝점에 +1을 하지 않으며 정답·예측 모두 같은 규칙을 사용합니다.
- 예측 점수 내림차순으로 **IoU ≥ 0.5**인 아직 매칭되지 않은 GT 하나와 연결합니다. 동일 GT에 두 번 탐지하면 나머지는 FP입니다.
- precision/recall/F1은 이미지 평균이 아니라 전체 TP/FP/FN 합으로 계산합니다.
- `mean_gt_best_box_iou`는 각 GT와 가장 잘 겹치는 예측의 IoU 평균입니다. 예측이 없으면 0입니다. 중복 사용을 허용하며 FP를 벌점으로 반영하지 않으므로 F1과 함께 봐야 합니다. 픽셀 IoU 또는 mAP가 아닙니다.
- `geometry_baseline`은 **동일한 새 탐지기에서 최종 밝기·질감 판정만 끈 비교군**입니다. 이전 픽셀 탐지기의 성능이 아닙니다.
- 빈 objects 배열을 기본 평가에서 음성으로 취급합니다. 미라벨링일 가능성을 감안해 `excluding_empty_annotations`도 제공합니다.
- 전체 데이터 초기 점검 결과이며 독립 테스트 성능을 주장하지 않습니다. 이후 임계값을 조정할 때는 개발/최종 평가 데이터를 분리해야 합니다.
