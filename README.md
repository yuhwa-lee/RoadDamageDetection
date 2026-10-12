## 저품질 도로 영상의 품질 개선 및 균열·포트홀 후보 영역 추출

PBL 모듈 1 · 4조 (이예은, 이유화, 황윤식)

1차과제수행계획서 https://docs.google.com/document/d/1w7huS31gQw7nimCrA5Cx006HOKKp_j_PKCoZPoF7iqI/edit?tab=t.0

2차과제수행계획서 https://docs.google.com/document/d/1gPKncIodOa8m3Pw-KXM9q3F3sMBTCdc97xOjchROhuQ/edit?tab=t.0

딥러닝 없이 영상처리 알고리즘만으로 저품질 도로 영상을 전처리하고 균열·포트홀 후보 영역을 추출한다.
전처리 전·후 결과(에지·특징점 수, 후보 수·면적, 처리 시간, 정답 대비 정확도)를 정량 비교한다.

---

## 빠른 실행 (통합 프로그램)

repo 최상위(cv2026)에서 실행한다. 모든 스크립트는 어디서 실행해도 repo 기준 경로를 쓴다.

```powershell
python src/pipeline.py train_data/IMG_1815.JPG               # 사진 한 장
python src/pipeline.py train_data                            # 폴더 (하위 폴더 포함)
python src/pipeline.py train_data/IMG_1815.JPG --show crack  # 그림에 균열만 표시 (계산·CSV는 동일)
python src/pipeline.py train_data --no-vis                   # 수치만
```

처리 흐름 (사진마다 전처리 전·후를 모두 실행)

```
원본 -> ROI(도로 마스크) -> 긴 변 640 -> 회색조
   ├─ 균열:   전처리 adapt_gauss_s2 -> 균열 추출기
   └─ 포트홀: 전처리 laplacian_a05 -> adapt_gauss_s2 -> 포트홀 추출기(Edge)
-> 시각화 + 정량 비교 CSV
```

출력 (`out_pipeline/<시각>/`, git 제외)

| 파일 | 내용 |
| --- | --- |
| `vis/<이름>.jpg` | 입력+정답 \| 전처리 전 \| 전처리 후. 균열 정답 초록, 포트홀 정답 자홍, 균열 검출 빨강, 포트홀 검출 파랑, 도로 마스크 노랑 |
| `per_image.csv` | 사진 x 조건(before/after): 에지 수, ORB 특징점 수, 후보 수·면적, 단계별 시간(ms), 균열 F2/P/R, 포트홀 IoU |
| `summary.csv` | 출처(제공/촬영/rdd) x 조건별 평균. 균열은 사진 평균 F2와 micro F2(채점기와 같은 방식) |

정답 라벨이 있으면 정답 윤곽과 점수를 함께 표시하고, 없으면 검출 결과만 표시한다.

---

## 폴더 구조

```
cv2026/
├── train_data/          개발 세트: 제공 13 + 촬영 10 (capture_pothole_01은 라벨 없음)
│   └── rdd_box/         RDD2020 추가 포트홀 5장 + boxes.json (포트홀 전용 라벨, 아래 규칙 참고)
├── test_data/           최종 평가 세트: 촬영 10장. 튜닝에 쓰지 않는다
├── labeled_data/        CVAT 라벨 (SegmentationClass, SegmentationObject, labelmap.txt)
├── src/                 코드 (아래 표)
├── docs/                실험 기록 (pothole_experiments.md)
├── pothole_edge_baseline.json   포트홀 closed_edges 설정 (비교용)
└── round_runner.py      루트에서 src/round_runner.py를 실행하기 위한 래퍼
```

### 데이터 규칙

- **test_data는 설정을 모두 고정한 뒤 최종 1회만 평가한다.** 장소(GPS) 기준으로 train과 나눴고, 야간 사진은 양쪽에 모두 있다.
- **train_data/rdd_box의 5장은 포트홀만 라벨링한다.** 균열이 보여도 칠하지 않는다.
  - 균열 채점(`round_runner.load_items`)은 `boxes.json`에 이름이 있는 사진을 이름 기준으로 항상 제외한다.
  - 픽셀 라벨은 `labeled_data/SegmentationClass/<원본과 같은 이름>.png`로 넣는다 (예: `Japan_001262.png`). 넣으면 포트홀 평가가 자동으로 박스 대신 픽셀 정답을 쓴다.
- `capture_pothole_02`는 이름과 달리 포트홀 없는 균열 사진이다. 라벨(균열만)이 맞으므로 평가에서 빼지 않는다.

---

## 코드 (`src/`)

| 파일 | 역할 |
| --- | --- |
| `pipeline.py` | 통합 실행: 전처리 -> 균열·포트홀 추출 -> 시각화 + 전·후 정량 비교 |
| `image_ROI_Resize.py` | ROI(도로 마스크) 검출과 resize |
| `resize_long_side.py` | 긴 변 640 축소 (비율 유지) |
| `candidates.py` | 전처리 후보 (노이즈·조도·흐림 축, 적응형 평활화 adapt_*) |
| `crack_extractor.py` | 균열 추출기 (black-hat -> 지역 MAD 임계 -> 정리 -> 문맥 가드 -> 모양 -> 골짜기 검사). 파라미터 근거는 docstring |
| `pothole_edge_extractor.py` | 포트홀 추출기 (주변보다 어두운 영역 -> 모양 -> 외곽 검증). 기본값 근거는 docstring |
| `round_runner.py` | 공용: 데이터 로딩, ROI 준비, 전처리 적용, 전처리 후보 채점 러너 |
| `crack_eval.py` | 균열 영역 채점 (골격 허용오차 d=5, F2 선정) |
| `pothole_eval.py` | 포트홀 픽셀 IoU 평가, `--audit-edge`로 단계별 진단 |
| `pothole_box_eval.py` | 포트홀 박스(객체) 평가 (IoU 0.5/0.3) |
| `pothole_axis_eval.py` | 포트홀 전처리 축·조합·순서 비교와 선택 규칙 판정(`--judge`) |
| `pixel_evaluation.py` | 픽셀 채점 공용 함수 (마스크 원본 좌표 복원 등) |
| `baseline_v2.py` | 초기 기준선 추출기 (러너 초기 버전용) |

---

## 평가 실행

### 균열

```powershell
python src/crack_eval.py                 # train, 전처리 없음
python src/crack_eval.py --axis noise    # 노이즈 축 후보를 none과 비교해 판정
python src/crack_eval.py --maps          # 영상별 오류 지도 저장
python src/crack_eval.py --split test    # 최종 평가 (1회만)
```

- 채점: 검출·정답 중심선을 d=5px(640 기준) 허용 오차로 매칭한 정밀도·재현율, 사진 합산(micro). 포트홀 주변 검출은 제외.
- 선정 점수 F2 (재현율을 2배 중시). 규칙: ① 재현율 붕괴(0.2) 없음 ② 정밀도 하락 0.03 이하 ③ F2 최대(0.01 이내 동률이면 덜 처리하는 쪽) ④ d=3/8 일관성.

### 포트홀

```powershell
python src/pothole_eval.py                                   # 픽셀 IoU (train)
python src/pothole_eval.py --config pothole_edge_baseline.json   # closed_edges 비교
python src/pothole_eval.py --audit-edge                      # 단계별 정답 포함률·탈락 사유
python src/pothole_eval.py --preprocess laplacian_a05+adapt_gauss_s2   # 최종 전처리 적용 평가
python src/pothole_box_eval.py                               # 박스(객체) 평가
python src/pothole_axis_eval.py --axis noise --out out_round/pothole_axis_v2.csv
python src/pothole_axis_eval.py --judge out_round/pothole_axis_v2.csv
```

- 주 점수: 포트홀 사진별 픽셀 IoU 평균. 보조: 정답 없는 사진의 오검출 면적. 박스 IoU>=0.5로 찾은 수는 참고.
- 전처리 선택 규칙: ① 한 장이라도 IoU 0.2 넘게 하락 시 탈락 ② 오검출 면적 증가 시 탈락 ③ IoU 최대(0.01 이내 동률이면 덜 처리) ④ 한 장 빼기 검사(모든 경우 none보다 0.01 넘게 높아야 통과).
- 라벨이 바뀌면 `pothole_axis_eval`의 입력 캐시가 자동으로 다시 만들어진다. 결과 CSV는 새 `--out` 이름을 쓴다.

### 전처리 후보 채점 러너 (초기 버전)

```powershell
python round_runner.py noise    # 노이즈 축 후보를 채점
python round_runner.py light    # 조도 축
python round_runner.py blur     # 블러 축
python round_runner.py all      # 세 축을 차례로
python round_runner.py noise --overlay Japan_001959,China_MotorBike_002159   # 결과 오버레이 저장
python round_runner.py noise --limit 3                                       # 앞의 3장만
python round_runner.py "combo:bilateral_c50|clahe_c2|unsharp_s1_a08"         # 조합 채점
```

---

## 라벨

`labeled_data/`: CVAT Segmentation 1.1 형식. 클래스 색은 `labelmap.txt` (crack 255,96,55 / pothole 51,221,255).

<img width="1100" height="333" alt="image" src="https://github.com/user-attachments/assets/b5e88c5b-782a-419b-aaf5-bd0da2c2bef4" />
<img width="1093" height="465" alt="image" src="https://github.com/user-attachments/assets/04deeb86-17e6-4693-acb8-1b913dfad9f5" />

---

## 환경

`environment.yml` (conda, Python 3.12). 실제로 쓰는 패키지: `opencv-contrib-python`, `numpy`, `pillow`.

```powershell
conda env create -f environment.yml
conda activate cv2026
```
