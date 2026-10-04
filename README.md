## 저품질 도로 영상의 품질 개선 및 균열·포트홀 후보 영역 추출

https://docs.google.com/document/d/1w7huS31gQw7nimCrA5Cx006HOKKp_j_PKCoZPoF7iqI/edit?tab=t.0

road_ground_truth_all 폴더 : 예시 사진 데이터 라벨 처리한 것. png파일, Segmentation 1.1
<img width="1100" height="333" alt="image" src="https://github.com/user-attachments/assets/b5e88c5b-782a-419b-aaf5-bd0da2c2bef4" />
<img width="1093" height="465" alt="image" src="https://github.com/user-attachments/assets/04deeb86-17e6-4693-acb8-1b913dfad9f5" />
## 사용법

`baseline_v2.py`, `candidates.py`와 같은 폴더에서 실행합니다.

```powershell
python round_runner.py noise    # 노이즈 축 후보를 채점
python round_runner.py light    # 조도 축
python round_runner.py blur     # 블러 축
python round_runner.py all      # 세 축을 차례로
```

### 옵션

```powershell
# 지정한 영상의 결과 오버레이 저장 (초록 = 정답, 빨강 = 후보 박스)
python round_runner.py noise --overlay Japan_001959,China_MotorBike_002159

# 앞의 3장만 사용 (빠른 동작 확인용)
python round_runner.py noise --limit 3

# 2라운드: 조합 채점 (이름을 | 로 이어서 앞에서부터 차례로 적용)
python round_runner.py "combo:bilateral_c50|clahe_c2|unsharp_s1_a08"
```
