road_ground_truth_all 폴더 : 예시 사진 데이터 라벨 처리한 것. png파일, Segmentation 1.1
<img width="1100" height="333" alt="image" src="https://github.com/user-attachments/assets/b5e88c5b-782a-419b-aaf5-bd0da2c2bef4" />
<img width="1093" height="465" alt="image" src="https://github.com/user-attachments/assets/04deeb86-17e6-4693-acb8-1b913dfad9f5" />
사용법 (baseline_v2.py, candidates.py와 같은 폴더에서)
    python round_runner.py noise                 # 노이즈 축 후보를 채점
    python round_runner.py light                 # 조도 축
    python round_runner.py blur                  # 블러 축
    python round_runner.py all                   # 세 축을 차례로
    python round_runner.py noise --overlay Japan_001959,China_MotorBike_002159
    python round_runner.py noise --limit 3       # 앞의 3장만 (빠른 동작 확인용)
    python round_runner.py "combo:bilateral_c50|clahe_c2|unsharp_s1_a08"    # 2라운드: 조합 채점(2*2*2)
