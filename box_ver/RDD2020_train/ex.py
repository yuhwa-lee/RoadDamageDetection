import cv2 as cv
import json
import os

img_path = "train/img/United_States_003339.jpg"
ann_path = "train/ann/United_States_003339.jpg.json"

# 이미지 읽기
img = cv.imread(img_path)

# 라벨 JSON 읽기
with open(ann_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 객체 하나씩 처리
for obj in data["objects"]:

    label = obj["classTitle"]

    points = obj["points"]["exterior"]

    x1, y1 = points[0]
    x2, y2 = points[1]

    # Bounding Box
    cv.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    # 클래스 이름
    cv.putText(
        img,
        label,
        (x1, y1 - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

cv.imshow("RDD2020 Label", img)

cv.waitKey(0)
cv.destroyAllWindows()