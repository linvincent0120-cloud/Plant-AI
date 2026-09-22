import cv2
import numpy as np

# 讀取植物照片
image = cv2.imread("plant.jpg")

if image is None:
    print("找不到 plant.jpg")
    exit()

# =========================
# 1. 找出綠色區域
# =========================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)

# =========================
# 2. 稍微整理遮罩
# =========================

kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

# =========================
# 3. 找植物輪廓
# =========================

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 複製原圖，準備畫結果
result = image.copy()

# =========================
# 4. 計算植物面積
# =========================

total_plant_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # 忽略太小的雜訊
    if area > 500:

        total_plant_area += area

        # 畫出植物輪廓
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 0, 255),
            3
        )

# =========================
# 5. 計算植物佔整張照片比例
# =========================

total_image_area = image.shape[0] * image.shape[1]

plant_ratio = (
    total_plant_area /
    total_image_area *
    100
)

# =========================
# 6. 找植物的整體外接矩形
# =========================

if contours:

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > 500
    ]

    if valid_contours:

        all_points = np.vstack(valid_contours)

        x, y, w, h = cv2.boundingRect(all_points)

        # 畫出植物整體範圍
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        print(f"植物寬度：{w} px")
        print(f"植物高度：{h} px")

# =========================
# 7. 顯示結果
# =========================

print(f"植物面積：約 {total_plant_area:.0f} px²")
print(f"植物覆蓋比例：{plant_ratio:.2f}%")

# 儲存結果
cv2.imwrite(
    "plant_analysis.jpg",
    result
)

cv2.imwrite(
    "green_mask.jpg",
    mask
)

print("分析完成！")
print("結果已儲存為 plant_analysis.jpg")