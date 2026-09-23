from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def analyze_plant(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]

    # BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 綠色範圍
    lower_green = np.array([25, 40, 30])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 去除小雜訊
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_pixels = cv2.countNonZero(mask)
    total_pixels = width * height

    green_ratio = (green_pixels / total_pixels) * 100

    # 找植物範圍
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    plant_area = 0
    plant_width = 0
    plant_height = 0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        plant_area = cv2.contourArea(largest_contour)

        x, y, w, h = cv2.boundingRect(largest_contour)

        plant_width = w
        plant_height = h

    # 簡單狀態判斷
    if green_ratio >= 15:
        status = "植物影像正常"
    elif green_ratio >= 5:
        status = "植物綠色區域較少"
    else:
        status = "可能需要進一步觀察"

    return {
        "width": width,
        "height": height,
        "green_pixels": green_pixels,
        "green_ratio": round(green_ratio, 2),
        "plant_area": round(plant_area, 2),
        "plant_width": plant_width,
        "plant_height": plant_height,
        "status": status
    }


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_url = None

    if request.method == "POST":

        if "plant_image" not in request.files:
            return render_template(
                "index.html",
                error="沒有收到圖片"
            )

        file = request.files["plant_image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="請先選擇植物照片"
            )

        filename = "plant.jpg"

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        result = analyze_plant(filepath)

        image_url = "/static/uploads/" + filename

    return render_template(
        "index.html",
        result=result,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )