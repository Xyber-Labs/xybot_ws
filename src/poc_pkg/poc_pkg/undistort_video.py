import cv2
import json
import numpy as np

# Загрузка параметров калибровки камеры
calibration_file = "/home/ubuntu/ros2_ws/src/poc_pkg/poc_pkg/camera_calibration_data.json"
with open(calibration_file, "r") as f:
    calibration_data = json.load(f)

camera_matrix = np.array(calibration_data["camera_matrix"])
dist_coeff = np.array(calibration_data["dist_coeff"])

# Открытие видеопотока с камеры
cap = cv2.VideoCapture('/dev/usb_cam')

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Применение параметров калибровки для устранения искажений
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeff)

    # Отображение исходного и скорректированного изображения
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Undistorted Frame", undistorted_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()