import os
import numpy as np
import json
import cv2

# Загрузка параметров калибровки камеры
calibration_file = "/home/ubuntu/ros2_ws/src/poc_pkg/poc_pkg/camera_calibration_data.json"
with open(calibration_file, "r") as f:
    calibration_data = json.load(f)

camera_matrix = np.array(calibration_data["camera_matrix"])
dist_coeff = np.array(calibration_data["dist_coeff"])

def main():
    # Инициализация камеры
    cap = cv2.VideoCapture('/dev/usb_cam')
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Директория для сохранения изображений
    save_dir = "~/ros2_ws/src/poc_pkg/poc_pkg/captured_frames"
    os.makedirs(save_dir, exist_ok=True)

    frame_count = 0

    while True:
        # Захват кадра
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.undistort(frame, camera_matrix, dist_coeff)
        
        # Отображение кадра
        cv2.imshow('Camera', frame)

        # Обработка нажатий клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Выход из программы при нажатии 'q'
            break
        elif key == ord(' '):
            # Сохранение кадра при нажатии пробела
            filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            frame_count += 1

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()