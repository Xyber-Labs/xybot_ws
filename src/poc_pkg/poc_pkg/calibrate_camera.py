import cv2
import numpy as np
import os

# Параметры, которые необходимо задавать вручную
CHESSBOARD_SIZE = (13, 9)  # Размеры шахматной доски (внутренние углы)
SQUARE_SIZE = 0.013  # Размер квадрата в метрах
IMAGES_FOLDER = "~/ros2_ws/src/poc_pkg/poc_pkg/captured_frames"  # Путь к папке с изображениями шахматной доски

# Критерии для алгоритма нахождения углов
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка мировых координат для углов шахматной доски (0,0,0), (1,0,0), (2,0,0), ...
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Списки для хранения точек в мире и точек на изображении
objpoints = []  # Точки в мире (3D)
imgpoints = []  # Точки на изображении (2D)

# Чтение изображений из папки
images = [os.path.join(IMAGES_FOLDER, fname) for fname in os.listdir(IMAGES_FOLDER) if fname.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)

        # Улучшение точности углов
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Отображение углов на изображении (для проверки)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Калибровка камеры
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Сохранение параметров калибровки
calibration_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "rvecs": [rvec.tolist() for rvec in rvecs],
    "tvecs": [tvec.tolist() for tvec in tvecs]
}


output_file = "camera_calibration_data.json"
with open(output_file, "w") as f:
    import json
    json.dump(calibration_data, f, indent=4)
print(f"параметры: {calibration_data}")
print(f"Калибровка завершена. Параметры сохранены в {output_file}")