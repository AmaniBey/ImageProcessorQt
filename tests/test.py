import cv2
import numpy as np

# 初始化视频捕获
cap = cv2.VideoCapture(1)  # 替换为 0 使用摄像头

# 创建 MOG2 背景减除器
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 应用背景减除器
    fgmask = fgbg.apply(frame)

    # 对前景掩码进行阈值处理
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # 对阈值图像进行形态学操作，去除噪声
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    for contour in contours:
        # 忽略小面积的轮廓
        if cv2.contourArea(contour) < 500:
            continue

        # 获取边界框并绘制
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # 按下 'q' 键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
