import cv2
import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage

from component.GlobalContext import gc
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class ProcessFrameThread(QThread):
    def __init__(self):
        super().__init__()
        self.isRunning = True
        self.exclusion_polygons = None
        self.points = None
        gc.drawMaskSignale.connect(self.updated_exclusion_mask)

    def run(self):
        while self.isRunning:
            if gc.cap is not None:
                ret, frame = gc.cap.read()
                if not ret:
                    if gc.openmodel == 1:
                        gc.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.error(" 摄像头读取失败，跳过处理")
                    self.isRunning = False
                    continue
                frame = cv2.resize(frame, (gc.target_width, gc.target_height))



                origin_frame = frame.copy()

                if self.points is not None:
                    frame = self.handle_exclusion_points(frame,self.points)

                frame = self.process_frame(frame, origin_frame)

                # 判断图像是单通道还是三通道
                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    # 单通道图像，使用灰度格式
                    frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_Grayscale8)
                else:
                    # 三通道图像，使用 RGB 格式（同时转换 BGR 到 RGB）
                    frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888).rgbSwapped()

                origin_frame = QImage(origin_frame.data, origin_frame.shape[1], origin_frame.shape[0],
                                      QImage.Format.Format_RGB888).rgbSwapped()
                gc.updateFrameSignal.emit(origin_frame, frame)

    def stop(self):
        self.isRunning = False

    def process_frame(self, frame, origin_frame):
        # resize
        if gc.cf.get_resize_enabled():
            frame = cv2.resize(frame, (gc.cf.get_resize_width(), gc.cf.get_resize_height()))
        # background
        if gc.cf.get_background_enable():
            alg = gc.cf.get_background_algorithms()
            frame = alg.apply(frame)

        # color_convert
        if gc.cf.get_color_convert_enabled():
            if gc.cf.get_background_enable() == False and gc.cf.get_contour_enabled() == False and gc.cf.get_binary_enabled() == False and gc.cf.get_canny_enabled() == False:
                frame = cv2.cvtColor(frame, gc.cf.get_color_convert_mode())
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # open and close
        if gc.cf.get_open_enabled():
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN,
                                     np.ones((gc.cf.get_open_kernel_size(), gc.cf.get_open_kernel_size()), np.uint8),
                                     iterations=gc.cf.get_open_iterations())
        if gc.cf.get_close_enabled():
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                                     np.ones((gc.cf.get_close_kernel_size(), gc.cf.get_close_kernel_size()), np.uint8),
                                     iterations=gc.cf.get_close_iterations())

        # binary
        if gc.cf.get_binary_enabled():
            _, frame = cv2.threshold(frame, gc.cf.get_binary_threshold(), gc.cf.get_binary_threshold(),
                                     cv2.THRESH_BINARY)
        # erode
        if gc.cf.get_erode_enabled():
            frame = cv2.erode(frame, np.ones((gc.cf.get_erode_kernel_size(), gc.cf.get_erode_kernel_size()), np.uint8),
                              iterations=gc.cf.get_erode_iterations())
        # dilate
        if gc.cf.get_dilate_enabled():
            frame = cv2.dilate(frame,
                               np.ones((gc.cf.get_dilate_kernel_size(), gc.cf.get_dilate_kernel_size()), np.uint8),
                               iterations=gc.cf.get_dilate_iterations())

        # blur
        if gc.cf.get_gaussian_enabled():
            frame = cv2.GaussianBlur(frame, (gc.cf.get_blur_kernel_size(), gc.cf.get_blur_kernel_size()), 1.5)

        # median
        if gc.cf.get_median_enabled():
            frame = cv2.medianBlur(frame, gc.cf.get_blur_kernel_size())

        # canny
        if gc.cf.get_canny_enabled():
            frame = cv2.Canny(frame, gc.cf.get_canny_threshold1(), gc.cf.get_canny_threshold2())

        # contour
        if gc.cf.get_contour_enabled():
            contours, _ = cv2.findContours(frame, gc.cf.get_contour_mode(), gc.cf.get_contour_method())
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if gc.cf.get_contour_min_area() < area < gc.cf.get_contour_max_area():
                    cv2.drawContours(origin_frame, [cnt], -1, (0, 255, 0), 2)
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0
                    cv2.putText(origin_frame, f"{int(area)}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def updated_exclusion_mask(self, points):
        self.points = points

    def handle_exclusion_points(self, frame, points):
        """在帧上绘制圆点排除区域，返回遮罩后的帧（圆点区域被排除/变黑）"""
        if frame is None or len(points) == 0:
            return frame

        # 1. 将 QPoint 转为 (x, y) 坐标
        points_cv = [(p.x(), p.y()) for p in points]

        # 2. 创建空白掩码（尺寸与帧相同）
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 3. 在掩码上绘制圆点（白色=要排除的区域）
        alpha = 0.5  # 透明度（0=全透明，1=不透明）
        radius = 15  # 圆点半径
        for (x, y) in points_cv:
            cv2.circle(frame, (x, y), radius, (255, 255, 255, int(255 * alpha)), -1)  # 白色+透明度

        # 4. 反转掩码（使圆点区域=0，其他区域=255）
        mask = cv2.bitwise_not(mask)  # 关键修改！

        # 5. 应用遮罩：圆点区域变黑，其他区域保留原图
        result = cv2.bitwise_and(frame, frame, mask=mask)
        return result