import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import gc
import tkinter as tk
from tkinter import ttk, messagebox
import os
import serial
import serial.tools.list_ports
import time
import cv2
import numpy as np
from threading import Thread


PORT = '/dev/ttyUSB0'


# 设置支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
sys.setrecursionlimit(10000)

# ================= 串口控制 =================
def send_hex_data(port, hex_data, baudrate=115200, timeout=1):
    try:
        if not os.path.exists(port):
            print(f"错误: 串口设备 {port} 不存在")
            return False
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            if isinstance(hex_data, str):
                message = bytes.fromhex(hex_data.replace(" ", ""))
            elif isinstance(hex_data, list):
                message = bytes(hex_data)
            else:
                message = hex_data
            ser.write(message)
            print(f"已发送: {message.hex().upper()}")
            return True
    except Exception as e:
        print(f"发送错误: {e}")
        return False

def Motor_Control_tb(Pulse_count, ID=0x01, A=0x00, Speed=0x05, PORT=PORT):
    msg = [ID, 0xFD]
    direction = 1 if Speed >= 0 else 0
    Speed = abs(Speed)
    msg += [direction, Speed // 256, Speed % 256, A,
            (Pulse_count >> 24) & 0xFF, (Pulse_count >> 16) & 0xFF,
            (Pulse_count >> 8) & 0xFF, Pulse_count & 0xFF,
            0x01, 0x01, 0x6B]
    send_hex_data(PORT, msg)

def generate_motor_commands(x, y, speed = 256):
    """
    根据目标位移 (x, y) 和速度 speed，生成两个电机的控制命令字符串。

    参数:
        x (int): x方向位移（右为正）
        y (int): y方向位移（上为正）
        speed (int): 电机速度大小（正负表示方向）

    返回:
        list[str]: 两个电机的命令字符串
    """

    # 避免 speed = 0 的除以 0 错误
    if speed == 0:
        raise ValueError("Speed 不能为0")

    # 电机 0x01 控制 x（右）
    pulse_x = int(abs(x))
    speed_x = speed if x >= 0 else -speed

    # 电机 0x02 控制 y（下），但坐标系中 y 上为正，所以要取 -y
    pulse_y = int(abs(y))
    speed_y = -speed if y >= 0 else speed  # y > 0 表示向上，对应电机方向为负

    Motor_Control_tb(ID = 0x01, Pulse_count = pulse_x, Speed = speed_x)
    Motor_Control_tb(ID = 0x02, Pulse_count = pulse_y, Speed = speed_y)
    time.sleep(0.0001)
    send_hex_data(PORT, [0x00, 0xFF, 0x66, 0x6B])

class AllContourVisualizer:
    def __init__(self, image_path, threshold_method='otsu', blur_kernel=(3, 3),
                 min_contour_area=50, scale_factor=1.0):
        self.image_path = image_path
        self.threshold_method = threshold_method
        self.blur_kernel = blur_kernel
        self.min_contour_area = min_contour_area  # 减小阈值以保留更多小轮廓
        self.scale_factor = scale_factor
        self.image = None
        self.gray_image = None
        self.blurred_image = None
        self.binary_image = None
        self.contours = None  # 存储所有轮廓（外部+内部）
        self.processed_path = None
        self.all_points = []
        self.segment_points = []  # 分段点（仅在无法一笔画时标记）

    def load_image(self):
        """加载图像并缩放"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"无法加载图像: {self.image_path}")
        if self.scale_factor != 1.0:
            self.image = cv2.resize(
                self.image, (0, 0), fx=self.scale_factor, fy=self.scale_factor,
                interpolation=cv2.INTER_AREA
            )
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def preprocess_image(self):
        """轻度模糊，保留内部轮廓细节"""
        if self.gray_image is None:
            self.load_image()
        # 使用小核模糊，避免内部细节丢失
        self.blurred_image = cv2.GaussianBlur(self.gray_image, self.blur_kernel, 0)
        return self.blurred_image

    def binarize_image(self, threshold_value=127):
        """二值化时保留内部细节"""
        if self.blurred_image is None:
            self.preprocess_image()
        # 降低阈值对比度，保留内部轮廓
        if self.threshold_method == 'otsu':
            _, self.binary_image = cv2.threshold(
                self.blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            # 手动阈值可适当降低，保留更多内部线条
            _, self.binary_image = cv2.threshold(
                self.blurred_image, threshold_value, 255, cv2.THRESH_BINARY_INV
            )
        # 轻度形态学操作，仅去除噪点，保留内部轮廓
        kernel = np.ones((3, 3), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        return self.binary_image

    def extract_all_contours(self):
        """提取所有轮廓（包括外部轮廓和内部轮廓）"""
        if self.binary_image is None:
            self.binarize_image()
        # 关键参数：RETR_TREE 提取所有轮廓，并保留层级关系
        contours, hierarchy = cv2.findContours(
            self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE  # 保留所有点，不简化
        )
        # 过滤过小的噪点轮廓，但保留有意义的内部轮廓
        self.contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                self.contours.append(cnt)
        return self.contours

    def optimize_path(self):
        """优化路径：优先一笔画，包含所有轮廓（外部+内部）"""
        if self.contours is None:
            self.extract_all_contours()
        if not self.contours:
            print("未检测到任何轮廓")
            return

        # 提取所有轮廓的点列表（保持连续性）
        contour_points = []
        for cnt in self.contours:
            points = cnt.reshape(-1, 2).tolist()
            contour_points.append(points)

        # 路径优化：就近连接所有轮廓（包括内部轮廓）
        processed = [0]
        unprocessed = list(range(1, len(contour_points)))
        self.processed_path = [contour_points[0]]

        # 记录分段点（不同轮廓的连接点）
        self.segment_points = []
        current_end_idx = len(contour_points[0]) - 1  # 第一个轮廓的终点索引

        while unprocessed:
            current_end = contour_points[processed[-1]][-1]
            nearest_idx = -1
            min_dist = float('inf')

            # 寻找最近的下一个轮廓（包括内部轮廓）
            for idx in unprocessed:
                start = contour_points[idx][0]
                dist = math.hypot(start[0] - current_end[0], start[1] - current_end[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx

            # 添加下一个轮廓
            processed.append(nearest_idx)
            unprocessed.remove(nearest_idx)
            self.processed_path.append(contour_points[nearest_idx])

            # 记录分段点
            self.segment_points.append(current_end_idx)
            current_end_idx += len(contour_points[nearest_idx])
            self.segment_points.append(current_end_idx)

        # 合并所有点
        self.all_points = []
        for points in self.processed_path:
            self.all_points.extend(points)

        return self.processed_path

    def static_visualization(self):
        """静态可视化所有轮廓，颜色渐变表示顺序，红色标记分段点"""
        if self.all_points is None:
            self.optimize_path()
        if not self.all_points:
            print("没有可显示的轮廓点")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title('所有轮廓路径（颜色表示顺序，红色为分段点）')
        ax.axis('off')

        # 背景显示二值化图像
        ax.imshow(self.binary_image, cmap='gray', alpha=0.3)

        # 颜色渐变表示绘制顺序（蓝→红）
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.all_points)))

        # 绘制所有轮廓路径
        for i in range(len(self.all_points) - 1):
            x = [self.all_points[i][0], self.all_points[i+1][0]]
            y = [self.all_points[i][1], self.all_points[i+1][1]]
            ax.plot(x, y, color=colors[i], linewidth=1.5, alpha=0.9)

        # 标记分段点（红色）
        if self.segment_points:
            seg_x = [self.all_points[i][0] for i in self.segment_points if 0 <= i < len(self.all_points)]
            seg_y = [self.all_points[i][1] for i in self.segment_points if 0 <= i < len(self.all_points)]
            ax.scatter(seg_x, seg_y, color='red', s=30, zorder=3,
                      edgecolors='black', linewidths=0.5)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=0, vmax=len(self.all_points)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='绘制顺序（蓝→红）', shrink=0.8)

        plt.tight_layout()
        plt.show()
    def get_path_info(self):
        """返回图像宽高、路径段数、每段的点列表"""
        if self.processed_path is None:
            self.optimize_path()
        height, width = self.binary_image.shape[:2]
        segment_count = len(self.processed_path)
        return (width, height), segment_count, self.processed_path

    def transform_coordinates(self, scale=1.0, origin=(0, 0)):
        """
        将坐标变换为：
        1. 先找到所有轮廓点的左下角点 (min_x, max_y)
        2. 所有点相对该点进行平移
        3. 再将这个左下角映射到 origin
        4. 最后进行缩放（以 origin 为原点）
        """
        if self.processed_path is None:
            self.optimize_path()

        # 收集所有点
        all_points = [pt for segment in self.processed_path for pt in segment]
        xs = [x for x, y in all_points]
        ys = [y for x, y in all_points]

        min_x = min(xs)
        max_y = max(ys)  # 图像坐标系中，y越大表示越下

        ox, oy = origin
        transformed_path = []

        for segment in self.processed_path:
            new_segment = []
            for (x, y) in segment:
                rel_x = x - min_x
                rel_y = max_y - y  # 因为 y 向下，需“反转”
                new_x = ox + rel_x * scale
                new_y = oy + rel_y * scale
                new_segment.append((new_x, new_y))
            transformed_path.append(new_segment)

        return transformed_path


def extract_transformed_segments(image_path, origin=(0, 0), scale=1.0,
                                 threshold_method='otsu', blur_kernel=(3, 3),
                                 min_contour_area=50, scale_factor=1.0):
    """
    主函数接口：
    - 输入图像路径、原点坐标、缩放因子
    - 返回转换后的轮廓段列表（大列表，每一段是若干坐标点）

    返回：
        transformed_segments: List[List[Tuple[float, float]]]
    """
    visualizer = AllContourVisualizer(
        image_path=image_path,
        threshold_method=threshold_method,
        blur_kernel=blur_kernel,
        min_contour_area=min_contour_area,
        scale_factor=scale_factor
    )

    # # 图像处理流程
    # try:
    #     visualizer.load_image()
    #     visualizer.preprocess_image()
    #     visualizer.binarize_image()
    #     visualizer.extract_all_contours()
    #     visualizer.optimize_path()
    # except Exception as e:
    #     print(f"处理失败: {e}")
    #     return []

    # 坐标变换
    transformed_segments = visualizer.transform_coordinates(
        scale=scale,
        origin=origin
    )
    return transformed_segments


if __name__ == "__main__":
    PORT = '/dev/ttyUSB0'
    send_hex_data(PORT, [0x01, 0x9A, 0x00, 0x00, 0x6B])
    send_hex_data(PORT, [0x02, 0x9A, 0x00, 0x00, 0x6B])
    
    # image_path = "/root/D-race/Z-zong/nailong.jpg"
    # origin = (-2250,-1600)
    # scale = 1.3


    # image_path = "/root/D-race/Z-zong/yuanshen.jpg"
    # origin = (-2250,-1600)
    # scale = 0.8

    # image_path = "/root/D-race/Z-zong/mihoyo.jpg"
    # origin = (-2100,-1600)
    # scale = 0.5

    time.sleep(0.2)
    generate_motor_commands(origin[0],origin[1])
    time.sleep(0.2)
    segments = extract_transformed_segments(image_path, origin=origin, scale=scale)

    print(f"共 {len(segments)} 段")
    print("第一段前5个点：")
    print(segments[0][:5])

    # 可选：计算变换后的边界框
    all_points = [pt for seg in segments for pt in seg]
    xs = [x for x, y in all_points]
    ys = [y for x, y in all_points]

    stride = 3  # 每隔几个点发送一次指令（例如 3 表示每隔 3 个点发一次）

    for segment in segments:
        length = len(segment)
        for i in range(0, length - 1, stride):
            x, y = segment[i]
            generate_motor_commands(x, y)

        # 保证最后一个点也执行（如果它不是刚好 stride 的倍数）
        if length > 0:
            x, y = segment[-1]
            generate_motor_commands(x, y)

    generate_motor_commands(2000,2000) # 移到外面

