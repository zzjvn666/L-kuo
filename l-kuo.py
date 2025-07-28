import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import gc

# 设置支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
sys.setrecursionlimit(10000)

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


# 使用示例
if __name__ == "__main__":
    image_path = "mihoyo.jpg"
    visualizer = AllContourVisualizer(
        image_path,
        scale_factor=0.5,
        min_contour_area=50  # 较小的值，保留更多内部轮廓
    )

    try:
        visualizer.load_image()
        visualizer.preprocess_image()
        visualizer.binarize_image()
        visualizer.extract_all_contours()  # 提取所有轮廓（外部+内部）
        visualizer.optimize_path()
        visualizer.static_visualization()
    except Exception as e:
        print(f"程序运行出错: {e}")