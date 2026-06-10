import time
from http import HTTPStatus
import cv2
import numpy as np
import requests
import torch
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PySide6.QtCore import Qt, QPoint, QObject, Signal, QThread
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from dashscope import VideoSynthesis
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from UI.ui_DunhuangRestorationApp import *
from PySide6.QtWidgets import *
import os
from PIL import Image
from PIL import ImageQt
from XDoG import *
from SAM_Adapter_Model import *
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, \
    DDIMScheduler

import sys
import os

# ======== SinSR 模型相关 =======
# 👉 添加 SinSR-main 到路径
# sys.path.append(r"D:\作业\毕业设计\SinSR-main")
sys.path.append(r"../ImageSR/SinSR")

from argparse import Namespace
# from inference import get_configs
# from sampler import Sampler
from ImageSR.SinSR.inference import get_configs
from ImageSR.SinSR.sampler import Sampler
import torchvision.transforms._functional_tensor


# =============================

def numpy_to_qpixmap(np_array):
    print("ndim", np_array.ndim)
    if np_array.ndim == 3:
        print("shape", np_array.shape)
        if np_array.shape[2] == 4:
            print("1numpy_to_qpixmap has alpha")
            pil_image = Image.fromarray(np_array, mode='RGBA')
            width, height = pil_image.size
            data = pil_image.tobytes("raw", "BGRA")
            qimage = QImage(data, width, height, QImage.Format.Format_ARGB32_Premultiplied)
            qpixmap = QPixmap()
            qpixmap.convertFromImage(qimage, Qt.ImageConversionFlags.NoFormatConversion)
            # # 使用特点格式创建 QPixmap，保持 alpha 通道
            # qpixmap = ImageQt.toqpixmap(pil_image)
            print("qpixmap format:", qpixmap.toImage().format())
        else:
            pil_image = Image.fromarray(np_array)
            qpixmap = ImageQt.toqpixmap(pil_image)
    return qpixmap


def qpixmap_to_numpy(qpixmap):
    """
    QPixmap -> numpy
    如果原图含 alpha，则强制保持4通道
    """
    print("qpixmap", qpixmap.size(), qpixmap.hasAlpha())
    pil_image = ImageQt.fromqpixmap(qpixmap)
    if qpixmap.hasAlpha():
        print("2qpixmap_to_numpy has alpha")
        pil_image = pil_image.convert("RGBA")
        np_array = np.array(pil_image)
    else:
        print("2qpixmap_to_numpy no alpha")
        np_array = np.array(pil_image)
    return np_array


def create_cursor_pixmap(color, size):
    diameter = size
    pixmap = QPixmap(diameter, diameter)
    pixmap.fill(Qt.transparent)  # 背景透明
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, diameter, diameter)  # 画圆
    painter.end()
    return pixmap


class LineGraphicsView(QGraphicsView):
    def __init__(self, size=512):
        super().__init__()

        self.viewport().installEventFilter(self)
        self.setMouseTracking(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # 设置视图属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.layer1 = QGraphicsPixmapItem()

        self.scene.addItem(self.layer1)

        self.line_path = None
        self.line_pixmap = None
        self.line = None

        self.line_drawing = False
        self.last_point = QPoint()
        self.size = size

        self.paintbrush_size = 1  # 画笔大小
        self.tool = "pointer"  # 默认选择指针工具，也就是鼠标

        self.init_empty_line(self.size, self.size)

    def init_empty_line(self, width, height):
        # 初始化空的线稿
        self.line = np.zeros((height, width, 3), dtype=np.uint8) + 255
        self.line_pixmap = numpy_to_qpixmap(self.line)
        self.layer1.setPixmap(self.line_pixmap)
        # 设置场景大小
        self.scene.setSceneRect(0, 0, width, height)

        print(self.line.shape, self.line.dtype, type(self.line), np.min(self.line), np.max(self.line))

    def upload_line(self):
        self.line_path, _ = QFileDialog.getOpenFileName(self, '选择线稿图像', './', '图像文件(*.jpg *.png *.jpeg)')
        if not self.line_path:
            return
        line = Image.open(self.line_path).convert('RGB')
        self.line = np.array(line)

        self.line_pixmap = numpy_to_qpixmap(self.line)
        print("上传")
        print(self.line.shape, self.line.dtype, type(self.line), np.min(self.line), np.max(self.line))
        self.layer1.setPixmap(self.line_pixmap)

    def export_line(self):
        if self.line is None:
            QMessageBox.warning(self, "提示", "没有线稿可导出！")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, '保存线稿图像', './line_art.png',
                                                   '图像文件(*.jpg *.png *.jpeg)')
        if not save_path:
            return
        line_img = Image.fromarray(self.line)
        line_img.save(save_path)
        QMessageBox.information(self, "提示", f"线稿已保存到: {save_path}")

    def line_art_extraction(self, image_path):
        if image_path is None:
            QMessageBox.warning(self, "提示", "请先上传图像！")
            return
        # 图像预处理流程
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图
            transforms.ToTensor(),  # 将图像转换为PyTorch张量
        ])
        with Image.open(image_path) as img:
            image_tensor = transform(img).unsqueeze(0)
            xdog_result = XDoGFilter(image_tensor, sigma1=1.0, sigma2=1.2, phi=0.1, epsilon=0, sharpen=100)
            xdog_result_img = xdog_result.squeeze().detach().numpy()
            xdog_result_img = (xdog_result_img).astype(np.uint8)
            # print(xdog_result_img.shape, type(xdog_result_img))

            xdog_result_img = np.stack([xdog_result_img] * 3, axis=-1)

            self.line = xdog_result_img
            print("提取")
            print(self.line.shape, self.line.dtype, type(self.line), np.min(self.line), np.max(self.line))

            pil_img_xdog = Image.fromarray(xdog_result_img)
            self.line_pixmap = ImageQt.toqpixmap(pil_img_xdog)
            self.layer1.setPixmap(self.line_pixmap)

    def draw_line_point(self, point):
        self.line_pixmap = numpy_to_qpixmap(self.line)
        # 创建一个 QPainter，绘制目标是 line
        painter = QPainter(self.line_pixmap)

        # 指针无效果
        if self.tool == "pointer":
            pass
        # 画笔
        elif self.tool == "line_paintbrush":
            pen = QPen(Qt.black, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPoint(point)
        # 橡皮擦
        elif self.tool == "line_eraser":
            pen = QPen(Qt.white, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPoint(point)
        painter.end()
        self.layer1.setPixmap(self.line_pixmap)
        self.line = qpixmap_to_numpy(self.line_pixmap)
        print(self.line.shape, self.line.dtype, type(self.line), np.min(self.line), np.max(self.line))

    def draw_line_line(self, from_point, to_point):
        self.line_pixmap = numpy_to_qpixmap(self.line)
        # 创建一个 QPainter，绘制目标是 line
        painter = QPainter(self.line_pixmap)

        if self.tool == "pointer":
            pass
        elif self.tool == "line_paintbrush":
            pen = QPen(Qt.black, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(from_point, to_point)
        elif self.tool == "line_eraser":
            pen = QPen(Qt.white, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(from_point, to_point)
        painter.end()
        self.layer1.setPixmap(self.line_pixmap)
        self.line = qpixmap_to_numpy(self.line_pixmap)
        print(self.line.shape, self.line.dtype, type(self.line), np.min(self.line), np.max(self.line))

    def eventFilter(self, source, event):
        # 检查事件源是否是 graphicsView_line的视口
        if source == self.viewport():
            # 鼠标按下事件
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    print("鼠标按下")
                    self.line_drawing = True
                    view_pos = event.pos()
                    scene_pos = self.mapToScene(view_pos)
                    self.last_point = scene_pos.toPoint()
                    # self.mask_paint(self.last_point)
                    self.draw_line_point(self.last_point)
                    return True

            # 鼠标移动事件
            elif event.type() == QtCore.QEvent.MouseMove:
                if self.line_drawing:
                    print("鼠标移动")
                    view_pos = event.pos()
                    scene_pos = self.mapToScene(view_pos)
                    point = scene_pos.toPoint()
                    self.draw_line_line(self.last_point, point)
                    self.last_point = point
                    return True

            # 鼠标释放事件
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    # self.mask_drawing = False
                    self.line_drawing = False
                    return True

            # 鼠标滚轮事件
            elif event.type() == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.paintbrush_size += 1
                else:
                    self.paintbrush_size = max(1, self.paintbrush_size - 1)

                if self.tool in ["line_paintbrush", "line_eraser"]:
                    # 更新光标
                    color = {
                        "line_paintbrush": Qt.black,
                        "line_eraser": Qt.white,
                    }[self.tool]
                    pixmap = create_cursor_pixmap(color, self.paintbrush_size)
                    self.viewport().setCursor(QCursor(pixmap))

                return True

        return super().eventFilter(source, event)


class MaskGraphicsView(QGraphicsView):
    def __init__(self, size=512):
        super().__init__()

        self.viewport().installEventFilter(self)
        self.setMouseTracking(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # 设置视图属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.layer1 = QGraphicsPixmapItem()
        self.layer2 = QGraphicsPixmapItem()

        self.layer1.setZValue(0)  # 底层
        self.layer2.setZValue(1)  # 顶层

        self.scene.addItem(self.layer1)
        self.scene.addItem(self.layer2)

        self.mask_pixmap = None
        self.background_pixmap = None

        self.mask_transparent = None
        self.mask = None
        self.auto_mask = None

        self.mask_drawing = False
        self.last_point = QPoint()
        self.size = size

        self.paintbrush_size = 1  # 画笔大小
        self.tool = "pointer"  # 默认选择指针工具，也就是鼠标

        self.init_empty_mask(self.size, self.size)

        self.layer2.setOpacity(0.5)  # 设置透明度为50%

    def export_mask(self):
        if self.mask is None:
            QMessageBox.warning(self, "提示", "没有遮罩可导出！")
            return
        else:
            save_path, _ = QFileDialog.getSaveFileName(
                self, '保存遮罩图像', './mask.png',
                '图像文件(*.jpg *.png *.jpeg)'
            )

            # 检查用户是否取消了保存对话框
            if not save_path:
                return

            try:
                mask = self.mask.astype(np.uint8)
                mask = Image.fromarray(mask, mode='L')
                mask.save(save_path)

                QMessageBox.information(self, "成功", f"遮罩已成功保存到: \n{save_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存遮罩失败: \n{str(e)}")
                return

    # 将透明遮罩转换为黑白遮罩
    def transparent_to_black(self):
        if self.mask_transparent is None:
            QMessageBox.warning(self, "提示", "请先绘制遮罩！")
            return

        alpha_channel = self.mask_transparent[:, :, 3]
        self.mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
        print(self.mask.shape, self.mask.dtype, type(self.mask), np.min(self.mask), np.max(self.mask))

    def init_empty_mask(self, width, height):
        # 初始化空的透明遮罩
        self.mask_pixmap = QPixmap(width, height)
        self.mask_pixmap.fill(Qt.transparent)
        self.layer2.setPixmap(self.mask_pixmap)
        # 设置场景大小
        self.scene.setSceneRect(0, 0, width, height)
        # 保存numpy的mask
        self.mask_transparent = qpixmap_to_numpy(self.mask_pixmap)

        print(self.mask_transparent.shape, self.mask_transparent.dtype, type(self.mask_transparent),
              np.min(self.mask_transparent), np.max(self.mask_transparent))

        self.transparent_to_black()

    def clean_all_mask(self):
        self.mask_pixmap.fill(Qt.transparent)
        self.mask_transparent = qpixmap_to_numpy(self.mask_pixmap)
        self.layer2.setPixmap(numpy_to_qpixmap(self.mask_transparent))
        self.transparent_to_black()

    # 全遮罩
    def all_mask(self):
        self.mask_pixmap.fill(Qt.transparent)
        # self.mask_pixmap.fill(QColor(255, 255, 255, 255))

        painter = QPainter(self.mask_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.mask_pixmap.rect(), QColor(255, 255, 255, 255))
        painter.end()

        self.mask_transparent = qpixmap_to_numpy(self.mask_pixmap)

        print("all_mask")
        print(self.mask_transparent.shape, self.mask_transparent.dtype, type(self.mask_transparent),
              np.min(self.mask_transparent), np.max(self.mask_transparent))

        self.layer2.setPixmap(numpy_to_qpixmap(self.mask_transparent))

        self.transparent_to_black()

    # 使用检测遮罩
    def detect_mask(self):
        if self.auto_mask is None:
            QMessageBox.warning(self, "提示", "请先进行损坏区域检测！")
            return
        print(self.auto_mask.shape, self.auto_mask.dtype, type(self.auto_mask),
              np.min(self.auto_mask), np.max(self.auto_mask))

        self.mask_transparent = cv2.cvtColor(self.auto_mask, cv2.COLOR_GRAY2RGBA)

        self.mask_transparent[:, :, 3] = np.where(self.auto_mask > 0, 255, 0)

        self.layer2.setPixmap(numpy_to_qpixmap(self.mask_transparent))
        self.transparent_to_black()

    def load_background_image(self, image):
        # 加载背景图像到底层
        self.layer1.setPixmap(image)
        self.layer2.setPixmap(numpy_to_qpixmap(self.mask_transparent))

    def draw_mask_point(self, point):
        mask = numpy_to_qpixmap(self.mask_transparent)
        print()
        print("1", mask.toImage().format())
        # 创建一个 QPainter，绘制目标是 mask
        painter = QPainter(mask)

        # 指针无效果
        if self.tool == "pointer":
            pass
        # 画笔
        elif self.tool == "mask_paintbrush":
            pen = QPen(Qt.white, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPoint(point)
        # 橡皮擦
        elif self.tool == "mask_eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.transparent, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPoint(point)
        painter.end()
        print("2", mask.toImage().format())
        self.layer2.setPixmap(mask)
        self.mask_transparent = qpixmap_to_numpy(mask)
        print(self.mask_transparent.shape, self.mask_transparent.dtype, type(self.mask_transparent),
              np.min(self.mask_transparent), np.max(self.mask_transparent))
        self.transparent_to_black()

    def draw_mask_line(self, from_point, to_point):
        mask = numpy_to_qpixmap(self.mask_transparent)
        # 创建一个 QPainter，绘制目标是 mask
        painter = QPainter(mask)

        if self.tool == "pointer":
            pass
        elif self.tool == "mask_paintbrush":
            pen = QPen(Qt.white, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(from_point, to_point)
        elif self.tool == "mask_eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.transparent, self.paintbrush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(from_point, to_point)
        painter.end()
        self.layer2.setPixmap(mask)
        self.mask_transparent = qpixmap_to_numpy(mask)
        self.transparent_to_black()

    def eventFilter(self, source, event):
        # 检查事件源是否是 graphicsView_mask_manual的视口
        if source == self.viewport():
            # 鼠标按下事件
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    print("鼠标按下")
                    self.mask_drawing = True
                    view_pos = event.pos()
                    scene_pos = self.mapToScene(view_pos)
                    self.last_point = scene_pos.toPoint()
                    self.draw_mask_point(self.last_point)
                    return True

            # 鼠标移动事件
            elif event.type() == QtCore.QEvent.MouseMove:
                if self.mask_drawing:
                    print("鼠标移动")
                    view_pos = event.pos()
                    scene_pos = self.mapToScene(view_pos)
                    point = scene_pos.toPoint()
                    self.draw_mask_line(self.last_point, point)
                    self.last_point = point
                    return True

            # 鼠标释放事件
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.mask_drawing = False
                    return True

            # 鼠标滚轮事件
            elif event.type() == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.paintbrush_size += 1
                else:
                    self.paintbrush_size = max(1, self.paintbrush_size - 1)

                if self.tool in ["mask_paintbrush", "mask_eraser"]:
                    # 更新光标
                    color = {
                        "mask_paintbrush": Qt.white,
                        "mask_eraser": Qt.black,
                    }[self.tool]
                    pixmap = create_cursor_pixmap(color, self.paintbrush_size)
                    self.viewport().setCursor(QCursor(pixmap))

                return True

        return super().eventFilter(source, event)


def display_image(img, GraphicsView):
    # 设置场景
    scene = QtWidgets.QGraphicsScene()
    # 将图片添加到场景中
    scene.addPixmap(img)
    # 将场景设置到 GraphicsView 中
    GraphicsView.setScene(scene)

    # 让图片填满 GraphicsView
    GraphicsView.setSceneRect(scene.itemsBoundingRect())  # 设置场景范围为图片大小
    GraphicsView.fitInView(scene.itemsBoundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # 可选：当 GraphicsView 尺寸变化时，自动缩放
    GraphicsView.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    GraphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


# def init_graphicsView(self, GraphicsView):
#     self.display_mask_scene = QtWidgets.QGraphicsScene(self)
#
#     self.image_item = QtWidgets.QGraphicsPixmapItem()
#     self.image_item.setZValue(0)  # 底层
#     self.display_mask_scene.addItem(self.image_item)
#
#     # 顶层 mask item
#     self.mask_item = QtWidgets.QGraphicsPixmapItem()
#     self.mask_item.setZValue(1)  # 顶层
#     self.display_mask_scene.addItem(self.mask_item)
#
#     GraphicsView.setScene(self.display_mask_scene)
#     # 让图片填满 GraphicsView
#     GraphicsView.setSceneRect(self.display_mask_scene.itemsBoundingRect())  # 设置场景范围为图片大小
#     GraphicsView.fitInView(self.display_mask_scene.itemsBoundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
#     # 可选：当 GraphicsView 尺寸变化时，自动缩放
#     GraphicsView.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#     GraphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


# ================= 损坏区域检测模块 =============
# ================= 多线程 ==================
class SegmentationWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, image_path, sam_base_model_path, sam_finetune_model_path, image_size, device):
        super().__init__()
        self.image_path = image_path
        self.sam_base_model_path = sam_base_model_path
        self.sam_finetune_model_path = sam_finetune_model_path
        self.image_size = image_size
        self.device = device

    def run(self):
        try:
            model = SAMAdapterModel(model_type="vit_h", checkpoint=self.sam_base_model_path)
            model.load_state_dict(torch.load(self.sam_finetune_model_path, weights_only=True))
            model = model.to(self.device)
            model.eval()

            # image = cv2.imread(self.image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 用 PIL 读图（稳定支持中文路径）
            image = Image.open(self.image_path).convert("RGB")
            # PIL → numpy（RGB, uint8）
            image = np.array(image)

            transform = A.Compose([
                A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            transformed = transform(image=image)
            image = transformed['image']

            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # [b, 1, 512, 512]
                pred_masks = model(image)
                print("pred_masks:")
                print(pred_masks.shape)

            # 释放显存
            torch.cuda.empty_cache()

            # [b, 512, 512, 1]
            pred_masks = torch.clamp(pred_masks, -1, 1).detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
            print(pred_masks.shape)

            # [b, 512, 512]
            if pred_masks.shape[-1] == 1:
                pred_masks = pred_masks.squeeze(-1)
                print(pred_masks.shape)

            # 二值化01后转换成0-255
            mask = (pred_masks[0] > 0).astype(np.uint8) * 255
            # 不二值化直接转换成0-255
            # mask = (pred_masks[0] * 255).astype(np.uint8)

            self.finished.emit(mask)
        except Exception as e:
            self.error.emit(str(e))


class SegmentationModule(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.ui = parent.ui
        self.device = parent.device
        self.image_size = parent.image_size

        # ========= 病害区域检测模块 =========
        # 模型路径
        self.sam_base_model_path = r"../ImageSeg/checkpoints/sam_vit_h_4b8939.pth"
        self.sam_finetune_model_path = r"../ImageSeg/checkpoints/best_iou_model.pth"
        self.sam_finetune_model_path = r"../ImageSeg/checkpoints/k7_SAM_Focal_Tversky_Loss.pth"

        # 图像路径
        self.image_path = None
        self.init_image = None
        self.image = None

        # 初始化参数
        # 模型路径设置
        self.ui.lineEdit_SAM_base_model_path.setText(self.sam_base_model_path)
        self.ui.lineEdit_SAM_finetune_model_path.setText(self.sam_finetune_model_path)

        # 实时更新参数
        self.ui.lineEdit_SAM_base_model_path.textChanged.connect(
            lambda text: setattr(self, "sam_base_model_path", text)
        )
        self.ui.lineEdit_SAM_finetune_model_path.textChanged.connect(
            lambda text: setattr(self, "sam_finetune_model_path", text)
        )

        # 按钮事件
        # 上传图像按钮
        self.ui.pushButton_upload_image.clicked.connect(
            lambda: self.upload_image(self.ui.graphicsView_init_image)
        )
        # 损坏区域检测按钮
        self.ui.pushButton_seg_detect.clicked.connect(
            lambda: self.Image_seg_detect()
        )
        # ====================================

    # 上传图像
    def upload_image(self, GraphicsView):
        self.image_path, _ = QFileDialog.getOpenFileName(self.parent, '选择图像', './', '图像文件(*.jpg *.png *.jpeg)')
        if not self.image_path:
            return
        self.init_image = np.array(Image.open(self.image_path).convert("RGB"))
        # 将图像加载为 QPixmap
        img = QPixmap(self.image_path)
        self.ui.graphicsView_mask_manual.load_background_image(img)
        display_image(img, GraphicsView)

    # 损坏区域检测
    def Image_seg_detect(self):
        if self.image_path is None:
            QMessageBox.warning(self.parent, "提示", "请先上传图像！")
            return

        self.ui.pushButton_seg_detect.setEnabled(False)

        self.thread = QThread()
        self.worker = SegmentationWorker(
            self.image_path,
            self.sam_base_model_path,
            self.sam_finetune_model_path,
            self.image_size,
            self.device
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_segmentation_finished)
        self.worker.error.connect(self.on_segmentation_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_segmentation_finished(self, mask):
        self.ui.pushButton_seg_detect.setEnabled(True)

        # self.ui.graphicsView_mask_manual.mask = mask
        self.ui.graphicsView_mask_manual.auto_mask = mask
        # print(self.mask.shape)

        mask = Image.fromarray(mask, mode='L')
        print(mask, type(mask))
        mask = ImageQt.toqpixmap(mask)
        print(mask, type(mask))

        display_image(mask, self.ui.graphicsView_mask)

    def on_segmentation_error(self, error_message):
        self.ui.pushButton_seg_detect.setEnabled(True)
        QMessageBox.critical(self.parent, "错误", f"损坏区域检测失败: \n{error_message}")


# ================== 壁画修复模块 =============
# ================== 多线程 ==================
class InpaintingWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, image, line, mask, anti_mask, sd_model_path, controlnet_model_path, step, device):
        super().__init__()
        self.image = image
        self.line = line
        self.mask = mask
        self.anti_mask = anti_mask
        self.sd_model_path = sd_model_path
        self.controlnet_model_path = controlnet_model_path
        self.step = step
        self.device = device

    def run(self):
        try:
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_path,
                # torch_dtype=weight_dtype,
                local_files_only=True,
            )

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.sd_model_path,
                controlnet=controlnet,
                # torch_dtype=weight_dtype,
                safety_checker=None,
                local_files_only=True,
            )

            noise_scheduler = DDIMScheduler.from_pretrained(self.sd_model_path, subfolder="scheduler")
            noise_scheduler.set_timesteps(self.step, device=self.device)

            vae = pipe.vae
            unet = pipe.unet
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder

            vae.eval()
            unet.eval()
            text_encoder.eval()
            controlnet.eval()

            vae.to(self.device)
            unet.to(self.device)
            text_encoder.to(self.device)
            controlnet.to(self.device)

            prompt = [""] * 64
            inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(self.device)
            with torch.no_grad():
                encoder_hidden_states_full = text_encoder(input_ids)[0]

            with torch.no_grad():
                encoder_hidden_states = encoder_hidden_states_full[:self.image.shape[0]]

                # 编码到潜空间
                latent_images = vae.encode(self.image).latent_dist.sample()
                latent_images = latent_images * vae.config.scaling_factor

                masks = torch.nn.functional.interpolate(self.mask, size=latent_images.shape[2:], mode='nearest')
                anti_masks = torch.nn.functional.interpolate(self.anti_mask, size=latent_images.shape[2:],
                                                             mode='nearest')

                x_t = torch.randn_like(latent_images).to(self.device)
                # 从标准正态分布中采样噪声 ε~N(0,I)
                fixed_noises = x_t.clone()

                print("noise_scheduler.config.num_train_timesteps:", noise_scheduler.config.num_train_timesteps)
                print("noise_scheduler.timesteps:", noise_scheduler.timesteps)
                for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
                    t_batch = torch.tensor([t] * latent_images.shape[0]).to(self.device)
                    print("t_batch:", t_batch)
                    gt_noisy = noise_scheduler.add_noise(latent_images, fixed_noises, t_batch)

                    x_t = x_t * masks + gt_noisy * anti_masks

                    controlnet_images = self.line

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x_t,
                        t_batch,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_images,
                        return_dict=False,
                    )

                    predicted_noise = unet(
                        x_t,
                        t_batch,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    prev_samples = []
                    for i in range(x_t.shape[0]):
                        step_output = noise_scheduler.step(
                            predicted_noise[i:i + 1],
                            t_batch[i],
                            x_t[i:i + 1],
                            return_dict=True,
                        )
                        prev_samples.append(step_output.prev_sample)

                    x_t = torch.cat(prev_samples, dim=0)
                pred_images = vae.decode(x_t / vae.config.scaling_factor).sample

            pred_images = pred_images.detach().cpu()
            pred_images = pred_images * 0.5 + 0.5
            pred_images = torch.clip(pred_images, 0, 1)

            # pred_images.shape before permute: torch.Size([1, 3, 512, 512])
            print("pred_images.shape before permute:", pred_images.shape)
            pred_images = pred_images.squeeze().permute(1, 2, 0)
            pred_images = (pred_images.numpy() * 255).astype(np.uint8)

            self.finished.emit(pred_images)

        except Exception as e:
            self.error.emit(str(e))
            return


class InpaintingModule(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.ui = parent.ui

        self.device = parent.device
        self.image_size = parent.image_size

        # 模型路径
        self.sd_model_path = r"runwayml/stable-diffusion-v1-5"
        self.controlnet_model_path = r"../ImageRestoration/checkpoints/controlnet_best_PSNR"

        # 扩散步数
        self.step = 2

        # 初始化参数
        self.ui.lineEdit_sd_model_path.setText(self.sd_model_path)
        self.ui.lineEdit_controlnet_model_path.setText(self.controlnet_model_path)
        self.ui.lineEdit_step.setText(str(self.step))

        # 实时更新参数
        self.ui.lineEdit_sd_model_path.textChanged.connect(
            lambda text: setattr(self, "sd_model_path", text)
        )
        self.ui.lineEdit_controlnet_model_path.textChanged.connect(
            lambda text: setattr(self, "controlnet_model_path", text)
        )
        self.ui.lineEdit_step.textChanged.connect(
            lambda: setattr(self, "step", int(self.ui.lineEdit_step.text()))
        )

        # 按钮事件
        # 图像修复
        self.ui.pushButton_image_restoration.clicked.connect(
            lambda: self.image_restoration(
                self.parent.seg_module.image_path,
                self.image_size,
                self.sd_model_path,
                self.controlnet_model_path,
                self.step,
                self.device
            )
            if self.ui.lineEdit_step.text().isdigit()
            else QMessageBox.warning(self.parent, "提示", "推理步数必须为整数！")
        )
        # 修复图像导出
        self.ui.pushButton_export_restore_image.clicked.connect(
            lambda: self.export_restore_image()
        )

    def image_restoration(self, image_path, image_size, sd_model_path, controlnet_model_path, step, device):
        if image_path is None:
            QMessageBox.warning(self.parent, "提示", "请先加载图像！")
            return

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        ])
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        line_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        image = Image.open(image_path).convert("RGB")
        line = Image.fromarray(self.ui.graphicsView_line.line)
        mask = Image.fromarray(self.ui.graphicsView_mask_manual.mask)

        # 统一变换
        image = transform(image)
        line = transform(line)
        mask = transform(mask)
        # 分别进行图像和掩码的后续变换
        image = image_transform(image)
        line = line_transform(line)
        mask = mask_transform(mask)
        # 二值化到0和1
        mask = (mask > 0.5).float()

        anti_mask = 1 - mask

        # [1, B, H, W]
        image = image.unsqueeze(0).to(device)
        line = line.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        anti_mask = anti_mask.unsqueeze(0).to(device)

        self.thread = QThread()
        self.worker = InpaintingWorker(image, line, mask, anti_mask, sd_model_path, controlnet_model_path, step, device)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.on_inpainting_finished)
        self.worker.error.connect(self.on_inpainting_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_inpainting_finished(self, pred_images):
        self.image = pred_images
        # 显示修复结果
        pred_images_pixmap = numpy_to_qpixmap(pred_images)
        display_image(pred_images_pixmap, self.ui.graphicsView_image_restoration)

    def on_inpainting_error(self, error_message):
        QMessageBox.critical(self.parent, "错误", f"图像修复失败: \n{error_message}")
        return

    def export_restore_image(self):
        if self.image is None:
            QMessageBox.warning(self.parent, "提示", "没有修复结果可导出！")
            return
        else:
            save_path, _ = QFileDialog.getSaveFileName(
                self.parent, '保存修复图像', './restored_image.png',
                '图像文件(*.jpg *.png *.jpeg)'
            )

            # 检查用户是否取消了保存对话框
            if not save_path:
                return

            try:
                restored_image = Image.fromarray(self.image)
                restored_image.save(save_path)

                QMessageBox.information(self.parent, "成功", f"修复图像已成功保存到: \n{save_path}")

            except Exception as e:
                QMessageBox.critical(self.parent, "错误", f"保存修复图像失败: \n{str(e)}")
                return


# ================== 超分辨率模块 =============
# ================== 多线程 ==================
class SRWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            configs, chop_stride = get_configs(self.args)

            resshift_sampler = Sampler(
                configs,
                chop_size=self.args.chop_size,
                chop_stride=chop_stride,
                chop_bs=1,
                use_fp16=True,
                seed=self.args.seed,
                ddim=self.args.ddim
            )

            resshift_sampler.inference(self.args.in_path, self.args.out_path, bs=1, noise_repeat=False,
                                       one_step=self.args.one_step)

            files = os.listdir(self.args.out_path)
            sr_output_image_path = os.path.join(self.args.out_path, files[0])

            self.finished.emit(sr_output_image_path)
        except Exception as e:
            self.error.emit(str(e))


class SuperResolutionModule(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.ui = parent.ui

        # ========= 超分辨率模块 =========
        # 模型路径
        self.sr_model_path = r"../ImageSR/SinSR/weights/SinSR_v1.pth"
        # 输入图像路径
        self.sr_input_image_path = None
        # 输出文件夹路径
        self.sr_output_image_dir_path = r"results"

        # 初始化参数
        self.ui.lineEdit_sr_model_path.setText(self.sr_model_path)

        # 定义图像显示的场景
        self.scene_sr_input = QGraphicsScene(self.parent)
        self.ui.graphicsView_image_sr_input.setScene(self.scene_sr_input)
        self.scene_sr_output = QGraphicsScene(self.parent)
        self.ui.graphicsView_image_sr_output.setScene(self.scene_sr_output)

        # 实时更新参数
        self.ui.lineEdit_sr_model_path.textChanged.connect(
            lambda text: setattr(self, "sr_model_path", text)
        )

        # 按钮事件
        # 上传修复图像
        self.ui.pushButton_upload_sr_image.clicked.connect(
            lambda: self.upload_image_to_sr(self.ui.graphicsView_image_sr_input)
        )
        # 超分按钮
        self.ui.pushButton_image_sr.clicked.connect(
            lambda: self.image_sr()
        )
        # 超分图像导出
        self.ui.pushButton_export_sr_image.clicked.connect(
            lambda: self.export_sr_image()
        )
        # ====================================

    def upload_image_to_sr(self, GraphicsView):
        self.sr_input_image_path, _ = QFileDialog.getOpenFileName(self.parent, '选择图像', './',
                                                                  '图像文件(*.jpg *.png *.jpeg)')
        if not self.sr_input_image_path:
            return

        self.scene_sr_input.clear()

        img = QPixmap(self.sr_input_image_path)

        sr_image_item = QGraphicsPixmapItem()
        sr_image_item.setPixmap(img)
        self.scene_sr_input.addItem(sr_image_item)

        # self.image_item = self.scene_video.addPixmap(img)
        GraphicsView.setSceneRect(sr_image_item.boundingRect())
        GraphicsView.fitInView(sr_image_item.boundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def image_sr(self):

        if os.path.exists(self.sr_output_image_dir_path):
            for f in os.listdir(self.sr_output_image_dir_path):
                os.remove(os.path.join(self.sr_output_image_dir_path, f))
        else:
            os.makedirs(self.sr_output_image_dir_path, exist_ok=True)

        args = Namespace(
            in_path=self.sr_input_image_path,
            out_path=self.sr_output_image_dir_path,
            ref_path=None,
            steps=15,
            config=None,
            infer_steps=15,
            scale=4,
            seed=12345,
            one_step=True,
            # ckpt="checkpoints/SinSR_v1.pth",
            ckpt=self.sr_model_path,
            chop_size=256,
            task="SinSR",
            ddim=False
        )

        self.thread = QThread()
        self.worker = SRWorker(args)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_sr_finished)
        self.worker.error.connect(self.on_sr_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_sr_finished(self, sr_output_image_path):
        self.scene_sr_output.clear()

        img = QPixmap(sr_output_image_path)

        sr_image_item = QGraphicsPixmapItem()
        sr_image_item.setPixmap(img)

        self.scene_sr_output.addItem(sr_image_item)

        # self.image_item = self.scene_video.addPixmap(img)
        self.ui.graphicsView_image_sr_output.setSceneRect(sr_image_item.boundingRect())
        self.ui.graphicsView_image_sr_output.fitInView(sr_image_item.boundingRect(),
                                                       QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def on_sr_error(self, error_message):
        QMessageBox.critical(self.parent, "错误", f"超分辨率处理失败: \n{error_message}")
        return

    def export_sr_image(self):
        items = self.scene_sr_output.items()
        if not items:
            QMessageBox.warning(self.parent, "提示", "没有超分辨率图像可导出！")
            return
        else:
            save_path, _ = QFileDialog.getSaveFileName(
                self.parent, '保存超分辨率图像', './sr_image.png',
                '图像文件(*.jpg *.png *.jpeg)'
            )

            # 检查用户是否取消了保存对话框
            if not save_path:
                return

            try:
                sr_image_item = items[0]
                sr_pixmap = sr_image_item.pixmap()
                sr_pixmap.save(save_path)

                QMessageBox.information(self.parent, "成功", f"修复图像已成功保存到: \n{save_path}")

            except Exception as e:
                QMessageBox.critical(self.parent, "错误", f"保存修复图像失败: \n{str(e)}")
                return


# ============= 动态化模块 ==============
# ================== 多线程 ============
class ImageToVideoWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress_message = Signal(str)

    def __init__(self, api_key, wan_model, prompt, i2v_image_path, resolution, duration):
        super().__init__()
        self.api_key = api_key
        self.wan_model = wan_model
        self.prompt = prompt
        self.i2v_image_path = i2v_image_path
        self.resolution = resolution
        self.duration = duration

    def run(self):
        # print("Work线程ID:", QThread.currentThread())
        # try:
        #     self.progress_message.emit("正在生成视频...")
        #
        #     # ================= 测试逻辑开始 =================
        #     print("test: 模拟 API 调用")
        #
        #     # 1. 模拟进度
        #     import time
        #     for i in range(1, 101, 10):
        #         print("test: 进度", i)
        #         self.progress_message.emit(f"视频处理中... {i}%")
        #
        #     # 2. 模拟完成（请确保你有一个测试视频文件，或者先传空字符串）
        #     # 如果你没有测试视频，暂时用 "" 代替，后续在 on_video_finished 中跳过播放
        #     test_video_path = r"D:\作业\毕业设计\Diffusion\ImageToVideo\video\1772513017.mp4"  # 或者放一个真实的 mp4 路径，例如 "C:/test.mp4"
        #
        #     self.progress_message.emit("处理完成！")
        #     self.finished.emit(test_video_path)  # 必须发出 finished 信号
        #     return
        #     # ================= 测试逻辑结束 =================
        # except Exception as e:
        #     self.error.emit(f"视频生成过程中发生错误: {str(e)}")
        #     return

        try:
            self.progress_message.emit("正在生成视频...")
            rsp = VideoSynthesis.call(
                api_key=self.api_key,
                # model='wan2.6-i2v-flash',
                model=self.wan_model,
                prompt=self.prompt,
                img_url=self.i2v_image_path,
                resolution=self.resolution,
                duration=self.duration,
                shot_type="single",
                # prompt_extend=True,
                watermark=True
            )

            if rsp.status_code != HTTPStatus.OK:
                self.error.emit(f"视频生成失败: {rsp.status_code, rsp.code, rsp.message}")
                return

            # 获取视频URL
            video_url = rsp.output.video_url
            self.progress_message.emit("视频生成成功，开始下载...")

            # 下载视频到当前目录的video文件夹，重命名为时间戳.mp4
            os.makedirs("video", exist_ok=True)
            video_path = os.path.join("video", f"{int(time.time())}.mp4")

            response = requests.get(video_url, stream=True)
            if response.status_code != 200:
                self.error.emit(f"视频下载失败:")
                return

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded_size / total_size * 100)
                            self.progress_message.emit(f"视频下载中... {percent}%")
            self.progress_message.emit("视频下载完成！")
            self.finished.emit(video_path)
        except Exception as e:
            self.error.emit(f"视频生成或下载过程中发生错误: {str(e)}")


class ImageToVideoModule(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.ui = parent.ui

        # ========= 动态化模块 =========
        # API
        self.api_key = "sk-296073781ee44f6887460531d93b6b69"

        # self.image_to_video = ImageToVideo()
        # 输入图片路径
        self.i2v_image_path = None
        # 输出视频路径
        self.video_path = None

        # 生成参数
        self.duration = 10
        self.prompt = """基于输入的敦煌壁画原图（支持全景及局部特写），在严格保持原作画幅比例、构图边界及全部元素造型与空间关系完全不变的前提下，对画面进行高保真静态还原与受控动态化处理：初始帧必须为100%一致的静态原画，完整保留矿物颜料的原生色彩、斑驳肌理与历史年代质感，不进行任何风格迁移、细节重绘或内容补全；随后以缓慢、连续且无突变的时间序列对画面进行渐进式唤醒，按照由背景至前景、由次要至主要的视觉层级依次引入极低幅度动态，使画面从静止状态平滑过渡至稳定循环；在循环阶段采用类Live2D的平面轻量动画形式，仅对原画中既有元素施加符合其物理与艺术属性的微动效果，包括人物衣袂、披帛与飘带的柔和摆动、眉眼的极缓开合及手部的细微舒展，装饰纹样中祥云、火焰、水波、莲纹与宝相花的轻柔起伏与周期性舒展，以及器物的极弱振动与能量感波动，同时允许壁画原生背景肌理产生不改变结构的微弱明暗呼吸变化；整个过程中严格限制为低振幅、慢节奏、连续平滑的运动表达，确保无新增元素、无越界变形、无镜头运动、无闪烁抖动及无风格偏移，最终形成庄重典雅、稳定循环且符合文化遗产展示规范的敦煌壁画动态呈现效果。"""

        # 初始化参数
        self.ui.textEdit_api_key.setPlainText(self.api_key)

        self.ui.comboBox_resolution.setCurrentText("720P")
        self.ui.comboBox_model.setCurrentText("wan2.6-i2v-flash")

        self.ui.lineEdit_duration.setText(str(self.duration))
        self.ui.textEdit_prompt.setPlainText(self.prompt)

        self.wan_model = self.ui.comboBox_model.currentText()
        self.resolution = self.ui.comboBox_resolution.currentText()

        self.scene_video = QGraphicsScene(self.parent)
        self.ui.graphicsView_video.setScene(self.scene_video)

        # 实时更新参数
        self.ui.textEdit_api_key.textChanged.connect(
            lambda: setattr(self, "api_key", self.ui.textEdit_api_key.toPlainText())
        )
        self.ui.textEdit_prompt.textChanged.connect(
            lambda: setattr(self, "prompt", self.ui.textEdit_prompt.toPlainText())
        )
        self.ui.comboBox_model.currentTextChanged.connect(
            lambda text: setattr(self, "wan_model", text)
        )
        self.ui.comboBox_resolution.currentTextChanged.connect(
            lambda text: setattr(self, "resolution", text)
        )
        self.ui.lineEdit_duration.textChanged.connect(
            lambda text: setattr(self, "duration", int(self.ui.lineEdit_duration.text()))
        )

        # 按钮事件
        # 上传图像
        self.ui.pushButton_upload_i2v_image.clicked.connect(
            lambda: self.upload_restore_image(self.ui.graphicsView_video)
        )
        # 动态化按钮
        self.ui.pushButton_image_to_video.clicked.connect(
            lambda: self.image_to_video()
            if self.ui.lineEdit_duration.text().isdigit()
            else QMessageBox.warning(self.parent, "提示", "视频时长必须为整数！")
        )
        # 播放暂停按钮
        self.ui.pushButton_play_pause.clicked.connect(
            lambda: self.toggle_play_pause()
        )
        # ====================================

    def toggle_play_pause(self):
        if not hasattr(self, "player") or self.player is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_play_button(self, state):
        if not hasattr(self, "player") or self.player is None:
            return
        if state == QMediaPlayer.PlayingState:
            self.ui.pushButton_play_pause.setText("⏸ 暂停")
        else:
            self.ui.pushButton_play_pause.setText("▶ 播放")

    # 释放资源
    def release_video_resources(self):
        if hasattr(self, "player") and self.player:
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.deleteLater()
            self.player = None
        if hasattr(self, "audio") and self.audio:
            self.audio.deleteLater()
            self.audio = None

    # 上传修复好的图片用于动态化
    def upload_restore_image(self, GraphicsView):
        self.i2v_image_path, _ = QFileDialog.getOpenFileName(self.parent, '选择图像', './',
                                                             '图像文件(*.jpg *.png *.jpeg)')
        if not self.i2v_image_path:
            return

        self.release_video_resources()

        self.scene_video.clear()

        img = QPixmap(self.i2v_image_path)

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setPixmap(img)
        self.scene_video.addItem(self.image_item)

        # self.image_item = self.scene_video.addPixmap(img)
        GraphicsView.setSceneRect(self.image_item.boundingRect())
        GraphicsView.fitInView(self.image_item.boundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def image_to_video(self):
        # self.video_path, _ = QFileDialog.getOpenFileName(self, '选择视频', '.', '视频文件(*.mp4 *.avi *.mov)')

        if self.i2v_image_path is None:
            QMessageBox.warning(self.parent, "提示", "请先上传图像！")
            return

        # 禁用按钮，防止重复点击
        self.ui.pushButton_image_to_video.setEnabled(False)

        print(self.wan_model)
        print(self.resolution)
        print(self.duration)

        self.thread = QThread()
        self.worker = ImageToVideoWorker(
            api_key=self.api_key,
            wan_model=self.wan_model,
            prompt=self.prompt,
            i2v_image_path=self.i2v_image_path,
            resolution=self.resolution,
            duration=self.duration
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.on_video_finished)
        self.worker.error.connect(self.on_video_error)
        self.worker.progress_message.connect(self.on_video_progress)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_video_progress(self, message):
        self.ui.textEdit_download_status.setText(message)

    def on_video_finished(self, video_path):
        print("主线程ID:", QThread.currentThread())
        # 恢复按钮状态
        self.ui.pushButton_image_to_video.setEnabled(True)

        self.video_path = video_path

        self.release_video_resources()

        self.scene_video.clear()

        self.video_item = QGraphicsVideoItem()
        self.video_item.setSize(self.ui.graphicsView_video.size())
        self.scene_video.addItem(self.video_item)

        self.ui.graphicsView_video.setSceneRect(self.video_item.boundingRect())
        self.ui.graphicsView_video.fitInView(self.video_item.boundingRect(),
                                             QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        self.player = QMediaPlayer(self.parent)
        self.audio = QAudioOutput(self.parent)

        self.player.setVideoOutput(self.video_item)
        self.player.setAudioOutput(self.audio)

        self.player.setLoops(QMediaPlayer.Infinite)

        self.player.playbackStateChanged.connect(self.update_play_button)

        self.player.setSource(QUrl.fromLocalFile(self.video_path))

        self.player.play()

    def on_video_error(self, error_message):
        self.ui.pushButton_image_to_video.setEnabled(True)
        QMessageBox.critical(self.parent, "错误", error_message)


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 图片大小及设备设置
        self.image_size = 512
        self.device = "cuda"

        self.ui.lineEdit_size.setText(str(self.image_size))
        self.ui.lineEdit_device.setText(self.device)

        self.ui.lineEdit_size.textChanged.connect(
            lambda: setattr(self, "image_size",
                            int(self.ui.lineEdit_size.text())
                            if self.ui.lineEdit_size.text().isdigit()
                            else QMessageBox.warning(self, "提示", "图像大小必须为整数！"))
        )
        self.ui.lineEdit_device.textChanged.connect(
            lambda: setattr(self, "device", self.ui.lineEdit_device.text())
        )

        # 损坏区域检测模块
        self.seg_module = SegmentationModule(self)
        # 图像修复模块
        self.inpainting_module = InpaintingModule(self)

        # 超分辨率模块
        self.sr_module = SuperResolutionModule(self)
        # 动态化模块
        self.i2v_module = ImageToVideoModule(self)

        # ========= 线稿与掩码编辑模块 =========
        # 替换 graphicsView_line 为自定义的类
        old_graphicsView_line = self.ui.graphicsView_line
        layout = old_graphicsView_line.parent().layout()
        self.ui.graphicsView_line = LineGraphicsView(self.image_size)
        layout.replaceWidget(old_graphicsView_line, self.ui.graphicsView_line)
        old_graphicsView_line.deleteLater()

        # 替换 graphicsView_mask_manual 为自定义的类
        old_graphicsView_mask_manual = self.ui.graphicsView_mask_manual
        layout = old_graphicsView_mask_manual.parent().layout()
        self.ui.graphicsView_mask_manual = MaskGraphicsView(self.image_size)
        layout.replaceWidget(old_graphicsView_mask_manual, self.ui.graphicsView_mask_manual)
        old_graphicsView_mask_manual.deleteLater()

        # 线稿工具按钮事件
        self.ui.pushButton_pointer.clicked.connect(
            lambda: self.select_tool("pointer")
        )
        self.ui.pushButton_paintbrush.clicked.connect(
            lambda: self.select_tool("line_paintbrush")
        )
        self.ui.pushButton_eraser.clicked.connect(
            lambda: self.select_tool("line_eraser")
        )

        # 遮罩工具按钮事件
        self.ui.pushButton_pointer_2.clicked.connect(
            lambda: self.select_tool("pointer")
        )
        self.ui.pushButton_mask_paintbrush.clicked.connect(
            lambda: self.select_tool("mask_paintbrush")
        )
        self.ui.pushButton_mask_eraser.clicked.connect(
            lambda: self.select_tool("mask_eraser")
        )

        # 线稿提取按钮
        self.ui.pushButton_line_art_extraction.clicked.connect(
            lambda: self.ui.graphicsView_line.line_art_extraction(self.seg_module.image_path)
        )
        # 上传线稿按钮
        self.ui.pushButton_upload_line.clicked.connect(
            lambda: self.ui.graphicsView_line.upload_line()
        )
        # 自动遮罩按钮
        self.ui.pushButton_auto_mask.clicked.connect(
            lambda: self.ui.graphicsView_mask_manual.detect_mask()
        )
        # 全部遮罩按钮
        self.ui.pushButton_all_mask.clicked.connect(
            lambda: self.ui.graphicsView_mask_manual.all_mask()
        )
        # 清除遮罩按钮
        self.ui.pushButton_clean_all_mask.clicked.connect(
            lambda: self.ui.graphicsView_mask_manual.clean_all_mask()
        )
        # 线稿导出
        self.ui.pushButton_export_line.clicked.connect(
            lambda: self.ui.graphicsView_line.export_line()
        )
        # 遮罩导出
        self.ui.pushButton_export_mask.clicked.connect(
            lambda: self.ui.graphicsView_mask_manual.export_mask()
        )

        # ====================================

        # ======================== 模型导入模块 =================================
        # ========================== 模型导入按钮事件 ============================
        self.ui.pushButton_SAM_base_model_chose.clicked.connect(
            lambda: self.chose_model(
                self.ui.lineEdit_SAM_base_model_path,
                "基础SAM模型"
            )
        )
        self.ui.pushButton_SAM_finetune_model_chose.clicked.connect(
            lambda: self.chose_model(
                self.ui.lineEdit_SAM_finetune_model_path,
                "微调SAM模型"
            )
        )
        self.ui.pushButton_sd_model_chose.clicked.connect(
            lambda: self.chose_model(
                self.ui.lineEdit_sd_model_path,
                "Stable Diffusion模型"
            )
        )
        self.ui.pushButton_controlnet_model_chose.clicked.connect(
            lambda: self.chose_model(
                self.ui.lineEdit_controlnet_model_path,
                "ControlNet模型"
            )
        )
        self.ui.pushButton_sr_model_chose.clicked.connect(
            lambda: self.chose_model(
                self.ui.lineEdit_sr_model_path,
                "超分模型"
            )
        )
        # 应用模型按钮
        self.ui.pushButton_application.clicked.connect(
            lambda: self.application_model(
                self.seg_module.sam_base_model_path,
                self.seg_module.sam_finetune_model_path,
                self.inpainting_module.sd_model_path,
                self.inpainting_module.controlnet_model_path,
                self.sr_module.sr_model_path,
                self.i2v_module.api_key
            )
        )
        # ==============================================================

    def chose_model(self, lineEdit, model_name):
        try:
            model_path, _ = QFileDialog.getOpenFileName(self, '选择模型', '.',
                                                        '模型文件(*.pth *.pt)')
            if not model_path:
                return
            lineEdit.setText(model_path)

            if model_name == "基础SAM模型":
                self.seg_module.sam_base_model_path = model_path
            if model_name == "微调SAM模型":
                self.seg_module.sam_finetune_model_path = model_path
            if model_name == "Stable Diffusion模型":
                self.inpainting_module.sd_model_path = model_path
            if model_name == "ControlNet模型":
                self.inpainting_module.controlnet_model_path = model_path
            if model_name == "超分模型":
                self.sr_module.sr_model_path = model_path

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载{model_name}失败: {str(e)}")

    def application_model(self, sam_file, seg_file, sd_file, controlnet_file, sr_file, api_key):
        try:
            msg = []
            if sam_file:
                msg.append(f"✔ SAM基础模型已加载: {sam_file}")
            else:
                msg.append("❌ SAM基础模型未加载")

            if seg_file:
                msg.append(f"✔ 损坏区域检测模型已加载: {seg_file}")
            else:
                msg.append("❌ 损坏区域检测模型未加载")

            if sd_file:
                msg.append(f"✔ Stable Diffusion模型已加载: {sd_file}")
            else:
                msg.append("❌ Stable Diffusion模型未加载")

            if controlnet_file:
                msg.append(f"✔ ControlNet模型已加载: {controlnet_file}")
            else:
                msg.append("❌ ControlNet模型未加载")

            if sr_file:
                msg.append(f"✔ 超分辨率模型已加载: {sr_file}")
            else:
                msg.append("❌ 超分辨率模型未加载")

            if api_key:
                msg.append("✔ 通义万相API Key已设置")
            else:
                msg.append("❌ 通义万相API Key未设置")

            self.ui.textEdit_status.setText("\n".join(msg))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"error: {str(e)}")

    # 选用工具
    def select_tool(self, tool):
        self.ui.graphicsView_mask_manual.tool = tool
        self.ui.graphicsView_line.tool = tool

        if tool == "pointer":
            cursor = Qt.ArrowCursor
            self.ui.graphicsView_line.viewport().setCursor(cursor)
            self.ui.graphicsView_mask_manual.viewport().setCursor(cursor)
        elif tool == "line_paintbrush":
            pixmap = create_cursor_pixmap(Qt.black, self.ui.graphicsView_line.paintbrush_size)
            cursor = QCursor(pixmap)
            self.ui.graphicsView_line.viewport().setCursor(cursor)
        elif tool == "line_eraser":
            pixmap = create_cursor_pixmap(Qt.white, self.ui.graphicsView_line.paintbrush_size)
            cursor = QCursor(pixmap)
            self.ui.graphicsView_line.viewport().setCursor(cursor)

        elif tool == "mask_paintbrush":
            pixmap = create_cursor_pixmap(Qt.white, self.ui.graphicsView_mask_manual.paintbrush_size)
            cursor = QCursor(pixmap)
            self.ui.graphicsView_mask_manual.viewport().setCursor(cursor)
        elif tool == "mask_eraser":
            pixmap = create_cursor_pixmap(Qt.black, self.ui.graphicsView_mask_manual.paintbrush_size)
            cursor = QCursor(pixmap)
            self.ui.graphicsView_mask_manual.viewport().setCursor(cursor)


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()
