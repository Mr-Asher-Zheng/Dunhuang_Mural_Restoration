import matplotlib
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# img_path = r"../XDoG/output/xdog_result.jpg"
img_path = r"../image/img_898crop_0_1.png"
RGB_image = np.array(Image.open(img_path).convert("RGB"))
L_image = np.array(Image.open(img_path).convert("L"))

# ============== 先降噪再增强对比度 ==============
# 降噪
blurred_image = cv2.GaussianBlur(np.array(L_image), (5, 5), 1)
# 增强对比度
contrast_image = cv2.equalizeHist(blurred_image)

# ============= 先增强对比度再降噪 ==============
# 增强对比度
# contrast_image = cv2.equalizeHist(np.array(L_image))
# # 降噪
# blurred_image = cv2.GaussianBlur(contrast_image, (3, 3), 1)

# ============= Canny边缘检测 ==============
canny_image = cv2.Canny(np.array(contrast_image), 50, 150, apertureSize=3, L2gradient=True)
canny_image = 255 - canny_image  # 反转边缘图，使边缘为白色，背景为黑色

# 显示结果并保存
plt.figure(figsize=(10, 10))
plt.subplot(231), plt.imshow(RGB_image), plt.title("原图")
plt.subplot(232), plt.imshow(L_image, cmap="gray"), plt.title("灰度图")
plt.subplot(233), plt.imshow(blurred_image, cmap="gray"), plt.title("降噪后灰度图")
plt.subplot(234), plt.imshow(contrast_image, cmap="gray"), plt.title("增强对比度后灰度图")
plt.subplot(235), plt.imshow(canny_image, cmap="gray"), plt.title("Canny边缘")
# 保存Canny边缘图到根目录
plt.imsave("Canny_edge.png", canny_image, cmap="gray")

plt.axis("off"),
plt.tight_layout()
plt.show()


# # 连通域分析 + 小区域去噪
# import cv2
# import numpy as np
#
# # 找连通区域
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L_image, connectivity=8)
#
# # 按面积过滤
# min_area = 100  # 根据图像大小调整
# filtered = np.zeros_like(L_image)
# for i in range(1, num_labels):  # 跳过背景
#     if stats[i, cv2.CC_STAT_AREA] >= min_area:
#         filtered[labels == i] = 255
#
# cv2.imwrite('filtered.png', filtered)

