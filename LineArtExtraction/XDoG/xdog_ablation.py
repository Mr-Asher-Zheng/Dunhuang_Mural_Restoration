from XDoG import *

# ================== 参数对比试验 =======================================================================
os.makedirs('figures', exist_ok=True)
pil_image = Image.open(r"../image/img_898crop_0_1.png")
image_tensor = transform(pil_image).unsqueeze(0)

# 设置一系列参数进行对比试验
sigma1_initial = 1.0
sigma2_initial = 1.2
phi_initial = 0.1
epsilon_initial = 0
sharpen_initial = 100

# sigma2 = sigma1 * 2
# 使用phi_initial、epsilon_initial和sharpen_initial
sigma1_list1 = [1, 2.5, 5]
sigma2_list1 = [2, 5, 10]

# 设置不同的sigma1和sigma2的差值进行对比试验
# 使用phi_initial、epsilon_initial和sharpen_initial
sigma1_list2 = [1, 1, 1]
sigma2_list2 = [2, 5, 10]

# 设置不同的phi、epsilon和sharpen参数进行对比试验
# 使用sigma1_initial和sigma2_initial
phi_list = [0.005, 0.1, 0.5]
epsilon_list = [0, 100, 200]
sharpen_list = [10, 50, 100]


def concat_and_annotate(images, texts, save_path, gap=20):
    """
    将多张PIL图像横向拼接，并在图像下方标注参数
    images: PIL.Image list
    texts: 文字说明 list
    save_path: 保存路径
    gap: 图片之间的间距（像素），默认20像素
    """
    # 1. 基础校验
    assert len(images) == len(texts), "图像数量与文字说明数量必须一致"
    if not images:
        return

    # 2. 计算拼接画布尺寸
    # 获取所有图片的宽高
    img_widths, img_heights = zip(*(img.size for img in images))
    # 总宽度 = 所有图片宽度之和 + (图片数量-1)*间距
    total_width = sum(img_widths) + (len(images) - 1) * gap
    max_img_height = max(img_heights)  # 最高图片的高度
    text_area_height = 60  # 预留底部文字区域高度
    canvas_height = max_img_height + text_area_height

    # 3. 创建白色背景画布
    canvas = Image.new('RGB', (total_width, canvas_height), color='white')

    # 4. 横向粘贴图片（增加间距）
    x_offset = 0  # 当前图片的x轴偏移量
    for img in images:
        canvas.paste(img, (x_offset, 0))  # 粘贴到画布顶部
        # 偏移量 = 当前偏移 + 图片宽度 + 间距
        x_offset += img.width + gap

    # 5. 绘制文字标注
    draw = ImageDraw.Draw(canvas)

    # 尝试加载系统字体（兼容Windows/Linux/macOS）
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Windows首选
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)  # Linux
        except IOError:
            font = ImageFont.load_default()  # 兜底默认字体

    # 逐个绘制文字（在对应图片下方居中，适配间距）
    x_offset = 0
    for img, text in zip(images, texts):
        # 计算文字边界框 (兼容Pillow 9.0+)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 计算文字坐标：水平居中 + 垂直居中于文字区域
        text_x = x_offset + (img.width - text_w) // 2
        text_y = max_img_height + (text_area_height - text_h) // 2

        draw.text((text_x, text_y), text, fill='black', font=font)
        # 文字偏移量同样要加上间距
        x_offset += img.width + gap

    # 6. 保存结果
    canvas.save(save_path)


# 一共出15张图进行对比试验
# 每个对比3张图，把每个对比的图横向拼接在一起，底下标注参数设置
images = []
texts = []

images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
for s1, s2 in zip(sigma1_list1, sigma2_list1):
    xdog_result = XDoGFilter(image_tensor, s1, s2, phi_initial, epsilon_initial, sharpen_initial)
    xdog_result_img = xdog_result.squeeze().detach().numpy()
    xdog_result_img = (xdog_result_img).astype(np.uint8)
    pil_img_xdog = Image.fromarray(xdog_result_img)
    images.append(pil_img_xdog)
    texts.append(f"sigma1={s1}, sigma2={s2}")
concat_and_annotate(images, texts, 'figures/xdog_sigma_comparison.jpg')

images.clear()
texts.clear()
images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
for s1, s2 in zip(sigma1_list2, sigma2_list2):
    xdog_result = XDoGFilter(image_tensor, s1, s2, phi_initial, epsilon_initial, sharpen_initial)
    xdog_result_img = xdog_result.squeeze().detach().numpy()
    xdog_result_img = (xdog_result_img).astype(np.uint8)
    pil_img_xdog = Image.fromarray(xdog_result_img)
    images.append(pil_img_xdog)
    texts.append(f"sigma1={s1}, sigma2={s2}")
concat_and_annotate(images, texts, 'figures/xdog_sigma_diff_comparison.jpg')

images.clear()
texts.clear()
images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
for p in phi_list:
    xdog_result = XDoGFilter(image_tensor, sigma1_initial, sigma2_initial, p, epsilon_initial, sharpen_initial)
    xdog_result_img = xdog_result.squeeze().detach().numpy()
    xdog_result_img = (xdog_result_img).astype(np.uint8)
    pil_img_xdog = Image.fromarray(xdog_result_img)
    images.append(pil_img_xdog)
    texts.append(f"phi={p}")
concat_and_annotate(images, texts, 'figures/xdog_phi_comparison.jpg')

images.clear()
texts.clear()
images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
for e in epsilon_list:
    xdog_result = XDoGFilter(image_tensor, sigma1_initial, sigma2_initial, phi_initial, e, sharpen_initial)
    xdog_result_img = xdog_result.squeeze().detach().numpy()
    xdog_result_img = (xdog_result_img).astype(np.uint8)
    pil_img_xdog = Image.fromarray(xdog_result_img)
    images.append(pil_img_xdog)
    texts.append(f"epsilon={e}")
concat_and_annotate(images, texts, 'figures/xdog_epsilon_comparison.jpg')

images.clear()
texts.clear()
images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
for s in sharpen_list:
    xdog_result = XDoGFilter(image_tensor, sigma1_initial, sigma2_initial, phi_initial, epsilon_initial, s)
    xdog_result_img = xdog_result.squeeze().detach().numpy()
    xdog_result_img = (xdog_result_img).astype(np.uint8)
    pil_img_xdog = Image.fromarray(xdog_result_img)
    images.append(pil_img_xdog)
    texts.append(f"sharpen={s}")
concat_and_annotate(images, texts, 'figures/xdog_sharpen_comparison.jpg')

images.clear()
texts.clear()
images.append(pil_image)  # 原图
texts.append("Original Image")  # 原图说明
# 最终选定的参数组合：sigma1=1.0, sigma2=1.2, phi=0.1, epsilon=0, sharpen=100
xdog_result = XDoGFilter(image_tensor, sigma1_initial, sigma2_initial, phi_initial, epsilon_initial, sharpen_initial)
xdog_result_img = xdog_result.squeeze().detach().numpy()
xdog_result_img = (xdog_result_img).astype(np.uint8)
pil_img_xdog = Image.fromarray(xdog_result_img)
images.append(pil_img_xdog)
texts.append(f"Final XDoG Result")
concat_and_annotate(images, texts, 'figures/xdog_final_result.jpg')
