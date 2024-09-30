import cv2
import numpy as np
import gradio as gr

from scipy.spatial.distance import cdist

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img


# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标

    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点

    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点

    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1
        )  # 绿色箭头表示映射

    return marked_image


def rbf_kernel(p, x, d=1000000):
    """
    Radial Basis Function (RBF) kernel.
    """
    # 计算 p 和 x 之间的欧氏距离的平方
    dist_sq = cdist(p, x, "sqeuclidean")

    # 计算 RBF 核矩阵
    A = 1 / (dist_sq.T + d)
    return A


# 执行仿射变换


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: 基于RBF 实现 image warping
    h, w, _ = warped_image.shape

    # 计算RBF核矩阵
    A = rbf_kernel(target_pts, target_pts)

    # 计算系数
    coef = np.linalg.solve(A, source_pts - target_pts)

    # 创建网格
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # 计算变形后的网格点
    B = rbf_kernel(target_pts, grid_points)
    q0 = B.dot(coef)
    q = np.floor(q0) + grid_points

    # 找到所有满足条件的点
    pFlag = np.all(q > 0, axis=1) & (q[:, 0] < w) & (q[:, 1] < h)
    mapPId = np.zeros(h * w, dtype=int)

    for i in range(h * w):
        if pFlag[i]:
            mapPId[i] = np.ravel_multi_index((int(q[i, 1]), int(q[i, 0])), (h, w))

    # 变形图像
    warped_image_flat = warped_image.reshape(h * w, 3)
    im2 = np.ones((h * w, 3), dtype=np.uint8) * 255

    for i in range(h * w):
        if mapPId[i] != 0:
            im2[i, :] = warped_image_flat[mapPId[i], :]

    warped_image = im2.reshape(h, w, 3)
    return warped_image


def run_warping():
    global points_src, points_dst, image  ### fetch global variables

    warped_image = point_guided_deformation(
        image, np.array(points_src), np.array(points_dst)
    )

    return warped_image


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                source="upload",
                label="上传图片",
                interactive=True,
                width=800,
                height=200,
            )
            point_select = gr.Image(
                label="点击选择控制点和目标点", interactive=True, width=800, height=800
            )

        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)

    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

# 启动 Gradio 应用
demo.launch()
