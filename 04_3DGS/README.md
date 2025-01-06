# Assignment 4 - Implement Simplified 3D Gaussian Splatting

## 实验目的

本实验旨在通过多视图图像重建3D场景，并使用简化的3D高斯散点（3D Gaussian Splatting, 3DGS）技术进行渲染。实验步骤包括从多视图图像中恢复相机姿态和3D点，初始化3D高斯分布，并将其投影到2D图像空间进行体积渲染。


## Resources:
- [Paper: 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [3DGS Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [Colmap for Structure-from-Motion](https://colmap.github.io/index.html)

---

## Startup process

### Step 1. Structure-from-Motion

首先，我们使用 Colmap 从多视图图像中恢复相机姿态和一组3D点。具体步骤如下：

1. 运行以下命令，使用 Colmap 进行结构化运动恢复：
    ```sh
    python mvs_with_colmap.py --data_dir data/chair
    ```

2. 调试重建结果，运行以下命令：
    ```sh
    python debug_mvs_by_projecting_pts.py --data_dir data/chair
    ```

### Step 2. A Simplified 3D Gaussian Splatting (Your Main Part)
从第一步的调试输出中可以看到，3D点对于渲染整个图像来说是稀疏的。我们将每个点扩展为一个3D高斯分布，以覆盖更多的3D空间。

#### 2.1 3D高斯分布初始化

参考[原始论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)，将3D点转换为3D高斯分布。我们需要为每个点定义协方差矩阵，初始高斯分布的中心就是这些点。根据公式（6），定义协方差时需要定义缩放矩阵 S 和旋转矩阵 R。由于我们需要使用3D高斯分布进行体积渲染，还需要为每个高斯分布定义不透明度和颜色属性。体积渲染过程使用公式（1）、（2）、（3）进行计算。

#### 2.2 将3D高斯分布投影为2D高斯分布

根据公式（5），我们需要通过世界到相机的变换矩阵 *_W_* 和投影变换的雅可比矩阵 *_J_* 将3D高斯分布投影到图像空间。

#### 2.3 计算高斯值

我们需要计算2D高斯分布的值以进行体积渲染。2D高斯分布的公式如下：

$$
  f(\mathbf{x}; \boldsymbol{\mu}\_{i}, \boldsymbol{\Sigma}\_{i}) = \frac{1}{2 \pi \sqrt{ | \boldsymbol{\Sigma}\_{i} |}} \exp \left ( {-\frac{1}{2}} (\mathbf{x} - \boldsymbol{\mu}\_{i})^T \boldsymbol{\Sigma}\_{i}^{-1} (\mathbf{x} - \boldsymbol{\mu}\_{i}) \right )
$$

其中，$\mathbf{x}$ 是表示像素位置的2D向量，$\boldsymbol{\mu}$ 是表示第 $i$ 个2D高斯分布均值的2D向量，$\boldsymbol{\Sigma}$ 是2D高斯分布的协方差矩阵。指数部分 $P_{(\mathbf{x}, i)}$ 为：

$$
  P_{(\mathbf{x}, i)} = {-\frac{1}{2}} (\mathbf{x} - \boldsymbol{\mu}\_{i})^T \mathbf{\Sigma}\_{i}^{-1} (\mathbf{x} - \boldsymbol{\mu}\_{i})
$$

#### 2.4 体积渲染（α-混合）

根据公式（1-3），使用这些有序的 `N` 个2D高斯分布，我们可以计算每个像素位置的alpha值和透射率值。

2D高斯分布 $i$ 在单个像素位置 $\mathbf{x}$ 的alpha值可以通过以下公式计算：

$$
  \alpha_{(\mathbf{x}, i)} = o_i*f(\mathbf{x}; \boldsymbol{\mu}\_{i}, \boldsymbol{\Sigma}\_{i})
$$

其中，$o_i$ 是每个高斯分布的不透明度，是一个可学习的参数。

给定 `N` 个有序的2D高斯分布，2D高斯分布 $i$ 在单个像素位置 $\mathbf{x}$ 的透射率值可以通过以下公式计算：

$$
  T_{(\mathbf{x}, i)} = \prod_{j \lt i} (1 - \alpha_{(\mathbf{x}, j)})
$$

### 实现3DGS模型

完成上述步骤后，运行以下命令构建3DGS模型：

```sh
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

## Results
1. chair
![pic1](/04_3DGS/data/chair/r_59.png)
2. lego
![pic2](/04_3DGS/data/lego/r_38.png)
<video width="320" height="240" controls>
    <source src="/04_3DGS/data/lego/debug_rendering.mp4" type="video/mp4">
</video>

