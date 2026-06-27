# TransRetarget
Code of IROS 2026"TransRetarget: A Human-Robot Hand Motion Retargeting Framework Based on Temporal-Spatial Transformer"

# TransHandR 安装与环境说明
## 环境要求

- Python 3.10
- 操作系统：Windows 11

## 安装步骤

1. 安装 Python 3.10（建议使用 Anaconda 或 Miniconda 管理环境）
2. 创建新环境并激活：
	```bash
	conda create -n handsformer python=3.10
	conda activate handsformer
	```
3. 安装 requirements.txt 中的依赖：
	```bash
	pip install -r requirements.txt
	```

> 注意：如遇到 torch/torchvision 安装问题，请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的 wheel 包进行安装。

# 使用说明
代码结构：参考代码结构.md