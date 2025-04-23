# seg-feat-toolkit
Semantic segmentation &amp; classical feature extraction toolkit (MMSeg + OpenCV)
项目简介

本仓库整合了两大功能：

基于 MMSegmentation 的语义分割批处理、可视化与统计分析

经典图像内容特征（边缘密度、颜色熵、颜色度、颜色矩、GLCM 纹理）批量提取

所有逻辑均封装在同一个脚本 semantic_segmentation_and_features.py 中，命令行友好，可直接在单机 GPU/CPU 上运行。

目录结构

repo_root/
├── semantic_segmentation_and_features.py
├── requirements.txt          # ✅ 建议提供（下方依赖列表可复制）
├── checkpoints/              # ⬇️ 放置已下载/训练的 *.pth
├── configs/                  # ⬇️ 放置 *.py 模型配置（MMSeg 官方或自定义）
├── images/                   # ⬇️ 输入图像目录（任意子目录皆可）
├── docs/                     # 📖 如果需要额外技术文档，可在此扩充
└── LICENSE                   # 📝 选择开源协议（MIT/Apache‑2.0...）

依赖环境

pip install -U torch torchvision torchaudio  # 先装 PyTorch（根据显卡/CPU 选择对应版本）

# 其余依赖
pip install mmcv-full==2.* mmsegmentation==1.* opencv-python-headless
pip install scikit-image matplotlib pandas pywavelets

🍃 建议将以上内容写入 requirements.txt 方便一键安装。

快速开始

1️⃣ 语义分割批处理

python semantic_segmentation_and_features.py seg \
       --config     configs/deeplabv3plus_r101-d8_512x1024.py \
       --checkpoint checkpoints/deeplabv3plus_r101-d8.pth \
       --images     images/ \
       --output     outputs/seg_results \
       --opacity    0.8     # 可选，默认 0.8

输出内容

seg_*.png ：原图与分割掩膜混合可视化

stats/overall_proportions.csv ：全数据集各类像素占比

stats/per_image_pixel_proportions.csv ：逐图像占比明细

2️⃣ 图像特征批量提取

python semantic_segmentation_and_features.py feat \
       --images images/ \
       --output outputs/features.xlsx

输出 features.xlsx（每行一张图，列为各特征）。

常见问题 FAQ

问题

解决方案

找不到 mmcv._ext

安装 mmcv-full 时需与 CUDA 版本匹配；或使用 CPU 版：pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VERSION}/index.html

CUDA out of memory

① 减小 --images 中图片尺寸；② 使用 CUDA_VISIBLE_DEVICES= 令脚本在 CPU 上运行

类占比统计不准确

确认 interested_classes 中的类别与所用数据集标签一致

贡献指南 (CONTRIBUTING)

Fork & Clone 本仓库

新建分支 git checkout -b feat/<feature-name>

提交代码前运行 pre-commit run --all-files（建议配置 pre‑commit）

提交 PR 时请附测试用例 & 更新文档

许可证 (License)

建议选择 MIT 或 Apache‑2.0，放入 LICENSE 文件并在此处声明。

ENGLISH QUICK‑START

For English users: a concise version is provided. For details please read the Chinese part above.

# Semantic Segmentation
python semantic_segmentation_and_features.py seg \
    --config configs/deeplabv3plus_r101-d8_512x1024.py \
    --checkpoint checkpoints/deeplabv3plus_r101-d8.pth \
    --images images/ \
    --output outputs/seg_results

# Feature Extraction
python semantic_segmentation_and_features.py feat \
    --images images/ \
    --output outputs/features.xlsx

Dependencies

Python ≥ 3.8

PyTorch + torchvision

mmcv‑full 2.x, mmsegmentation 1.x

OpenCV, scikit‑image, matplotlib, pandas, pywavelets

pip install -r requirements.txt

License

MIT (suggested) – feel free to open issues/PRs.
