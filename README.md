# Gaussian Splatting Playground

<p align="center">
  <img src="2dgs/outputs/output.gif" width="480" alt="2D Gaussian Splatting fitting">
</p>

회전·anisotropic Gaussian Splatting을 직접 구현하고, 백엔드·응용까지 확장하는 작업 모음.

## Projects

### [2dgs/](./2dgs) — 2D Gaussian Splatting

이미지 한 장을 회전·anisotropic 2D 가우시안들의 합으로 근사.
Adam + L1 + D-SSIM 손실, density control (clone / split / prune) 포함.
[상세 README →](./2dgs/README.md)

### 3D Gaussian Splatting Pipeline

COLMAP 기반 카메라 포즈 추정 → 3DGS 학습 (7000 iterations) → novel view synthesis.
구현: [`3D Gaussian Splatting Colab.ipynb`](./3D%20Gaussian%20Splatting%20Colab.ipynb)
*(결과 영상 추가 예정)*

### [2dgs-accelerated/](./2dgs-accelerated) — Multi-backend Renderer *(진행중)*

2dgs 렌더러를 OpenMP / std::thread / CUDA / Metal로 가속.

### [stop_and_shoot/](./stop_and_shoot) — ROS2 / C++ 응용 *(진행중)*
