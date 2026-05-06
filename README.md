# 3D Gaussian Splatting Pipeline

## Overview
COLMAP 기반 카메라 포즈 추정부터 3DGS 렌더링까지 구현

## Pipeline
1. Video capture (30fps, 1min)
2. Frame extraction (ffmpeg, 2fps)
3. Camera pose estimation (COLMAP)
4. 3DGS training (7000 iterations)
5. Novel view synthesis
