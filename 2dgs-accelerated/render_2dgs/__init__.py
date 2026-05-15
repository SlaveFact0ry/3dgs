
import numpy as np
import torch

BACKEND = "python"
_render_fn = None

try:
    from render_2dgs.cuda.render_2dgs_cuda import render as _render_cuda
    BACKEND = "cuda"
    _render_fn = _render_cuda
except ImportError:
    try:
        from render_2dgs.metal.render_2dgs_metal import render as _render_metal
        BACKEND = "metal"
        _render_fn = _render_metal
    except ImportError:
        try:
            from render_2dgs.cpu_omp.render_2dgs_cpu_omp import render as _render_cpu_omp
            BACKEND = "cpu_omp"
            _render_fn = _render_cpu_omp
        except ImportError:
            try:
                from render_2dgs.cpu_stdthread.render_2dgs_cpu_stdthread import render as _render_cpu_stdthread
                BACKEND = "cpu_stdthread"
                _render_fn = _render_cpu_stdthread
            except ImportError:
                pass


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def render(mus, sigmas, thetas, opacities, rgbs, H: int, W: int) -> np.ndarray:
    """
    시각화 전용 렌더러 (autograd 없음).
    학습 루프의 loss 계산은 기존 render_gaussians_2d() 사용.

    Returns
    -------
    img : np.ndarray, shape (H, W, 3), float32, range [0, 1]
    """
    if BACKEND == "python":
        return _render_python(mus, sigmas, thetas, opacities, rgbs, H, W)

    return _render_fn(
        _to_np(mus), _to_np(sigmas), _to_np(thetas),
        _to_np(opacities), _to_np(rgbs),
        H, W
    )


def _render_python(mus, sigmas, thetas, opacities, rgbs, H, W):
    """Python fallback — 기존 노트북 코드와 동일 로직"""
    from render_2dgs.python_fallback import render_gaussians_2d
    img = render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs)
    return img.detach().cpu().numpy()