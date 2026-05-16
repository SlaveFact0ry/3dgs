from .render import make_pixel_grid, render_gaussians_2d
from .density import (
    prune_gaussians,
    remove_from_optimizer,
    clone_gaussians_if_high_grad,
    split_gaussians_if_large,
    extend_optimizer_with_new_points,
)
from .losses import ssim, diff_image
from .viz import (
    show_images,
    show_image_grid,
    save_training_video,
    plot_gaussian_positions,
)

__all__ = [
    "make_pixel_grid",
    "render_gaussians_2d",
    "prune_gaussians",
    "remove_from_optimizer",
    "clone_gaussians_if_high_grad",
    "split_gaussians_if_large",
    "extend_optimizer_with_new_points",
    "ssim",
    "diff_image",
    "show_images",
    "show_image_grid",
    "save_training_video",
    "plot_gaussian_positions",
]
