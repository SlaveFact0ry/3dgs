import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse


def show_images(images, titles, figsize=None, vmin=0.0, vmax=1.0):
    n = len(images)
    figsize = figsize or (3.6 * n, 3.6)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes = [axes] if n == 1 else axes
    for ax, im, title in zip(axes, images, titles):
        if torch.is_tensor(im):
            im = im.detach().cpu().numpy()
        ax.imshow(im, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_image_grid(images, titles, cols=4, cell_size=3.2, vmin=0.0, vmax=1.0):
    n = len(images)
    cols = int(max(1, cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            im = images[i]
            if torch.is_tensor(im):
                im = im.detach().cpu().numpy()
            ax.imshow(im, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_training_video(frames, filename, fps=30, title="Training Progress"):
    if len(frames) == 0:
        print(f"No frames to save for {filename}")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.axis("off")

    im_display = ax.imshow(frames[0], vmin=0.0, vmax=1.0, interpolation="nearest")
    text = ax.text(0.5, -0.05, "Iteration: 0", transform=ax.transAxes, ha="center", fontsize=12)

    def update(frame_idx):
        im_display.set_array(frames[frame_idx])
        text.set_text(f"Iteration: {frame_idx}")
        return [im_display, text]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Video saved to {filename}")


def plot_gaussian_positions(ax, mus, sigmas, thetas, image, color="red", title=""):
    mus_np = mus.detach().cpu().numpy() if torch.is_tensor(mus) else mus
    img_norm = image / image.max()
    img_np = img_norm.detach().cpu().numpy() if torch.is_tensor(img_norm) else img_norm

    ax.imshow(img_np, extent=[-1, 1, -1, 1], origin="upper", alpha=0.6)
    ax.scatter(mus_np[:, 0], mus_np[:, 1], c=color, s=100, marker="x", linewidths=3, label="Gaussian centers")

    n = len(mus_np)
    for i in range(n):
        ellipse = Ellipse(
            (mus_np[i, 0], mus_np[i, 1]),
            width=sigmas[i, 0].item() * 3,
            height=sigmas[i, 1].item() * 3,
            angle=np.degrees(thetas[i].item()),
            edgecolor=color,
            facecolor="none",
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(ellipse)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
