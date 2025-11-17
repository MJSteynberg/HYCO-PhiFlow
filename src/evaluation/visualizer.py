"""Small animation utilities used by the evaluation CLI.

This module intentionally implements a minimal subset of the
deprecated visualization utilities so the new `src/evaluation`
package can be used to create quick animations from Scene / DataManager
data without touching the rest of the repository.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _ensure_time_first(tensor: torch.Tensor) -> torch.Tensor:
    """Rearrange tensor to [T, C, H, W] or [T, H, W].

    Accepts tensors with common shapes returned by DataManager:
       - [C, T, H, W] -> permute(1,0,2,3)
       - [T, C, H, W] -> unchanged
       - [T, H, W] -> unchanged
       - [C, H, W] -> add time dim
    """
    if tensor.ndim == 4:
        # [C, T, H, W] ? Check an heuristic: if tensor.shape[0] in {1,2}
        # and tensor.shape[1] > 1 then assume [C, T, H, W]
        if tensor.shape[0] <= 3 and tensor.shape[1] > 1 and tensor.shape[1] != tensor.shape[0]:
            tensor = tensor.permute(1, 0, 2, 3)
        else:
            # assume [T, C, H, W]
            pass
    elif tensor.ndim == 3:
        # [T, H, W] - add channel dim
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 2:
        # [H, W] - add time and channel dims
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def _compute_magnitude_if_vector(t: np.ndarray) -> np.ndarray:
    """If the channel dimension is 2, compute the magnitude; otherwise return first channel."""
    if t.ndim == 4 and t.shape[1] == 2:
        # t [T, C=2, H, W]
        return np.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2)
    if t.ndim == 4 and t.shape[1] == 1:
        return t[:, 0]
    if t.ndim == 3:
        # [T, H, W]
        return t
    raise ValueError(f"Unsupported tensor shape {t.shape}")


def animate_field(
    tensor: torch.Tensor,
    save_path: Optional[str] = None,
    fps: int = 10,
    dpi: int = 100,
):
    """Create and save a GIF animation for a single field (ground truth or predictions).

    The tensor is expected in one of the shapes handled by `_ensure_time_first`.
    The function computes the magnitude for 2-channel fields and plots 2D frames
    with matplotlib `imshow`.
    """
    t = _ensure_time_first(tensor).detach().cpu().numpy()

    # Ensure shape is [T, C, H, W] or [T, H, W]
    if t.ndim == 4 and t.shape[1] in (1, 2):
        frames = _compute_magnitude_if_vector(t)
    elif t.ndim == 3:
        frames = t
    else:
        raise ValueError(f"Unsupported tensor shape after normalization: {t.shape}")

    # Transpose spatial dims if needed: phi uses [x,y], matplotlib expects [row,col]
    frames = np.transpose(frames, (0, 2, 1))

    num_frames = frames.shape[0]
    vmin = np.nanmin(frames)
    vmax = np.nanmax(frames)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    im = ax.imshow(frames[0], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")
    title = ax.set_title("Frame 0")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx: int):
        im.set_array(frames[frame_idx])
        title.set_text(f"Frame {frame_idx}/{num_frames-1}")
        return (im, title)

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path), writer="pillow", fps=fps)

    plt.close(fig)

    return save_path
