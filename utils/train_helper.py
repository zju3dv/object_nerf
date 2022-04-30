import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def visualize_depth(depth, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x) if vmin == None else vmin  # get minimum depth
    ma = np.max(x) if vmax == None else vmax
    x = np.clip(x, mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_val_image(img_wh, batch, results, typ="fine"):
    W, H = img_wh
    rgbs = batch["rgbs"]
    img_inst = (
        results[f"rgb_instance_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
    )  # (3, H, W)
    img_full = results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # img_bg = results[f'rgb_bg_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    if batch["depths"].sum() == 0:
        vis_min, vis_max = None, None
    else:
        vis_min, vis_max = (
            batch["depths"].min().item() + 0.3,
            batch["depths"].max().item(),
        )
    depth_inst = visualize_depth(
        results[f"depth_instance_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)
    depth = visualize_depth(
        results[f"depth_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)
    gt_depth = visualize_depth(batch["depths"].view(H, W), vmin=vis_min, vmax=vis_max)
    opacity = visualize_depth(
        results[f"opacity_instance_{typ}"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    stack = torch.stack(
        [img_gt, img_inst, img_full, depth_inst, depth, gt_depth, opacity]
    )  # (4, 3, H, W)
    return stack
