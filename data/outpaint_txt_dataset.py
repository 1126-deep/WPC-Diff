# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import glob
import math
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

try:
    import cv2
except Exception:
    cv2 = None


def _pil_gray_u8(path: str, *, size: int) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    img = img.resize((int(size), int(size)), resample=Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _u8_to_m11_tensor(u8: np.ndarray) -> torch.Tensor:
    x = u8.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x)[None]


def _u8_to_01(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def _center_radius_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) < 10:
        raise RuntimeError("mask too small / empty")
    cy = float(ys.mean())
    cx = float(xs.mean())
    area = float(len(xs))
    r = math.sqrt(area / math.pi)
    return (cy, cx), r


def align_anchor_to_heat(
    anchor_u8: np.ndarray,
    heat_u8: np.ndarray,
    *,
    thr_anchor: float = 0.08,
    thr_heat: float = 0.35,
) -> np.ndarray:
    if cv2 is None:
        return anchor_u8
    h, w = anchor_u8.shape
    if heat_u8.shape != (h, w):
        return anchor_u8

    a01 = _u8_to_01(anchor_u8)
    h01 = _u8_to_01(heat_u8)

    src_mask = (a01 > float(thr_anchor)).astype(np.float32)
    tgt_mask = (h01 > float(thr_heat)).astype(np.float32)

    try:
        (cy_s, cx_s), r_s = _center_radius_from_mask(src_mask)
        (cy_t, cx_t), r_t = _center_radius_from_mask(tgt_mask)
    except Exception:
        return anchor_u8

    scale = float(r_t / max(r_s, 1e-6))
    tx = float(cx_t - scale * cx_s)
    ty = float(cy_t - scale * cy_s)
    M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)

    return cv2.warpAffine(
        anchor_u8,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def geom_maps_from_heat_m11(heat_m11: torch.Tensor) -> torch.Tensor:
    assert heat_m11.ndim == 3 and int(heat_m11.shape[0]) == 1
    device = heat_m11.device
    h = int(heat_m11.shape[1])
    w = int(heat_m11.shape[2])

    heat01 = ((heat_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    m = heat01
    m_sum = m.sum().clamp(min=1e-6)

    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=heat_m11.dtype).view(h, 1).expand(h, w)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=heat_m11.dtype).view(1, w).expand(h, w)

    cy = (yy * m[0]).sum() / m_sum
    cx = (xx * m[0]).sum() / m_sum

    area = m_sum
    r_px = torch.sqrt(area / float(math.pi))
    r_norm = (r_px / (0.5 * float(h))).clamp(min=1e-3)

    dx_raw = (xx - cx)
    dy_raw = (yy - cy)
    dist_raw = torch.sqrt(dx_raw * dx_raw + dy_raw * dy_raw) / r_norm
    dist_raw = dist_raw.clamp(0.0, 2.0)

    dx = (dx_raw / 2.0).clamp(-1.0, 1.0)
    dy = (dy_raw / 2.0).clamp(-1.0, 1.0)
    dist = (dist_raw - 1.0).clamp(-1.0, 1.0)

    geom = torch.stack([dx, dy, dist], dim=0)
    return geom


def _mask_nonblack_u8(heat_u8: np.ndarray) -> np.ndarray:
    return heat_u8.astype(np.uint8) > 0


def _anchor_canvas_u8(anchor_patch_u8: np.ndarray, heat_u8: np.ndarray) -> np.ndarray:
    h, w = anchor_patch_u8.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    m = _mask_nonblack_u8(heat_u8)
    if not bool(np.any(m)):
        return anchor_patch_u8
    ys, xs = np.where(m)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = max(0, min(h - 1, y0))
    x0 = max(0, min(w - 1, x0))
    y1 = max(y0 + 1, min(h, y1))
    x1 = max(x0 + 1, min(w, x1))

    patch_rs = Image.fromarray(anchor_patch_u8).resize((x1 - x0, y1 - y0), resample=Image.Resampling.BILINEAR)
    patch_rs = np.array(patch_rs, dtype=np.uint8)
    roi_m = m[y0:y1, x0:x1]
    region = canvas[y0:y1, x0:x1]
    region[roi_m] = patch_rs[roi_m]
    canvas[y0:y1, x0:x1] = region
    return canvas


class OutpaintTxtDataset(Dataset):
    def __init__(self, root_dir: str, *, split: str = "train", image_size: int = 192):
        super().__init__()
        self.root_dir = str(root_dir)
        self.split = str(split)
        self.image_size = int(image_size)

        self.align_thr_anchor = 0.08
        self.align_thr_heat = 0.35

        split_path = os.path.join(self.root_dir, f"{self.split}.txt")
        with open(split_path, "r") as f:
            self.names = [line.strip() for line in f.readlines() if line.strip()]

        self.anchor_dir = os.path.join(self.root_dir, "anchor_img")
        self.heat_dir = os.path.join(self.root_dir, "heat")
        self.target_dir = os.path.join(self.root_dir, "target_img")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[int(index)]
        anchor_path = os.path.join(self.anchor_dir, f"{name}.png")
        heat_path = os.path.join(self.heat_dir, f"{name}.png")
        target_path = os.path.join(self.target_dir, f"{name}.png")

        anchor_u8 = _pil_gray_u8(anchor_path, size=self.image_size)
        heat_u8 = _pil_gray_u8(heat_path, size=self.image_size)
        target_u8 = _pil_gray_u8(target_path, size=self.image_size)

        anchor_u8 = align_anchor_to_heat(
            anchor_u8,
            heat_u8,
            thr_anchor=float(self.align_thr_anchor),
            thr_heat=float(self.align_thr_heat),
        )

        target = _u8_to_m11_tensor(target_u8).float()
        cond0 = _u8_to_m11_tensor(anchor_u8).float()
        cond1 = _u8_to_m11_tensor(heat_u8).float()
        geom = geom_maps_from_heat_m11(cond1)
        cond = torch.cat([cond0, cond1, geom], dim=0)
        return {"a": target, "b": cond}


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        if transforms_ is False or transforms_ is None:
            self.transform = None
        else:
            self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/a" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/b" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = np.load(self.files_A[index % len(self.files_A)], allow_pickle=True)
        image_B = np.load(self.files_B[index % len(self.files_B)], allow_pickle=True)

        item_A = torch.from_numpy(image_A).float()
        item_B = torch.from_numpy(image_B).float()

        if item_A.ndim == 2:
            item_A = item_A.unsqueeze(0)
        if item_B.ndim == 2:
            item_B = item_B.unsqueeze(0)

        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
