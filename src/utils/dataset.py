import os
import cv2
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

from src.utils.ops import xyxyxyxy2xywhr

IMG_FORMATS = "bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"


def img2label_paths(img_paths):
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class OBBDataset(Dataset):
    """OBB dataset: label format = class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1, 4 corners)."""

    def __init__(self, img_path, imgsz=640, data=None, augment=True):
        self.imgsz = imgsz
        self.augment = augment
        self.data = data or {}
        self.nc = int(self.data.get("nc", 80))
        self.names = self.data.get("names", {i: str(i) for i in range(self.nc)})
        if isinstance(img_path, str):
            img_path = [img_path]
        self.im_files = []
        for p in img_path:
            p = Path(p)
            if p.is_dir():
                for ext in IMG_FORMATS:
                    self.im_files.extend(p.rglob(f"*.{ext}"))
            else:
                self.im_files.append(p)
        self.im_files = sorted([str(x) for x in self.im_files if Path(x).suffix[1:].lower() in IMG_FORMATS])
        assert self.im_files, f"No images found in {img_path}"
        self.label_files = img2label_paths(self.im_files)
        self.labels = [self._load_label(lb) for lb in self.label_files]

    def _load_label(self, path):
        path = Path(path)
        if not path.exists():
            return {"cls": np.zeros((0, 1), dtype=np.float32), "segments": np.zeros((0, 4, 2), dtype=np.float32)}
        with open(path) as f:
            lines = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if not lines:
            return {"cls": np.zeros((0, 1), dtype=np.float32), "segments": np.zeros((0, 4, 2), dtype=np.float32)}
        classes = []
        segments = []
        for x in lines:
            classes.append(float(x[0]))
            pts = np.array(x[1:], dtype=np.float32)
            if len(pts) >= 8:
                segments.append(pts.reshape(-1, 2))
            else:
                segments.append(np.zeros((4, 2), dtype=np.float32))
        return {
            "cls": np.array(classes, dtype=np.float32).reshape(-1, 1),
            "segments": np.stack(segments) if segments else np.zeros((0, 4, 2), dtype=np.float32),
        }

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, i):
        img = cv2.imread(self.im_files[i])
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        label = self.labels[i]
        cls = label["cls"]
        segments = label["segments"]
        n = len(cls)
        if n == 0:
            bboxes = np.zeros((0, 5), dtype=np.float32)
        else:
            seg = segments if segments.ndim == 3 else segments.reshape(-1, 4, 2)[:n]
            if seg.shape[1] >= 4:
                corners = seg[:, :4, :].reshape(n, 8)
            else:
                corners = np.zeros((n, 8), dtype=np.float32)
            bboxes = xyxyxyxy2xywhr(torch.from_numpy(corners)).numpy()
            if bboxes[:, 4].max() > 2:
                bboxes[:, 4] *= np.pi / 180
        img, ratio, (dw, dh) = self._letterbox(img)
        if n and bboxes.size:
            bboxes[:, 0] = (bboxes[:, 0] * w0 * ratio + dw) / self.imgsz
            bboxes[:, 1] = (bboxes[:, 1] * h0 * ratio + dh) / self.imgsz
            bboxes[:, 2] = bboxes[:, 2] * w0 * ratio / self.imgsz
            bboxes[:, 3] = bboxes[:, 3] * h0 * ratio / self.imgsz
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        return {
            "img": img,
            "cls": torch.from_numpy(cls).float() if n else torch.zeros(0, 1),
            "bboxes": torch.from_numpy(bboxes).float() if n else torch.zeros(0, 5),
            "batch_idx": torch.zeros(n, 1) if n else torch.zeros(0, 1),
            "im_file": self.im_files[i],
            "ori_shape": (h0, w0),
            "ratio_pad": (ratio, (dw, dh)),
        }

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(round(h * r)), int(round(w * r))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dh = (self.imgsz - new_h) / 2
        dw = (self.imgsz - new_w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            v = [b[k] for b in batch]
            if k == "img":
                out[k] = torch.stack(v, 0)
            elif k in ("cls", "bboxes"):
                out[k] = torch.cat(v, 0)
            elif k == "batch_idx":
                start = 0
                batch_idx = []
                for b in batch:
                    n = len(b["cls"])
                    batch_idx.append(torch.full((n, 1), start, dtype=torch.float32))
                    start += 1
                out[k] = torch.cat(batch_idx, 0)
            else:
                out[k] = v
        return out
