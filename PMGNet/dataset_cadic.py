# dataset.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset

DEFAULT_TARGET_SHAPE = (1, 128, 128, 128)

class CTCACSimpleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        phase: str = "train",
        target_shape: tuple = DEFAULT_TARGET_SHAPE,
        label_suffix: str = "label"   # 如果标签前缀是 labelnew_ ，改成 "labelnew"
    ):
        """
        期望目录结构:
        data_root/
          train/
            imagesTr/image_XXX.nii.gz
            labelsTr/label_XXX.nii.gz
          val/
            imagesTr/...
            labelsTr/...
        """
        self.img_dir = os.path.join(data_root, phase, "imagesTr")
        self.seg_dir = os.path.join(data_root, phase, "labelsTr")
        self.target_shape = target_shape
        self.label_suffix = label_suffix

        # 扫描所有 image 文件，提取 case_id (XXX)
        self.case_ids = [
            fname.replace("image_", "").split(".nii")[0]
            for fname in os.listdir(self.img_dir)
            if fname.startswith("image_") and fname.endswith((".nii", ".nii.gz"))
        ]
        self.case_ids.sort()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        case = self.case_ids[idx]
        img_path = os.path.join(self.img_dir, f"image_{case}.nii.gz")
        seg_path = os.path.join(self.seg_dir, f"{self.label_suffix}_{case}.nii.gz")

        img = self._load_nifti(img_path)   # (D,H,W), float32
        seg = self._load_nifti(seg_path)   # (D,H,W), int

        # 增加 channel 维度 -> (1, D, H, W)
        img_t = torch.from_numpy(img[None]).float()
        seg_t = torch.from_numpy(seg[None].astype(np.int64))

        # 中心裁剪或对称 pad 到 target_shape
        img_t = self._pad_or_crop(img_t, self.target_shape)
        seg_t = self._pad_or_crop(seg_t, self.target_shape)

        # 返回字典
        return {"image": img_t, "label": seg_t, "case_id": case}

    def _load_nifti(self, path: str) -> np.ndarray:
        try:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))  # (D,H,W)
        except Exception:
            nb = nib.load(path)
            data = nb.get_fdata(dtype=np.float32)
            arr = data.transpose(2, 1, 0)  # (X,Y,Z) -> (Z,Y,X) = (D,H,W)
        return arr

    def _pad_or_crop(self, x: torch.Tensor, target_shape: tuple):
        """
        x: Tensor (C,D,H,W)
        target_shape: (C,D,H,W)
        """
        _, d0, h0, w0 = x.shape
        _, dt, ht, wt = target_shape

        # 中心裁剪
        if d0 > dt:
            sd = (d0 - dt) // 2
            x = x[:, sd:sd+dt, :, :]
        if h0 > ht:
            sh = (h0 - ht) // 2
            x = x[:, :, sh:sh+ht, :]
        if w0 > wt:
            sw = (w0 - wt) // 2
            x = x[:, :, :, sw:sw+wt]

        # 对称 pad
        pad_d = max(dt - x.shape[1], 0)
        pad_h = max(ht - x.shape[2], 0)
        pad_w = max(wt - x.shape[3], 0)
        pad_cfg = (
            pad_w//2, pad_w - pad_w//2,
            pad_h//2, pad_h - pad_h//2,
            pad_d//2, pad_d - pad_d//2,
            0, 0
        )
        if any((pad_d, pad_h, pad_w)):
            x = F.pad(x, pad_cfg, mode='constant', value=0)
        return x
if __name__ == "__main__":
    root = "/home/ubuntu/ZZM/PGMNet/train_cadic/data"
    print("\n=== TEST TRAIN MODE ===")
    ds = CTCACSimpleDataset(root)
    print(f"patch count: {len(ds)}")
    sample = ds[0]

    img = sample['image']

    seg = sample['label']

    case_id = sample['case_id']
    print("img.shape:", img.shape)
    print("seg.shape:", seg.shape)
    print("\n=== TEST VAL MODE ===")
    ds2 = CTCACSimpleDataset(root)
    print(f"patch count: {len(ds2)}")