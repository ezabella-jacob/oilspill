import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.augment = augment
        self.filenames = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)

        # Mask filename might have a different extension than the image -
        # try the same name first, fall back to matching by stem if needed.
        mask_path = os.path.join(self.mask_dir, fname)
        if not os.path.exists(mask_path):
            stem = os.path.splitext(fname)[0]
            candidates = [f for f in os.listdir(self.mask_dir) if f.startswith(stem)]
            if not candidates:
                raise FileNotFoundError(f"No matching mask found for {fname}")
            mask_path = os.path.join(self.mask_dir, candidates[0])

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)  # grayscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)  # binarize: assume >127 = oil (255), else 0

        H, W = image.shape
        ph = pw = min(self.patch_size, H, W)
        top = np.random.randint(0, H - ph + 1) if H > ph else 0
        left = np.random.randint(0, W - pw + 1) if W > pw else 0
        image = image[top:top+ph, left:left+pw]
        mask = mask[top:top+ph, left:left+pw]

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        image = (image - image.mean()) / (image.std() + 1e-6)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W) - single channel
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


if __name__ == "__main__":
    import sys
    image_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    ds = OilSpillDataset(image_dir, mask_dir, patch_size=256, augment=True)
    print("Samples:", len(ds))
    img, msk = ds[0]
    print("Image shape:", img.shape, "Mask shape:", msk.shape)
    print("Mask unique values:", msk.unique())