import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from dataset import OilSpillDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load("best_unet.pth", map_location=device))
model.eval()


def characterize_spill(mask_binary, pixel_size_m=None):
    """
    mask_binary: 2D numpy array, 0/1 values
    pixel_size_m: real-world meters per pixel, if known (from actual Sentinel-1
        metadata - typically ~10m for Sentinel-1 GRD). None if unknown (as with
        this Kaggle training patches, which have no geo metadata attached).
    Returns a dict of geometric properties.
    """
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"found": False}

    # Take the largest contour - the main detected slick
    largest = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(largest)

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return {"found": False}
    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    # Orientation via minimum-area rotated rectangle
    rect = cv2.minAreaRect(largest)
    (rect_cx, rect_cy), (width_px, height_px), angle_deg = rect

    # Elongation ratio - how "streak-like" vs "blob-like" the shape is
    elongation = max(width_px, height_px) / (min(width_px, height_px) + 1e-6)

    result = {
        "found": True,
        "area_px": float(area_px),
        "centroid_px": (float(centroid_x), float(centroid_y)),
        "orientation_deg": float(angle_deg),
        "elongation_ratio": float(elongation),
        "bounding_width_px": float(width_px),
        "bounding_height_px": float(height_px),
    }

    if pixel_size_m is not None:
        result["area_m2"] = area_px * (pixel_size_m ** 2)
        result["area_km2"] = result["area_m2"] / 1_000_000

    return result


if __name__ == "__main__":
    test_ds = OilSpillDataset(
        r"C:\Users\ezabe\Downloads\dataset\test\sentinel\image",
        r"C:\Users\ezabe\Downloads\dataset\test\sentinel\label",
        patch_size=256,
        augment=False,
    )

    img, mask = test_ds[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    pred_binary = (pred > 0.5).astype(np.uint8)

    # Sentinel-1 GRD real pixel size is typically ~10m - using it here as an
    # EXAMPLE only, since these specific patches aren't real georeferenced
    # Sentinel-1 scenes. Swap in the real value once using actual downloaded data.
    stats = characterize_spill(pred_binary, pixel_size_m=10.0)
    print("Predicted spill characterization:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nSame stats on GROUND TRUTH mask (for comparison):")
    gt_stats = characterize_spill(mask[0].numpy().astype(np.uint8), pixel_size_m=10.0)
    for k, v in gt_stats.items():
        print(f"  {k}: {v}")