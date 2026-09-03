import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset import OilSpillDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMAGE_DIR = r"C:\Users\ezabe\Downloads\dataset\train\sentinel\image"
MASK_DIR = r"C:\Users\ezabe\Downloads\dataset\train\sentinel\label"

full_dataset = OilSpillDataset(IMAGE_DIR, MASK_DIR, patch_size=256, augment=True)
print("Total samples:", len(full_dataset))

val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
).to(device)

dice_fn = smp.losses.DiceLoss(mode="binary")
bce_fn = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return dice_fn(pred, target) + bce_fn(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 3  # small run to prove it works; increase later
CKPT_PATH = "best_unet.pth"
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for i, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        if i % 50 == 0:
            print(f"  epoch {epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.4f}")
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_ds)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  Saved new best model -> {CKPT_PATH}")

print("Done.")