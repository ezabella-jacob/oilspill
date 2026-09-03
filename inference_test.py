import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from dataset import OilSpillDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load("best_unet.pth", map_location=device))
model.eval()

test_ds = OilSpillDataset(
    r"C:\Users\ezabe\Downloads\dataset\test\sentinel\image",
    r"C:\Users\ezabe\Downloads\dataset\test\sentinel\label",
    patch_size=256,
    augment=False,
)

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i in range(4):
    img, mask = test_ds[i]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()

    axes[i, 0].imshow(img[0], cmap='gray')
    axes[i, 0].set_title('Input SAR patch')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(mask[0], cmap='gray')
    axes[i, 1].set_title('Ground truth')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_title('Model prediction')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png', dpi=120)
plt.show()
print("Saved to test_predictions.png")