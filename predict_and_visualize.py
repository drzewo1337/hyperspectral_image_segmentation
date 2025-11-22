import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from load_data import load_data, normalize, pad_with_zeros


def predict_whole_scene(model, dataset_name, patch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)

    # Załaduj dane
    data, labels = load_data(dataset_name)
    data = normalize(data)
    
    h, w, b = data.shape
    margin = patch_size // 2
    padded_data = pad_with_zeros(data, margin)
    output = np.zeros((h, w), dtype=np.uint8)

    is_3d = False
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            is_3d = True
            break

    for i in range(h):
        for j in range(w):
            if labels[i, j] == 0:
                continue
            
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            patch = np.expand_dims(patch, axis=0)

            if is_3d:
                patch = np.transpose(patch, (0, 3, 1, 2))  # (1, B, H, W)
                patch = np.expand_dims(patch, axis=1)  # (1, 1, B, H, W)
            else:
                patch = np.transpose(patch, (0, 3, 1, 2))  # (1, B, H, W)
            
            patch = torch.tensor(patch, dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(patch)
                output[i, j] = pred.argmax(1).item() + 1

    return output, labels


def visualize(pred_map, true_map, dataset_name):
    num_classes_map = {
        'Indian': 16,
        'PaviaU': 9,
        'PaviaC': 9,
        'KSC': 13,
        'Salinas': 16
    }
    
    num_classes = num_classes_map.get(dataset_name, 16)

    colors = plt.cm.tab20(np.linspace(0, 1, num_classes + 1))
    colors[0] = [0, 0, 0, 1]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=np.arange(num_classes + 2) - 0.5, ncolors=num_classes + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(true_map, cmap=cmap, norm=norm)
    axs[0].set_title(f"Ground Truth - {dataset_name}")
    axs[0].axis("off")

    axs[1].imshow(pred_map, cmap=cmap, norm=norm)
    axs[1].set_title(f"Predicted Map - {dataset_name}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

