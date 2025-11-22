# Hyperspectral Image Segmentation

The project investigates hyperspectral imaging for segmentation driven by material-specific spectral signatures. Hyperspectral data comprise hundreds of narrow wavelength bands (e.g., 400–1000 nm). Distinct reflectance profiles enable robust discrimination of materials within a scene.

## Models

- **InceptionHSINet** (model1.py): 3D CNN with Inception-like architecture
- **SimpleHSINet** (model2.py): Simple 2D CNN
- **CNNFromDiagram** (model3.py): 2D CNN based on diagram architecture

## Datasets

- Indian Pines: 145×145, 200 bands, 16 classes
- PaviaU: 610×340, 103 bands, 9 classes
- PaviaC: 1096×715, 102 bands, 9 classes
- KSC: 512×614, 176 bands, 13 classes
- Salinas: 512×217, 204 bands, 16 classes

## Usage

### Training a single model

```bash
# Train InceptionHSINet on Indian dataset
python run_training.py --model InceptionHSINet --dataset Indian --epochs 50

# Train SimpleHSINet on PaviaU dataset
python run_training.py --model SimpleHSINet --dataset PaviaU --epochs 50 --batch_size 32

# Train CNNFromDiagram with custom parameters
python run_training.py --model CNNFromDiagram --dataset Indian --epochs 100 --lr 0.0001
```

### Available arguments

- `--model`: Model name (InceptionHSINet, SimpleHSINet, CNNFromDiagram)
- `--dataset`: Dataset name (Indian, PaviaU, PaviaC, KSC, Salinas)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--patch_size`: Patch size (default: 16)
- `--val_split`: Validation split ratio (default: 0.2)
- `--device`: Device to use (cuda/cpu). If None, auto-detects GPU

### Output files

- `best_model_{model_name}_{dataset_name}.pth`: Best model checkpoint
- `training_log_{model_name}_{dataset_name}.csv`: Training history

## Requirements

- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
