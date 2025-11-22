"""
Główny skrypt do trenowania modeli hiperspektralnych lokalnie
Użycie:
    python run_training.py --model InceptionHSINet --dataset Indian --epochs 50
    python run_training.py --model SimpleHSINet --dataset PaviaU --epochs 50
    python run_training.py --model CNNFromDiagram --dataset Indian --epochs 50
"""
import argparse
import torch
from load_data import get_loaders, DATASET_INFO
from train import train
from model1 import InceptionHSINet
from model2 import SimpleHSINet
from model3 import CNNFromDiagram


# Mapowanie nazw modeli do klas
MODELS = {
    'InceptionHSINet': InceptionHSINet,
    'SimpleHSINet': SimpleHSINet,
    'CNNFromDiagram': CNNFromDiagram
}

# Mapowanie typów modeli (2D vs 3D)
MODEL_TYPES = {
    'InceptionHSINet': '3d',
    'SimpleHSINet': '2d',
    'CNNFromDiagram': '2d'
}

# Dostępne datasety
AVAILABLE_DATASETS = list(DATASET_INFO.keys())


def create_model(model_name, num_bands, num_classes, patch_size=16):
    """Tworzy model na podstawie nazwy"""
    model_class = MODELS.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    if model_name == 'InceptionHSINet':
        return model_class(in_channels=1, num_classes=num_classes)
    elif model_name == 'SimpleHSINet':
        return model_class(input_channels=num_bands, num_classes=num_classes)
    elif model_name == 'CNNFromDiagram':
        return model_class(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Train hyperspectral image segmentation models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(MODELS.keys()),
                       help='Model name to train')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=AVAILABLE_DATASETS,
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patch_size', type=int, default=8,
                       help='Patch size (default: 8)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). If None, auto-detect')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patch size: {args.patch_size}")
    print(f"Validation split: {args.val_split}")
    print(f"{'='*80}\n")
    
    # Get model type
    model_type = MODEL_TYPES[args.model]
    
    # Load data and create loaders
    print("Loading dataset...")
    # num_workers=0 = główny wątek (mniej CPU, bezpieczniejsze)
    # num_workers=2-4 = równoległe ładowanie (więcej CPU, ale szybsze)
    num_workers = 0  # Zmień na 2-4 jeśli masz wystarczająco RAM i chcesz przyspieszyć
    train_loader, val_loader, dataset_info = get_loaders(
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        model_type=model_type,
        num_workers=num_workers
    )
    
    # Create model
    num_bands = dataset_info['num_bands']
    num_classes = dataset_info['num_classes']
    
    print(f"\nModel configuration:")
    print(f"  Input bands: {num_bands}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Model type: {model_type}")
    
    model = create_model(args.model, num_bands, num_classes, args.patch_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Train
    print("Starting training...")
    print(f"{'='*80}\n")
    
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_name=args.model,
        dataset_name=args.dataset
    )
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    print(f"Model saved to: best_model_{args.model}_{args.dataset}.pth")
    print(f"Training log saved to: training_log_{args.model}_{args.dataset}.csv")


if __name__ == '__main__':
    main()

