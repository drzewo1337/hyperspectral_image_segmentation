"""
Skrypt do wizualizacji wyników wszystkich 15 eksperymentów
Wyświetla ground truth vs predictions dla każdego modelu i datasetu
"""
import sys
import os
import torch
import matplotlib.pyplot as plt

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn'))

from utils.load_data import get_loaders, DATASET_INFO
from scripts.visualize.predict_and_visualize import predict_whole_scene, visualize
from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram

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

# Wszystkie modele i datasety
ALL_MODELS = ['InceptionHSINet', 'SimpleHSINet', 'CNNFromDiagram']
ALL_DATASETS = ['Indian', 'PaviaU', 'PaviaC', 'KSC', 'Salinas']

# Parametry
PATCH_SIZE = 8


def create_model(model_name, num_bands, num_classes, patch_size=8):
    """Tworzy model na podstawie nazwy"""
    model_class = MODELS.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    if model_name == 'InceptionHSINet':
        return model_class(in_channels=1, num_classes=num_classes)
    elif model_name == 'SimpleHSINet':
        return model_class(input_channels=num_bands, num_classes=num_classes)
    elif model_name == 'CNNFromDiagram':
        return model_class(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_trained_model(model_name, dataset_name, num_bands, num_classes, patch_size=8, device=None):
    """Ładuje wytrenowany model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(model_name, num_bands, num_classes, patch_size)
    
    # Ścieżka do modelu w nowej strukturze
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'models')
    model_path = os.path.join(models_dir, f"best_model_{model_name}_{dataset_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"⚠ Model {model_path} nie istnieje - pomijam")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Załadowano model: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Błąd ładowania modelu {model_path}: {e}")
        return None


def visualize_experiment(model_name, dataset_name, device=None):
    """Wizualizuje wyniki pojedynczego eksperymentu"""
    print(f"\n{'='*80}")
    print(f"Visualizing: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pobierz informacje o datasecie
    num_bands = DATASET_INFO[dataset_name]['num_bands']
    num_classes = DATASET_INFO[dataset_name]['num_classes']
    
    # Załaduj model
    model = load_trained_model(model_name, dataset_name, num_bands, num_classes, PATCH_SIZE, device)
    if model is None:
        return False
    
    try:
        # Wykonaj predykcję
        print(f"Wykonywanie predykcji...")
        pred_map, true_map = predict_whole_scene(model, dataset_name, patch_size=PATCH_SIZE, device=device)
        
        # Wyświetl wizualizację
        print(f"Wyświetlanie wyników...")
        fig = visualize(pred_map, true_map, dataset_name)
        
        # Zapisz do pliku
        # Ścieżka do wizualizacji w nowej strukturze
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        viz_dir = os.path.join(base_dir, 'results', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        output_file = os.path.join(viz_dir, f"visualization_{model_name}_{dataset_name}.png")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Zapisano wizualizację do: {output_file}")
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"✗ Błąd podczas wizualizacji: {e}")
        return False


def main():
    """Główna funkcja - wizualizuje wszystkie eksperymenty"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"VISUALIZATION OF ALL EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total experiments: {len(ALL_MODELS) * len(ALL_DATASETS)}")
    print(f"{'='*80}\n")
    
    results = []
    successful = 0
    failed = 0
    
    for model_name in ALL_MODELS:
        for dataset_name in ALL_DATASETS:
            success = visualize_experiment(model_name, dataset_name, device)
            if success:
                successful += 1
                results.append(f"✓ {model_name} on {dataset_name}")
            else:
                failed += 1
                results.append(f"✗ {model_name} on {dataset_name}")
    
    # Podsumowanie
    print(f"\n{'='*80}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*80}\n")
    
    print("Detailed results:")
    for result in results:
        print(f"  {result}")
    
    print(f"\n{'='*80}")
    print("All visualizations completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

