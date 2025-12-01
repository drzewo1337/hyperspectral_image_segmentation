"""
Wizualizacja wyników segmentacji
Wyświetla ground truth (z lewej) i wynik segmentacji (z prawej) dla różnych datasetów
"""
import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Użyj backendu bez GUI
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import load_data, pad_with_zeros
from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram

# Import funkcji clusteringu
import importlib.util
clustering_path = os.path.join(os.path.dirname(__file__), '7_clustering_segmentation.py')
spec = importlib.util.spec_from_file_location("clustering_segmentation", clustering_path)
clustering_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clustering_module)
segment_with_dbscan = clustering_module.segment_with_dbscan

# Mapowanie nazw modeli do klas
MODELS = {
    'InceptionHSINet': InceptionHSINet,
    'SimpleHSINet': SimpleHSINet,
    'CNNFromDiagram': CNNFromDiagram
}

def predict_segmentation_map(model, data, labels, patch_size=16, model_type='2d', device=None):
    """
    Przewiduje mapę segmentacji używając DBSCAN clustering
    
    Args:
        model: wytrenowany model z metodą extract_features()
        data: obraz hiperspektralny (H, W, B)
        labels: ground truth labels (H, W) - używane tylko do określenia obszaru
        patch_size: rozmiar patchy
        model_type: '2d' lub '3d'
        device: urządzenie (cuda/cpu)
    
    Returns:
        segmentation_map: mapa segmentacji (H, W) z grupami z clusteringu
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Użyj DBSCAN do segmentacji
    segmentation_map, n_clusters = segment_with_dbscan(
        model, data, labels, patch_size=patch_size, model_type=model_type,
        eps=0.5, min_samples=5, device=device, batch_size=64
    )
    
    return segmentation_map

def visualize_segmentation_results(model_path, model_name, dataset_name, target_bands, 
                                   preprocessed_data, patch_size=16, device=None):
    """
    Wizualizuje wyniki segmentacji: ground truth (lewo) vs prediction (prawo)
    
    Args:
        model_path: ścieżka do wytrenowanego modelu
        model_name: nazwa modelu
        dataset_name: nazwa datasetu
        target_bands: liczba kanałów
        preprocessed_data: dict z preprocessowanymi danymi
        patch_size: rozmiar patchy
        device: urządzenie
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Załaduj dane
    key = (dataset_name, target_bands)
    if key not in preprocessed_data:
        print(f"⚠ Brak danych dla {dataset_name} z {target_bands} kanałami")
        return None
    
    data = preprocessed_data[key]['data']
    labels = preprocessed_data[key]['labels']
    info = preprocessed_data[key]['info']
    
    # Załaduj model
    model_class = MODELS[model_name]
    model_type = '3d' if model_name == 'InceptionHSINet' else '2d'
    
    num_bands = data.shape[2]
    num_classes = info['num_classes']
    
    if model_name == 'InceptionHSINet':
        model = model_class(in_channels=1, num_classes=num_classes)
    elif model_name == 'SimpleHSINet':
        model = model_class(input_channels=num_bands, num_classes=num_classes)
    elif model_name == 'CNNFromDiagram':
        model = model_class(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
    
    # Załaduj wagi
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Załadowano model z {model_path}")
    else:
        print(f"⚠ Model nie istnieje: {model_path}")
        return None
    
    # Przewiduj mapę segmentacji
    print(f"Przewidywanie segmentacji dla {dataset_name}...")
    segmentation_map = predict_segmentation_map(
        model, data, labels, patch_size=patch_size, 
        model_type=model_type, device=device
    )
    
    # Utwórz wizualizację
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground truth (lewo)
    axes[0].imshow(labels, cmap='tab20', vmin=0, vmax=num_classes)
    axes[0].set_title(f'Ground Truth - {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction (prawo)
    axes[1].imshow(segmentation_map, cmap='tab20', vmin=0, vmax=num_classes)
    axes[1].set_title(f'Prediction - {model_name} ({target_bands} bands)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Zapisz
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'new_flow', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'segmentation_{model_name}_{dataset_name}_{target_bands}bands.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Zapisano wizualizację do {filepath}")
    
    plt.close()
    
    return filepath

def visualize_all_results(all_results, preprocessed_data, patch_size=16):
    """
    Wizualizuje wyniki dla wszystkich modeli i datasetów z wyników testów
    
    Args:
        all_results: dict z wynikami (target_bands -> lista wyników)
        preprocessed_data: dict z preprocessowanymi danymi
        patch_size: rozmiar patchy
    """
    print("=" * 80)
    print("WIZUALIZACJA WYNIKÓW SEGMENTACJI")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Znajdź wszystkie unikalne kombinacje model-dataset-bands
    for target_bands, results in all_results.items():
        print(f"\nWizualizacja dla {target_bands} kanałów...")
        
        for result in results:
            model_name = result['model_name']
            test_dataset = result['test_dataset']
            validation_dataset = result.get('validation_dataset', result.get('final_test_dataset'))
            
            # Ścieżka do modelu - nazwa zgodna z train.py
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'models')
            dataset_name_for_model = f"Split{result['split_id']}_bands{target_bands}"
            model_filename = f"best_model_{model_name}_{dataset_name_for_model}.pth"
            model_path = os.path.join(models_dir, model_filename)
            
            if model_path is None:
                print(f"  ⚠ Nie znaleziono modelu dla {model_name}, split {result['split_id']}, {target_bands} bands")
                continue
            
            # Wizualizuj dla test dataset
            print(f"  Wizualizacja: {model_name} na {test_dataset} (test)")
            visualize_segmentation_results(
                model_path, model_name, test_dataset, target_bands,
                preprocessed_data, patch_size=patch_size, device=device
            )
            
            # Wizualizuj dla validation dataset
            print(f"  Wizualizacja: {model_name} na {validation_dataset} (validation)")
            visualize_segmentation_results(
                model_path, model_name, validation_dataset, target_bands,
                preprocessed_data, patch_size=patch_size, device=device
            )
    
    print("\n✓ Wszystkie wizualizacje zapisane")

if __name__ == "__main__":
    print("Ten skrypt powinien być wywoływany z głównego flow scriptu")

