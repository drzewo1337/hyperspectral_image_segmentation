"""
Moduł do tworzenia zbiorów testowych z danych hiperspektralnych
Dzieli 5 datasetów na: 3 do treningu, 1 do testu, 1 do walidacji
Tworzy 5 różnych kombinacji (permutacji)
"""
import sys
import os
import numpy as np

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from utils.load_data import load_data, DATASET_INFO, pad_with_zeros, normalize
from preprocessing import preprocess_data


def extract_patches_from_dataset(data, labels, patch_size=8):
    """
    Ekstrahuje patchy z pojedynczego datasetu
    
    Args:
        data: numpy array o kształcie (H, W, B)
        labels: numpy array o kształcie (H, W)
        patch_size: rozmiar patchy
    
    Returns:
        patches: numpy array o kształcie (N, H, W, B)
        targets: numpy array o kształcie (N,)
    """
    # Padding
    margin = patch_size // 2
    padded_data = pad_with_zeros(data, margin)
    
    # Ekstrakcja patchy
    h, w = labels.shape
    patches = []
    targets = []
    
    for i in range(h):
        for j in range(w):
            label = labels[i, j]
            if label == 0:  # Ignoruj tło
                continue
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            targets.append(label)
    
    return np.array(patches), np.array(targets)


def create_dataset_splits(target_bands=90, patch_size=8, n_splits=5, export_data=False, export_dir=None):
    """
    Tworzy n_splits różnych podziałów datasetów na train/test/val
    Każdy podział: 3 datasety do treningu, 1 do testu, 1 do walidacji
    
    Args:
        target_bands: docelowa liczba pasm po preprocessing
        patch_size: rozmiar patchy
        n_splits: liczba różnych podziałów (domyślnie 5)
        export_data: czy eksportować przetworzone dane do plików
        export_dir: katalog do eksportu danych (jeśli None, używa domyślnego)
    
    Returns:
        splits: lista słowników z kluczami: train_datasets, test_dataset, val_dataset, 
               train_patches, train_labels, test_patches, test_labels, val_patches, val_labels
    """
    # Wszystkie dostępne datasety
    all_datasets = ['Indian', 'PaviaU', 'KSC', 'Salinas', 'PaviaC']
    
    # Eksportuj przetworzone dane jeśli wymagane
    if export_data:
        sys.path.insert(0, os.path.dirname(__file__))
        from export_data import export_all_processed_datasets
        if export_dir is None:
            export_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
        os.makedirs(export_dir, exist_ok=True)
        export_all_processed_datasets(all_datasets, target_bands, export_dir)
    
    # Utwórz 5 różnych kombinacji
    # Będziemy rotować które datasety są train/test/val
    splits = []
    
    # Generuj różne kombinacje
    # Kombinacja 1: Train: Indian+PaviaU+KSC, Test: Salinas, Val: PaviaC
    # Kombinacja 2: Train: Indian+PaviaU+Salinas, Test: KSC, Val: PaviaC
    # Kombinacja 3: Train: Indian+KSC+Salinas, Test: PaviaU, Val: PaviaC
    # Kombinacja 4: Train: PaviaU+KSC+Salinas, Test: Indian, Val: PaviaC
    # Kombinacja 5: Train: Indian+PaviaU+KSC, Test: PaviaC, Val: Salinas
    
    split_configs = [
        (['Indian', 'PaviaU', 'KSC'], 'Salinas', 'PaviaC'),
        (['Indian', 'PaviaU', 'Salinas'], 'KSC', 'PaviaC'),
        (['Indian', 'KSC', 'Salinas'], 'PaviaU', 'PaviaC'),
        (['PaviaU', 'KSC', 'Salinas'], 'Indian', 'PaviaC'),
        (['Indian', 'PaviaU', 'KSC'], 'PaviaC', 'Salinas'),
    ]
    
    for i, (train_names, test_name, val_name) in enumerate(split_configs[:n_splits]):
        print(f"\n{'='*80}")
        print(f"Tworzenie podziału {i+1}/{n_splits}")
        print(f"Train: {', '.join(train_names)}")
        print(f"Test: {test_name}")
        print(f"Val: {val_name}")
        print(f"{'='*80}")
        
        # Załaduj i przetwórz dane treningowe
        train_patches_list = []
        train_labels_list = []
        
        for dataset_name in train_names:
            print(f"\nŁadowanie {dataset_name}...")
            data, labels = load_data(dataset_name)
            data_processed = preprocess_data(data, target_bands, normalize=True)
            patches, targets = extract_patches_from_dataset(data_processed, labels, patch_size)
            train_patches_list.append(patches)
            train_labels_list.append(targets)
            print(f"  {dataset_name}: {len(patches)} patchy")
        
        # Połącz dane treningowe
        train_patches = np.concatenate(train_patches_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        print(f"\nTrain łącznie: {len(train_patches)} patchy")
        
        # Załaduj i przetwórz dane testowe
        print(f"\nŁadowanie {test_name}...")
        test_data, test_labels = load_data(test_name)
        test_data_processed = preprocess_data(test_data, target_bands, normalize=True)
        test_patches, test_targets = extract_patches_from_dataset(
            test_data_processed, test_labels, patch_size
        )
        print(f"  {test_name}: {len(test_patches)} patchy")
        
        # Załaduj i przetwórz dane walidacyjne
        print(f"\nŁadowanie {val_name}...")
        val_data, val_labels = load_data(val_name)
        val_data_processed = preprocess_data(val_data, target_bands, normalize=True)
        val_patches, val_targets = extract_patches_from_dataset(
            val_data_processed, val_labels, patch_size
        )
        print(f"  {val_name}: {len(val_patches)} patchy")
        
        splits.append({
            'split_id': i + 1,
            'train_datasets': train_names,
            'test_dataset': test_name,
            'val_dataset': val_name,
            'train_patches': train_patches,
            'train_labels': train_labels,
            'test_patches': test_patches,
            'test_labels': test_targets,
            'val_patches': val_patches,
            'val_labels': val_targets
        })
        
        print(f"\n✓ Podział {i+1} utworzony")
    
    return splits


def create_test_datasets_from_multiple_sources(target_bands=90, patch_size=8, n_splits=5, export_data=False, export_dir=None):
    """
    Tworzy n_splits różnych podziałów datasetów
    Każdy podział: 3 datasety do treningu, 1 do testu, 1 do walidacji
    
    Args:
        target_bands: docelowa liczba pasm po preprocessing
        patch_size: rozmiar patchy
        n_splits: liczba różnych podziałów (domyślnie 5)
        export_data: czy eksportować przetworzone dane do plików
        export_dir: katalog do eksportu danych
    
    Returns:
        splits: lista słowników z informacjami o podziałach
    """
    return create_dataset_splits(
        target_bands=target_bands, 
        patch_size=patch_size, 
        n_splits=n_splits,
        export_data=export_data,
        export_dir=export_dir
    )

