"""
Moduł do eksportowania przetworzonych danych po redukcji wymiarów
"""
import os
import numpy as np
import scipy.io as sio
from datetime import datetime


def export_processed_data(data, labels, dataset_name, target_bands, output_dir):
    """
    Eksportuje przetworzone dane do plików .npy i .mat
    
    Args:
        data: numpy array o kształcie (H, W, target_bands) - przetworzone dane
        labels: numpy array o kształcie (H, W) - etykiety
        dataset_name: nazwa datasetu
        target_bands: docelowa liczba pasm
        output_dir: katalog wyjściowy
    """
    # Utwórz katalog dla tego datasetu i liczby pasm
    dataset_dir = os.path.join(output_dir, f'{dataset_name}_bands{target_bands}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Zapisz jako .npy (szybki format numpy)
    data_file_npy = os.path.join(dataset_dir, f'{dataset_name}_data_bands{target_bands}.npy')
    labels_file_npy = os.path.join(dataset_dir, f'{dataset_name}_labels.npy')
    
    np.save(data_file_npy, data)
    np.save(labels_file_npy, labels)
    
    print(f"  ✓ Zapisano .npy: {data_file_npy}")
    print(f"  ✓ Zapisano .npy: {labels_file_npy}")
    
    # Zapisz jako .mat (kompatybilność z istniejącym kodem)
    data_file_mat = os.path.join(dataset_dir, f'{dataset_name}_data_bands{target_bands}.mat')
    labels_file_mat = os.path.join(dataset_dir, f'{dataset_name}_labels.mat')
    
    # Użyj tej samej konwencji nazewnictwa co oryginalne pliki
    data_key = f'{dataset_name.lower()}_data_bands{target_bands}'
    labels_key = f'{dataset_name.lower()}_labels'
    
    sio.savemat(data_file_mat, {data_key: data})
    sio.savemat(labels_file_mat, {labels_key: labels})
    
    print(f"  ✓ Zapisano .mat: {data_file_mat}")
    print(f"  ✓ Zapisano .mat: {labels_file_mat}")
    
    # Zapisz metadane
    metadata = {
        'dataset_name': dataset_name,
        'target_bands': target_bands,
        'data_shape': data.shape,
        'labels_shape': labels.shape,
        'data_dtype': str(data.dtype),
        'labels_dtype': str(labels.dtype),
        'export_date': datetime.now().isoformat(),
        'data_min': float(np.min(data)),
        'data_max': float(np.max(data)),
        'data_mean': float(np.mean(data)),
        'data_std': float(np.std(data)),
        'unique_labels': int(len(np.unique(labels[labels > 0])))  # Liczba klas (bez tła)
    }
    
    metadata_file = os.path.join(dataset_dir, f'{dataset_name}_metadata_bands{target_bands}.npz')
    np.savez(metadata_file, **metadata)
    
    print(f"  ✓ Zapisano metadane: {metadata_file}")
    
    return dataset_dir


def export_all_processed_datasets(dataset_names, target_bands, output_dir):
    """
    Eksportuje wszystkie przetworzone datasety dla danej liczby pasm
    
    Args:
        dataset_names: lista nazw datasetów
        target_bands: docelowa liczba pasm
        output_dir: katalog wyjściowy
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    
    from utils.load_data import load_data
    from preprocessing import preprocess_data
    
    print(f"\n{'='*80}")
    print(f"EKSPORTOWANIE PRZETWORZONYCH DANYCH - {target_bands} pasm")
    print(f"{'='*80}\n")
    
    exported_datasets = []
    
    for dataset_name in dataset_names:
        print(f"\nPrzetwarzanie {dataset_name}...")
        
        # Załaduj dane
        data, labels = load_data(dataset_name)
        print(f"  Oryginalne dane: shape={data.shape}, bands={data.shape[2]}")
        
        # Preprocessing
        data_processed = preprocess_data(data, target_bands, normalize=True)
        print(f"  Przetworzone dane: shape={data_processed.shape}, bands={data_processed.shape[2]}")
        
        # Eksportuj
        dataset_dir = export_processed_data(
            data_processed, 
            labels, 
            dataset_name, 
            target_bands, 
            output_dir
        )
        
        exported_datasets.append({
            'dataset_name': dataset_name,
            'target_bands': target_bands,
            'output_dir': dataset_dir,
            'data_shape': data_processed.shape,
            'labels_shape': labels.shape
        })
    
    print(f"\n{'='*80}")
    print(f"EKSPORT ZAKOŃCZONY")
    print(f"{'='*80}")
    print(f"Wyeksportowano {len(exported_datasets)} datasetów do: {output_dir}")
    print(f"{'='*80}\n")
    
    return exported_datasets


def load_exported_data(dataset_name, target_bands, data_dir):
    """
    Ładuje wyeksportowane przetworzone dane
    
    Args:
        dataset_name: nazwa datasetu
        target_bands: liczba pasm
        data_dir: katalog z wyeksportowanymi danymi
    
    Returns:
        data: numpy array (H, W, target_bands)
        labels: numpy array (H, W)
        metadata: słownik z metadanymi
    """
    dataset_dir = os.path.join(data_dir, f'{dataset_name}_bands{target_bands}')
    
    # Załaduj dane
    data_file = os.path.join(dataset_dir, f'{dataset_name}_data_bands{target_bands}.npy')
    labels_file = os.path.join(dataset_dir, f'{dataset_name}_labels.npy')
    metadata_file = os.path.join(dataset_dir, f'{dataset_name}_metadata_bands{target_bands}.npz')
    
    data = np.load(data_file)
    labels = np.load(labels_file)
    metadata = dict(np.load(metadata_file))
    
    return data, labels, metadata

