"""
Krok 4: Testy 3 odtworzonych modeli
Testuje 3 modele na wygenerowanych datasetach używając DBSCAN clustering do segmentacji
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import mode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import pad_with_zeros, DATASET_INFO
from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram
from scripts.train.train import train

# Import funkcji clusteringu - użyj sys.path aby zaimportować moduł z numerem
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

class PreprocessedHSIDataset(Dataset):
    """
    Dataset dla preprocessowanych danych hiperspektralnych
    """
    def __init__(self, data_dict, dataset_names, patch_size=16, model_type='2d'):
        """
        Args:
            data_dict: dict z kluczami (dataset_name, target_bands) -> (data, labels, info)
            dataset_names: lista nazw datasetów do użycia
            patch_size: rozmiar patchy
            model_type: '2d' lub '3d'
        """
        self.patch_size = patch_size
        self.model_type = model_type
        
        patches_list = []
        targets_list = []
        all_num_classes = set()
        
        # Najpierw znajdź wspólną liczbę kanałów dla wszystkich datasetów
        # (powinna być taka sama, bo preprocessing ujednolicił)
        target_bands_list = [data_dict[k]['data'].shape[2] for k in data_dict.keys() if k[0] in dataset_names]
        if not target_bands_list:
            raise ValueError(f"Brak danych dla datasetów: {dataset_names}")
        
        # Użyj pierwszej dostępnej liczby kanałów (wszystkie powinny być takie same)
        target_bands = target_bands_list[0]
        if len(set(target_bands_list)) > 1:
            print(f"  ⚠ Ostrzeżenie: Różne liczby kanałów w datasetach: {set(target_bands_list)}, używam {target_bands}")
        
        for dataset_name in dataset_names:
            # Znajdź klucz z odpowiednią liczbą kanałów
            dataset_keys = [k for k in data_dict.keys() if k[0] == dataset_name and data_dict[k]['data'].shape[2] == target_bands]
            if not dataset_keys:
                # Jeśli nie ma dokładnie target_bands, użyj pierwszej dostępnej
                dataset_keys = [k for k in data_dict.keys() if k[0] == dataset_name]
                if not dataset_keys:
                    continue
            
            key = dataset_keys[0]
            data = data_dict[key]['data']
            labels = data_dict[key]['labels']
            info = data_dict[key]['info']
            all_num_classes.add(info['num_classes'])
            
            # Padding
            margin = patch_size // 2
            padded_data = pad_with_zeros(data, margin)
            
            # Ekstrakcja patchy
            h, w, _ = data.shape
            for i in range(h):
                for j in range(w):
                    label = labels[i, j]
                    if label == 0:  # Ignoruj tło
                        continue
                    patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                    patches_list.append(patch)
                    targets_list.append(label - 1)  # -1 bo klasy zaczynają od 1
        
        if len(patches_list) == 0:
            raise ValueError(f"Brak danych dla datasetów: {dataset_names}")
        
        self.patches = np.array(patches_list)
        self.targets = np.array(targets_list)
        
        # Upewnij się, że wszystkie patchy mają tę samą liczbę kanałów
        if len(self.patches.shape) == 4:
            current_bands = self.patches.shape[-1]
            if current_bands != target_bands:
                if current_bands < target_bands:
                    # Padding zerami
                    padding = np.zeros((self.patches.shape[0], self.patches.shape[1], self.patches.shape[2], target_bands - current_bands))
                    self.patches = np.concatenate([self.patches, padding], axis=-1)
                else:
                    # Obcięcie
                    self.patches = self.patches[:, :, :, :target_bands]
        
        # Przygotuj dane w odpowiednim formacie
        if model_type == '3d':
            # Conv3D: (N, 1, B, H, W)
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)
            self.patches = np.expand_dims(self.patches, axis=1)  # (N, 1, B, H, W)
        else:
            # Conv2D: (N, B, H, W)
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)
        
        # Konwertuj do torch tensor
        self.patches = torch.from_numpy(self.patches).float()
        self.targets = torch.from_numpy(self.targets).long()
        
        self.num_bands = self.patches.shape[1] if model_type == '2d' else self.patches.shape[2]
        self.num_classes = max(all_num_classes) if all_num_classes else 16
        
        print(f"Dataset {dataset_names}: {len(self)} samples, bands={self.num_bands}, classes={self.num_classes}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.targets[idx]

def test_models(preprocessed_data, splits, target_bands, patch_size=16, batch_size=64, epochs=50, lr=0.001):
    """
    Testuje 3 modele na wygenerowanych datasetach
    
    Args:
        preprocessed_data: dict z preprocessowanymi danymi
        splits: lista podziałów datasetów
        target_bands: docelowa liczba kanałów
        patch_size: rozmiar patchy
        batch_size: rozmiar batcha
        epochs: liczba epok
        lr: learning rate
    
    Returns:
        results: lista wyników dla każdego modelu i split
    """
    print("=" * 80)
    print(f"KROK 4: Testy 3 odtworzonych modeli (target_bands={target_bands})")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    results = []
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split {split['split_id']}:")
        print(f"  Train: {', '.join(split['train_datasets'])}")
        print(f"  Test: {split['test_dataset']}")
        print(f"  Validation: {split.get('validation_dataset', split.get('final_test_dataset'))}")
        print(f"{'='*60}")
        
        # Przygotuj dane treningowe
        train_keys = [(name, target_bands) for name in split['train_datasets']]
        train_data_dict = {k: preprocessed_data[k] for k in train_keys if k in preprocessed_data}
        
        if not train_data_dict:
            print(f"  ⚠ Brak danych dla train datasets z {target_bands} kanałami")
            continue
        
        # Określ typ modelu na podstawie liczby kanałów
        # Dla każdego modelu sprawdź czy potrzebuje 2d czy 3d
        for model_name, model_class in MODELS.items():
            print(f"\n  Model: {model_name}")
            
            # Określ typ modelu
            model_type = '3d' if model_name == 'InceptionHSINet' else '2d'
            
            try:
                # Dataset treningowy - użyj tylko danych z target_bands
                train_keys_filtered = {k: v for k, v in train_data_dict.items() if k[1] == target_bands}
                if not train_keys_filtered:
                    print(f"    ⚠ Brak danych treningowych dla {target_bands} kanałów")
                    continue
                
                train_dataset = PreprocessedHSIDataset(
                    train_keys_filtered, split['train_datasets'], 
                    patch_size=patch_size, model_type=model_type
                )
                
                # Podział na train/val
                val_len = int(len(train_dataset) * 0.2)
                train_len = len(train_dataset) - val_len
                train_set, val_set = random_split(train_dataset, [train_len, val_len])
                
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                
                # Utwórz model
                num_bands = train_dataset.num_bands
                num_classes = train_dataset.num_classes
                
                if model_name == 'InceptionHSINet':
                    model = model_class(in_channels=1, num_classes=num_classes)
                elif model_name == 'SimpleHSINet':
                    model = model_class(input_channels=num_bands, num_classes=num_classes)
                elif model_name == 'CNNFromDiagram':
                    model = model_class(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
                
                # Trenuj model - użyj meta-learning jeśli dostępne
                print(f"    Trenowanie... (bands={num_bands}, classes={num_classes})")
                
                # Sprawdź czy używać meta-learning (trening na różnych liczbach klas)
                use_meta_learning = True  # Można zmienić na False dla standardowego treningu
                
                if use_meta_learning:
                    try:
                        # Import meta-learning modułu
                        import importlib.util
                        meta_learning_path = os.path.join(os.path.dirname(__file__), '8_meta_learning_training.py')
                        spec = importlib.util.spec_from_file_location("meta_learning_training", meta_learning_path)
                        meta_learning_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(meta_learning_module)
                        train_meta_learning_simple = meta_learning_module.train_meta_learning_simple
                        FlexibleHSIDataset = meta_learning_module.FlexibleHSIDataset
                        
                        # Utwórz flexible dataset (obsługuje różne liczby klas)
                        flexible_train_dataset = FlexibleHSIDataset(
                            train_keys_filtered, split['train_datasets'],
                            patch_size=patch_size, model_type=model_type
                        )
                        flexible_val_len = int(len(flexible_train_dataset) * 0.2)
                        flexible_train_len = len(flexible_train_dataset) - flexible_val_len
                        flexible_train_set, flexible_val_set = random_split(
                            flexible_train_dataset, [flexible_train_len, flexible_val_len]
                        )
                        flexible_train_loader = DataLoader(flexible_train_set, batch_size=batch_size, shuffle=True)
                        flexible_val_loader = DataLoader(flexible_val_set, batch_size=batch_size, shuffle=False)
                        
                        print(f"    Meta-learning: trening na różnych liczbach klas")
                        trained_model = train_meta_learning_simple(
                            model, flexible_train_loader, flexible_val_loader,
                            epochs=epochs, lr=lr, device=device
                        )
                    except Exception as e:
                        print(f"    ⚠ Meta-learning nie działa, używam standardowego treningu: {e}")
                        trained_model = train(
                            model, train_loader, val_loader, epochs=epochs, lr=lr,
                            device=device, model_name=model_name,
                            dataset_name=f"Split{split['split_id']}_bands{target_bands}"
                        )
                else:
                    # Standardowy trening
                    trained_model = train(
                        model, train_loader, val_loader, epochs=epochs, lr=lr,
                        device=device, model_name=model_name,
                        dataset_name=f"Split{split['split_id']}_bands{target_bands}"
                    )
                
                # Test na test dataset - użyj DBSCAN clustering
                test_key = (split['test_dataset'], target_bands)
                test_acc = 0.0
                test_n_clusters = 0
                test_n_samples = 0
                if test_key in preprocessed_data:
                    test_data = preprocessed_data[test_key]['data']
                    test_labels = preprocessed_data[test_key]['labels']
                    
                    # Segmentacja przez DBSCAN
                    segmentation_map, n_clusters = segment_with_dbscan(
                        trained_model, test_data, test_labels,
                        patch_size=patch_size, model_type=model_type,
                        eps=0.5, min_samples=5, device=device, batch_size=batch_size
                    )
                    
                    # Ewaluacja clusteringu (porównanie z ground truth)
                    test_acc = evaluate_clustering(segmentation_map, test_labels)
                    test_n_clusters = n_clusters
                    test_n_samples = np.sum(test_labels > 0)
                    
                    print(f"    Test DBSCAN: accuracy={test_acc:.2f}%, clusters={n_clusters}")
                else:
                    print(f"    ⚠ Brak danych testowych")
                
                # Test na validation dataset (walidacja) - użyj DBSCAN clustering
                validation_dataset_name = split.get('validation_dataset', split.get('final_test_dataset'))
                final_test_key = (validation_dataset_name, target_bands)
                final_test_acc = 0.0
                final_test_n_clusters = 0
                final_test_n_samples = 0
                if final_test_key in preprocessed_data:
                    validation_data = preprocessed_data[final_test_key]['data']
                    validation_labels = preprocessed_data[final_test_key]['labels']
                    
                    # Segmentacja przez DBSCAN
                    segmentation_map, n_clusters = segment_with_dbscan(
                        trained_model, validation_data, validation_labels,
                        patch_size=patch_size, model_type=model_type,
                        eps=0.5, min_samples=5, device=device, batch_size=batch_size
                    )
                    
                    # Ewaluacja clusteringu
                    final_test_acc = evaluate_clustering(segmentation_map, validation_labels)
                    final_test_n_clusters = n_clusters
                    final_test_n_samples = np.sum(validation_labels > 0)
                    
                    print(f"    Validation DBSCAN: accuracy={final_test_acc:.2f}%, clusters={n_clusters}")
                else:
                    print(f"    ⚠ Brak danych validation")
                
                # Zapisz wynik
                result = {
                    'split_id': split['split_id'],
                    'model_name': model_name,
                    'target_bands': target_bands,
                    'train_datasets': split['train_datasets'],
                    'test_dataset': split['test_dataset'],
                    'validation_dataset': validation_dataset_name,
                    'final_test_dataset': validation_dataset_name,  # Dla kompatybilności
                    'test_accuracy': test_acc,
                    'test_n_clusters': test_n_clusters,
                    'validation_accuracy': final_test_acc,
                    'validation_n_clusters': final_test_n_clusters,
                    'final_test_accuracy': final_test_acc,  # Dla kompatybilności
                    'test_n_samples': test_n_samples,
                    'validation_n_samples': final_test_n_samples,
                    'final_test_n_samples': final_test_n_samples  # Dla kompatybilności
                }
                results.append(result)
                
            except Exception as e:
                print(f"    ✗ Błąd: {e}")
                import traceback
                traceback.print_exception(type(e), e, e.__traceback__)
                continue
    
    return results

def evaluate_clustering(segmentation_map, ground_truth):
    """
    Ewaluacja clusteringu przez porównanie z ground truth
    Mapuje klastry na klasy używając najczęstszego dopasowania
    
    Args:
        segmentation_map: mapa segmentacji z clusteringu (H, W)
        ground_truth: ground truth labels (H, W)
    
    Returns:
        accuracy: dokładność po mapowaniu klastrów na klasy
    """
    # Znajdź piksele z danymi (nie tło)
    mask = ground_truth > 0
    
    if np.sum(mask) == 0:
        return 0.0
    
    # Pobierz segmenty i klasy dla pikseli z danymi
    segments = segmentation_map[mask]
    classes = ground_truth[mask]
    
    # Mapuj klastry na klasy używając najczęstszego dopasowania
    from scipy.stats import mode
    
    unique_segments = np.unique(segments)
    unique_segments = unique_segments[unique_segments > 0]  # Usuń tło
    
    if len(unique_segments) == 0:
        return 0.0
    
    # Dla każdego klastra znajdź najczęstszą klasę
    cluster_to_class = {}
    for seg in unique_segments:
        seg_mask = segments == seg
        if np.sum(seg_mask) > 0:
            most_common_class = mode(classes[seg_mask], keepdims=True)[0][0]
            cluster_to_class[seg] = most_common_class
    
    # Mapuj segmenty na klasy
    mapped_segments = np.zeros_like(segments)
    for seg, cls in cluster_to_class.items():
        mapped_segments[segments == seg] = cls
    
    # Oblicz accuracy
    correct = np.sum(mapped_segments == classes)
    total = len(classes)
    
    return 100.0 * correct / total if total > 0 else 0.0

if __name__ == "__main__":
    print("Ten skrypt powinien być wywoływany z głównego flow scriptu")

