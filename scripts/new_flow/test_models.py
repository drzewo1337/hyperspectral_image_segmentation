"""
Moduł do testowania odtworzonych modeli na zbiorach testowych
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'cnn'))

from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram


class TestDataset(Dataset):
    """Dataset dla danych testowych (patchy)"""
    def __init__(self, X, y, patch_size=8, model_type='2d'):
        """
        Args:
            X: numpy array o kształcie (N, H, W, B) - patchy
            y: numpy array o kształcie (N,) - etykiety
            patch_size: rozmiar patchy
            model_type: '2d' lub '3d'
        """
        self.X = X
        self.y = y
        self.patch_size = patch_size
        self.model_type = model_type
        
        # Przygotuj dane w odpowiednim formacie
        if model_type == '3d':
            # Conv3D: (N, 1, B, H, W) - pasma jako depth dimension
            self.X = np.transpose(self.X, (0, 3, 1, 2))  # (N, B, H, W)
            self.X = np.expand_dims(self.X, axis=1)  # (N, 1, B, H, W)
        else:
            # Conv2D: (N, B, H, W) - pasma jako channels
            self.X = np.transpose(self.X, (0, 3, 1, 2))  # (N, B, H, W)
        
        # Konwertuj do tensorów
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long() - 1  # -1 bo klasy zaczynają od 1
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_trained_model(model_name, model_path, num_bands, num_classes, patch_size=8, device='cpu'):
    """
    Ładuje wytrenowany model z pliku
    
    Args:
        model_name: nazwa modelu ('InceptionHSINet', 'SimpleHSINet', 'CNNFromDiagram')
        model_path: ścieżka do pliku modelu
        num_bands: liczba pasm
        num_classes: liczba klas
        patch_size: rozmiar patchy
        device: urządzenie (cpu/cuda)
    
    Returns:
        model: załadowany model
    """
    # Utwórz model
    if model_name == 'InceptionHSINet':
        model = InceptionHSINet(in_channels=1, num_classes=num_classes)
    elif model_name == 'SimpleHSINet':
        model = SimpleHSINet(input_channels=num_bands, num_classes=num_classes)
    elif model_name == 'CNNFromDiagram':
        model = CNNFromDiagram(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Załaduj wagę
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  ✓ Załadowano model z {model_path}")
    else:
        print(f"  ⚠ Plik modelu nie istnieje: {model_path}")
        print(f"  Używam modelu bez wag (losowe inicjalizacja)")
    
    model.to(device)
    model.eval()
    
    return model


def test_model_on_dataset(model, test_dataset, device, batch_size=64):
    """
    Testuje model na zbiorze testowym
    
    Args:
        model: model do testowania
        test_dataset: dataset testowy
        device: urządzenie
        batch_size: rozmiar batcha
    
    Returns:
        accuracy: dokładność modelu
        predictions: przewidywania
        true_labels: prawdziwe etykiety
    """
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Ogranicz przewidywania do zakresu klas w zbiorze testowym
            max_class = labels.max().item() + 1
            predicted = torch.clamp(predicted, 0, max_class - 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return accuracy, np.array(all_predictions), np.array(all_labels)


def test_models_on_datasets(model_names, model_paths, dataset_splits, 
                           target_bands, patch_size=8, 
                           device=None, batch_size=64):
    """
    Testuje wiele modeli na wielu podziałach datasetów
    
    Args:
        model_names: lista nazw modeli
        model_paths: słownik {model_name: path} lub lista ścieżek
        dataset_splits: lista słowników z informacjami o podziałach (train/test/val)
        target_bands: docelowa liczba pasm
        patch_size: rozmiar patchy
        device: urządzenie
        batch_size: rozmiar batcha
    
    Returns:
        results: słownik z wynikami
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mapowanie typów modeli
    MODEL_TYPES = {
        'InceptionHSINet': '3d',
        'SimpleHSINet': '2d',
        'CNNFromDiagram': '2d'
    }
    
    # Import DATASET_INFO dla num_classes
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from utils.load_data import DATASET_INFO
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Testowanie modelu: {model_name}")
        print(f"{'='*80}")
        
        model_type = MODEL_TYPES[model_name]
        
        # Pobierz ścieżkę modelu
        if isinstance(model_paths, dict):
            model_path = model_paths.get(model_name)
        elif isinstance(model_paths, list):
            model_path = model_paths[model_names.index(model_name)]
        else:
            # Domyślna ścieżka
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'results', 'models',
                f'best_model_{model_name}_*.pth'
            )
        
        if model_path is None:
            print(f"  ⚠ Brak ścieżki dla modelu {model_name}, pomijam")
            continue
        
        model_results = []
        
        # Testuj na każdym podziale
        for split in dataset_splits:
            split_id = split['split_id']
            test_dataset_name = split['test_dataset']
            val_dataset_name = split['val_dataset']
            
            # Pobierz num_classes z test dataset (używamy max z wszystkich klas)
            num_classes = max([DATASET_INFO[ds]['num_classes'] for ds in split['train_datasets'] + [test_dataset_name, val_dataset_name]])
            
            print(f"\n  Podział {split_id}:")
            print(f"    Train: {', '.join(split['train_datasets'])}")
            print(f"    Test: {test_dataset_name}")
            print(f"    Val: {val_dataset_name}")
            
            # Załaduj model
            model = load_trained_model(
                model_name, model_path, target_bands, num_classes, 
                patch_size, device
            )
            
            # Testuj na zbiorze testowym
            test_patches = split['test_patches']
            test_labels = split['test_labels']
            test_dataset = TestDataset(test_patches, test_labels, patch_size, model_type)
            
            test_accuracy, test_predictions, test_true_labels = test_model_on_dataset(
                model, test_dataset, device, batch_size
            )
            
            # Waliduj na zbiorze walidacyjnym
            val_patches = split['val_patches']
            val_labels = split['val_labels']
            val_dataset = TestDataset(val_patches, val_labels, patch_size, model_type)
            
            val_accuracy, val_predictions, val_true_labels = test_model_on_dataset(
                model, val_dataset, device, batch_size
            )
            
            split_result = {
                'split_id': split_id,
                'train_datasets': split['train_datasets'],
                'test_dataset': test_dataset_name,
                'val_dataset': val_dataset_name,
                'test_accuracy': test_accuracy,
                'val_accuracy': val_accuracy,
                'test_n_samples': len(test_labels),
                'val_n_samples': len(val_labels)
            }
            
            model_results.append(split_result)
            
            print(f"    Test accuracy: {test_accuracy:.2f}% ({len(test_labels)} próbek)")
            print(f"    Val accuracy: {val_accuracy:.2f}% ({len(val_labels)} próbek)")
        
        results[model_name] = model_results
    
    return results


def train_and_test_models_on_splits(model_names, models_dict, model_types_dict, 
                                    dataset_splits, target_bands, patch_size=8,
                                    epochs=50, batch_size=16, lr=0.001, val_split=0.2,
                                    device=None):
    """
    Trenuje i testuje modele na każdym podziale datasetów
    Podobnie jak run_cross_dataset_training.py
    
    Args:
        model_names: lista nazw modeli
        models_dict: słownik {model_name: model_class}
        model_types_dict: słownik {model_name: model_type}
        dataset_splits: lista słowników z informacjami o podziałach (train/test/val)
        target_bands: docelowa liczba pasm
        patch_size: rozmiar patchy
        epochs: liczba epok
        batch_size: rozmiar batcha
        lr: learning rate
        val_split: validation split ratio (używany do podziału train na train/val)
        device: urządzenie
    
    Returns:
        results: słownik z wynikami
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import funkcji treningowej
    train_script_path = os.path.join(os.path.dirname(__file__), '..', 'train', 'train.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train = train_module.train
    
    # Import DATASET_INFO
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from utils.load_data import DATASET_INFO
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        model_class = models_dict[model_name]
        model_type = model_types_dict[model_name]
        model_results = []
        
        # Dla każdego podziału
        for split in dataset_splits:
            split_id = split['split_id']
            train_datasets = split['train_datasets']
            test_dataset_name = split['test_dataset']
            val_dataset_name = split['val_dataset']
            
            train_datasets_str = "+".join(train_datasets)
            dataset_name_str = f"Split{split_id}_Train_{train_datasets_str}"
            
            print(f"\n  Podział {split_id}:")
            print(f"    Train: {', '.join(train_datasets)}")
            print(f"    Test: {test_dataset_name}")
            print(f"    Val: {val_dataset_name}")
            
            # Pobierz num_classes (max z wszystkich datasetów)
            num_classes = max([DATASET_INFO[ds]['num_classes'] for ds in train_datasets + [test_dataset_name, val_dataset_name]])
            
            # Utwórz model
            if model_name == 'InceptionHSINet':
                model = model_class(in_channels=1, num_classes=num_classes)
            elif model_name == 'SimpleHSINet':
                model = model_class(input_channels=target_bands, num_classes=num_classes)
            elif model_name == 'CNNFromDiagram':
                model = model_class(input_channels=target_bands, num_classes=num_classes, patch_size=patch_size)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Przygotuj dane treningowe
            train_patches = split['train_patches']
            train_labels = split['train_labels']
            
            # Podziel train na train/val
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(train_patches))
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_split,
                random_state=42,
                stratify=train_labels
            )
            
            train_patches_split = train_patches[train_indices]
            train_labels_split = train_labels[train_indices]  # TestDataset sam odejmie 1
            val_patches_split = train_patches[val_indices]
            val_labels_split = train_labels[val_indices]
            
            # Utwórz dataloadery
            train_dataset_obj = TestDataset(train_patches_split, train_labels_split + 1, patch_size, model_type)
            val_dataset_obj = TestDataset(val_patches_split, val_labels_split + 1, patch_size, model_type)
            
            train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False)
            
            # Trenuj model
            print(f"    Trening...")
            trained_model = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                model_name=model_name,
                dataset_name=dataset_name_str
            )
            
            # Testuj na zbiorze testowym
            test_patches = split['test_patches']
            test_labels = split['test_labels']
            test_dataset_obj = TestDataset(test_patches, test_labels, patch_size, model_type)
            
            test_accuracy, test_predictions, test_true_labels = test_model_on_dataset(
                trained_model, test_dataset_obj, device, batch_size
            )
            
            # Waliduj na zbiorze walidacyjnym
            val_patches = split['val_patches']
            val_labels = split['val_labels']
            val_dataset_obj = TestDataset(val_patches, val_labels, patch_size, model_type)
            
            val_accuracy, val_predictions, val_true_labels = test_model_on_dataset(
                trained_model, val_dataset_obj, device, batch_size
            )
            
            split_result = {
                'split_id': split_id,
                'train_datasets': train_datasets,
                'test_dataset': test_dataset_name,
                'val_dataset': val_dataset_name,
                'test_accuracy': test_accuracy,
                'val_accuracy': val_accuracy,
                'test_n_samples': len(test_labels),
                'val_n_samples': len(val_labels)
            }
            
            model_results.append(split_result)
            
            print(f"    Test accuracy: {test_accuracy:.2f}% ({len(test_labels)} próbek)")
            print(f"    Val accuracy: {val_accuracy:.2f}% ({len(val_labels)} próbek)")
        
        results[model_name] = model_results
    
    return results

