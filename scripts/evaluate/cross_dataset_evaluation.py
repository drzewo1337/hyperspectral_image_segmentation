import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.decomposition import PCA

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.load_data import load_data, normalize, pad_with_zeros, DATASET_INFO, get_loaders
import csv

class FlexibleHSIDataset(Dataset):

    def __init__(self, dataset_names, patch_size=16, model_type='2d', 
                 pca_components=None, normalize_per_dataset=True, target_bands=None):
        """
        Args:
            dataset_names: Lista nazw datasetów
            patch_size: Rozmiar patchy
            model_type: '2d' lub '3d'
            pca_components: Liczba komponentów PCA (None = bez PCA)
            normalize_per_dataset: Czy normalizować każdy dataset osobno
            target_bands: Docelowa liczba pasm (jeśli None, używa max_bands z datasetów)
        """
        self.dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        self.patch_size = patch_size
        self.model_type = model_type
        self.pca_components = pca_components
        self.normalize_per_dataset = normalize_per_dataset
        
        self.patches_list = []
        self.targets_list = []
        self.dataset_info = {}
        
        max_bands = 0
        all_num_classes = set()
        
        for dataset_name in self.dataset_names:
            info = DATASET_INFO[dataset_name]
            all_num_classes.add(info['num_classes'])
            max_bands = max(max_bands, info['num_bands'])
            self.dataset_info[dataset_name] = info
        
        # Jeśli target_bands jest podane, użyj go (dla test dataset)
        # W przeciwnym razie użyj max_bands (dla train dataset)
        if target_bands is not None:
            self.target_bands = target_bands
        elif pca_components is not None:
            self.target_bands = pca_components
        else:
            self.target_bands = max_bands
        
        self.pca_models = {}
        
        for dataset_name in self.dataset_names:
            data, labels = self._load_single_dataset(dataset_name)
            
            patches, targets = self._extract_patches(data, labels, dataset_name)
            
            # Dostosuj liczbę pasm do target_bands
            current_bands = patches.shape[-1]
            
            if pca_components is not None and current_bands != pca_components:
                patches, pca_model = self._apply_pca(patches, dataset_name)
                self.pca_models[dataset_name] = pca_model
            elif current_bands < self.target_bands:
                # Padding jeśli mniej pasm
                patches = self._pad_bands(patches, self.target_bands)
            elif current_bands > self.target_bands:
                # Redukcja jeśli więcej pasm (np. Salinas 204 -> 200)
                patches = patches[:, :, :, :self.target_bands]
                print(f"  {dataset_name}: Reduced bands {current_bands} -> {self.target_bands}")
            
            self.patches_list.append(patches)
            self.targets_list.append(targets)
        
        self.patches = np.concatenate(self.patches_list, axis=0)
        self.targets = np.concatenate(self.targets_list, axis=0)
        
        if model_type == '3d':
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)
            self.patches = np.expand_dims(self.patches, axis=1)  # (N, 1, B, H, W)
        else:
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)

        self.num_bands = self.patches.shape[1] if model_type == '2d' else self.patches.shape[2]
        self.num_classes = max(all_num_classes)
        
        print(f"FlexibleHSIDataset: {len(self)} samples from {self.dataset_names}")
        print(f"  Bands: {self.num_bands}, Max classes: {self.num_classes}")
        print(f"  Shape: {self.patches.shape}")
    
    def _load_single_dataset(self, dataset_name):
        data, labels = load_data(dataset_name)
        
        if self.normalize_per_dataset:
            data = normalize(data)
        
        return data, labels
    
    def _extract_patches(self, data, labels, dataset_name):
        margin = self.patch_size // 2
        padded_data = pad_with_zeros(data, margin)
        
        h, w, _ = data.shape
        patches = []
        targets = []
        
        for i in range(h):
            for j in range(w):
                label = labels[i, j]
                if label == 0:
                    continue
                patch = padded_data[i:i+self.patch_size, j:j+self.patch_size, :]
                patches.append(patch)
                targets.append(label - 1)
        
        return np.array(patches), np.array(targets)
    
    def _apply_pca(self, patches, dataset_name):
        N, H, W, B = patches.shape
        patches_flat = patches.reshape(N * H * W, B)
        
        pca = PCA(n_components=self.pca_components)
        patches_reduced = pca.fit_transform(patches_flat)
        patches_reduced = patches_reduced.reshape(N, H, W, self.pca_components)
        
        explained_var = sum(pca.explained_variance_ratio_)
        print(f"  {dataset_name} PCA: {B} -> {self.pca_components} bands (explained variance: {explained_var:.3f})")
        
        return patches_reduced, pca
    
    def _pad_bands(self, patches, target_bands):
        N, H, W, B = patches.shape
        if B < target_bands:
            padding = np.zeros((N, H, W, target_bands - B))
            patches = np.concatenate([patches, padding], axis=-1)
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)


def get_cross_dataset_loaders(train_datasets, test_datasets, patch_size=8, batch_size=16, 
                              val_split=0.2, model_type='2d', pca_components=None):
    train_dataset = FlexibleHSIDataset(
        train_datasets, patch_size=patch_size, model_type=model_type,
        pca_components=pca_components
    )
    
    # Pobierz target_bands z train dataset (używane przez model)
    # train_dataset.target_bands to liczba pasm użyta podczas tworzenia datasetu
    train_target_bands = train_dataset.target_bands
    
    test_loaders = {}
    test_info = {}
    
    for test_dataset_name in test_datasets:
        # WAŻNE: Użyj target_bands z train dataset, aby test dataset miał tę samą liczbę pasm
        test_dataset = FlexibleHSIDataset(
            [test_dataset_name], patch_size=patch_size, model_type=model_type,
            pca_components=pca_components, target_bands=train_target_bands
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[test_dataset_name] = test_loader
        test_info[test_dataset_name] = {
            'num_bands': test_dataset.num_bands,
            'num_classes': DATASET_INFO[test_dataset_name]['num_classes']
        }
    
    from torch.utils.data import random_split
    val_len = int(len(train_dataset) * val_split)
    train_len = len(train_dataset) - val_len
    train_set, val_set = random_split(train_dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loaders, {
        'train_info': {'num_bands': train_dataset.num_bands, 'num_classes': train_dataset.num_classes},
        'test_info': test_info
    }


def evaluate_on_test_datasets(model, test_loaders, device, model_name, train_datasets_str):
    results = {}
    
    model.eval()
    with torch.no_grad():
        for test_dataset_name, test_loader in test_loaders.items():
            correct = 0
            total = 0
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                pred = outputs.argmax(1)
                max_class = labels.max().item() + 1
                pred = torch.clamp(pred, 0, max_class - 1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            
            acc = 100.0 * correct / total if total > 0 else 0
            results[test_dataset_name] = acc
            
            print(f"  Test on {test_dataset_name}: {acc:.2f}% ({correct}/{total})")
    
    return results


def cross_dataset_experiment(model_class, model_name, train_datasets, test_datasets,
                            patch_size=16, batch_size=64, epochs=50, lr=0.001,
                            device=None, pca_components=None, model_type='2d'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_datasets_str = "+".join(train_datasets)
    test_datasets_str = "+".join(test_datasets)
    
    print(f"\n{'='*80}")
    print(f"Cross-Dataset Experiment: {model_name}")
    print(f"Train on: {train_datasets_str}")
    print(f"Test on: {test_datasets_str}")
    if pca_components:
        print(f"PCA: {pca_components} components")
    print(f"{'='*80}")
    
    # Get loaders
    train_loader, val_loader, test_loaders, info = get_cross_dataset_loaders(
        train_datasets, test_datasets, patch_size, batch_size, 
        model_type=model_type, pca_components=pca_components
    )
    
    # Create model
    num_bands = info['train_info']['num_bands']
    num_classes = info['train_info']['num_classes']
    
    # Create model with flexible parameters
    if model_name == 'InceptionHSINet':
        model = model_class(in_channels=1, num_classes=num_classes)
    elif model_name == 'SimpleHSINet':
        model = model_class(input_channels=num_bands, num_classes=num_classes)
    elif model_name == 'CNNFromDiagram':
        model = model_class(input_channels=num_bands, num_classes=num_classes, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model: {model_name}, Input: {num_bands} bands, {num_classes} classes")
    
    # Train - import funkcji treningowej
    train_script_path = os.path.join(os.path.dirname(__file__), '..', 'train', 'train.py')
    if not os.path.exists(train_script_path):
        raise FileNotFoundError(f"Training script not found: {train_script_path}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_func = train_module.train
    
    trained_model = train_func(
        model, train_loader, val_loader, epochs=epochs, lr=lr,
        device=device, model_name=model_name, 
        dataset_name=f"Train_{train_datasets_str}"
    )
    
    # Test on all test datasets
    test_results = evaluate_on_test_datasets(
        trained_model, test_loaders, device, model_name, train_datasets_str
    )
    
    # Save results
    result_entry = {
        'model': model_name,
        'train_datasets': train_datasets_str,
        'test_datasets': test_datasets_str,
        'pca_components': pca_components,
        'num_bands': num_bands,
        'num_classes': num_classes,
        **{f'test_acc_{ds}': acc for ds, acc in test_results.items()},
        'avg_test_acc': np.mean(list(test_results.values())) if test_results else 0
    }
    
    return result_entry, trained_model

