"""
Meta-learning: Trening modelu na różnych obrazach z różnymi liczbami klas
Model uczy się ekstrahować cechy (embedding), nie klasyfikować do konkretnych klas
Dzięki temu może segmentować nowe obrazy bez znajomości liczby klas
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import pad_with_zeros, DATASET_INFO
from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram

# Mapowanie nazw modeli do klas
MODELS = {
    'InceptionHSINet': InceptionHSINet,
    'SimpleHSINet': SimpleHSINet,
    'CNNFromDiagram': CNNFromDiagram
}

class FlexibleHSIDataset(Dataset):
    """
    Dataset który obsługuje różne liczby klas dla różnych obrazów
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
        dataset_info_list = []  # Zapisz info o każdym datasetcie
        
        # Znajdź wspólną liczbę kanałów
        target_bands_list = [data_dict[k]['data'].shape[2] for k in data_dict.keys() if k[0] in dataset_names]
        if not target_bands_list:
            raise ValueError(f"Brak danych dla datasetów: {dataset_names}")
        
        target_bands = target_bands_list[0]
        
        for dataset_name in dataset_names:
            dataset_keys = [k for k in data_dict.keys() if k[0] == dataset_name and data_dict[k]['data'].shape[2] == target_bands]
            if not dataset_keys:
                dataset_keys = [k for k in data_dict.keys() if k[0] == dataset_name]
                if not dataset_keys:
                    continue
            
            key = dataset_keys[0]
            data = data_dict[key]['data']
            labels = data_dict[key]['labels']
            info = data_dict[key]['info']
            
            # Padding
            margin = patch_size // 2
            padded_data = pad_with_zeros(data, margin)
            
            # Ekstrakcja patchy
            h, w, _ = data.shape
            dataset_patches = []
            dataset_targets = []
            
            for i in range(h):
                for j in range(w):
                    label = labels[i, j]
                    if label == 0:
                        continue
                    patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                    dataset_patches.append(patch)
                    dataset_targets.append(label - 1)  # 0-based
            
            # Zapisz info o datasetcie (dla każdego patchy)
            for _ in range(len(dataset_patches)):
                dataset_info_list.append({
                    'dataset_name': dataset_name,
                    'num_classes': info['num_classes'],
                    'max_class': info['num_classes'] - 1  # 0-based max class
                })
            
            patches_list.extend(dataset_patches)
            targets_list.extend(dataset_targets)
        
        if len(patches_list) == 0:
            raise ValueError(f"Brak danych dla datasetów: {dataset_names}")
        
        self.patches = np.array(patches_list)
        self.targets = np.array(targets_list)
        self.dataset_info = dataset_info_list
        
        # Upewnij się, że wszystkie patchy mają tę samą liczbę kanałów
        if len(self.patches.shape) == 4:
            current_bands = self.patches.shape[-1]
            if current_bands != target_bands:
                if current_bands < target_bands:
                    padding = np.zeros((self.patches.shape[0], self.patches.shape[1], self.patches.shape[2], target_bands - current_bands))
                    self.patches = np.concatenate([self.patches, padding], axis=-1)
                else:
                    self.patches = self.patches[:, :, :, :target_bands]
        
        # Przygotuj dane w odpowiednim formacie
        if model_type == '3d':
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))
            self.patches = np.expand_dims(self.patches, axis=1)
        else:
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))
        
        self.patches = torch.from_numpy(self.patches).float()
        self.targets = torch.from_numpy(self.targets).long()
        
        self.num_bands = self.patches.shape[1] if model_type == '2d' else self.patches.shape[2]
        
        print(f"FlexibleHSIDataset: {len(self)} samples from {dataset_names}")
        print(f"  Bands: {self.num_bands}, Different num_classes per dataset")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.targets[idx], self.dataset_info[idx]

def train_meta_learning(model, train_loader, val_loader, epochs=50, lr=0.001, device=None):
    """
    Trening meta-learning: model uczy się na różnych obrazach z różnymi liczbami klas
    
    Strategia:
    1. Model ekstrahuje cechy (embedding) - wspólne dla wszystkich obrazów
    2. Dla każdego obrazu tworzymy dynamiczny klasyfikator (tylko dla treningu)
    3. Model uczy się, że podobne obiekty mają podobne embeddingi
    4. Po treningu używamy tylko extract_features() + DBSCAN
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Użyjemy tylko feature extraction - nie klasyfikacji!
    # Cel: nauczyć model ekstrahować dobre cechy
    criterion = nn.MSELoss()  # Będziemy używać contrastive learning
    
    print(f"Meta-learning training: {epochs} epochs")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Grupuj batchy według datasetu (te same num_classes)
        batch_by_dataset = {}
        for batch_idx, (inputs, labels, dataset_info) in enumerate(train_loader):
            dataset_name = dataset_info[0]['dataset_name']
            if dataset_name not in batch_by_dataset:
                batch_by_dataset[dataset_name] = []
            batch_by_dataset[dataset_name].append((inputs, labels, dataset_info))
        
        # Trenuj na każdym datasetcie osobno
        for dataset_name, batches in batch_by_dataset.items():
            for inputs, labels, dataset_info in batches:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Ekstrahuj cechy
                features = model.extract_features(inputs)
                
                # Contrastive learning: podobne klasy → podobne embeddingi
                # Różne klasy → różne embeddingi
                optimizer.zero_grad()
                
                # Prosty loss: embeddingi tej samej klasy powinny być podobne
                # Uproszczona wersja - w praktyce używa się bardziej zaawansowanych lossów
                unique_classes = torch.unique(labels)
                loss = 0
                
                for cls in unique_classes:
                    cls_mask = labels == cls
                    if torch.sum(cls_mask) > 1:
                        cls_features = features[cls_mask]
                        # Średni embedding dla klasy
                        cls_center = cls_features.mean(dim=0, keepdim=True)
                        # Loss: embeddingi powinny być blisko centrum klasy
                        loss += torch.mean((cls_features - cls_center) ** 2)
                
                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
        # Validation - sprawdź jakość embeddingów
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, dataset_info in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    features = model.extract_features(inputs)
                    
                    # Prosty validation loss
                    unique_classes = torch.unique(labels)
                    for cls in unique_classes:
                        cls_mask = labels == cls
                        if torch.sum(cls_mask) > 1:
                            cls_features = features[cls_mask]
                            cls_center = cls_features.mean(dim=0, keepdim=True)
                            val_loss += torch.mean((cls_features - cls_center) ** 2).item()
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model

def train_meta_learning_simple(model, train_loader, val_loader, epochs=50, lr=0.001, device=None):
    """
    Prostsza wersja: trenuj model na wszystkich datasetach jednocześnie
    Używa maksymalnej liczby klas, ale model uczy się tylko ekstrahować cechy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Znajdź maksymalną liczbę klas
    max_classes = 0
    for _, _, dataset_info in train_loader:
        max_classes = max(max_classes, dataset_info[0]['num_classes'])
        break
    
    # Utwórz dynamiczny klasyfikator (tylko dla treningu)
    # W praktyce użyjemy tylko extract_features()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Meta-learning training (simple): {epochs} epochs, max_classes={max_classes}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels, dataset_info in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Ekstrahuj cechy
            features = model.extract_features(inputs)
            
            # Dla każdego batcha, utwórz dynamiczny klasyfikator
            # (w praktyce używamy tylko cech, ale dla treningu potrzebujemy loss)
            # Uproszczona wersja: użyj embeddingów bezpośrednio
            # Normalizuj embeddingi
            features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            
            # Prosty loss: embeddingi tej samej klasy powinny być podobne
            optimizer.zero_grad()
            
            unique_classes = torch.unique(labels)
            loss = 0
            
            for cls in unique_classes:
                cls_mask = labels == cls
                if torch.sum(cls_mask) > 1:
                    cls_features = features_norm[cls_mask]
                    cls_center = cls_features.mean(dim=0, keepdim=True)
                    # Loss: embeddingi powinny być blisko centrum klasy
                    loss += torch.mean((cls_features - cls_center) ** 2)
            
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
    
    return model

if __name__ == "__main__":
    print("Ten skrypt powinien być wywoływany z głównego flow scriptu")



