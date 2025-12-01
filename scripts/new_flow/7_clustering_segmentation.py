"""
Segmentacja przez clustering
Używa feature extraction + clustering (DBSCAN/K-Means) do segmentacji nowych obrazów
"""
import sys
import os
import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import pad_with_zeros

def extract_features_for_image(model, data, labels, patch_size=16, model_type='2d', device=None, batch_size=64):
    """
    Ekstrahuje embeddingi (cechy) dla wszystkich pikseli obrazu
    
    Args:
        model: wytrenowany model z metodą extract_features()
        data: obraz hiperspektralny (H, W, B)
        labels: ground truth labels (H, W) - używane tylko do określenia obszaru
        patch_size: rozmiar patchy
        model_type: '2d' lub '3d'
        device: urządzenie (cuda/cpu)
        batch_size: rozmiar batcha do przetwarzania
    
    Returns:
        embeddings: numpy array (H, W, feature_dim) - embedding dla każdego piksela
        pixel_coords: lista (i, j) współrzędnych pikseli z danymi (nie tło)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model = model.to(device)
    
    # Padding
    margin = patch_size // 2
    padded_data = pad_with_zeros(data, margin)
    
    h, w, _ = data.shape
    
    # Zbierz wszystkie patchy i współrzędne
    patches = []
    pixel_coords = []
    
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:  # Tylko piksele z danymi (nie tło)
                patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                patches.append(patch)
                pixel_coords.append((i, j))
    
    if len(patches) == 0:
        raise ValueError("Brak pikseli z danymi w obrazie")
    
    patches = np.array(patches)
    
    # Przygotuj patchy w odpowiednim formacie
    if model_type == '3d':
        # Conv3D: (N, 1, B, H, W)
        patches_tensor = np.transpose(patches, (0, 3, 1, 2))  # (N, B, H, W)
        patches_tensor = np.expand_dims(patches_tensor, axis=1)  # (N, 1, B, H, W)
    else:
        # Conv2D: (N, B, H, W)
        patches_tensor = np.transpose(patches, (0, 3, 1, 2))  # (N, B, H, W)
    
    patches_tensor = torch.from_numpy(patches_tensor).float().to(device)
    
    # Ekstrahuj embeddingi w batchach
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size]
            batch_embeddings = model.extract_features(batch)
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Utwórz mapę embeddingów (H, W, feature_dim)
    feature_dim = embeddings.shape[1]
    embedding_map = np.zeros((h, w, feature_dim))
    
    for idx, (i, j) in enumerate(pixel_coords):
        embedding_map[i, j] = embeddings[idx]
    
    return embedding_map, pixel_coords

def segment_with_dbscan(model, data, labels, patch_size=16, model_type='2d', 
                       eps=0.5, min_samples=5, device=None, batch_size=64):
    """
    Segmentacja używając DBSCAN (automatyczna liczba klas)
    
    Args:
        model: wytrenowany model z metodą extract_features()
        data: obraz hiperspektralny (H, W, B)
        labels: ground truth labels (H, W)
        patch_size: rozmiar patchy
        model_type: '2d' lub '3d'
        eps: parametr DBSCAN - maksymalna odległość między pikselami w klastrze
        min_samples: parametr DBSCAN - minimalna liczba pikseli w klastrze
        device: urządzenie
        batch_size: rozmiar batcha
    
    Returns:
        segmentation_map: mapa segmentacji (H, W) z przypisanymi grupami
        n_clusters: liczba znalezionych klastrów
    """
    print(f"Segmentacja DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    # Ekstrahuj embeddingi
    embedding_map, pixel_coords = extract_features_for_image(
        model, data, labels, patch_size, model_type, device, batch_size
    )
    
    # Przygotuj dane do clusteringu
    embeddings_list = []
    coords_list = []
    
    for i, j in pixel_coords:
        embeddings_list.append(embedding_map[i, j])
        coords_list.append((i, j))
    
    embeddings_array = np.array(embeddings_list)
    
    # Normalizacja embeddingów
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_array)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings_normalized)
    
    # Utwórz mapę segmentacji
    h, w = data.shape[:2]
    segmentation_map = np.zeros((h, w), dtype=np.int32)
    
    for idx, (i, j) in enumerate(coords_list):
        cluster_id = cluster_labels[idx]
        if cluster_id >= 0:  # -1 to outliers w DBSCAN
            segmentation_map[i, j] = cluster_id + 1  # +1 bo 0 to tło
        else:
            segmentation_map[i, j] = 0  # Outliers jako tło
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = np.sum(cluster_labels == -1)
    
    print(f"  Znaleziono {n_clusters} klastrów, {n_outliers} outliers")
    
    return segmentation_map, n_clusters

def segment_with_kmeans(model, data, labels, patch_size=16, model_type='2d',
                       n_clusters=5, device=None, batch_size=64):
    """
    Segmentacja używając K-Means (wymaga podania liczby klas)
    
    Args:
        model: wytrenowany model z metodą extract_features()
        data: obraz hiperspektralny (H, W, B)
        labels: ground truth labels (H, W)
        patch_size: rozmiar patchy
        model_type: '2d' lub '3d'
        n_clusters: liczba klastrów (klas)
        device: urządzenie
        batch_size: rozmiar batcha
    
    Returns:
        segmentation_map: mapa segmentacji (H, W) z przypisanymi grupami
    """
    print(f"Segmentacja K-Means (n_clusters={n_clusters})...")
    
    # Ekstrahuj embeddingi
    embedding_map, pixel_coords = extract_features_for_image(
        model, data, labels, patch_size, model_type, device, batch_size
    )
    
    # Przygotuj dane do clusteringu
    embeddings_list = []
    coords_list = []
    
    for i, j in pixel_coords:
        embeddings_list.append(embedding_map[i, j])
        coords_list.append((i, j))
    
    embeddings_array = np.array(embeddings_list)
    
    # Normalizacja embeddingów
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_array)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_normalized)
    
    # Utwórz mapę segmentacji
    h, w = data.shape[:2]
    segmentation_map = np.zeros((h, w), dtype=np.int32)
    
    for idx, (i, j) in enumerate(coords_list):
        segmentation_map[i, j] = cluster_labels[idx] + 1  # +1 bo 0 to tło
    
    print(f"  Utworzono {n_clusters} segmentów")
    
    return segmentation_map

if __name__ == "__main__":
    print("Ten skrypt powinien być wywoływany z głównego flow scriptu")

