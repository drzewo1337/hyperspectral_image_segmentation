"""
Krok 2: Preprocessing - redukcja wymiarów do 10/30/60/90 kanałów (Gauss)
Redukcja wymiarów używając filtra Gaussa
"""
import sys
import os
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import normalize

# Docelowe liczby kanałów
TARGET_BANDS = [10, 20, 30]

def gaussian_band_reduction(data, target_bands, sigma=1.0):
    """
    Redukuje liczbę kanałów używając filtra Gaussa
    
    Args:
        data: numpy array o shape (H, W, B) gdzie B to liczba kanałów
        target_bands: docelowa liczba kanałów
        sigma: parametr sigma dla filtra Gaussa (domyślnie 1.0)
    
    Returns:
        data_reduced: numpy array o shape (H, W, target_bands)
    """
    H, W, B = data.shape
    
    if B <= target_bands:
        # Jeśli mniej kanałów niż docelowa liczba, zwróć oryginalne dane
        # (można dodać padding zerami, ale lepiej zachować oryginalne)
        return data
    
    # Oblicz krok próbkowania
    step = B / target_bands
    
    # Indeksy kanałów do wyboru
    indices = np.round(np.arange(0, B, step)).astype(int)
    indices = indices[:target_bands]  # Upewnij się, że nie przekraczamy
    
    # Wybierz kanały
    selected_bands = data[:, :, indices]
    
    # Zastosuj filtr Gaussa wzdłuż osi kanałów dla wygładzenia
    # Filtrujemy każdy kanał osobno w przestrzeni 2D (H, W)
    filtered_bands = np.zeros_like(selected_bands)
    for i in range(target_bands):
        filtered_bands[:, :, i] = ndimage.gaussian_filter(selected_bands[:, :, i], sigma=sigma)
    
    return filtered_bands

def preprocess_datasets(datasets, target_bands_list=TARGET_BANDS):
    """
    Preprocessing wszystkich datasetów dla różnych liczb kanałów
    
    Args:
        datasets: dict z danymi z kroku 1
        target_bands_list: lista docelowych liczb kanałów
    
    Returns:
        preprocessed: dict z kluczami (dataset_name, target_bands) -> (data, labels, info)
    """
    print("=" * 80)
    print("KROK 2: Preprocessing - redukcja wymiarów (Gauss)")
    print("=" * 80)
    
    preprocessed = {}
    
    for dataset_name, dataset_data in datasets.items():
        data = dataset_data['data']
        labels = dataset_data['labels']
        info = dataset_data['info']
        
        # Normalizacja
        print(f"\nPreprocessing {dataset_name}...")
        data_normalized = normalize(data)
        original_bands = data.shape[2]
        
        for target_bands in target_bands_list:
            if target_bands >= original_bands:
                # Jeśli docelowa liczba kanałów >= oryginalna, użyj oryginalnej
                data_reduced = data_normalized
                print(f"  {target_bands} kanałów: {original_bands} (oryginalne, bez redukcji)")
            else:
                # Redukcja przez filtr Gaussa
                data_reduced = gaussian_band_reduction(data_normalized, target_bands, sigma=1.0)
                print(f"  {target_bands} kanałów: {original_bands} -> {target_bands} (Gauss)")
            
            key = (dataset_name, target_bands)
            preprocessed[key] = {
                'data': data_reduced,
                'labels': labels,
                'info': {
                    **info,
                    'num_bands': data_reduced.shape[2],
                    'original_bands': original_bands
                }
            }
    
    print(f"\n✓ Preprocessing zakończony dla {len(target_bands_list)} konfiguracji kanałów")
    return preprocessed

if __name__ == "__main__":
    # Test - załaduj dane i przetestuj preprocessing
    import importlib.util
    import sys
    load_data_path = os.path.join(os.path.dirname(__file__), '1_load_data.py')
    spec = importlib.util.spec_from_file_location("load_data", load_data_path)
    load_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_data_module)
    datasets = load_data_module.load_all_datasets()
    preprocessed = preprocess_datasets(datasets)
    print(f"\nGotowe! Przetworzono {len(preprocessed)} kombinacji dataset-kanały")

