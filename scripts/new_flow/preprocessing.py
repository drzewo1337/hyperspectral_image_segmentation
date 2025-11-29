"""
Moduł do preprocessing danych hiperspektralnych z redukcją wymiarów metodą Gaussa
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler


def gaussian_band_reduction(data, target_bands):
    """
    Redukcja liczby kanałów/pasm metodą Gaussa
    
    Args:
        data: numpy array o kształcie (H, W, B) gdzie B to liczba pasm
        target_bands: docelowa liczba pasm (10, 30, 60, 90)
    
    Returns:
        data_reduced: numpy array o kształcie (H, W, target_bands)
    """
    h, w, b = data.shape
    
    if b == target_bands:
        return data
    
    if b < target_bands:
        # Jeśli mniej pasm niż docelowa liczba, zwróć dane bez zmian
        print(f"  Ostrzeżenie: Liczba pasm ({b}) jest mniejsza niż docelowa ({target_bands})")
        return data
    
    # Oblicz sigma dla filtra Gaussa
    # Sigma powinna być proporcjonalna do stosunku redukcji
    reduction_ratio = b / target_bands
    sigma = max(1.0, reduction_ratio / 2.0)
    
    # Zastosuj filtr Gaussa wzdłuż osi pasm
    data_reduced = gaussian_filter1d(data, sigma=sigma, axis=2)
    
    # Próbkowanie do docelowej liczby pasm
    indices = np.linspace(0, b - 1, target_bands).astype(int)
    data_reduced = data_reduced[:, :, indices]
    
    print(f"  Redukcja pasm: {b} -> {target_bands} (sigma={sigma:.2f})")
    
    return data_reduced


def preprocess_data(data, target_bands, normalize=True):
    """
    Pełny preprocessing danych: normalizacja + redukcja wymiarów Gaussa
    
    Args:
        data: numpy array o kształcie (H, W, B)
        target_bands: docelowa liczba pasm (10, 30, 60, 90)
        normalize: czy normalizować dane
    
    Returns:
        processed_data: przetworzone dane o kształcie (H, W, target_bands)
    """
    # Normalizacja
    if normalize:
        h, w, b = data.shape
        data_flat = data.reshape(-1, b)
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_flat)
        data = data_normalized.reshape(h, w, b)
    
    # Redukcja wymiarów metodą Gaussa
    data_reduced = gaussian_band_reduction(data, target_bands)
    
    return data_reduced

