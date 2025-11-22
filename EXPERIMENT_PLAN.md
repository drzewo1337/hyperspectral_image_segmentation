# Plan Eksperymentów - Szczegółowa Struktura

## Parametry Eksperymentów

### 1. Liczby pasm (redukcja przez filtr Gaussa)
- 103 pasma (minimalna wartość - PaviaU)
- 90 pasm
- 60 pasm  
- 30 pasm

### 2. Opcje redukcji wymiarów
- Bez PCA (tylko filtr Gaussa)
- Z PCA (różne liczby komponentów dla każdej liczby pasm)

### 3. Modele
- InceptionHSINet (3D CNN)
- SimpleHSINet (2D CNN)
- CNNFromDiagram (2D CNN)
- RNN/LSTM (nowy - do implementacji pozniej)
- MLP (nowy - do implementacji pozniej)

### 4. Podział datasetów
- **Train**: Indian + PaviaU + KSC (3 datasety)
- **Val**: 20% z train (split z połączonych danych)
- **Test**: Salinas (1 dataset)
- **Hold-out**: PaviaC (1 dataset - finalna ocena)

## Struktura Eksperymentów

### Wariant A: Wszystko w jednym skrypcie (40 eksperymentów)
```
4 liczby pasm × 2 opcje PCA × 5 modeli = 40 eksperymentów
```

**Zalety:**
- Jeden skrypt do zarządzania
- Łatwe porównanie wyników
- Automatyczne zapisywanie wszystkich wyników

**Wady:**
- Długi czas wykonania (40 × ~30 min = 20 godzin)
- Trudne debugowanie jeśli jeden eksperyment się zepsuje
- Wszystko w pamięci jednocześnie

### Wariant B: Podział na grupy (REKOMENDOWANY)
```
Grupa 1: Testy bez PCA (4 liczby pasm × 5 modeli = 20 eksperymentów)
Grupa 2: Testy z PCA (4 liczby pasm × 5 modeli = 20 eksperymentów)
```

**Zalety:**
- Możliwość uruchomienia grup osobno
- Łatwiejsze debugowanie
- Możliwość przerwania i wznowienia
- Lepsza organizacja wyników

**Wady:**
- Wymaga 2 skryptów (lub parametru w jednym)

### Wariant C: Podział per model (5 skryptów)
```
Dla każdego modelu: 4 liczby pasm × 2 opcje PCA = 8 eksperymentów
```

**Zalety:**
- Najłatwiejsze debugowanie
- Możliwość równoległego uruchomienia na różnych GPU
- Najlepsza organizacja

**Wady:**
- Więcej plików do zarządzania

## Rekomendacja: Wariant B (2 grupy)

### Struktura nazewnictwa wyników:
```
best_model_{model_name}_bands{num_bands}_pca{yes/no}_train{train_datasets}.pth
training_log_{model_name}_bands{num_bands}_pca{yes/no}_train{train_datasets}.csv
```

Przykłady:
- `best_model_InceptionHSINet_bands103_nopca_train_Indian+PaviaU+KSC.pth`
- `best_model_SimpleHSINet_bands60_pca30_train_Indian+PaviaU+KSC.pth`

### Organizacja folderów wyników:
```
results/
├── models/
│   ├── bands103/
│   │   ├── nopca/
│   │   └── pca/
│   ├── bands90/
│   ├── bands60/
│   └── bands30/
├── logs/
│   └── (taka sama struktura)
└── metrics/
    └── cross_dataset_comparison.csv
```

## Implementacja

### Skrypt główny: `run_cross_dataset_experiments.py`

Parametry:
- `--bands`: 103, 90, 60, 30
- `--use_pca`: True/False
- `--pca_components`: liczba komponentów (jeśli use_pca=True)
- `--models`: lista modeli lub "all"
- `--gaussian_sigma`: parametr filtra Gaussa

### Przykład użycia:
```bash
# Wszystkie eksperymenty bez PCA
python run_cross_dataset_experiments.py --bands all --use_pca False

# Wszystkie eksperymenty z PCA
python run_cross_dataset_experiments.py --bands all --use_pca True --pca_components 30

# Jeden konkretny eksperyment
python run_cross_dataset_experiments.py --bands 60 --use_pca True --pca_components 30 --models SimpleHSINet
```

## Kolejność wykonania

1. **Faza 1**: Testy bez PCA (20 eksperymentów)
   - 4 liczby pasm × 5 modeli
   
2. **Faza 2**: Testy z PCA (20 eksperymentów)
   - 4 liczby pasm × 5 modeli
   - PCA components = docelowa liczba pasm (103, 90, 60, 30)

3. **Faza 3**: Analiza i porównanie wyników
   - Agregacja metryk
   - Wizualizacje
   - Raport końcowy

## Czas wykonania (szacunkowy)

- 1 eksperyment: ~30-45 minut (50 epok, batch_size=16)
- 20 eksperymentów (grupa): ~10-15 godzin
- Wszystkie 40 eksperymentów: ~20-30 godzin

**Rekomendacja**: Uruchamiaj grupy osobno, możesz przerwać i wznowić.

