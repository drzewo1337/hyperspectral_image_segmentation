# Flowchart - Przepływ Danych w Nowym Flow

## Ogólny Przepływ

```
START
  ↓
1. WCZYTYWANIE DANYCH
  ↓
2. PREPROCESSING - Redukcja wymiarów (Gauss)
  ↓
3. TWORZENIE PODZIAŁÓW DATASETÓW (5 podziałów)
  ↓
4. TRENING I TESTOWANIE MODELI (dla każdego podziału)
  ↓
5. WALIDACJA WYNIKÓW
  ↓
END
```

## Szczegółowy Przepływ Danych

### KROK 1-2: Wczytanie Danych i Preprocessing

**Wejście:**
- 5 datasetów z `data/raw/`: Indian, PaviaU, KSC, Salinas, PaviaC
- Parametr: `target_bands` (10, 30, 60, 90)

**Proces:**
1. Dla każdego datasetu:
   - `load_data(dataset_name)` → ładuje `.mat` pliki
   - Zwraca: `data` (H, W, B), `labels` (H, W)

2. Preprocessing:
   - `preprocess_data(data, target_bands)`:
     - Normalizacja: `StandardScaler()` na wszystkich pikselach
     - Redukcja wymiarów: `gaussian_band_reduction(data, target_bands)`
       - Filtr Gaussa wzdłuż osi pasm
       - Próbkowanie do docelowej liczby pasm
   - Zwraca: `data_processed` (H, W, target_bands)

**Wyjście:**
- Przetworzone dane dla każdego datasetu z redukcją do `target_bands` pasm

---

### KROK 3: Tworzenie Podziałów Datasetów

**Wejście:**
- Przetworzone dane z KROKU 1-2
- Parametr: `n_splits = 5`

**Proces:**
1. Dla każdego z 5 podziałów:
   - **Podział 1**: Train: Indian+PaviaU+KSC, Test: Salinas, Val: PaviaC
   - **Podział 2**: Train: Indian+PaviaU+Salinas, Test: KSC, Val: PaviaC
   - **Podział 3**: Train: Indian+KSC+Salinas, Test: PaviaU, Val: PaviaC
   - **Podział 4**: Train: PaviaU+KSC+Salinas, Test: Indian, Val: PaviaC
   - **Podział 5**: Train: Indian+PaviaU+KSC, Test: PaviaC, Val: Salinas

2. Dla każdego podziału:
   - Ekstrakcja patchy z train datasetów:
     - `extract_patches_from_dataset()` dla każdego train datasetu
     - Padding: `pad_with_zeros(data, margin)` gdzie `margin = patch_size // 2`
     - Ekstrakcja: dla każdego piksela (i,j) z label != 0:
       - Patch: `padded_data[i:i+patch_size, j:j+patch_size, :]`
       - Label: `labels[i, j]`
     - Łączenie patchy z wszystkich train datasetów
   
   - Ekstrakcja patchy z test datasetu
   - Ekstrakcja patchy z val datasetu

**Wyjście:**
- Lista 5 słowników, każdy zawiera:
  ```python
  {
    'split_id': 1-5,
    'train_datasets': ['Indian', 'PaviaU', 'KSC'],
    'test_dataset': 'Salinas',
    'val_dataset': 'PaviaC',
    'train_patches': (N_train, H, W, target_bands),
    'train_labels': (N_train,),
    'test_patches': (N_test, H, W, target_bands),
    'test_labels': (N_test,),
    'val_patches': (N_val, H, W, target_bands),
    'val_labels': (N_val,)
  }
  ```

---

### KROK 4: Trening i Testowanie Modeli

**Wejście:**
- Podziały datasetów z KROKU 3
- Modele: InceptionHSINet, SimpleHSINet, CNNFromDiagram
- Parametry: epochs, batch_size, lr, val_split

**Proces dla każdego modelu i każdego podziału:**

1. **Przygotowanie danych:**
   - Podział train na train/val:
     - `train_test_split(train_patches, train_labels, test_size=val_split)`
     - Train: 80% patchy
     - Val: 20% patchy
   
   - Konwersja do formatu modelu:
     - `TestDataset(patches, labels, patch_size, model_type)`
     - Dla 2D: `(N, B, H, W)` - pasma jako channels
     - Dla 3D: `(N, 1, B, H, W)` - pasma jako depth dimension
     - Konwersja do torch tensors

2. **Utworzenie modelu:**
   - `InceptionHSINet`: `in_channels=1, num_classes=num_classes`
   - `SimpleHSINet`: `input_channels=target_bands, num_classes=num_classes`
   - `CNNFromDiagram`: `input_channels=target_bands, num_classes=num_classes, patch_size=patch_size`

3. **Trening:**
   - `train(model, train_loader, val_loader, epochs, lr, device)`
   - Dla każdej epoki:
     - Forward pass na train
     - Backward pass (loss.backward())
     - Update wag (optimizer.step())
     - Ewaluacja na val
     - Zapis najlepszego modelu (najlepsza val accuracy)
   - Zapis modelu: `best_model_{model_name}_Split{id}_Train_{datasets}.pth`
   - Zapis logu: `training_log_{model_name}_Split{id}_Train_{datasets}.csv`

4. **Testowanie:**
   - Załadowanie najlepszego modelu
   - Test na test dataset:
     - `test_model_on_dataset(model, test_dataset, device, batch_size)`
     - Obliczenie accuracy: `correct / total * 100`
   - Walidacja na val dataset:
     - `test_model_on_dataset(model, val_dataset, device, batch_size)`
     - Obliczenie accuracy

**Wyjście:**
- Słownik wyników:
  ```python
  {
    'InceptionHSINet': [
      {
        'split_id': 1,
        'train_datasets': ['Indian', 'PaviaU', 'KSC'],
        'test_dataset': 'Salinas',
        'val_dataset': 'PaviaC',
        'test_accuracy': 85.23,
        'val_accuracy': 82.15,
        'test_n_samples': 54129,
        'val_n_samples': 148152
      },
      ... (dla każdego podziału)
    ],
    'SimpleHSINet': [...],
    'CNNFromDiagram': [...]
  }
  ```

---

### KROK 5: Walidacja Danych

**Wejście:**
- Wyniki testów z KROKU 4

**Proces:**
1. `validate_results(test_results, min_accuracy, max_accuracy)`:
   - Dla każdego modelu:
     - Zbieranie wszystkich accuracy (test i val)
     - Sprawdzanie czy accuracy w zakresie [min, max]
     - Obliczanie statystyk:
       - Średnia accuracy
       - Odchylenie standardowe
       - Min/Max accuracy
       - Liczba testów

2. Generowanie raportu:
   - Status: VALID/INVALID
   - Lista błędów (accuracy < min)
   - Lista ostrzeżeń (accuracy > max)
   - Podsumowanie statystyk dla każdego modelu

3. Zapis raportu:
   - `validation_report_{target_bands}bands.json`
   - `all_test_results.json`

**Wyjście:**
- Raport walidacji w formacie JSON
- Wszystkie wyniki w `all_test_results.json`

---

## Struktura Plików Wyjściowych

```
results/new_flow/
├── all_test_results.json              # Wszystkie wyniki
├── validation_report_10bands.json     # Raport dla 10 pasm
├── validation_report_30bands.json     # Raport dla 30 pasm
├── validation_report_60bands.json     # Raport dla 60 pasm
└── validation_report_90bands.json     # Raport dla 90 pasm

results/models/
├── best_model_InceptionHSINet_Split1_Train_Indian+PaviaU+KSC.pth
├── best_model_SimpleHSINet_Split1_Train_Indian+PaviaU+KSC.pth
├── best_model_CNNFromDiagram_Split1_Train_Indian+PaviaU+KSC.pth
└── ... (dla każdego modelu i każdego podziału)

results/logs/
├── training_log_InceptionHSINet_Split1_Train_Indian+PaviaU+KSC.csv
├── training_log_SimpleHSINet_Split1_Train_Indian+PaviaU+KSC.csv
└── ... (dla każdego modelu i każdego podziału)
```

---

## Parametry Konfiguracyjne

- **target_bands**: [10, 30, 60, 90] - docelowe liczby pasm po redukcji
- **n_splits**: 5 - liczba różnych podziałów datasetów
- **patch_size**: 8 - rozmiar patchy
- **epochs**: 50 - liczba epok treningu
- **batch_size**: 16 - rozmiar batcha
- **lr**: 0.001 - learning rate
- **val_split**: 0.2 - proporcja danych walidacyjnych (20% z train)

---

## Kolejność Wykonania

1. **Dla każdej liczby pasm** (10, 30, 60, 90):
   - KROK 1-2: Wczytanie i preprocessing wszystkich 5 datasetów
   - KROK 3: Utworzenie 5 podziałów
   - KROK 4: Dla każdego modelu (3 modele):
     - Dla każdego podziału (5 podziałów):
       - Trening modelu
       - Test na test dataset
       - Walidacja na val dataset
   - KROK 5: Walidacja wyników

**Łącznie:** 4 liczby pasm × 3 modele × 5 podziałów = **60 eksperymentów treningowych**

