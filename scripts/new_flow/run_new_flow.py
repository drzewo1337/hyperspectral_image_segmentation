"""
Główny skrypt implementujący nowy flow:
1. Wczytanie danych
2. Preprocessing - redukcja wymiarów do 10/30/60/90 kanałów (Gauss)
3. Stworzenie zbioru testowego na podstawie danych - 5 datasetów
4. Trening i testy 3 modeli dla każdego podziału
5. Walidacja danych
"""
import sys
import os
import torch
import argparse
import json
import numpy as np
from datetime import datetime

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from utils.load_data import DATASET_INFO
from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram
from test_datasets import create_test_datasets_from_multiple_sources
from test_models import train_and_test_models_on_splits
from validate_results import validate_results, print_validation_report, save_validation_report


# Konfiguracja
AVAILABLE_DATASETS = list(DATASET_INFO.keys())
MODEL_NAMES = ['InceptionHSINet', 'SimpleHSINet', 'CNNFromDiagram']
MODELS = {
    'InceptionHSINet': InceptionHSINet,
    'SimpleHSINet': SimpleHSINet,
    'CNNFromDiagram': CNNFromDiagram
}
MODEL_TYPES = {
    'InceptionHSINet': '3d',
    'SimpleHSINet': '2d',
    'CNNFromDiagram': '2d'
}
TARGET_BANDS = [10, 30, 60, 90]  # Możliwe wartości redukcji wymiarów
N_TEST_DATASETS = 5  # Liczba zbiorów testowych

# Parametry treningu
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2


def main():
    parser = argparse.ArgumentParser(description='Nowy flow: preprocessing, testy, walidacja')
    # Usunięto --datasets, bo używamy wszystkich 5 datasetów z różnymi podziałami
    parser.add_argument('--target_bands', type=int, nargs='+', default=TARGET_BANDS,
                       choices=TARGET_BANDS,
                       help='Lista docelowych liczb pasm (10, 30, 60, 90)')
    parser.add_argument('--models', type=str, nargs='+', default=MODEL_NAMES,
                       choices=MODEL_NAMES,
                       help='Lista modeli do testowania')
    parser.add_argument('--n_test_datasets', type=int, default=N_TEST_DATASETS,
                       help='Liczba zbiorów testowych do utworzenia')
    parser.add_argument('--patch_size', type=int, default=8,
                       help='Rozmiar patchy')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Rozmiar batcha')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Liczba epok treningu')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=VAL_SPLIT,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default=None,
                       help='Urządzenie (cuda/cpu). Jeśli None, auto-detect')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Katalog wyjściowy dla wyników')
    parser.add_argument('--export_data', action='store_true',
                       help='Eksportuj przetworzone dane do plików (.npy, .mat)')
    parser.add_argument('--export_dir', type=str, default=None,
                       help='Katalog do eksportu przetworzonych danych (domyślnie: data/processed)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'new_flow')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("NOWY FLOW - PRZETWARZANIE I TESTOWANIE")
    print(f"{'='*80}")
    print(f"Wszystkie datasety: {', '.join(AVAILABLE_DATASETS)}")
    print(f"Docelowe pasma: {', '.join(map(str, args.target_bands))}")
    print(f"Modele: {', '.join(args.models)}")
    print(f"Liczba podziałów: {args.n_test_datasets}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Validation split: {args.val_split}")
    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Katalog wyjściowy: {output_dir}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    # Dla każdej liczby pasm
    for target_bands in args.target_bands:
        print(f"\n{'#'*80}")
        print(f"# KROK 1-2: WCZYTYWANIE I PREPROCESSING - {target_bands} pasm")
        print(f"{'#'*80}\n")
        
        # KROK 1-2: Wczytanie danych i preprocessing
        print("Tworzenie podziałów datasetów (3 train, 1 test, 1 val)...")
        dataset_splits = create_test_datasets_from_multiple_sources(
            target_bands=target_bands,
            patch_size=args.patch_size,
            n_splits=args.n_test_datasets,
            export_data=args.export_data,
            export_dir=args.export_dir
        )
        
        print(f"\n{'#'*80}")
        print(f"# KROK 3: UTWORZONO {args.n_test_datasets} PODZIAŁÓW DATASETÓW")
        print(f"{'#'*80}\n")
        
        # KROK 4: Trening i testy modeli
        print(f"\n{'#'*80}")
        print(f"# KROK 4: TRENING I TESTOWANIE {len(args.models)} MODELI")
        print(f"{'#'*80}\n")
        
        # Trenuj i testuj modele na każdym podziale
        test_results = train_and_test_models_on_splits(
            model_names=args.models,
            models_dict=MODELS,
            model_types_dict=MODEL_TYPES,
            dataset_splits=dataset_splits,
            target_bands=target_bands,
            patch_size=args.patch_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            device=device
        )
        
        all_results[target_bands] = test_results
        
        # KROK 5: Walidacja
        print(f"\n{'#'*80}")
        print(f"# KROK 5: WALIDACJA DANYCH - {target_bands} pasm")
        print(f"{'#'*80}\n")
        
        validation_report = validate_results(test_results, min_accuracy=0.0, max_accuracy=100.0)
        print_validation_report(validation_report)
        
        # Zapisz raport walidacji
        validation_path = os.path.join(output_dir, f'validation_report_{target_bands}bands.json')
        save_validation_report(validation_report, validation_path)
    
    # Zapisz wszystkie wyniki
    results_path = os.path.join(output_dir, 'all_test_results.json')
    
    # Konwertuj wyniki do formatu JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("FLOW ZAKOŃCZONY")
    print(f"{'='*80}")
    print(f"Wyniki zapisane do: {output_dir}")
    print(f"  - all_test_results.json")
    print(f"  - validation_report_*bands.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

