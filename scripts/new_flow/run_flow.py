"""
Główny skrypt łączący wszystkie kroki flow
Wykonuje pełny pipeline zgodnie z flowchartem:
1. Wczytanie danych
2. Preprocessing - redukcja wymiarów do 10/30/60/90 kanałów (Gauss)
3. Stworzenie zbioru testowego na podstawie danych - 5 datasetów
4. Testy 3 odtworzonych modeli (używając DBSCAN clustering do segmentacji)
5. Walidacja danych
"""
import sys
import os
import json
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modułów z numerami w nazwach plików
import importlib.util
import sys

# Helper function do importowania modułów z numerami
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Ścieżka do folderu new_flow
new_flow_dir = os.path.join(os.path.dirname(__file__))

# Import wszystkich modułów
load_data_module = import_module_from_file("load_data", os.path.join(new_flow_dir, "1_load_data.py"))
preprocessing_module = import_module_from_file("preprocessing", os.path.join(new_flow_dir, "2_preprocessing.py"))
create_test_datasets_module = import_module_from_file("create_test_datasets", os.path.join(new_flow_dir, "3_create_test_datasets.py"))
test_models_module = import_module_from_file("test_models", os.path.join(new_flow_dir, "4_test_models.py"))
validate_results_module = import_module_from_file("validate_results", os.path.join(new_flow_dir, "5_validate_results.py"))
visualize_segmentation_module = import_module_from_file("visualize_segmentation", os.path.join(new_flow_dir, "6_visualize_segmentation.py"))

# Zaimportuj funkcje
load_all_datasets = load_data_module.load_all_datasets
preprocess_datasets = preprocessing_module.preprocess_datasets
generate_dataset_splits = create_test_datasets_module.generate_dataset_splits
test_models = test_models_module.test_models
validate_results = validate_results_module.validate_results
visualize_all_results = visualize_segmentation_module.visualize_all_results

# Docelowe liczby kanałów
TARGET_BANDS = [10, 20, 30]

def run_full_flow(target_bands_list=None, patch_size=16, batch_size=128, epochs=50, lr=0.001):
    """
    Wykonuje pełny flow
    
    Args:
        target_bands_list: lista docelowych liczb kanałów (domyślnie [30])
        patch_size: rozmiar patchy
        batch_size: rozmiar batcha
        epochs: liczba epok treningu
        lr: learning rate
    """
    if target_bands_list is None:
        target_bands_list = TARGET_BANDS
    
    print("\n" + "=" * 80)
    print("START FLOW - Segmentacja obrazów hiperspektralnych")
    print("=" * 80)
    
    # Krok 1: Wczytanie danych
    datasets = load_all_datasets()
    
    # Krok 2: Preprocessing
    preprocessed_data = preprocess_datasets(datasets, target_bands_list=target_bands_list)
    
    # Krok 3: Stworzenie zbioru testowego
    splits = generate_dataset_splits()
    
    # Krok 4: Testy modeli
    all_results = {}
    
    for target_bands in target_bands_list:
        print(f"\n{'#'*80}")
        print(f"# Testowanie dla {target_bands} kanałów")
        print(f"{'#'*80}")
        
        results = test_models(
            preprocessed_data, splits, target_bands,
            patch_size=patch_size, batch_size=batch_size, epochs=epochs, lr=lr
        )
        
        all_results[target_bands] = results
        
        # Zapisz wyniki po każdej liczbie kanałów
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'new_flow')
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'all_test_results.json')
        
        # Załaduj istniejące wyniki jeśli istnieją
        existing_results = {}
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        
        # Zaktualizuj wyniki
        existing_results[str(target_bands)] = results
        with open(results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print(f"\n✓ Zapisano wyniki dla {target_bands} kanałów")
    
    # Krok 5: Walidacja danych
    validation_report = validate_results(all_results)
    
    # Krok 6: Wizualizacja wyników segmentacji
    print("\n" + "=" * 80)
    print("KROK 6: Wizualizacja wyników segmentacji")
    print("=" * 80)
    visualize_all_results(all_results, preprocessed_data, patch_size=patch_size)
    
    print("\n" + "=" * 80)
    print("FLOW ZAKOŃCZONY")
    print("=" * 80)
    print(f"✓ Wszystkie kroki wykonane pomyślnie")
    print(f"✓ Wyniki zapisane w results/new_flow/")
    print(f"✓ Wizualizacje zapisane w results/new_flow/visualizations/")
    
    return validation_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Uruchom pełny flow segmentacji hiperspektralnej')
    parser.add_argument('--bands', type=int, nargs='+', default=[30],
                        help='Lista docelowych liczb kanałów (domyślnie: 30)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Rozmiar patchy (domyślnie: 16)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Rozmiar batcha (domyślnie: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Liczba epok treningu (domyślnie: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (domyślnie: 0.001)')
    
    args = parser.parse_args()
    
    validation_report = run_full_flow(
        target_bands_list=args.bands,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    print("\nNajlepszy model:", validation_report.get('best_model', 'N/A'))
    print("Najlepszy wynik:", validation_report.get('best_score', 'N/A'))

