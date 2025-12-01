"""
Krok 3: Stworzenie zbioru testowego na podstawie danych - 5 datasetów
Generuje 5 różnych podziałów datasetów na train/test/final_test
"""
import sys
import os
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATASET_NAMES = ['Indian', 'PaviaU', 'PaviaC', 'KSC', 'Salinas']

def generate_dataset_splits():
    """
    Generuje 5 różnych podziałów datasetów
    
    Strategia: 
    - Train: 3 datasety (do treningu)
    - Test: 1 dataset (do testowania)
    - Validation: 1 dataset (do walidacji końcowej)
    
    Uwaga: "final_test_dataset" w kodzie to faktycznie validation dataset (walidacja)
    
    Zwraca: lista słowników z kluczami: split_id, train_datasets, test_dataset, validation_dataset
    """
    print("=" * 80)
    print("KROK 3: Stworzenie zbioru testowego - 5 datasetów")
    print("=" * 80)
    
    splits = []
    
    # Generuj wszystkie możliwe kombinacje 3 datasetów do treningu
    # Z pozostałych 2, jeden będzie test, drugi final_test
    train_combinations = list(itertools.combinations(DATASET_NAMES, 3))
    
    for split_id, train_datasets in enumerate(train_combinations, 1):
        train_list = list(train_datasets)
        remaining = [d for d in DATASET_NAMES if d not in train_list]
        
        # Dla każdej kombinacji train, generuj różne podziały test/final_test
        # Użyjemy wszystkich możliwych kombinacji z pozostałych 2 datasetów
        test_combinations = list(itertools.permutations(remaining, 2))
        
        for test_idx, (test_dataset, validation_dataset) in enumerate(test_combinations):
            split = {
                'split_id': split_id * 10 + test_idx + 1,  # Unikalny ID
                'train_datasets': train_list,
                'test_dataset': test_dataset,  # Test dataset
                'validation_dataset': validation_dataset,  # Validation dataset (walidacja)
                'final_test_dataset': validation_dataset  # Dla kompatybilności wstecznej
            }
            splits.append(split)
    
    # Jeśli mamy więcej niż 5 kombinacji, wybierz 5 pierwszych
    # (lub możemy wybrać różne strategie)
    if len(splits) > 5:
        # Wybierz 5 różnych podziałów, preferując różne kombinacje train
        selected_splits = []
        seen_train_combos = set()
        
        for split in splits:
            train_key = tuple(sorted(split['train_datasets']))
            if train_key not in seen_train_combos or len(selected_splits) < 5:
                selected_splits.append(split)
                seen_train_combos.add(train_key)
                if len(selected_splits) >= 5:
                    break
        
        splits = selected_splits[:5]
    
    # Wyświetl wygenerowane podziały
    print(f"\nWygenerowano {len(splits)} podziałów datasetów:\n")
    for split in splits:
        train_str = "+".join(split['train_datasets'])
        print(f"  Split {split['split_id']}:")
        print(f"    Train: {train_str}")
        print(f"    Test: {split['test_dataset']}")
        print(f"    Validation: {split['validation_dataset']}")
    
    print(f"\n✓ Wygenerowano {len(splits)} podziałów datasetów")
    return splits

if __name__ == "__main__":
    splits = generate_dataset_splits()
    print(f"\nGotowe! Wygenerowano {len(splits)} podziałów")

