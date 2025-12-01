"""
Krok 1: Wczytanie danych
Ładuje wszystkie 5 datasetów hiperspektralnych
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.load_data import load_data, DATASET_INFO

DATASET_NAMES = ['Indian', 'PaviaU', 'PaviaC', 'KSC', 'Salinas']

def load_all_datasets():
    """
    Ładuje wszystkie 5 datasetów
    Zwraca: dict z kluczami będącymi nazwami datasetów, wartościami są tuple (data, labels)
    """
    datasets = {}
    
    print("=" * 80)
    print("KROK 1: Wczytanie danych")
    print("=" * 80)
    
    for dataset_name in DATASET_NAMES:
        print(f"\nŁadowanie {dataset_name}...")
        data, labels = load_data(dataset_name)
        datasets[dataset_name] = {
            'data': data,
            'labels': labels,
            'info': DATASET_INFO[dataset_name]
        }
        print(f"  ✓ Załadowano: shape={data.shape}, bands={data.shape[2]}, classes={DATASET_INFO[dataset_name]['num_classes']}")
    
    print(f"\n✓ Wczytano wszystkie {len(datasets)} datasetów")
    return datasets

if __name__ == "__main__":
    datasets = load_all_datasets()
    print(f"\nGotowe! Załadowano {len(datasets)} datasetów")

