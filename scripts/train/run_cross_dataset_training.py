"""
Skrypt do treningu cross-dataset dla 3 modeli
Train: Indian + PaviaU + KSC (3 datasety)
Val: 20% z train (split z połączonych danych)
Test: Salinas (1 dataset)
Hold-out: PaviaC (1 dataset - do finalnej oceny później)
"""
import sys
import os
import torch

# Dodaj ścieżki do sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluate'))

from models.cnn.model1 import InceptionHSINet
from models.cnn.model2 import SimpleHSINet
from models.cnn.model3 import CNNFromDiagram
from scripts.evaluate.cross_dataset_evaluation import cross_dataset_experiment

# Mapowanie typów modeli (2D vs 3D)
MODEL_TYPES = {
    'InceptionHSINet': '3d',
    'SimpleHSINet': '2d',
    'CNNFromDiagram': '2d'
}

# Konfiguracja eksperymentu
TRAIN_DATASETS = ['Indian', 'PaviaU', 'KSC']  # 3 datasety do treningu
TEST_DATASETS = ['Salinas']  # 1 dataset do testowania
HOLDOUT_DATASETS = ['PaviaC']  # 1 dataset do finalnej oceny (na później)

# Parametry treningu
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATCH_SIZE = 8
VAL_SPLIT = 0.2

# Wszystkie modele do przetestowania
MODELS = {
    'InceptionHSINet': InceptionHSINet,
    'SimpleHSINet': SimpleHSINet,
    'CNNFromDiagram': CNNFromDiagram
}


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"CROSS-DATASET TRAINING")
    print(f"{'='*80}")
    print(f"Train datasets: {', '.join(TRAIN_DATASETS)}")
    print(f"Test datasets: {', '.join(TEST_DATASETS)}")
    print(f"Hold-out datasets: {', '.join(HOLDOUT_DATASETS)} (do finalnej oceny)")
    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Validation split: {VAL_SPLIT}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Trenuj każdy model
    for model_name, model_class in MODELS.items():
        print(f"\n{'#'*80}")
        print(f"# Training: {model_name}")
        print(f"{'#'*80}\n")
        
        model_type = MODEL_TYPES[model_name]
        
        try:
            result_entry, trained_model = cross_dataset_experiment(
                model_class=model_class,
                model_name=model_name,
                train_datasets=TRAIN_DATASETS,
                test_datasets=TEST_DATASETS,
                patch_size=PATCH_SIZE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                device=device,
                pca_components=None,  # Bez PCA na razie
                model_type=model_type
            )
            
            results.append(result_entry)
            
            print(f"\n✓ Completed: {model_name}")
            print(f"  Test accuracy: {result_entry.get('avg_test_acc', 0):.2f}%")
            
        except Exception as e:
            print(f"\n✗ Failed: {model_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Podsumowanie
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Test Acc (Salinas)':<20} {'Avg Test Acc':<20}")
    print(f"{'-'*80}")
    
    for result in results:
        model_name = result['model']
        test_acc_salinas = result.get('test_acc_Salinas', 0)
        avg_test_acc = result.get('avg_test_acc', 0)
        print(f"{model_name:<20} {test_acc_salinas:>18.2f}% {avg_test_acc:>18.2f}%")
    
    print(f"{'='*80}\n")
    
    # Zapisz wyniki do CSV
    import csv
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'metrics')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(results_dir, 'cross_dataset_results.csv')
    
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Results saved to: {csv_file}")
    
    print(f"\n{'='*80}")
    print("Cross-dataset training completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()



