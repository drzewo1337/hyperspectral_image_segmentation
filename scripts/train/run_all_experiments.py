"""
Skrypt do automatycznego uruchomienia wszystkich eksperymentów
3 modele × 5 datasetów = 15 eksperymentów
"""
import os
import subprocess
import sys
import time
from datetime import datetime

# Wszystkie modele
MODELS = ['InceptionHSINet', 'SimpleHSINet', 'CNNFromDiagram']

# Wszystkie datasety
#DATASETS = ['Indian', 'PaviaU', 'PaviaC', 'KSC', 'Salinas']
DATASETS = ['PaviaC']
# Parametry treningu
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATCH_SIZE = 8
VAL_SPLIT = 0.2


def run_experiment(model, dataset, epochs, batch_size, lr, patch_size, val_split):
    """Uruchamia pojedynczy eksperyment"""
    print(f"\n{'='*80}")
    print(f"Starting: {model} on {dataset}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Ścieżka do skryptu treningowego
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_training_script = os.path.join(script_dir, 'run_training.py')
    
    cmd = [
        sys.executable,
        run_training_script,
        '--model', model,
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--patch_size', str(patch_size),
        '--val_split', str(val_split)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Completed: {model} on {dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {model} on {dataset}")
        print(f"Error: {e}")
        return False


def main():
    total_experiments = len(MODELS) * len(DATASETS)
    completed = 0
    failed = 0
    
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"BATCH TRAINING: {total_experiments} experiments")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Epochs per experiment: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*80}\n")
    
    results = []
    
    for model in MODELS:
        for dataset in DATASETS:
            experiment_name = f"{model}_{dataset}"
            exp_start_time = time.time()
            
            success = run_experiment(
                model, dataset, EPOCHS, BATCH_SIZE, 
                LEARNING_RATE, PATCH_SIZE, VAL_SPLIT
            )
            
            exp_time = time.time() - exp_start_time
            
            if success:
                completed += 1
                status = "✓ SUCCESS"
            else:
                failed += 1
                status = "✗ FAILED"
            
            results.append({
                'model': model,
                'dataset': dataset,
                'status': status,
                'time': f"{exp_time/60:.1f} min"
            })
            
            print(f"\n{status}: {experiment_name} ({exp_time/60:.1f} min)")
            print(f"Progress: {completed + failed}/{total_experiments} experiments")
    
    total_time = time.time() - start_time
    
    # Podsumowanie
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"{'='*80}\n")
    
    # Szczegółowe wyniki
    print("Detailed results:")
    print("-" * 80)
    for result in results:
        print(f"{result['status']:15} {result['model']:20} on {result['dataset']:10} ({result['time']})")
    
    print(f"\n{'='*80}")
    print("All experiments finished!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

