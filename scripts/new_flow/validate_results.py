"""
Moduł do walidacji wyników testów modeli
"""
import numpy as np
import pandas as pd
from collections import defaultdict


def validate_results(test_results, min_accuracy=0.0, max_accuracy=100.0):
    """
    Waliduje wyniki testów modeli
    
    Args:
        test_results: słownik z wynikami testów
        min_accuracy: minimalna akceptowalna dokładność
        max_accuracy: maksymalna akceptowalna dokładność
    
    Returns:
        validation_report: raport walidacji
    """
    validation_report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    all_accuracies = []
    
    for model_name, model_results in test_results.items():
        model_accuracies = []
        model_val_accuracies = []
        
        # model_results to lista wyników dla każdego podziału
        for result in model_results:
            test_accuracy = result.get('test_accuracy', 0)
            val_accuracy = result.get('val_accuracy', 0)
            split_id = result.get('split_id', 0)
            test_dataset = result.get('test_dataset', 'unknown')
            val_dataset = result.get('val_dataset', 'unknown')
            
            # Test accuracy
            all_accuracies.append(test_accuracy)
            model_accuracies.append(test_accuracy)
            
            if test_accuracy < min_accuracy:
                validation_report['valid'] = False
                validation_report['errors'].append(
                    f"{model_name} - Podział {split_id} (test: {test_dataset}): "
                    f"dokładność test {test_accuracy:.2f}% jest poniżej minimum {min_accuracy}%"
                )
            
            if test_accuracy > max_accuracy:
                validation_report['warnings'].append(
                    f"{model_name} - Podział {split_id} (test: {test_dataset}): "
                    f"dokładność test {test_accuracy:.2f}% jest powyżej maksimum {max_accuracy}%"
                )
            
            # Val accuracy
            all_accuracies.append(val_accuracy)
            model_val_accuracies.append(val_accuracy)
            
            if val_accuracy < min_accuracy:
                validation_report['valid'] = False
                validation_report['errors'].append(
                    f"{model_name} - Podział {split_id} (val: {val_dataset}): "
                    f"dokładność val {val_accuracy:.2f}% jest poniżej minimum {min_accuracy}%"
                )
            
            if val_accuracy > max_accuracy:
                validation_report['warnings'].append(
                    f"{model_name} - Podział {split_id} (val: {val_dataset}): "
                    f"dokładność val {val_accuracy:.2f}% jest powyżej maksimum {max_accuracy}%"
                )
        
        if model_accuracies:
            validation_report['summary'][model_name] = {
                'mean_test_accuracy': np.mean(model_accuracies),
                'std_test_accuracy': np.std(model_accuracies),
                'min_test_accuracy': np.min(model_accuracies),
                'max_test_accuracy': np.max(model_accuracies),
                'mean_val_accuracy': np.mean(model_val_accuracies) if model_val_accuracies else 0,
                'std_val_accuracy': np.std(model_val_accuracies) if model_val_accuracies else 0,
                'min_val_accuracy': np.min(model_val_accuracies) if model_val_accuracies else 0,
                'max_val_accuracy': np.max(model_val_accuracies) if model_val_accuracies else 0,
                'n_splits': len(model_accuracies)
            }
    
    if all_accuracies:
        validation_report['summary']['overall'] = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'n_tests': len(all_accuracies)
        }
    
    return validation_report


def print_validation_report(validation_report):
    """Wypisuje raport walidacji"""
    print(f"\n{'='*80}")
    print("RAPORT WALIDACJI")
    print(f"{'='*80}")
    
    if validation_report['valid']:
        print("✓ Status: WALIDACJA PRZESZŁA")
    else:
        print("✗ Status: WALIDACJA NIE PRZESZŁA")
    
    if validation_report['errors']:
        print(f"\nBłędy ({len(validation_report['errors'])}):")
        for error in validation_report['errors']:
            print(f"  ✗ {error}")
    
    if validation_report['warnings']:
        print(f"\nOstrzeżenia ({len(validation_report['warnings'])}):")
        for warning in validation_report['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation_report['summary']:
        print(f"\nPodsumowanie:")
        for key, stats in validation_report['summary'].items():
            if key == 'overall':
                print(f"\n  Ogółem:")
                print(f"    Średnia dokładność: {stats.get('mean_accuracy', 0):.2f}%")
                print(f"    Odchylenie std: {stats.get('std_accuracy', 0):.2f}%")
                print(f"    Min: {stats.get('min_accuracy', 0):.2f}%")
                print(f"    Max: {stats.get('max_accuracy', 0):.2f}%")
                print(f"    Liczba testów: {stats.get('n_tests', 0)}")
            else:
                print(f"\n  {key}:")
                print(f"    Test - Średnia: {stats.get('mean_test_accuracy', 0):.2f}%")
                print(f"    Test - Std: {stats.get('std_test_accuracy', 0):.2f}%")
                print(f"    Test - Min: {stats.get('min_test_accuracy', 0):.2f}%")
                print(f"    Test - Max: {stats.get('max_test_accuracy', 0):.2f}%")
                print(f"    Val - Średnia: {stats.get('mean_val_accuracy', 0):.2f}%")
                print(f"    Val - Std: {stats.get('std_val_accuracy', 0):.2f}%")
                print(f"    Val - Min: {stats.get('min_val_accuracy', 0):.2f}%")
                print(f"    Val - Max: {stats.get('max_val_accuracy', 0):.2f}%")
                print(f"    Liczba podziałów: {stats.get('n_splits', 0)}")
    
    print(f"{'='*80}\n")


def save_validation_report(validation_report, output_path):
    """Zapisuje raport walidacji do pliku"""
    import json
    
    # Konwertuj numpy types na Python types dla JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Konwertuj raport
    serializable_report = {}
    for key, value in validation_report.items():
        if isinstance(value, dict):
            serializable_report[key] = {
                k: convert_to_serializable(v) for k, v in value.items()
            }
        elif isinstance(value, list):
            serializable_report[key] = value
        else:
            serializable_report[key] = convert_to_serializable(value)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Raport walidacji zapisany do: {output_path}")

