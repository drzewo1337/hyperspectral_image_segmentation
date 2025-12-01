"""
Krok 5: Walidacja danych
Porównuje wyniki 3 modeli i generuje raporty
"""
import sys
import os
import json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def validate_results(all_results, output_dir=None):
    """
    Waliduje i analizuje wyniki testów
    
    Args:
        all_results: dict z kluczami target_bands -> lista wyników
        output_dir: katalog do zapisania wyników (opcjonalny)
    
    Returns:
        validation_report: dict z raportem walidacji
    """
    print("=" * 80)
    print("KROK 5: Walidacja danych")
    print("=" * 80)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'new_flow')
    os.makedirs(output_dir, exist_ok=True)
    
    validation_report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Grupuj wyniki według liczby kanałów
    for target_bands, results in all_results.items():
        print(f"\nAnaliza wyników dla {target_bands} kanałów...")
        
        if not results:
            validation_report['warnings'].append(f"Brak wyników dla {target_bands} kanałów")
            continue
        
        # Grupuj wyniki według modelu
        model_results = defaultdict(list)
        for result in results:
            model_name = result['model_name']
            model_results[model_name].append(result)
        
        summary = {}
        
        for model_name, model_result_list in model_results.items():
            print(f"\n  Model: {model_name}")
            
            # Zbierz metryki
            test_accuracies = [r['test_accuracy'] for r in model_result_list]
            # Użyj validation_accuracy jeśli dostępne, w przeciwnym razie final_test_accuracy
            final_test_accuracies = [r.get('validation_accuracy', r.get('final_test_accuracy', 0)) for r in model_result_list]
            
            if not test_accuracies:
                continue
            
            summary[model_name] = {
                'mean_test_accuracy': float(np.mean(test_accuracies)),
                'std_test_accuracy': float(np.std(test_accuracies)),
                'min_test_accuracy': float(np.min(test_accuracies)),
                'max_test_accuracy': float(np.max(test_accuracies)),
                'mean_val_accuracy': float(np.mean(final_test_accuracies)),
                'std_val_accuracy': float(np.std(final_test_accuracies)),
                'min_val_accuracy': float(np.min(final_test_accuracies)),
                'max_val_accuracy': float(np.max(final_test_accuracies)),
                'n_splits': len(model_result_list)
            }
            
            print(f"    Test accuracy: {summary[model_name]['mean_test_accuracy']:.2f}% ± {summary[model_name]['std_test_accuracy']:.2f}%")
            print(f"    Final test accuracy: {summary[model_name]['mean_val_accuracy']:.2f}% ± {summary[model_name]['std_val_accuracy']:.2f}%")
            print(f"    Liczba testów: {summary[model_name]['n_splits']}")
        
        # Oblicz ogólne statystyki
        all_test_accs = []
        all_final_test_accs = []
        for model_summary in summary.values():
            all_test_accs.extend([model_summary['mean_test_accuracy']])
            all_final_test_accs.extend([model_summary['mean_val_accuracy']])
        
        if all_test_accs:
            summary['overall'] = {
                'mean_accuracy': float(np.mean(all_test_accs + all_final_test_accs)),
                'std_accuracy': float(np.std(all_test_accs + all_final_test_accs)),
                'min_accuracy': float(np.min(all_test_accs + all_final_test_accs)),
                'max_accuracy': float(np.max(all_test_accs + all_final_test_accs)),
                'n_tests': len(all_test_accs + all_final_test_accs)
            }
        
        validation_report['summary'][str(target_bands)] = summary
        
        # Zapisz raport dla tej liczby kanałów
        report_file = os.path.join(output_dir, f'validation_report_{target_bands}bands.json')
        with open(report_file, 'w') as f:
            json.dump({str(target_bands): summary}, f, indent=2)
        print(f"\n  ✓ Zapisano raport do {report_file}")
    
    # Porównaj modele - znajdź najlepszy
    print("\n" + "=" * 80)
    print("PORÓWNANIE MODELI:")
    print("=" * 80)
    
    model_comparison = defaultdict(list)
    
    for target_bands, summary in validation_report['summary'].items():
        for model_name, metrics in summary.items():
            if model_name == 'overall':
                continue
            model_comparison[model_name].append({
                'bands': target_bands,
                'mean_test': metrics['mean_test_accuracy'],
                'mean_final_test': metrics['mean_val_accuracy']
            })
    
    # Oblicz średnie dla każdego modelu
    best_model = None
    best_score = -1
    
    for model_name, scores in model_comparison.items():
        avg_test = np.mean([s['mean_test'] for s in scores])
        avg_final = np.mean([s['mean_final_test'] for s in scores])
        overall_avg = (avg_test + avg_final) / 2
        
        print(f"\n  {model_name}:")
        print(f"    Średnia test accuracy: {avg_test:.2f}%")
        print(f"    Średnia final test accuracy: {avg_final:.2f}%")
        print(f"    Ogólna średnia: {overall_avg:.2f}%")
        
        if overall_avg > best_score:
            best_score = overall_avg
            best_model = model_name
    
    if best_model:
        print(f"\n  ✓ Najlepszy model: {best_model} (średnia: {best_score:.2f}%)")
        validation_report['best_model'] = best_model
        validation_report['best_score'] = float(best_score)
    
    # Zapisz pełny raport
    report_file = os.path.join(output_dir, 'validation_report_all.json')
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"\n✓ Zapisano pełny raport do {report_file}")
    
    return validation_report

if __name__ == "__main__":
    print("Ten skrypt powinien być wywoływany z głównego flow scriptu")

