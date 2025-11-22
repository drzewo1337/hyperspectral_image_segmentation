"""
Ładowanie danych hiperspektralnych dla 5 datasetów
Prosty kod jak w train.ipynb, ale dla wszystkich datasetów
"""
import os
import urllib.request
import ssl
import scipy.io as sio
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

# Obsługa SSL - wyłącz weryfikację certyfikatów
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

try:
    import requests
    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass

# URLs dla wszystkich 5 datasetów
DATASET_URLS = {
    'Indian': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt':   'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
    },
    'PaviaU': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt':   'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
    },
    'PaviaC': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
        'gt':   'https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat',
    },
    'KSC': {
        'data': 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
        'gt':   'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat',
    },
    'Salinas': {
        'data': 'https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        'gt':   'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
    }
}

# Możliwe nazwy kluczy w plikach .mat
DATASET_KEYS = {
    'Indian': {
        'data': ['indian_pines_corrected', 'Indian_pines_corrected'],
        'gt': ['indian_pines_gt', 'Indian_pines_gt']
    },
    'PaviaU': {
        'data': ['paviaU', 'PaviaU', 'pavia_u'],
        'gt': ['paviaU_gt', 'PaviaU_gt', 'pavia_u_gt']
    },
    'PaviaC': {
        'data': ['pavia', 'Pavia', 'paviaC', 'PaviaC'],
        'gt': ['pavia_gt', 'Pavia_gt', 'paviaC_gt', 'PaviaC_gt']
    },
    'KSC': {
        'data': ['KSC', 'ksc'],
        'gt': ['KSC_gt', 'ksc_gt']
    },
    'Salinas': {
        'data': ['salinas_corrected', 'Salinas_corrected', 'salinas'],
        'gt': ['salinas_gt', 'Salinas_gt']
    }
}

# Informacje o datasetach
DATASET_INFO = {
    'Indian': {'num_classes': 16, 'num_bands': 200},
    'PaviaU': {'num_classes': 9, 'num_bands': 103},
    'PaviaC': {'num_classes': 9, 'num_bands': 102},
    'KSC': {'num_classes': 13, 'num_bands': 176},
    'Salinas': {'num_classes': 16, 'num_bands': 204}
}


def download_file(url, filename, max_retries=5):
    """Pobiera plik jeśli nie istnieje - z obsługą wznawiania pobierania (resume)
    NIGDY nie usuwa częściowo pobranych plików - zawsze wznawia od miejsca przerwania
    """
    import requests
    import time
    
    # Sprawdź czy plik już istnieje i czy jest kompletny
    resume_pos = 0
    if os.path.exists(filename):
        resume_pos = os.path.getsize(filename)
        file_size_mb = resume_pos / 1024 / 1024
        
        # Sprawdź czy plik może być już kompletny (pobierz nagłówki bez pobierania całego pliku)
        try:
            head_response = requests.head(url, verify=False, timeout=30, allow_redirects=True)
            expected_size = int(head_response.headers.get('content-length', 0))
            
            if expected_size > 0:
                # Jeśli plik ma rozmiar zbliżony do oczekiwanego (tolerancja 1MB), uznaj za kompletny
                if abs(resume_pos - expected_size) < 1024 * 1024:  # 1MB tolerancja
                    print(f"{filename} już istnieje i wygląda na kompletny ({file_size_mb:.1f} MB)")
                    return
                elif resume_pos > expected_size * 0.95:  # 95% to prawdopodobnie kompletny
                    print(f"{filename} już istnieje ({file_size_mb:.1f} MB) - prawdopodobnie kompletny")
                    return
                else:
                    print(f"Wznawianie pobierania {filename} od {file_size_mb:.1f} MB...")
            else:
                # Nie znamy oczekiwanego rozmiaru, sprawdź czy plik wygląda na kompletny
                if resume_pos > 1024 * 1024:  # Jeśli > 1MB, prawdopodobnie częściowy
                    print(f"Wznawianie pobierania {filename} od {file_size_mb:.1f} MB...")
                else:
                    print(f"{filename} już istnieje ({file_size_mb:.1f} MB)")
                    return
        except:
            # Jeśli HEAD request nie działa, po prostu sprawdź rozmiar
            if resume_pos > 0:
                print(f"Wznawianie pobierania {filename} od {file_size_mb:.1f} MB...")
            else:
                print(f"{filename} już istnieje ({file_size_mb:.1f} MB)")
                return
    else:
        print(f"Pobieranie {filename}...")
    
    for attempt in range(max_retries):
        try:
            # Przygotuj nagłówki dla resume download
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            # Użyj requests z obsługą resume
            response = requests.get(url, verify=False, timeout=180, stream=True, 
                                  allow_redirects=True, headers=headers)
            
            # Sprawdź czy serwer obsługuje resume (206 Partial Content)
            if resume_pos > 0 and response.status_code == 206:
                print(f"  ✓ Serwer obsługuje resume - wznawianie od {resume_pos / 1024 / 1024:.1f} MB")
            elif resume_pos > 0 and response.status_code == 200:
                # Serwer nie obsługuje resume - sprawdź czy plik może być już kompletny
                content_length = int(response.headers.get('content-length', 0))
                if content_length > 0 and abs(resume_pos - content_length) < 1024 * 1024:  # 1MB tolerancja
                    print(f"  ✓ Plik wygląda na kompletny ({resume_pos / 1024 / 1024:.1f} MB)")
                    return
                else:
                    # Serwer nie obsługuje resume, ale zachowujemy istniejący plik
                    print(f"  ⚠ Serwer nie obsługuje resume")
                    print(f"  Zachowujemy istniejący plik ({resume_pos / 1024 / 1024:.1f} MB)")
                    print(f"  Próbujemy pobrać od nowa (nadpisze, ale lepsze niż tracić postęp)")
                    # Zaczynamy od nowa, ale plik pozostaje jako backup
                    resume_pos = 0
            else:
                response.raise_for_status()
            
            if resume_pos == 0:
                response.raise_for_status()
            
            # Sprawdź oczekiwany rozmiar
            content_range = response.headers.get('Content-Range', '')
            if content_range:
                # Format: "bytes 0-123456/123456789"
                total_size = int(content_range.split('/')[-1])
            else:
                content_length = int(response.headers.get('content-length', 0))
                if resume_pos > 0 and response.status_code == 206:
                    # Partial content - content-length to rozmiar części, nie całego pliku
                    total_size = resume_pos + content_length
                else:
                    total_size = content_length
            
            # Otwórz plik w trybie append jeśli wznawiamy
            if resume_pos > 0 and response.status_code == 206:
                mode = 'ab'  # Append tylko jeśli serwer obsługuje resume
            else:
                mode = 'wb'  # Write binary (nadpisze, ale mamy już backup w pamięci)
            
            downloaded = resume_pos if mode == 'ab' else 0
            
            with open(filename, mode) as f:
                try:
                    chunk_size = 64 * 1024  # 64KB chunks
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                # Wyświetl postęp co 2MB
                                if downloaded % (2 * 1024 * 1024) < chunk_size:
                                    print(f"  Pobrano {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)", end='\r')
                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                    # Jeśli transfer się przerwał, zapisz postęp
                    f.flush()
                    os.fsync(f.fileno())
                    raise Exception(f"Transfer przerwany: {downloaded} / {total_size} bytes pobrano")
            
            # Weryfikuj rozmiar pliku
            final_size = os.path.getsize(filename)
            file_size_mb = final_size / 1024 / 1024
            
            # Jeśli znamy oczekiwany rozmiar, sprawdź czy się zgadza
            if total_size > 0:
                if abs(final_size - total_size) > 1024 * 1024:  # Tolerancja 1MB
                    # Plik nie jest kompletny, ale NIE USUWAJ - zachowaj dla następnej próby
                    raise Exception(f"Niepełne pobieranie: {final_size} / {total_size} bytes (różnica: {abs(final_size - total_size) / 1024 / 1024:.1f} MB)")
            
            print(f"\n✓ Pobrano {filename} ({file_size_mb:.1f} MB)")
            return
                
        except ImportError:
            print(f"Błąd: requests nie jest zainstalowane.")
            raise
        except Exception as e:
            # NIGDY nie usuwaj pliku - zawsze zachowaj postęp
            error_msg = str(e)[:100]  # Skróć długie komunikaty
            
            # Zaktualizuj resume_pos na podstawie aktualnego rozmiaru pliku
            if os.path.exists(filename):
                resume_pos = os.path.getsize(filename)
                current_size_mb = resume_pos / 1024 / 1024
            else:
                resume_pos = 0
                current_size_mb = 0
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"\n✗ Błąd pobierania {filename} (próba {attempt + 1}/{max_retries}): {error_msg}")
                if resume_pos > 0:
                    print(f"  ✓ Zachowano postęp: {current_size_mb:.1f} MB - wznawianie od tego miejsca")
                print(f"⏳ Ponowna próba za {wait_time} sekund...")
                time.sleep(wait_time)
            else:
                print(f"\n✗ Błąd pobierania {filename} po {max_retries} próbach")
                if resume_pos > 0:
                    print(f"  ✓ Częściowo pobrany plik ({current_size_mb:.1f} MB) pozostaje w katalogu")
                    print(f"  Możesz uruchomić skrypt ponownie - wzniesie pobieranie od {current_size_mb:.1f} MB")
                print(f"   Lub pobrać plik ręcznie z: {url}")
                raise Exception(f"Nie udało się pobrać {filename} po {max_retries} próbach: {str(e)}")


def find_key_in_mat(mat_file, possible_keys):
    """Znajduje właściwy klucz w pliku .mat"""
    # Sprawdź dokładne dopasowanie
    if isinstance(possible_keys, str):
        possible_keys = [possible_keys]
    
    for key in possible_keys:
        if key in mat_file:
            return key
    
    # Jeśli nie znaleziono, zwróć pierwszy klucz który nie jest meta
    keys = [k for k in mat_file.keys() if not k.startswith('__')]
    if keys:
        print(f"Używam klucza: {keys[0]} (oczekiwano: {possible_keys})")
        return keys[0]
    
    raise ValueError(f"Nie znaleziono klucza w pliku .mat. Możliwe: {possible_keys}, dostępne: {list(mat_file.keys())}")


def load_data(dataset_name):
    """
    Ładuje dane hiperspektralne dla wybranego datasetu
    Zwraca: data, labels
    Prosty kod jak load_indian_pines() z train.ipynb
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_URLS.keys())}")
    
    urls = DATASET_URLS[dataset_name]
    keys = DATASET_KEYS[dataset_name]
    
    # Utwórz folder data/raw jeśli nie istnieje
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    # Nazwy plików z pełną ścieżką
    data_file = os.path.join(data_dir, f"{dataset_name}_data.mat")
    gt_file = os.path.join(data_dir, f"{dataset_name}_gt.mat")
    
    # Pobierz pliki
    download_file(urls['data'], data_file)
    download_file(urls['gt'], gt_file)
    
    # Załaduj dane z obsługą błędów
    try:
        mat_data = sio.loadmat(data_file)
    except (OSError, ValueError) as e:
        print(f"Błąd odczytu {data_file}: {e}")
        print(f"Usuwam uszkodzony plik i pobieram ponownie...")
        if os.path.exists(data_file):
            os.remove(data_file)
        download_file(urls['data'], data_file)
        mat_data = sio.loadmat(data_file)
    
    try:
        mat_gt = sio.loadmat(gt_file)
    except (OSError, ValueError) as e:
        print(f"Błąd odczytu {gt_file}: {e}")
        print(f"Usuwam uszkodzony plik i pobieram ponownie...")
        if os.path.exists(gt_file):
            os.remove(gt_file)
        download_file(urls['gt'], gt_file)
        mat_gt = sio.loadmat(gt_file)
    
    # Znajdź właściwe klucze
    data_key = find_key_in_mat(mat_data, keys['data'] if isinstance(keys['data'], list) else [keys['data']])
    gt_key = find_key_in_mat(mat_gt, keys['gt'] if isinstance(keys['gt'], list) else [keys['gt']])
    
    data = mat_data[data_key]
    labels = mat_gt[gt_key]
    
    print(f"Załadowano {dataset_name}: shape={data.shape}, bands={data.shape[2]}")
    
    return data, labels


def normalize(data):
    """
    Normalizacja używając StandardScaler (ZMIANA względem train.ipynb który używał MinMaxScaler)
    Ta sama logika co w train.ipynb, tylko StandardScaler zamiast MinMaxScaler
    """
    h, w, b = data.shape
    data = data.reshape(-1, b)
    data = StandardScaler().fit_transform(data)
    return data.reshape(h, w, b)


def pad_with_zeros(data, margin):
    """Dodaje padding zerami wokół obrazu - dokładnie jak w train.ipynb"""
    return np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='constant')


# Dataset class - dokładnie jak w train.ipynb, ale elastyczny dla wszystkich datasetów


class HSI_Dataset(Dataset):
    """
    Dataset dla danych hiperspektralnych - prosty kod jak w train.ipynb
    Elastyczny dla wszystkich 5 datasetów i różnych typów modeli (2D/3D)
    """
    def __init__(self, dataset_name, patch_size=16, model_type='2d'):
        """
        Args:
            dataset_name: 'Indian', 'PaviaU', 'PaviaC', 'KSC', 'Salinas'
            patch_size: rozmiar patchy
            model_type: '2d' dla Conv2D, '3d' dla Conv3D
        """
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.model_type = model_type
        
        # Załaduj dane
        data, labels = load_data(dataset_name)
        data = normalize(data)
        
        # Padding
        margin = patch_size // 2
        padded_data = pad_with_zeros(data, margin)
        
        # Ekstrakcja patchy - dokładnie jak w train.ipynb
        h, w, _ = data.shape
        self.patches = []
        self.targets = []
        
        for i in range(h):
            for j in range(w):
                label = labels[i, j]
                if label == 0:  # Ignoruj tło
                    continue
                patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                self.patches.append(patch)
                self.targets.append(label - 1)  # -1 bo klasy zaczynają od 1
        
        self.patches = np.array(self.patches)
        self.targets = np.array(self.targets)
        
        # Przygotuj dane w odpowiednim formacie
        if model_type == '3d':
            # Conv3D: (N, 1, B, H, W) - pasma jako depth dimension
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)
            self.patches = np.expand_dims(self.patches, axis=1)  # (N, 1, B, H, W)
        else:
            # Conv2D: (N, B, H, W) - pasma jako channels
            self.patches = np.transpose(self.patches, (0, 3, 1, 2))  # (N, B, H, W)
        
        # Konwertuj do torch tensor RAZ podczas inicjalizacji zamiast w __getitem__
        # To zmniejszy obciążenie CPU podczas treningu
        self.patches = torch.from_numpy(self.patches).float()
        self.targets = torch.from_numpy(self.targets).long()
        
        print(f"Dataset {dataset_name}: {len(self)} samples, shape={self.patches.shape}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Zwracamy bezpośrednio tensory - już są torch tensory, nie numpy
        # DataLoader z pin_memory=True i non_blocking=True w train.py przyspieszy transfer
        return self.patches[idx], self.targets[idx]


def get_loaders(dataset_name, batch_size=16, patch_size=8, val_split=0.2, model_type='2d', num_workers=0):
    """
    Zwraca train_loader, val_loader i info - prosty kod jak w train.ipynb
    
    Args:
        num_workers: Liczba wątków do ładowania danych (0 = główny wątek, 2-4 zalecane dla CPU)
    """
    dataset = HSI_Dataset(dataset_name, patch_size=patch_size, model_type=model_type)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    
    # Przygotuj info
    info = {
        'num_bands': dataset.patches.shape[1] if model_type == '2d' else dataset.patches.shape[2],
        'num_classes': DATASET_INFO[dataset_name]['num_classes']
    }
    
    # num_workers=0 oznacza główny wątek (bezpieczniejsze, ale wolniejsze)
    # num_workers=2-4 może przyspieszyć, ale wymaga więcej CPU i RAM
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True), 
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            info)

