"""
Funkcja treningowa - prosty kod jak w train.ipynb
"""
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time


def train(model, train_loader, val_loader, epochs=10, lr=0.001, 
          device=None, model_name="model", dataset_name="dataset"):
    """
    Trenowanie modelu - dokładnie jak w train.ipynb
    Prosty kod bez skomplikowanych rzeczy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Nazwa pliku logu
    log_filename = f"training_log_{model_name}_{dataset_name}.csv"
    
    best_val_acc = 0.0
    
    with open(log_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Accuracy"])

        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            model.train()
            total_loss, correct = 0, 0

            for inputs, labels in train_loader:
                # non_blocking=True przyspiesza transfer CPU->GPU (wymaga pin_memory=True w DataLoader)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = 100.0 * correct / len(train_loader.dataset)

            # Validation
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    # non_blocking=True przyspiesza transfer CPU->GPU
                    val_inputs, val_labels = val_inputs.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
                    val_outputs = model(val_inputs)
                    val_correct += (val_outputs.argmax(1) == val_labels).sum().item()
            
            val_acc = 100.0 * val_correct / len(val_loader.dataset)
            
            epoch_time = time.time() - epoch_start_time

            writer.writerow([epoch + 1, total_loss, train_acc, val_acc])
            
            # Zapisz najlepszy model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_filename = f"best_model_{model_name}_{dataset_name}.pth"
                torch.save(model.state_dict(), model_filename)
            
            # Wyświetl wyniki co epokę z czasem
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
    
    print(f"Zapisano log do {log_filename}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return model

