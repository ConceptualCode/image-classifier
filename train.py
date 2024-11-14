import torch
import torch.optim as optim
import torch.nn as nn
from evaluate import evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

def train_model(model, train_loader, val_loader, epochs, learning_rate, device, save_model=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    model = model.to(device)

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    best_val_loss = float('inf')  # Set initial best_val_loss to infinity
    patience = 0
    patience_limit = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * correct / total
        history['accuracy'].append(train_accuracy)
        history['loss'].append(train_loss / len(train_loader))

        # Validation step
        try:
            val_loss, val_acc, *_ = evaluate_model(model, val_loader, device, criterion)
        except ValueError:
            print("Evaluation error: Skipping this validation step due to invalid predictions.")
            val_loss, val_acc = float('inf'), 0 # Set default values if evaluation fails

        # Update scheduler
        scheduler.step(val_loss)

        # Checkpointing: Save the model if validation loss improves
        if save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}: New best validation loss achieved, model saved.")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping")
                break


        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}% Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")

    if save_model:
        print("Training complete. Best model saved as 'best_model.pth'.")
    else:
        print("Training complete. Model was not saved during hyperparameter tuning.")

    with open("training_history.json", "w") as f:
        json.dump(history, f)
    print("Training history saved to 'training_history.json'.")

    print("Training complete. Best model saved as 'best_model.pth'.")
    return history