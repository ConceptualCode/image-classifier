import optuna
from optuna.samplers import RandomSampler
import torch
import yaml
import argparse
from data import create_data_loaders
from model import create_model
from train import train_model

def objective(trial, data_dir, img_height, img_width, epochs):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5) 
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_data_loaders(data_dir, img_height, img_width, batch_size)

    model = create_model(num_classes=len(train_loader.dataset.dataset.classes), dropout_rate=dropout_rate)
    model.to(device)

    val_accuracy = train_model(model, train_loader,
                                val_loader, epochs,
                                learning_rate, device, 
                                save_model=False)
    
    return val_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search with Optuna for chess piece classification")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset")
    parser.add_argument('--img_height', type=int, default=224, help="Image height for resizing")
    parser.add_argument('--img_width', type=int, default=224, help="Image width for resizing")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for each trial")

    args = parser.parse_args()

    study = optuna.create_study(direction='maximize', sampler=RandomSampler())
    
    study.optimize(lambda trial: objective(trial, args.data_dir, args.img_height, args.img_width, args.epochs), n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)

    # Save best hyperparameters to a YAML file
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump(best_params, f)