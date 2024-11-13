import argparse
import torch
from data import create_data_loaders
from model import create_model
from train import train_model
from evaluate import plot_confusion_matrix
from save_model import save_model
from utils import count_images_in_directory, estimate_effective_dataset_size
import yaml


def load_best_hyperparameters(file_path='best_hyperparameters.yaml'):
    with open(file_path, 'r') as f:
        best_params = yaml.safe_load(f)
    return best_params

def main(data_dir, img_height, img_width, epochs, save_path, learning_rate, dropout_rate, batch_size, num_units, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best hyperparameters
    best_params = load_best_hyperparameters()

    # Extract hyperparameters
    learning_rate = best_params.get('learning_rate', learning_rate)  # Default in case missing
    dropout_rate = best_params.get('dropout_rate', dropout_rate)
    batch_size = best_params.get('batch_size', batch_size)
    num_units = best_params.get('num_units', num_units)

    # Count the original images in the dataset
    class_counts, total_images = count_images_in_directory(data_dir)

    # Estimate effective dataset size with augmentation for each class
    effective_class_sizes, total_effective_size = estimate_effective_dataset_size(class_counts, epochs)

    # Data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args.data_dir, args.img_height, args.img_width, args.batch_size)

    # Model
    model = create_model(num_classes=len(train_loader.dataset.dataset.classes), 
                         dropout_rate=dropout_rate,
                         num_units=num_units,
                         model_name=model_name,
                         pretrained=True)
    model.to(device)
    
    # Train
    history  = train_model(model, train_loader, val_loader, epochs, learning_rate, device)

    # Optional: Print final training and validation accuracy from history
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    # Save the model
    save_model(model, args.save_path)

    # Evaluation
    plot_confusion_matrix(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess Piece Classification with ResNet50 in PyTorch")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset")
    parser.add_argument('--img_height', type=int, default=224, help="Image height for resizing")
    parser.add_argument('--img_width', type=int, default=224, help="Image width for resizing")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--save_path', type=str, default='chess_classifier_model.pth', help="Path to save the trained model")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loading")
    parser.add_argument('--num_units', type=int, default=256, help="Number of units in the fully connected layer")
    parser.add_argument('--model_name', type=str, default='resnet18', help="Model architecture to use: 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0', 'vit_b_16'")


    args = parser.parse_args()
    main(
        args.data_dir,
        args.img_height,
        args.img_width,
        args.epochs,
        args.save_path,
        args.learning_rate,
        args.dropout_rate,
        args.batch_size,
        args.num_units,
        args.model_name
    )