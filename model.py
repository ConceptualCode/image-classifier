import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet18, resnet34, resnet50


def create_model(num_classes=6, dropout_rate=0.5, model_name='resnet18', pretrained=True, num_units=256):

    """
    Create a ResNet model for image classification with customizable dropout and fully connected layer.

    Parameters:
    - num_classes: Number of output classes for classification.
    - dropout_rate: Dropout rate for the fully connected layer.
    - model_name: Choose between 'resnet18', 'resnet34', and 'resnet50'.
    - pretrained: If True, load pretrained weights on ImageNet.
    - num_units: Number of units in the fully connected layer.

    Returns:
    - model: The customized ResNet model.
    """
    
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("Unsupported model name. Choose from 'resnet18', 'resnet34', 'resnet50'.")

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(num_units, num_classes)
    )

    return model
