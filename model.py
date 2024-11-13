import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet18, resnet34

# def create_model(num_classes, dropout_rate=0.5, num_units=256):
#     model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#     for param in model.parameters():
#         param.requires_grad = False

#     model.fc = nn.Sequential(
#         nn.Linear(model.fc.in_features, num_units),
#         nn.ReLU(),
#         nn.Dropout(dropout_rate),
#         nn.Linear(num_units, num_classes)
#     )

#     return model

# def create_model(num_classes=6, dropout_rate=0.5, resnet_version='resnet18', pretrained=True, num_units=256):
#     if resnet_version == 'resnet18':
#         model = resnet18(pretrained=pretrained)
#     elif resnet_version == 'resnet34':
#         model = resnet34(pretrained=pretrained)
#     else:
#         raise ValueError("Unsupported ResNet version. Choose 'resnet18' or 'resnet34'.")

#     # Modify the final fully connected layer for our specific number of classes
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_ftrs, num_units),  # Optional: adjust the number of units here
#         nn.ReLU(),
#         nn.Dropout(dropout_rate),
#         nn.Linear(num_units, num_classes)
#     )
#     return model

import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, vgg16, densenet121,
    mobilenet_v2, inception_v3, efficientnet_b0, vit_b_16,
    ResNet50_Weights, VGG16_Weights, DenseNet121_Weights,
    MobileNet_V2_Weights, Inception_V3_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights
)

def create_model(num_classes=6, dropout_rate=0.5, model_name='resnet18', pretrained=True, num_units=256):
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    elif model_name == 'vgg16':
        model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    elif model_name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    elif model_name == 'inception_v3':
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT if pretrained else None, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_units, num_classes)
        )
        return model
    else:
        raise ValueError("Unsupported model name. Choose from 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'densenet121', 'mobilenet_v2', 'inception_v3', 'efficientnet_b0', 'vit_b_16'.")

    # Modify the final fully connected layer for ResNet models
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(num_units, num_classes)
    )

    return model
