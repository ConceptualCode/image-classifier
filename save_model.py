import torch
import os

# def save_model(model, path='models/chess_piece_classification_model.pth'):
#     torch.save(model.state_dict(), path)
#     print(f"Model saved to {path}")

def save_model(model, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path='chess_piece_classification_model.pth', device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval() 
    print(f"Model loaded from {path}")
    return model
