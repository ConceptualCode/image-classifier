# Chess Piece Classification Project

This project is a chess piece classification system that leverages a deep learning model to classify images of chess pieces, specifically: Bishop, King, Knight, Pawn, Queen, and Rook. The project includes a FastAPI-based REST API for serving the model and a Streamlit web interface for easy user interaction. Both the API and the Streamlit interface are containerized with Docker and managed through Docker Compose.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup Instructions](#setup-instructions)
3. [Usage Instructions](#usage-instructions)
4. [API Details](#api-details)
5. [Streamlit Interface](#streamlit-interface)
6. [Monitoring Performance Metrics](#monitoring-performance-metrics)
7. [Dockerization](#dockerization)
8. [Troubleshooting](#troubleshooting)

---

### Project Structure

```plaintext
chess-classification/
├── best_hyperparameters.yaml       # Configuration for model parameters
├── data/                           # Placeholder for datasets (not included in repo)
├── models/
│   └── chess_classifier.pth        # Trained PyTorch model weights
├── api.py                          # FastAPI application for model serving
├── model.py                        # Model loading and preprocessing
├── streamlit_app.py                # Streamlit interface for user interaction
├── Dockerfile.api                  # Dockerfile for API and model service
├── Dockerfile.streamlit            # Dockerfile for Streamlit service
├── docker-compose.yml              # Docker Compose file to manage services
├── requirements.txt                # Python dependencies
├── requirements_st.txt             # Streamlit dependencies
├── README.md                       # Project documentation
├── main.py                         # Main entry point for the API application
├── hyperparameter_search.py        # Script for hyperparameter tuning
├── save_model.py                   # Utility for saving model state
├── train.py                        # Script to train the model
├── utils.py                        # Utility functions
├── notebook.ipynb                  # Jupyter notebook for experiments and analysis
└── data.py                         # Data handling and preprocessing
```

#### Setup Instructions:

#### Prerequisites

* Python 3.8 or higher
* Docker and Docker Compose
* Basic familiarity with command-line operations

#### Clone the Repository

```bash
git clone https://github.com/yourusername/chess-classification.git
cd chess-classification
```

Local Installation (Optional)
#### Set up a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

#### Install dependencies:

```bash
pip install -r requirements.txt
```
#### Run the model training (if needed):

To train the model, ensure data is placed in the data/ directory, then execute the following command:
```bash
python train.py
```
The trained model will be saved to models/chess_classifier.pth. Configuration for training can be modified in best_hyperparameters.yaml.

#### Usage Instructions
##### Docker Setup
1. Pull the Docker images:

   If the images are available on Docker Hub, users can pull them directly:
```bash
    docker pull tonysoro/chess_api:latest
    docker pull tonysoro/chess_streamlit:latest
```
2. Run the Docker containers:

   Start the services with Docker Compose:

```bash
docker-compose up
```
3. To shut down services:

```bash
docker-compose down
```

This will start two services:
- FastAPI API running at http://localhost:8000
- Streamlit app running at http://localhost:8501

### Docker Compose Configuration
The Docker Compose file (docker-compose.yml) maps the root directory to /app within the containers, allowing configuration files and model weights to be easily accessible. The following environment variables are required:

- MODEL_PATH: Path to the trained model (e.g., /app/models/chess_classifier.pth)
- CONFIG_PATH: Path to the hyperparameter config file (e.g., /app/best_hyperparameters.yaml)


### API Details

The FastAPI server exposes a single POST endpoint at `/predict` for image classification.

* **Endpoint**: `POST /predict`
* **Input**: `file` parameter (JPEG/PNG image file of a chess piece)
* **Response**: JSON object with `class_label` and `confidence` indicating the predicted class and model confidence.

#### Sample Request

```python
import requests

url = "http://localhost:8000/predict"
image_path = "path/to/chess_piece_image.jpg"

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

print(response.json())
```

```json
{
    "class_label": "Knight",
    "confidence": 0.98
}
```

### Streamlit Interface
The Streamlit interface provides an easy-to-use web application for uploading images and viewing classification results.

- Upload Image: Use the file uploader to upload a JPEG or PNG image.
- Classify Image: Click the "Classify Image" button to send the image to the model for classification.
- View Results: The prediction, confidence level, latency, and throughput are displayed on the page.

#### Monitoring Performance Metrics
The Streamlit interface tracks two performance metrics for each prediction:

- Latency: Time taken for each classification request.
- Throughput: Displays the processing rate of the API (useful when sending multiple images).

### Project Notes
Ensure that best_hyperparameters.yaml is correctly set with the model parameters.
Check that chess_classifier.pth is loaded in the correct model path for accurate predictions.

### Troubleshooting
#### Common Issues
- Streamlit not loading: Ensure API service is running, and verify API_URL in the Streamlit environment variable points to the API’s address.
- Model not found error: Verify MODEL_PATH points to chess_classifier.pth in the models/ directory.
- Permission issues on Linux: Use sudo for Docker commands or adjust Docker’s permissions.

### Updating Model or Configuration
If you update chess_classifier.pth or best_hyperparameters.yaml, restart the Docker containers:

```bash
docker-compose down
docker-compose up
```

This ensures the latest model and configurations are loaded.