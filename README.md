# Agile Effort Estimation: Deep Learning Approach

A machine learning solution for automatically estimating story points in Agile software development using BERT embeddings and neural networks. This project builds upon the research paper "Agile Effort Estimation: Have We Solved the Problem Yet? Insights From A Replication Study" published in IEEE Transactions on Software Engineering.

## 🎯 Overview

This project provides an automated approach to story point estimation using:
- **BERT Embeddings** for semantic understanding of user stories
- **Deep Neural Networks** for regression-based prediction
- **FastAPI** for production-ready REST API deployment
- **Docker** for containerized deployment

## 🏗️ Project Structure

```
Agile_Effort_Estimation/
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # MIT License
├── 🐍 spe.py                    # Main training script
├── 🚀 app.py                    # FastAPI application
├── 📋 requirements.txt          # Python dependencies
├── 🐳 Dockerfile               # Docker configuration
├── 🤖 story_point_model.pth    # Trained model weights
└── 📁 datasets/                # Training datasets (referenced)
    └── Choet_Dataset/
        └── MESOS_deep-se.csv
```

## ✨ Key Features

- **BERT-based Text Encoding**: Leverages DistilBERT for semantic understanding
- **Deep Neural Network**: 4-layer fully connected network for story point prediction
- **Text Preprocessing**: Advanced NLP preprocessing with spaCy
- **REST API**: Production-ready FastAPI endpoint
- **Docker Support**: Containerized deployment
- **Comprehensive Evaluation**: MAE, RMSE, and R² metrics
- **Visualization**: Training progress and prediction analysis

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Build the Docker image
docker build -t agile-estimator .

# Run the container
docker run -p 8000:8000 agile-estimator

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "User login functionality",
       "description": "As a user, I want to log into the system securely"
     }'
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd Agile_Effort_Estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Train the model (optional - if you have the dataset)
python spe.py

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📊 Model Architecture

### Neural Network Structure
```python
StoryPointEstimator(
  (fc1): Linear(768 → 512)    # BERT embedding input
  (fc2): Linear(512 → 256)    # Hidden layer 1
  (fc3): Linear(256 → 128)    # Hidden layer 2
  (fc4): Linear(128 → 1)      # Output layer (story points)
  (relu): ReLU()              # Activation function
)
```

### Training Pipeline
1. **Data Preprocessing**: Text cleaning and lemmatization
2. **BERT Encoding**: Convert text to 768-dimensional embeddings
3. **Model Training**: Neural network with MSE loss
4. **Evaluation**: Comprehensive metrics calculation
5. **Model Persistence**: Save trained weights

## 🔧 API Usage

### Endpoints

#### `POST /predict`
Predict story points for a given user story.

**Request Body:**
```json
{
  "title": "User story title",
  "description": "Detailed user story description"
}
```

**Response:**
```json
{
  "predicted_story_points": 5.25
}
```

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Story Point Estimation API is running!"
}
```

### Example Usage

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# User story data
data = {
    "title": "User Authentication",
    "description": "As a user, I want to securely log into the application using my credentials so that I can access my personal dashboard"
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"Predicted Story Points: {result['predicted_story_points']}")
```

## 📈 Model Performance

The model is evaluated using standard regression metrics:

- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors
- **R² Score**: Coefficient of determination

Training includes:
- 70% training data
- 20% validation data  
- 10% test data
- Adam optimizer with learning rate 0.001
- MSE loss function

## 🛠️ Development

### Training Your Own Model

1. **Prepare Dataset**: CSV with columns `title`, `description`, `storypoint`
2. **Update Path**: Modify `dataset_path` in `spe.py`
3. **Run Training**: Execute `python spe.py`
4. **Model Artifacts**: Generated files include:
   - `X_bert_embeddings.npy`: BERT embeddings
   - `y_storypoints.npy`: Target labels
   - `story_point_model.pth`: Trained model weights

### Customization Options

- **Model Architecture**: Modify layers in `StoryPointEstimator` class
- **BERT Model**: Change to different BERT variants (BERT-base, RoBERTa, etc.)
- **Training Parameters**: Adjust epochs, batch size, learning rate
- **Text Preprocessing**: Customize cleaning and lemmatization

## 🔍 Research Background

This implementation is based on the research paper:
> "Agile Effort Estimation: Have We Solved the Problem Yet? Insights From A Replication Study"
> Published in IEEE Transactions on Software Engineering

**Key Contributions:**
- Replication study of existing effort estimation techniques
- Evaluation of deep learning approaches for story point estimation
- Comparison with traditional estimation methods
- Insights into the current state of agile effort estimation

## 📋 Dependencies

**Core Libraries:**
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers (BERT)
- `fastapi` - Modern web framework for APIs
- `spacy` - Advanced NLP processing
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computing

**Development:**
- `uvicorn` - ASGI server
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `tqdm` - Progress bars

## 🐳 Docker Configuration

The Docker setup includes:
- **Base Image**: Python 3.10 slim
- **Port**: 8000 (FastAPI default)
- **Dependencies**: Automatically installed from requirements.txt
- **Model**: Pre-trained weights included in container


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original research by the authors of the IEEE TSE paper
- Hugging Face for pre-trained BERT models
- FastAPI team for the excellent web framework
- spaCy team for NLP processing capabilities

