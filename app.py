from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

# Define request body model
class StoryPointRequest(BaseModel):
    title: str
    description: str

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load trained story point model
class StoryPointEstimator(torch.nn.Module):
    def __init__(self, input_dim=768):
        super(StoryPointEstimator, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Load trained model
model = StoryPointEstimator()
model.load_state_dict(torch.load("story_point_model.pth"))
model.eval()

# FastAPI App
app = FastAPI()

@app.post("/predict")
def predict_story_point(request: StoryPointRequest):
    """Predict story points from title & description."""
    text = request.title + " " + request.description
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()
        prediction = model(torch.tensor(embeddings).unsqueeze(0)).item()
    
    return {"predicted_story_points": round(prediction, 2)}

@app.get("/")
def root():
    return {"message": "Story Point Estimation API is running!"}

