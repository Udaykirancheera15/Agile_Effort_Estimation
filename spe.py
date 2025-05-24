#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:20:40 2025

@author: cheera
"""

import pandas as pd
import os
dataset_path = "datasets/Choet_Dataset/MESOS_deep-se.csv"
df = pd.read_csv(os.path.expanduser(dataset_path))
print(df.info())
print(df.head())

import re
import pandas as pd
def clean_text(text):
    """Cleans text by removing special characters and extra spaces."""
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # special characters
    return text.strip()
df["description"].fillna("", inplace=True)
df["text"] = df["title"] + " " + df["description"]
df["text"] = df["text"].apply(clean_text)
df = df[["text", "storypoint"]]
print(df.head())

import spacy
nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    """Tokenizes and lemmatizes text using Spacy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
df["text"] = df["text"].apply(lemmatize_text)
print(df.head())

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import numpy as np
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
def get_bert_embeddings(text_list):
    """Converts text into BERT embeddings."""
    embeddings = []
    for text in tqdm(text_list, desc="Encoding Text"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())  # CLS token representation
    return np.array(embeddings)
X = get_bert_embeddings(df["text"].tolist())
y = df["storypoint"].values
np.save("X_bert_embeddings.npy", X)
np.save("y_storypoints.npy", y)
print("✅ BERT embeddings successfully generated and saved!")

from sklearn.model_selection import train_test_split
X = np.load("X_bert_embeddings.npy")
y = np.load("y_storypoints.npy")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Testing set: {X_test.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
class StoryPointEstimator(nn.Module):
    def __init__(self, input_dim=768):
        super(StoryPointEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Output: Story Points
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # Regression output

model = StoryPointEstimator()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✅ Model initialized!")

import torch
from torch.utils.data import DataLoader, TensorDataset
# NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
# DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("✅ Data converted to PyTorch tensors & DataLoaders created!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_predictions = model(X_val_batch)
            val_loss += criterion(val_predictions, y_val_batch).item()
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
print("✅ Model training complete!")

import pandas as pd
import torch
model.eval()
data_list = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch).cpu().numpy().flatten()
        
        for actual, predicted, text in zip(y_batch.numpy().flatten(), predictions, X_test):
            data_list.append([actual, round(predicted, 2), text])
df_results = pd.DataFrame(data_list, columns=["Actual Story Point", "Predicted Story Point", "Text Description"])
pd.set_option("display.max_colwidth", 100)  # Ensure text is visible
print(df_results.head(10))  # Show first 10 predictions

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model.eval()
# Store true vs. predicted values
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch).cpu().numpy().flatten()

        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(predictions)

# Convert lists to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"✅ Model Evaluation Metrics:\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

torch.save(model.state_dict(), "story_point_model.pth")
print("✅ Model saved as story_point_model.pth!")

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color="green", edgecolor="black", alpha=0.7)
plt.axvline(x=0, color="red", linestyle="--", label="Zero Error")
plt.xlabel("Prediction Error (Residual)")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.legend()
plt.show()
