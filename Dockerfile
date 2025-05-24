# Use a specific Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY story_point_model.pth .
COPY requirements.txt .

# Install dependencies with the correct Python version
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

