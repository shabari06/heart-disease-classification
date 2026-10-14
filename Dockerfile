# Use a Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application and trained model
COPY app.py .
COPY heart_model.pkl .

# Expose the FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]