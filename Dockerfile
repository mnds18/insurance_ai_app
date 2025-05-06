# Dockerfile for insurance_ai_app with Streamlit UI and MLflow UI

FROM python:3.11-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . /app

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install streamlit mlflow

# Expose Streamlit and MLflow ports
EXPOSE 8501 5000

# Command to launch both MLflow UI and Streamlit
CMD streamlit run streamlit_app.py & \
    mlflow ui --port 5000 --host 0.0.0.0
