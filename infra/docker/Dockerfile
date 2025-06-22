# Dockerfile for NeuroForge
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY frontend/ ./frontend
COPY infra/ ./infra
COPY middleware/ ./middleware
COPY hardware_profiles/ ./hardware_profiles
COPY mock_data/ ./mock_data
COPY agents/ ./agents
COPY tests/ ./tests
COPY pipeline.py .
COPY README.md .


# Run the Streamlit dashboard
CMD ["streamlit", "run", "frontend/dashboard.py"]