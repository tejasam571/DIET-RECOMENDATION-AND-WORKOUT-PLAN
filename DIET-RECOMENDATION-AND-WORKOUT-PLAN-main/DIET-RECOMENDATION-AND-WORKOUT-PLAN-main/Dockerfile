# Use Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from local machine into /app inside container
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Streamlit’s default port
EXPOSE 8501

# Healthcheck to verify the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "diet.py", "--server.port=8501", "--server.address=0.0.0.0"]
