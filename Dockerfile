FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add requests for Telegram notifications
RUN pip install --no-cache-dir requests

# Copy application code
COPY . .

# Create directories for data and results
RUN mkdir -p data results logs

# Environment variable defaults
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""

# Ensure head-less matplotlib backend inside the container
ENV MPLBACKEND=Agg

# Entry point
ENTRYPOINT ["python", "dql_trading.py"] 