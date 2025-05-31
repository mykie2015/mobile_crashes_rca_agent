# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config/ ./config/
COPY src/ ./src/
COPY data/ ./data/
COPY examples/ ./examples/
COPY setup.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p logs data/cache data/outputs data/outputs/images

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port (if needed for web interface)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.mobile_crash_agent"] 