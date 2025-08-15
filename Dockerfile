# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Java for tabula-py, other tools)
RUN apt-get update && apt-get install -y \
    default-jre \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads and outputs directories
RUN mkdir -p uploads outputs logs

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PORT=5000

# Expose port (will be overridden by hosting platform)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application (use PORT environment variable for hosting platforms)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --keep-alive 2 app:app
