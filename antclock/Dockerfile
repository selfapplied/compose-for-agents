FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with trusted host flags (for environments with SSL issues)
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Try to install AntClock from GitHub, but don't fail if it's not available
RUN pip install --no-cache-dir git+https://github.com/selfapplied/antclock.git || \
    echo "Warning: Could not install AntClock from GitHub. Demo will run in fallback mode."

# Copy application code
COPY agent.py .

# Make agent executable
RUN chmod +x agent.py

# Run the agent
CMD ["python", "agent.py"]
