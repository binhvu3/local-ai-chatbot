# Dockerfile

FROM python:3.10-slim

# 1) Install system tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      tar \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Install Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3) Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4) Copy application code
COPY . .
RUN chmod +x run_tests.sh

# 5) Create uploads directory
RUN mkdir -p uploads && chmod 777 uploads

EXPOSE 5001

# 6) Launch Ollama daemon, pre-pull model, init DB, and start Flask
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama pull deepseek-r1:1.5b && flask init_db && flask run --host=0.0.0.0 --port=5001"]