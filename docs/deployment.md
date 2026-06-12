# IntelliGraph-RAG — Deployment & Production Guide

This guide explains how to install, deploy, and containerize the **IntelliGraph-RAG** platform in local development or production environments.

---

## 1. Local Virtual Environment Setup

### System Dependencies
The platform uses **Tesseract OCR** as a fallback for scanned files. You must install the Tesseract binary on your host machine:

* **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr libgl1-mesa-glx libglib2.0-0 curl
  ```
* **macOS**:
  ```bash
  brew install tesseract
  ```
* **Windows**:
  - Download and run the installer from the GitHub Tesseract Wiki.
  - Add the installation path (usually `C:\Program Files\Tesseract-OCR`) to your system's Environment Variables `PATH`.

### Application Setup
```bash
# Clone and enter the project folder
git clone https://github.com/your-username/IntelliGraph-RAG.git
cd IntelliGraph-RAG

# Create and activate python virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On macOS/Linux

# Install libraries
pip install -r requirements.txt
```

---

## 2. Docker Containerization

The platform includes a **Multi-stage Dockerfile** that optimizes build cache dependencies and keeps the final image size minimized:

### Dockerfile Summary
* **Stage 1 (`builder`)**: Uses `python:3.11-slim` to compile wheels and install virtualenv dependencies.
* **Stage 2 (`runner`)**: Uses `python:3.11-slim` to pull the compiled virtualenv, installs the host binary `tesseract-ocr` and OpenGL layout libraries, copies the code layers, and starts Streamlit.

### docker-compose.yml Orchestration
To persist data stores, sessions, and pricing settings across container lifecycles, map the local storage volume inside the container:

```yaml
version: '3.8'

services:
  rag-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: intelligent-doc-assistant
    ports:
      - "8501:8501"
    volumes:
      - ./rag_storage:/app/rag_storage  # Volume mount maps host storage
    env_file:
      - .env                            # Injects credentials from local secrets file
    restart: unless-stopped
```

### Launch Commands
```bash
# Build and run containers in background
docker-compose up --build -d

# Check container status
docker-compose ps

# View execution logs
docker-compose logs -f
```

---

## 3. Storage Persistence Architecture

All state folders are mapped locally under the `./rag_storage/` folder, structured as follows:

```text
rag_storage/
├── store.json              # Chunks metadata and document index states
├── faiss.index             # Compressed FAISS index binary
├── graph.json              # NetworkX directed graph database JSON
├── platform.log            # System execution traces
├── telemetry.jsonl         # Observability dashboard metrics logs
├── pricing_config.json     # Saved LLM pricing configurations
└── sessions/               # Chat histories JSON files
```

Always ensure the user running the Docker daemon or Streamlit server has write permissions on the directory. On Linux, grant ownership if needed:
```bash
sudo chown -R 1000:1000 ./rag_storage
```
