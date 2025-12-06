# Mood Ring - Production Dockerfile
# For persistent GPU deployment (RunPod, Vast.ai, Lambda Labs, etc.)

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies + uv
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies with uv (much faster than pip)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY vectors/ ./vectors/

# Pre-download model (optional - can also mount from volume)
# RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-14B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-14B')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/config || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

