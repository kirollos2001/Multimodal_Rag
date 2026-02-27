FROM python:3.10-slim

WORKDIR /app


COPY requirements.txt .
# Install PyTorch CPU first to avoid downloading massive CUDA binaries
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
