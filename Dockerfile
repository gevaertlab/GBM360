FROM python:3.9-slim

WORKDIR /GBM360-Lite

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main_app.py", "--server.maxUploadSize=3072", "--server.port=8501", "--server.address=0.0.0.0"]
