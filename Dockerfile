FROM python:3.9-slim

EXPOSE 8502

WORKDIR /GBM360-STREAMLIT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "spatial_app.py", "--server.maxUploadSize=3072", "--server.port=8502", "--server.address=0.0.0.0"]
