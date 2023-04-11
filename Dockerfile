FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY model.h5 model.h5
COPY app.py app.py
COPY support.py support.py

RUN mkdir .streamlit
COPY .streamlit/config.toml .streamlit/config.toml

RUN pip3 install -r requirements.txt

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]