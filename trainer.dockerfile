# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/figures/ reports/figures/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torchvision==0.10.0 -f https://download.pytorch.org/whl/torchvision/ 
RUN pip install libpng-bins

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
