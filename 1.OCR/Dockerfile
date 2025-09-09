#
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        iputils-ping \
        curl \
        poppler-utils \
        nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ./OCR

# Copy file requirements.txt
COPY ./requirements.txt /OCR/requirements.txt

# Install Dependencies
RUN pip install --no-cache-dir -r /OCR/requirements.txt

# Copy semua file ke working directory
COPY ./app /OCR/app

# Expose port yang digunakan oleh aplikasi FastAPI
EXPOSE 8001

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]