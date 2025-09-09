#
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        nvidia-utils-450 \
        git \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        iputils-ping \
        curl \
        nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 
WORKDIR ./LLM

# 
COPY ./requirements.txt /LLM/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /LLM/requirements.txt

# 
COPY ./app /LLM/app

# 
EXPOSE 8003

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]