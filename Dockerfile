# # Use an Ubuntu base image
# FROM ubuntu:22.04
FROM rocker/r-ver:4.3.3

# Set environment variables to avoid user interaction during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Create a directory for data storage
RUN mkdir /data

# Optional: Set appropriate permissions
RUN chmod 755 /data


# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libgomp1 \
    libblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpcre2-dev \
    libbz2-dev \
    liblzma-dev \
    libz-dev \
    libreadline-dev \
    libncurses5-dev \
    libgfortran5 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment
RUN conda create -n fastapi-env python=3.10 -y

# Copy requirements.txt before installing dependencies
COPY requirements.txt /app/requirements.txt

# Activate the environment and install Python dependencies, including PyTorch
RUN /bin/bash -c "source activate fastapi-env && \
    pip install -r /app/requirements.txt"

# Install R packages
COPY R_dependencies.R /R_dependencies.R
RUN Rscript /R_dependencies.R

# Set the working directory
WORKDIR /app

# Copy the entire app directory (including app.py and R model)
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["bash", "-c", "source activate fastapi-env && uvicorn app:app --host 0.0.0.0 --port 8000"]