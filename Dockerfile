# Use a base image with R version 4.3.3
FROM rocker/r-ver:4.3.3

# Set environment variables to avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Miniconda
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
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && rm -rf /var/lib/apt/lists/*

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN conda create -n fastapi-env python=3.10 -y \
    && conda run -n fastapi-env pip install -r /app/requirements.txt

# Install R dependencies using R script
COPY R_dependencies.R /app/R_dependencies.R
RUN Rscript /app/R_dependencies.R

# Set the working directory and copy necessary files
WORKDIR /app
COPY app.py /app/
COPY follic_model.RData /app/
COPY resources/ /app/resources/
COPY templates/ /app/templates/
COPY local_data /app/local_data

# Set permissions for the `resources` and `local_data` directories
RUN chmod -R 755 /app/resources /app/local_data

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["conda", "run", "--no-capture-output", "-n", "fastapi-env", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
