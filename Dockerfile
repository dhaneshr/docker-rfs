FROM rocker/r-ver:4.4.0

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
    libxt-dev \
    libcurl4 \
    libpng-dev \
    liblz4-dev \
    libdeflate-dev && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"

# Install R dependencies
COPY R_dependencies.R /app/R_dependencies.R
RUN Rscript /app/R_dependencies.R && \
    Rscript -e "if (!requireNamespace('rms', quietly = TRUE) || !requireNamespace('riskRegression', quietly = TRUE) || !requireNamespace('prodlim', quietly = TRUE)) { stop('One or more R packages failed to install.') }"

ENV R_LIBS_SITE="/usr/local/lib/R/site-library"

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN conda create -n fastapi-env python=3.10 -y && \
    conda run -n fastapi-env pip install -r /app/requirements.txt && \
    conda clean -afy

# Set work directory and copy application files
WORKDIR /app
COPY app.py /app/
COPY fgr-python.py /app/
COPY FGR_predict.R /app/
COPY final_FGR_clean.RData /app/
COPY resources/ /app/resources/
COPY templates/ /app/templates/
COPY local_data /app/local_data
COPY static/ /app/static/

# Set permissions
RUN chmod -R 755 /app/resources /app/local_data /app/static

# Expose port for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app
CMD ["/bin/bash", "-c", "source activate fastapi-env && uvicorn app:app --host 0.0.0.0 --port 8000"]
