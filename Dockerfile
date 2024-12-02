FROM rocker/r-ver:4.4.0

# Pass the architecture as an argument
ARG TARGETARCH
ENV TARGETARCH=$TARGETARCH

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
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    wget \
    libxt-dev \
    libcurl4 \
    libpng-dev \
    liblz4-dev \
    libdeflate-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add architecture-specific logic (optional)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "Installing ARM64-specific libraries or packages"; \
    else \
        echo "Installing AMD64-specific libraries or packages"; \
    fi

# Update Python alternatives to point to Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install R dependencies
COPY R_dependencies.R /app/R_dependencies.R
RUN Rscript /app/R_dependencies.R && \
    Rscript -e "if (!requireNamespace('rms', quietly = TRUE) || !requireNamespace('riskRegression', quietly = TRUE) || !requireNamespace('prodlim', quietly = TRUE)) { stop('One or more R packages failed to install.') }"

ENV R_LIBS_SITE="/usr/local/lib/R/site-library"

# Create a Python virtual environment and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r /app/requirements.txt

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
CMD ["/bin/bash", "-c", "source /app/venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000"]
