
# Limb Amputation Risk Predictor Model

## 
This repository contains a Dockerfile for building and deploying a containerized R and Python for the limb amputation risk prediction model based on Random Survival Forest for Competing Risk / Fine-Gray Subdistribution Risk Model. It supports multi-platform builds for both `AMD64` and `ARM64` architectures using Docker or Podman.

## Features

- R environment with necessary dependencies (`rocker/r-ver:4.4.0` as base image).
- Python 3.10 setup with virtual environment and package installation.
- Multi-platform support for AMD64 and ARM64 architectures.
- Configurable build process for architecture-specific dependencies.

---

## Prerequisites

1. Install either **Docker** or **Podman**.
2. Enable **Docker Buildx** for Docker users to support multi-platform builds.

---

## Building the Image

### **Using Docker**

#### 1. Enable Docker Buildx (if not already set up)

Check if Buildx is installed:
```bash
docker buildx version
```

If not, create and enable a builder instance:
```bash
docker buildx create --use
```

#### 2. Build the Image

##### Build for a Specific Architecture
- For `AMD64`:
  ```bash
  docker buildx build --platform linux/amd64 -t your-image-name:amd64 .
  ```
- For `ARM64`:
  ```bash
  docker buildx build --platform linux/arm64 -t your-image-name:arm64 .
  ```

##### Multi-Platform Build
To build and push a single multi-platform image:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --push .
```

##### Save the Image Locally
If you want to use the image locally instead of pushing to a registry:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --load .
```

---

### **Using Podman**

#### 1. Enable Podman Buildx (if not already set up)

Verify Buildx support in Podman:
```bash
podman buildx version
```

#### 2. Build the Image

##### Build for a Specific Architecture
- For `AMD64`:
  ```bash
  podman buildx build --platform linux/amd64 -t your-image-name:amd64 .
  ```
- For `ARM64`:
  ```bash
  podman buildx build --platform linux/arm64 -t your-image-name:arm64 .
  ```

##### Multi-Platform Build
To build and push a single multi-platform image:
```bash
podman buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --push .
```

##### Save the Image Locally
If you want to use the image locally instead of pushing to a registry:
```bash
podman buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --load .
```

---

## Running the Container

### Using Docker
```bash
docker run -d -p 8000:8000 --name your-container-name your-image-name:multiarch
```

### Using Podman
```bash
podman run -d -p 8000:8000 --name your-container-name your-image-name:multiarch
```

Once the container is running, the application will be accessible at `http://localhost:8000`.

---

## Development Tips

### Cleaning Up Unused Resources
To remove unused Docker/Podman resources:
```bash
docker system prune --all       # For Docker
podman system prune --all       # For Podman
```

### Debugging Multi-Platform Images
To verify the architecture of your built image:
```bash
docker run --rm your-image-name:amd64 uname -m    # Expected output: x86_64
docker run --rm your-image-name:arm64 uname -m    # Expected output: aarch64
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, suggest improvements, or report issues!

--- 

### Example Commands Summary
| Task                                  | Docker Command                                                                                  | Podman Command                                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Build for AMD64                       | `docker buildx build --platform linux/amd64 -t your-image-name:amd64 .`                       | `podman buildx build --platform linux/amd64 -t your-image-name:amd64 .`                      |
| Build for ARM64                       | `docker buildx build --platform linux/arm64 -t your-image-name:arm64 .`                       | `podman buildx build --platform linux/arm64 -t your-image-name:arm64 .`                      |
| Build Multi-Arch Image                | `docker buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --push .` | `podman buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --push .` |
| Save Multi-Arch Image Locally         | `docker buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --load .` | `podman buildx build --platform linux/amd64,linux/arm64 -t your-image-name:multiarch --load .` |
| Run the Container Locally             | `docker run -d -p 8000:8000 --name your-container-name your-image-name:multiarch`             | `podman run -d -p 8000:8000 --name your-container-name your-image-name:multiarch`             |


