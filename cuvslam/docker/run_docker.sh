#!/bin/bash

# Parse command line arguments
UBUNTU_VERSION="${1:-22}"

# Validate input
if [[ "$UBUNTU_VERSION" != "22" && "$UBUNTU_VERSION" != "24" ]]; then
    echo "Usage: $0 [22|24]"
    echo "  22 - Ubuntu 22.04 with CUDA 12 (default)"
    echo "  24 - Ubuntu 24.04 with CUDA 13"
    exit 1
fi

# Set image tag based on selection
if [[ "$UBUNTU_VERSION" == "24" ]]; then
    IMAGE_TAG="pycuvslam:realsense-cu13"
    echo "Selected: Ubuntu 24.04 with CUDA 13"
else
    IMAGE_TAG="pycuvslam:realsense-cu12"
    echo "Selected: Ubuntu 22.04 with CUDA 12"
fi

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# On Jetson/aarch64 with CUDA 12, mount host CUDA since container CUDA headers may be incomplete
CUDA_MOUNT=""
if [[ "$ARCH" == "aarch64" && "$UBUNTU_VERSION" == "22" ]]; then
    if [[ -d "/usr/local/cuda-12.6" ]]; then
        echo "Jetson detected (CUDA 12): mounting host CUDA from /usr/local/cuda-12.6"
        CUDA_MOUNT="-v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro"
    fi
fi

# Ensure X11 forwarding is set up properly
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# Create xauth file if it doesn't exist
touch $XAUTH
xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - 2>/dev/null
chmod 777 $XAUTH

# Set up datasets path (create if doesn't exist)
DATASETS=$(realpath -s ~/datasets 2>/dev/null || echo ~/datasets)
mkdir -p "$DATASETS"

# Allow X11 connections from localhost
xhost +local:docker 2>/dev/null

echo "Starting cuVSLAM Docker container with RealSense support..."
echo "Image: $IMAGE_TAG"
echo "X11 forwarding enabled for GUI applications"
echo "Datasets directory: $DATASETS"

docker run -it \
  --runtime=nvidia \
  --rm \
  --gpus all \
  --privileged \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=$XAUTH \
  -e QT_X11_NO_MITSHM=1 \
  -e _X11_NO_MITSHM=1 \
  -e _MITSHM=0 \
  -v $XSOCK:$XSOCK \
  -v $XAUTH:$XAUTH \
  -v .:/cuvslam \
  -v "$DATASETS":"$DATASETS" \
  $(for dev in /dev/video*; do [ -e "$dev" ] && echo "--device=$dev:$dev"; done) \
  -v /dev/bus/usb:/dev/bus/usb \
  $CUDA_MOUNT \
  "$IMAGE_TAG"

# Clean up X11 permissions
xhost -local:docker 2>/dev/null
