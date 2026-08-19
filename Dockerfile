FROM ros:jazzy-ros-base

ENV ROS_DISTRO=jazzy

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
can-utils \
iproute2 \
python3-pip \
libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /root/workspace/

CMD ["bash"]
