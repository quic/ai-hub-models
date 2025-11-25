FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ARG DOCKER_USER
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID
ARG DOCKER_GROUP
ARG QAIHM_CI
ARG HF_TOKEN
ARG QAIRT_SDK_PATH
ARG QDC_API_TOKEN
ARG QAIHM_TEST_MODELS
ARG QAIHM_JUNIT_XML_PATH
ARG QAIHM_MODELS_JUNIT_XML_PATH
ARG GITHUB_STEP_SUMMARY

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    ca-certificates \
    tmux \
    vim \
    htop \
    zip \
    unzip \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev

# Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Add symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | bash && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Create a user and set up home directory
RUN useradd -m -u $DOCKER_USER_ID -d /home/$DOCKER_USER -s /bin/bash $DOCKER_USER

# Set ownership and permissions
RUN chown -R $DOCKER_USER:$DOCKER_USER /home/$DOCKER_USER
RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_GROUP


# Set environment variables
USER $DOCKER_USER
ENV QAIHM_CI=$QAIHM_CI
ENV HF_TOKEN=$HF_TOKEN
ENV AWS_DEFAULT_REGION=us-west-2
ENV AWS_REGION=us-west-2
ENV QAIRT_SDK_PATH=$QAIRT_SDK_PATH
ENV QDC_API_TOKEN=$QDC_API_TOKEN
ENV QAIHM_TEST_MODELS=$QAIHM_TEST_MODELS
ENV QAIHM_JUNIT_XML_PATH=$QAIHM_JUNIT_XML_PATH
ENV QAIHM_MODELS_JUNIT_XML_PATH=$QAIHM_MODELS_JUNIT_XML_PATH
ENV GITHUB_STEP_SUMMARY=$GITHUB_STEP_SUMMARY
