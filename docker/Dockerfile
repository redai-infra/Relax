ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.9
ARG TRAIN_IMAGE=train
FROM ${BASE_IMAGE} as base

ENV NVIDIA_PYTORCH_VERSION=25.06 \
    PYTORCH_BUILD_VERSION=2.9.0 \
    TARGETARCH=amd64 \
    PYVER=3.12 \
    PYSFX=3 \
    L4T=0 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TZ=Asia/Shanghai\
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8\
    OPENCV_VERSION=4.7.0 \
    DEBIAN_FRONTEND=noninteractive

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    build-essential patchelf tmux net-tools htop nano iputils-ping \
    autoconf automake libatlas-base-dev libgoogle-glog-dev libbz2-dev libc-ares2 \
    libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev libtool nasm protobuf-compiler pkg-config unzip sox \
    libsndfile1 libpng-dev libhdf5-dev gfortran rapidjson-dev ninja-build libedit-dev rsync && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y locales tzdata && \
    locale-gen "$LANG" && update-locale LANG="$LANG" LC_ALL="$LC_ALL" && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN cd /tmp && git clone -b update-transformers-v5 https://github.com/sgl-project/sglang.git && \
    cd sglang && rsync -avP ./python/sglang/ /sgl-workspace/sglang/python/sglang/ && \
    rm -rf /tmp/sglang

FROM base as sglang

WORKDIR /sgl-workspace/sglang

RUN pip install transformers==5.3.0 nvidia-cudnn-cu12==9.16.0.29

FROM base as train

RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir && \
    pip install --no-cache-dir flash-linear-attention==0.4.1 && \
    pip install --no-cache-dir tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

RUN MAX_JOBS=64 \
    cd /opt && git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && \
    git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && \
    git submodule update --init && \
    cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf /opt/flash-attention/

RUN pip -v install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.14.1" && \
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall && \
    pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation --no-cache-dir && \
    pip install "numpy<2" nvidia-cudnn-cu12==9.16.0.29 --no-cache-dir && \
    NVCC_APPEND_FLAGS="--threads 32" \
    pip -v install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

ARG TRAIN_IMAGE=train
FROM ${TRAIN_IMAGE} as relax

ARG PATCH_VERSION=latest
ARG ENABLE_SGLANG_PATCH=1

WORKDIR /root

ARG MEGATRON_BRIDGE_COMMIT=2faedbf6fe3c422835a44b2b360cadcb2a116a54
ENV MEGATRON_BRIDGE_COMMIT=${MEGATRON_BRIDGE_COMMIT} \
    PYTHONPATH=/root/Megatron-LM/

RUN rm -rf /root/Megatron-LM && git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && \
    cd /root/Megatron-Bridge/ && git checkout ${MEGATRON_BRIDGE_COMMIT} && \
    git submodule update --init --recursive && ./scripts/switch_mcore.sh dev && \
    mkdir /root/Megatron-LM &&\
    cp -r /root/Megatron-Bridge/src/megatron /root/Megatron-LM/ && \
    rsync -avP /root/Megatron-Bridge/3rdparty/Megatron-LM/megatron/ /root/Megatron-LM/megatron/ && \
    rm -rf /root/Megatron-Bridge

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt --no-cache-dir && \
    pip install --no-cache-dir tensordict==0.10.0 pyvers==0.1.0 --no-deps && \
    pip install "transferqueue @ git+https://github.com/redai-infra/TransferQueue.git" --no-deps

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    patch -p1 < /root/Megatron-LM/megatron.patch && \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    rm megatron.patch && \
    apt update && apt install -y jq

COPY docker/patch/${PATCH_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
    cd /sgl-workspace/sglang && git add . && \
    git update-index --refresh && \
    git apply sglang.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
    fi && \
    rm sglang.patch; \
fi

# FROM relax as relax-release

# WORKDIR /root/Relax

# COPY . .

# RUN pip install -e . --no-deps
