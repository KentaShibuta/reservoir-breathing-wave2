FROM python:3.12
USER root

WORKDIR /root/app/
COPY requirements.txt /root/app/

RUN apt -y update && \
    apt -y install libopencv-dev && \
    apt install -y cmake && \
    apt install -y ninja-build && \
    apt install -y patchelf && \
    apt install -y clang libc++-dev libc++abi-dev
RUN pip install --upgrade pip && \
    pip install -r /root/app/requirements.txt

ENV LD_LIBRARY_PATH=/root/app/lib/opencv-4.x/build/lib:/root/app/lib/onnxruntime/build/Linux/Release