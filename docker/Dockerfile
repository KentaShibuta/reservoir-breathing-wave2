FROM python:3.12
USER root

WORKDIR /root/app/
COPY requirements.txt /root/app/

RUN apt -y update && \
    apt -y install libopencv-dev
RUN pip install --upgrade pip && \
    pip install -r /root/app/requirements.txt