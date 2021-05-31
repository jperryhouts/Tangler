FROM tensorflow/tensorflow:latest
#FROM ubuntu:18.04

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget \
        parallel git build-essential ca-certificates \
        graphicsmagick-libmagick-dev-compat

WORKDIR /tmp

RUN wget http://www.imagemagick.org/download/ImageMagick.tar.gz && \
    tar -xvf ImageMagick.tar.gz && \
    cd ImageMagick-* && \
    ./configure --prefix=/usr && \
    make && \
    make install

RUN git clone https://github.com/jperryhouts/raveler.git && \
    cd raveler && \
    make cli && \
    mv build/cli/raveler /usr/bin

RUN mkdir -p /var/tangler

COPY do_ravel.sh /var/tangler/do_ravel.sh
COPY make_tfrecords.py /var/tangler/make_tfrecords.py

WORKDIR /var/tangler
