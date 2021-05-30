FROM ubuntu:18.04

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ca-certificates \
        graphicsmagick graphicsmagick-libmagick-dev-compat

WORKDIR /tmp

RUN git clone https://github.com/jperryhouts/raveler.git && \
    cd raveler && \
    make cli && \
    mv build/cli/raveler /usr/bin

RUN mkdir -p /var/tangler

COPY Makefile.image /var/tangler/Makefile

WORKDIR /var/tangler
