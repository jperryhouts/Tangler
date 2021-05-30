FROM ubuntu:18.04

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG userid="1000"
ARG groupid="1000"
ARG username="tangler"

USER root

RUN useradd -m -u ${userid} -g ${groupid} ${username}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential \
        graphicsmagick graphicsmagick-libmagick-dev-compat

WORKDIR /tmp

RUN git clone https://github.com/jperryhouts/raveler.git && \
    cd raveler && \
    make cli && \
    mv build/cli/raveler /usr/bin

USER ${username}

WORKDIR /home/${username}
