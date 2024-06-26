# pull official base image
FROM python:3.11-slim-bullseye

# set work directory
WORKDIR /src/app

# https://vsupalov.com/docker-arg-env-variable-guide/
# https://bobcares.com/blog/debian_frontendnoninteractive-docker/
ARG DEBIAN_FRONTEND=noninteractive
# Timezone
ENV TZ="Asia/Bangkok"
RUN apt clean 
RUN apt update && apt upgrade -y
# Set timezone
RUN apt install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Set locales
# https://leimao.github.io/blog/Docker-Locale/
RUN apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8 
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip

RUN apt-get update 
RUN apt-get install -y \
  python-tk \
  python3-tk \
  gcc \
  build-essential \
  python3-dev \
  && apt-get clean

# Copy requirements.txt
COPY ./requirements.txt .
# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# copy project
# COPY ./src /src/app

# clean apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# clean pip cache
RUN pip cache purge
CMD tail -f /dev/null
