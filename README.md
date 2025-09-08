# face_processing_pipeline

Utilities and pipeline for processing faces.


## Overview

This repo performs two main tasks:

- (1) it exposes `FaceProcessor` which can be used for identifying high-quality forward-looking faces, demonstrated in `pipeline_test.py`
- (2) it exposes various utilities for face morphing, demonstrated in `morph_test.py`

It also implements `swarm` the face blending project.


## Setup

MacOS (testing):

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements_macos.txt`
- `pip install git+https://github.com/ageitgey/face_recognition_models`

Raspbian:

- `python3 -m venv --system-site-packages .venv`
- `source .venv/bin/activate`
```
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libjpeg-dev libtiff-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y python3-dev python3-pip
sudo apt install -y libboost-all-dev
sudo apt install libcap-dev
sudo apt install -y cmake
```
- `python3 -m pip install --upgrade pip setuptools wheel`
- `python3 -m pip install --extra-index-url https://www.piwheels.org/simple dlib --no-cache-dir`
- `python3 -m pip install -r requirements_pi.txt`
- `python3 -m pip install git+https://github.com/ageitgey/face_recognition_models`

Ubuntu:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python3 -m pip install -r requirements_ubuntu.txt`
- `python3 -m pip install git+https://github.com/ageitgey/face_recognition_models`


## Test

Test the forward face pipeline:
- `python pipeline_test.py`

Test the morphing function:
- `python morph_test.py`


## Run in Production

Start a system service:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`
- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`
- `sudo loginctl enable-linger $(whoami)`

Show the logs:

- `journalctl --user -u display.service`

Clear logs:

- `sudo journalctl --unit=display.service --rotate`
- `sudo journalctl --vacuum-time=1s`


## Swarm

- `python display.py`
