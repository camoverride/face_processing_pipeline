# face_processing_pipeline

Utilities and pipeline for processing faces.


## Overview

This repo performs two main tasks:

- (1) it exposes `FaceProcessor` which can be used for identifying high-quality forward-looking faces, demonstrated in `pipeline_test.py`
- (2) it exposes various utilities for face morphing, demonstrated in `morph_test.py`

It also implements `swarm` the face blending project.


## Setup

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install setuptools`
- `pip install git+https://github.com/ageitgey/face_recognition_models`


## Test

Test the forward face pipeline:
- `python pipeline_test.py`

Test the morphing function:
- `python morph_test.py`


## Swarm

- `python display.py`


## TODO's

- [ ] Performance upgrade: allow for fewer landmarks (both in `FaceProcesser` and other utils) [link](https://github.com/camoverride/swarm/issues/1)
- [ ] Performance upgrade: allow for weaker face detection model as alternative to `MTCNN` [link](https://github.com/camoverride/swarm/issues/2)
